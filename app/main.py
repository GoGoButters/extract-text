"""
Text Extraction API for RAG.

Main FastAPI application module
"""

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.extractors import TextExtractor
from app.utils import (
    cleanup_recent_temp_files,
    cleanup_temp_files,
    sanitize_filename,
    setup_logging,
    validate_file_type,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants for FastAPI arguments
FILE_UPLOAD = File(...)

# Initialize text extractor
text_extractor = TextExtractor()


# Pydantic models
class Base64FileRequest(BaseModel):
    """Model for base64 file processing request."""

    encoded_base64_file: str = Field(
        "0J/RgNC40LLQtdGCINC80LjRgCEg0K3RgtC+INGC0LXRgdGCIGJhc2U2NArQntGH0LXQvdGMINC00LvQuNC90L3Ri9C5LCDRgSDQv9C10YDQtdC90L7RgdC+0Lwg0YHRgtGA0L7Qui4=",
        description="Base64 encoded file",
    )
    filename: str = Field("test.txt", description="Filename with extension")


class ExtractionOptions(BaseModel):
    """Text extraction settings for web pages (new in v1.10.2)."""

    # JavaScript and rendering
    enable_javascript: Optional[bool] = Field(
        True, description="Enable/disable JavaScript rendering"
    )
    js_render_timeout: Optional[int] = Field(
        10, description="JS rendering timeout in seconds"
    )
    web_page_delay: Optional[int] = Field(
        3, description="Delay after JS load in seconds"
    )

    # Lazy Loading
    enable_lazy_loading_wait: Optional[bool] = Field(
        True, description="Enable lazy loading wait"
    )
    max_scroll_attempts: Optional[int] = Field(
        3, description="Maximum scroll attempts"
    )

    # Image processing
    process_images: Optional[bool] = Field(
        True, description="Process images via OCR"
    )
    enable_base64_images: Optional[bool] = Field(
        True, description="Process base64 images"
    )
    min_image_size_for_ocr: Optional[int] = Field(
        22500, description="Minimum image size for OCR (pixels)"
    )
    max_images_per_page: Optional[int] = Field(
        20, description="Maximum images per page"
    )

    # Timeouts
    web_page_timeout: Optional[int] = Field(
        30, description="Page load timeout in seconds"
    )
    image_download_timeout: Optional[int] = Field(
        15, description="Image download timeout in seconds"
    )

    # Network settings
    follow_redirects: Optional[bool] = Field(
        True, description="Follow redirects"
    )
    max_redirects: Optional[int] = Field(
        5, description="Maximum redirects"
    )


class URLRequest(BaseModel):
    """Model for web page processing request (updated in v1.10.2)."""

    url: str = Field(
        "https://habr.com/ru/companies/softonit/articles/911520/",
        description="Web page URL for text extraction",
    )
    user_agent: Optional[str] = Field(
        "Text Extraction Bot 1.0",
        description="Custom User-Agent (optional, for backward compatibility)",
    )
    extraction_options: Optional[ExtractionOptions] = Field(
        None, description="Text extraction settings (optional)"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    logger.info(f"Starting Text Extraction API v{settings.VERSION}")

    # Clean up temporary files on startup
    cleanup_temp_files()

    yield

    # Graceful shutdown: properly close thread pool
    logger.info("Shutting down Text Extraction API")
    try:
        if hasattr(text_extractor, "_thread_pool"):
            logger.info("Closing thread pool...")
            text_extractor._thread_pool.shutdown(wait=True)
            logger.info("Thread pool closed successfully")
    except Exception as e:
        logger.warning(f"Error closing thread pool: {str(e)}")

    # Final cleanup of temporary files
    try:
        cleanup_temp_files()
    except Exception as e:
        logger.warning(f"Error during final cleanup: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Text Extraction API for RAG",
    description="API for extracting text from files in various formats",
    version=settings.VERSION,
    lifespan=lifespan,
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    contact={
        "name": "Vitalii Barilko",
        "email": "support@softonit.ru",
        "url": "https://softonit.ru",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware for request logging."""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url} "
            f"in {process_time:.3f}s"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error processing request {request.method} {request.url} "
            f"in {process_time:.3f}s: {str(e)}"
        )
        raise


@app.get("/")
async def root() -> Dict[str, str]:
    """API information."""
    return {
        "api_name": "Text Extraction API for RAG",
        "version": settings.VERSION,
        "contact": "Vitalii Barilko",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """API health check."""
    return {"status": "ok"}


@app.get("/v1/supported-formats")
async def supported_formats() -> Dict[str, list]:
    """Supported file formats."""
    return settings.SUPPORTED_FORMATS


@app.post("/v1/extract/file")
async def extract_text(file: UploadFile = FILE_UPLOAD):
    """Extract text from file."""
    try:
        # Sanitize filename
        original_filename = file.filename or "unknown_file"
        safe_filename_for_processing = sanitize_filename(original_filename)

        logger.info(f"Received file for processing: {original_filename}")

        # Check file size header (DoS protection)
        if file.size is None:
            logger.warning(
                f"File {original_filename} missing Content-Length header"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "Missing Content-Length header. Please ensure file size is specified in request.",
                },
            )

        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            logger.warning(
                f"File {original_filename} too large: {file.size} bytes"
            )
            raise HTTPException(
                status_code=413, detail="File size exceeds maximum allowed size"
            )

        # Read file content
        content = await file.read()

        # Check for empty file
        if not content:
            logger.warning(f"File {original_filename} is empty")
            raise HTTPException(status_code=422, detail="File is empty")

        # Check if file extension matches its content
        is_valid, validation_error = validate_file_type(content, original_filename)
        if not is_valid:
            logger.warning(
                f"File {original_filename} failed type validation: {validation_error}"
            )
            return JSONResponse(
                status_code=415,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "File extension does not match its content. Possible file type spoofing.",
                },
            )

        # Text extraction - CRITICAL FIX: perform in thread pool with timeout
        start_time = time.time()
        try:
            extracted_files = await asyncio.wait_for(
                run_in_threadpool(
                    text_extractor.extract_text, content, safe_filename_for_processing
                ),
                timeout=settings.PROCESSING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout processing file {original_filename}: limit of {settings.PROCESSING_TIMEOUT_SECONDS} seconds exceeded"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": f"File processing exceeded the time limit ({settings.PROCESSING_TIMEOUT_SECONDS} seconds).",
                },
            )
        finally:
            # Additional cleanup of temporary files after processing
            try:
                cleanup_recent_temp_files()
            except Exception as cleanup_error:
                logger.warning(
                    f"Error cleaning up temporary files: {str(cleanup_error)}"
                )

        process_time = time.time() - start_time

        # Calculate total text length
        total_text_length = sum(
            len(file_data.get("text", "")) for file_data in extracted_files
        )

        logger.info(
            f"Text successfully extracted from {original_filename} in {process_time:.3f}s. "
            f"Files processed: {len(extracted_files)}, total text length: {total_text_length} characters"
        )

        return {
            "status": "success",
            "filename": original_filename,
            "count": len(extracted_files),
            "files": extracted_files,
        }

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "Unsupported file format" in error_msg:
            logger.warning(f"Unsupported file format: {original_filename}")
            return JSONResponse(
                status_code=415,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "Unsupported file format.",
                },
            )
        else:
            logger.error(
                f"Error processing file {original_filename}: {error_msg}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "File is corrupted or format is not supported.",
                },
            )
    except Exception as e:
        # Determine filename for logging
        filename_for_error = getattr(file, "filename", "unknown_file") or "unknown_file"
        logger.error(
            f"Error processing file {filename_for_error}: {str(e)}", exc_info=True
        )
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "filename": filename_for_error,
                "message": "File is corrupted or format is not supported.",
            },
        )


@app.post("/v1/extract/base64")
async def extract_text_base64(request: Base64FileRequest):
    """Extract text from base64-encoded file."""
    try:
        # Sanitize filename
        original_filename = request.filename
        safe_filename_for_processing = sanitize_filename(original_filename)

        logger.info(f"Received base64 file for processing: {original_filename}")

        # Decode base64
        try:
            content = base64.b64decode(request.encoded_base64_file)
        except Exception as e:
            logger.warning(
                f"Error decoding base64 for file {original_filename}: {str(e)}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "Invalid base64 format. Ensure the file is correctly encoded in base64.",
                },
            )

        # Check file size
        file_size = len(content)
        if file_size > settings.MAX_FILE_SIZE:
            logger.warning(
                f"File {original_filename} too large: {file_size} bytes"
            )
            raise HTTPException(
                status_code=413, detail="File size exceeds maximum allowed size"
            )

        # Check for empty file
        if not content:
            logger.warning(f"File {original_filename} is empty")
            raise HTTPException(status_code=422, detail="File is empty")

        # Check if file extension matches its content
        is_valid, validation_error = validate_file_type(content, original_filename)
        if not is_valid:
            logger.warning(
                f"File {original_filename} failed type validation: {validation_error}"
            )
            return JSONResponse(
                status_code=415,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "File extension does not match its content. Possible file type spoofing.",
                },
            )

        # Text extraction - perform in thread pool with timeout
        start_time = time.time()
        try:
            extracted_files = await asyncio.wait_for(
                run_in_threadpool(
                    text_extractor.extract_text, content, safe_filename_for_processing
                ),
                timeout=settings.PROCESSING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout processing file {original_filename}: limit of {settings.PROCESSING_TIMEOUT_SECONDS} seconds exceeded"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": f"File processing exceeded the time limit ({settings.PROCESSING_TIMEOUT_SECONDS} seconds).",
                },
            )
        finally:
            # Additional cleanup of temporary files after processing base64 file
            try:
                cleanup_recent_temp_files()
            except Exception as cleanup_error:
                logger.warning(
                    f"Error cleaning up temporary files: {str(cleanup_error)}"
                )

        process_time = time.time() - start_time

        # Calculate total text length
        total_text_length = sum(
            len(file_data.get("text", "")) for file_data in extracted_files
        )

        logger.info(
            f"Text successfully extracted from base64 file {original_filename} in {process_time:.3f}s. "
            f"Files processed: {len(extracted_files)}, total text length: {total_text_length} characters"
        )

        return {
            "status": "success",
            "filename": original_filename,
            "count": len(extracted_files),
            "files": extracted_files,
        }

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "Unsupported file format" in error_msg:
            logger.warning(f"Неподдерживаемый формат файла: {original_filename}")
            return JSONResponse(
                status_code=415,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "Неподдерживаемый формат файла.",
                },
            )
        else:
            logger.error(
                f"Ошибка при обработке файла {original_filename}: {error_msg}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "filename": original_filename,
                    "message": "Файл поврежден или формат не поддерживается.",
                },
            )
    except Exception as e:
        logger.error(
            f"Error processing base64 file {original_filename}: {str(e)}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "filename": original_filename,
                "message": "File is corrupted or format is not supported.",
            },
        )


@app.post("/v1/extract/url")
async def extract_text_from_url(request: URLRequest):
    """Extract text from web page (updated in v1.10.2)."""
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Validate URL
    if not url.startswith(("http://", "https://")):
        logger.warning(f"Invalid URL: {url}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "url": url,
                "message": "URL must start with http:// or https://",
            },
        )

    logger.info(f"Starting text extraction from URL: {url}")

    # Use user_agent from the root level
    user_agent = request.user_agent

    try:
        # Extract text in thread pool with timeout
        start_time = time.time()
        try:
            extracted_files = await asyncio.wait_for(
                run_in_threadpool(
                    text_extractor.extract_from_url,
                    url,
                    user_agent,
                    request.extraction_options,
                ),
                timeout=settings.PROCESSING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout processing URL {url}: limit of {settings.PROCESSING_TIMEOUT_SECONDS} seconds exceeded"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "url": url,
                    "message": f"Web page processing exceeded the time limit ({settings.PROCESSING_TIMEOUT_SECONDS} seconds).",
                },
            )

        process_time = time.time() - start_time

        # Calculate total text length
        total_text_length = sum(
            len(file_data.get("text", "")) for file_data in extracted_files
        )

        logger.info(
            f"Text successfully extracted from URL {url} in {process_time:.3f}s. "
            f"Files processed: {len(extracted_files)}, total text length: {total_text_length} characters"
        )

        return {
            "status": "success",
            "url": url,
            "count": len(extracted_files),
            "files": extracted_files,
        }

    except ValueError as e:
        error_msg = str(e)

        # Determine error type for correct HTTP code
        if "internal IP" in error_msg.lower() or "prohibited" in error_msg.lower():
            logger.warning(f"Request to blocked URL {url}: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "url": url,
                    "message": "Access to internal IP addresses is prohibited for security reasons.",
                },
            )
        elif "timeout" in error_msg.lower():
            logger.warning(f"Timeout loading URL {url}: {error_msg}")
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "url": url,
                    "message": "Failed to load page: timeout exceeded.",
                },
            )
        elif "connection" in error_msg.lower() or "failed to load" in error_msg.lower():
            logger.warning(f"Connection error to URL {url}: {error_msg}")
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "url": url,
                    "message": f"Failed to load page: {error_msg}",
                },
            )
        else:
            logger.error(f"Error processing URL {url}: {error_msg}")
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "url": url,
                    "message": f"HTML parsing error: {error_msg}",
                },
            )
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "url": url,
                "message": f"Web page processing error: {str(e)}",
            },
        )



if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        log_level="info",
        reload=settings.DEBUG,
    )
