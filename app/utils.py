"""Application utilities."""

import glob
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# Cross-platform support: resource module only exists on Unix
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    resource = None  # type: ignore
    RESOURCE_AVAILABLE = False

import magic
from werkzeug.utils import secure_filename

from .config import settings

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup structured logging."""
    # Create formatter for logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Setup uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)

    # Disable duplicate logs
    uvicorn_logger.propagate = False


def get_file_extension(filename: str) -> Optional[str]:
    """Get file extension."""
    if not filename or "." not in filename:
        return None

    # Handle composite extensions (tar.gz, tar.bz2, tar.xz)
    filename_lower = filename.lower()
    if filename_lower.endswith(".tar.gz") or filename_lower.endswith(".tgz"):
        return "tar.gz"
    elif filename_lower.endswith(".tar.bz2") or filename_lower.endswith(".tbz2"):
        return "tar.bz2"
    elif filename_lower.endswith(".tar.xz") or filename_lower.endswith(".txz"):
        return "tar.xz"

    return filename.split(".")[-1].lower()


def is_supported_format(filename: str, supported_formats: dict) -> bool:
    """Check if file format is supported."""
    extension = get_file_extension(filename)
    if not extension:
        return False

    for format_group in supported_formats.values():
        if extension in format_group:
            return True

    return False


def is_archive_format(filename: str, supported_formats: dict) -> bool:
    """Check if the file is an archive."""
    extension = get_file_extension(filename)
    if not extension:
        return False

    archives = supported_formats.get("archives", [])
    return extension in archives


def safe_filename(filename: str) -> str:
    """Safe filename for logs."""
    if not filename:
        return "unknown_file"

    # Remove potentially dangerous characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in "._-":
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    return "".join(safe_chars)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for security with support for various character sets.

    Removes dangerous characters for path traversal attacks.
    """
    if not filename:
        return "unknown_file"

    # Remove dangerous characters for path traversal
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Remove other potentially dangerous characters
    dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\0"]
    for char in dangerous_chars:
        filename = filename.replace(char, "")

    # Remove control characters
    filename = "".join(char for char in filename if ord(char) >= 32)

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    # If the filename is empty after sanitization, return a safe name
    if not filename:
        return "sanitized_file"

    # Limit filename length (max 255 characters for most FS)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        # Preserve extension and truncate name
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext

    return filename


def validate_file_type(content: bytes, filename: str) -> tuple[bool, Optional[str]]:
    """
    Validate that the file extension matches its content.
    """
    if not content or not filename:
        return False, "File or filename is missing"

    try:
        # Get file extension
        file_extension = get_file_extension(filename)
        if not file_extension:
            return False, "Failed to determine file extension"

        # Determine content MIME type
        mime_type = magic.from_buffer(content, mime=True)

        # Dictionary for extension and MIME type matching
        extension_to_mime = {
            # Images
            "jpg": ["image/jpeg"],
            "jpeg": ["image/jpeg"],
            "png": ["image/png"],
            "gif": ["image/gif", "image/png"],  # Sometimes GIF is identified as PNG
            "bmp": ["image/bmp", "image/x-ms-bmp"],
            "tiff": ["image/tiff", "image/png"],  # Sometimes TIFF is identified as PNG
            "tif": ["image/tiff", "image/png"],
            # Documents
            "pdf": ["application/pdf"],
            "doc": ["application/msword"],
            "docx": [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ],
            "rtf": ["application/rtf", "text/rtf"],
            "odt": ["application/vnd.oasis.opendocument.text"],
            # Spreadsheets
            "xls": ["application/vnd.ms-excel"],
            "xlsx": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ],
            "csv": ["text/csv", "text/plain"],
            "ods": ["application/vnd.oasis.opendocument.spreadsheet"],
            # Presentations
            "ppt": ["application/vnd.ms-powerpoint"],
            "pptx": [
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ],
            # Text files
            "txt": ["text/plain"],
            "html": ["text/html"],
            "htm": ["text/html"],
            "md": ["text/plain", "text/markdown"],
            "json": ["application/json", "text/plain"],
            "xml": ["application/xml", "text/xml"],
            "yaml": ["text/plain", "application/x-yaml"],
            "yml": ["text/plain", "application/x-yaml"],
            # Archives
            "zip": ["application/zip"],
            "rar": ["application/vnd.rar"],
            "7z": ["application/x-7z-compressed"],
            "tar": ["application/x-tar"],
            "gz": ["application/gzip"],
            "bz2": ["application/x-bzip2"],
            "xz": ["application/x-xz"],
            # Source code (various MIME types)
            "py": ["text/plain", "text/x-script.python", "text/x-python"],
            "js": ["text/plain", "application/javascript", "text/javascript"],
            "ts": ["text/plain", "text/x-typescript", "application/typescript"],
            "java": ["text/plain", "text/x-java", "text/x-java-source"],
            "c": ["text/plain", "text/x-c", "text/x-csrc"],
            "cpp": ["text/plain", "text/x-c", "text/x-c++", "text/x-c++src"],
            "h": ["text/plain", "text/x-c", "text/x-chdr"],
            "cs": ["text/plain", "text/x-c++", "text/x-csharp"],
            "php": ["text/plain", "text/x-php", "application/x-php"],
            "rb": ["text/plain", "text/x-ruby", "application/x-ruby"],
            "go": ["text/plain", "text/x-c", "text/x-go"],
            "rs": ["text/plain", "text/x-c", "text/x-rust"],
            "swift": ["text/plain", "text/x-c", "text/x-swift"],
            "kt": ["text/plain", "text/x-c", "text/x-kotlin"],
            "scala": ["text/plain", "text/x-scala"],
            "sql": ["text/plain", "text/x-sql"],
            "sh": ["text/plain", "text/x-shellscript", "application/x-shellscript"],
            "css": ["text/css", "text/plain"],
            "scss": ["text/plain", "text/x-scss"],
            "sass": ["text/plain", "text/x-sass"],
            "less": ["text/plain", "text/x-less"],
            "ini": ["text/plain", "text/x-ini"],
            "cfg": ["text/plain"],
            "conf": ["text/plain"],
            "config": ["text/plain"],
            "toml": ["text/plain", "application/toml"],
            "properties": ["text/plain"],
            "dockerfile": ["text/plain"],
            "makefile": ["text/plain", "text/x-makefile"],
            "gitignore": ["text/plain"],
            "bsl": ["text/plain"],
            "os": ["text/plain"],
        }

        # Get allowed MIME types for extension
        expected_mimes = extension_to_mime.get(file_extension, [])

        # If extension is not in our dictionary, consider it valid
        if not expected_mimes:
            return True, None

        # Check for match
        if mime_type in expected_mimes:
            return True, None

        # Special cases for text files and source code
        text_based_extensions = [
            "txt",
            "md",
            "py",
            "js",
            "java",
            "c",
            "cpp",
            "h",
            "cs",
            "php",
            "rb",
            "go",
            "rs",
            "swift",
            "kt",
            "scala",
            "sql",
            "sh",
            "ini",
            "cfg",
            "conf",
            "config",
            "toml",
            "properties",
            "dockerfile",
            "makefile",
            "gitignore",
            "bsl",
            "os",
            "yaml",
            "yml",
            "ts",
            "jsx",
            "tsx",
            "scss",
            "sass",
            "less",
            "latex",
            "tex",
            "rst",
            "adoc",
            "asciidoc",
            "jsonc",
            "jsonl",
            "ndjson",
        ]

        if mime_type == "text/plain" and file_extension in text_based_extensions:
            return True, None

        # Special cases for various source code MIME types
        source_code_mimes = [
            "text/x-c",
            "text/x-script.python",
            "text/x-java",
            "text/x-php",
            "text/x-shellscript",
            "text/x-c++",
            "text/x-python",
            "text/x-ruby",
            "text/x-go",
            "text/x-rust",
            "text/x-swift",
            "text/x-kotlin",
            "text/x-scala",
            "text/x-sql",
            "text/x-scss",
            "text/x-sass",
            "text/x-less",
            "text/x-ini",
            "text/x-makefile",
            "text/x-typescript",
            "text/x-csrc",
            "text/x-c++src",
            "text/x-chdr",
            "text/x-csharp",
            "text/x-java-source",
            "application/x-shellscript",
            "application/javascript",
            "text/javascript",
            "text/css",
            "application/x-php",
            "application/x-ruby",
            "application/toml",
            "application/typescript",
        ]

        if mime_type in source_code_mimes and file_extension in text_based_extensions:
            return True, None

        return False, (
            f"File extension '.{file_extension}' does not match its content (MIME type: {mime_type})"
        )

    except Exception as e:
        # In case of MIME type determination error, consider file invalid (fail-closed)
        logger.warning(f"Error validating file {filename}: {str(e)}")
        return False, f"Failed to determine file type: {str(e)}"


def cleanup_temp_files() -> None:
    """
    Cleanup temporary files at application startup.

    Removes temporary files that might have been left over from previous runs.
    """
    try:
        # Get system temporary directory
        temp_dir = tempfile.gettempdir()

        # Patterns to find temporary files of our application
        patterns = [
            "tmp*.pdf",
            "tmp*.doc",
            "tmp*.docx",
            "tmp*.ppt",
            "tmp*.pptx",
            "tmp*.odt",
            "tmp*.xlsx",
            "tmp*.xls",
            "tmp*.csv",
            "tmp*.txt",
            "tmp*.zip",
            "tmp*.rar",
            "tmp*.7z",
            "tmp*.tar",
            "tmp*.gz",
            "tmp*.bz2",
            "tmp*.xz",
            "tmp*.html",
            "tmp*.htm",
            "tmp*.xml",
            "tmp*.json",
            "tmp*.yaml",
            "tmp*.yml",
        ]

        files_removed = 0

        # Find and remove temporary files
        for pattern in patterns:
            full_pattern = os.path.join(temp_dir, pattern)
            for temp_file in glob.glob(full_pattern):
                try:
                    # Check if the file is older than 1 hour (3600 seconds)
                    file_age = os.path.getmtime(temp_file)
                    current_time = time.time()

                    if current_time - file_age > 3600:
                        os.unlink(temp_file)
                        files_removed += 1
                        logger.debug(f"Removed temporary file: {temp_file}")
                except OSError as e:
                    logger.warning(
                        f"Failed to remove temporary file {temp_file}: {str(e)}"
                    )

        # Find and remove temporary folders
        temp_dirs_patterns = ["tmp*", "extract_*", "temp_*"]

        dirs_removed = 0

        for pattern in temp_dirs_patterns:
            full_pattern = os.path.join(temp_dir, pattern)
            for temp_dir_path in glob.glob(full_pattern):
                if os.path.isdir(temp_dir_path):
                    try:
                        # Check if the folder is older than 1 hour
                        dir_age = os.path.getmtime(temp_dir_path)
                        current_time = time.time()

                        if current_time - dir_age > 3600:
                            shutil.rmtree(temp_dir_path, ignore_errors=True)
                            dirs_removed += 1
                            logger.debug(f"Removed temporary folder: {temp_dir_path}")
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove temporary folder {temp_dir_path}: {str(e)}"
                        )

        if files_removed > 0 or dirs_removed > 0:
            logger.info(
                f"Temporary file cleanup completed. Removed files: {files_removed}, folders: {dirs_removed}"
            )
        else:
            logger.info(
                "Temporary file cleanup completed. No old temporary files found."
            )

    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}", exc_info=True)


def cleanup_recent_temp_files() -> None:
    """
    Immediate cleanup of temporary files for the current process.

    Removes temporary files created in the last 10 minutes.
    """
    try:
        # Get system temporary directory
        temp_dir = tempfile.gettempdir()

        # Patterns to find temporary files of our application
        patterns = [
            "tmp*.pdf",
            "tmp*.doc",
            "tmp*.docx",
            "tmp*.ppt",
            "tmp*.pptx",
            "tmp*.odt",
            "tmp*.xlsx",
            "tmp*.xls",
            "tmp*.csv",
            "tmp*.txt",
            "tmp*.zip",
            "tmp*.rar",
            "tmp*.7z",
            "tmp*.tar",
            "tmp*.gz",
            "tmp*.bz2",
            "tmp*.xz",
            "tmp*.html",
            "tmp*.htm",
            "tmp*.xml",
            "tmp*.json",
            "tmp*.yaml",
            "tmp*.yml",
            "tmp*.png",
            "tmp*.jpg",
            "tmp*.jpeg",
            "tmp*.tiff",
            "tmp*.tif",
            "tmp*.bmp",
            "tmp*.gif",
        ]

        files_removed = 0
        current_time = time.time()

        # Find and remove recent temporary files (less than 10 minutes old)
        for pattern in patterns:
            full_pattern = os.path.join(temp_dir, pattern)
            for temp_file in glob.glob(full_pattern):
                try:
                    # Check if the file is less than 10 minutes old (600 seconds)
                    file_age = os.path.getmtime(temp_file)

                    if current_time - file_age <= 600:  # 10 minutes
                        os.unlink(temp_file)
                        files_removed += 1
                        logger.debug(f"Removed recent temporary file: {temp_file}")
                except OSError as e:
                    logger.debug(
                        f"Failed to remove temporary file {temp_file}: {str(e)}"
                    )

        # Find and remove recent temporary folders
        temp_dirs_patterns = ["tmp*", "extract_*", "temp_*"]
        dirs_removed = 0

        for pattern in temp_dirs_patterns:
            full_pattern = os.path.join(temp_dir, pattern)
            for temp_dir_path in glob.glob(full_pattern):
                if os.path.isdir(temp_dir_path):
                    try:
                        # Check if the folder is less than 10 minutes old
                        dir_age = os.path.getmtime(temp_dir_path)

                        if current_time - dir_age <= 600:  # 10 minutes
                            shutil.rmtree(temp_dir_path, ignore_errors=True)
                            dirs_removed += 1
                            logger.debug(
                                f"Removed recent temporary folder: {temp_dir_path}"
                            )
                    except OSError as e:
                        logger.debug(
                            f"Failed to remove temporary folder {temp_dir_path}: {str(e)}"
                        )

        if files_removed > 0 or dirs_removed > 0:
            logger.info(
                f"Recent temporary file cleanup completed. Removed files: {files_removed}, folders: {dirs_removed}"
            )

    except Exception as e:
        logger.warning(f"Error cleaning up recent temporary files: {str(e)}")


def run_subprocess_with_limits(
    command: list,
    timeout: int = 30,
    memory_limit: Optional[int] = None,
    capture_output: bool = True,
    text: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with resource limits.

    Args:
        command: Command to execute
        timeout: Timeout in seconds
        memory_limit: Memory limit in bytes (None to use defaults)
        capture_output: Whether to capture output
        text: Whether to use text mode
        **kwargs: Additional parameters for subprocess.run

    Returns:
        subprocess.CompletedProcess: Execution result

    Raises:
        subprocess.TimeoutExpired: On timeout
        subprocess.CalledProcessError: On execution error
        MemoryError: On memory limit exceeded
    """
    # On Windows or if limits are disabled, use standard launch
    if not settings.ENABLE_RESOURCE_LIMITS or not RESOURCE_AVAILABLE:
        if not RESOURCE_AVAILABLE:
            logger.debug("resource module unavailable (Windows?), launching without resource limits")
        return subprocess.run(
            command, timeout=timeout, capture_output=capture_output, text=text, **kwargs
        )

    # Determine memory limit
    if memory_limit is None:
        memory_limit = settings.MAX_SUBPROCESS_MEMORY

    def preexec_fn():
        """Function to set resource limits before execution."""
        try:
            # Set limit on virtual memory usage
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # Set limit on data size
            resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, memory_limit))

            # Set limit on CPU time (in seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (timeout * 2, timeout * 2))

            logger.debug(
                f"Resource limits set: memory={memory_limit}, CPU={timeout * 2}"
            )

        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")

    try:
        # Run process with limits
        result = subprocess.run(
            command,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            preexec_fn=preexec_fn,
            **kwargs,
        )

        return result

    except subprocess.TimeoutExpired:
        logger.error(f"Process exceeded timeout {timeout}s: {' '.join(command)}")
        raise
    except subprocess.CalledProcessError as e:
        # Check if error was due to memory limit exceeded
        if e.returncode == 137:  # SIGKILL, often means memory limit exceeded
            logger.error(
                f"Process exceeded memory limit {memory_limit} bytes: {' '.join(command)}"
            )
            raise MemoryError(f"Subprocess exceeded memory limit: {memory_limit} bytes")
        else:
            logger.error(
                f"Process exited with error {e.returncode}: {' '.join(command)}"
            )
            raise
    except Exception as e:
        logger.error(f"Error executing process: {' '.join(command)}, {str(e)}")
        raise


def validate_image_for_ocr(image_content: bytes) -> tuple[bool, Optional[str]]:
    """
    Validate image before OCR to prevent DoS attacks.

    Args:
        image_content: Image content

    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        import io

        from PIL import Image

        # Open image without fully loading into memory
        with Image.open(io.BytesIO(image_content)) as img:
            # Check resolution
            width, height = img.size
            total_pixels = width * height

            if total_pixels > settings.MAX_OCR_IMAGE_PIXELS:
                return (
                    False,
                    f"Image too large: {total_pixels} pixels (max: {settings.MAX_OCR_IMAGE_PIXELS})",
                )

            # Check format
            if img.format not in ["JPEG", "PNG", "TIFF", "BMP", "GIF"]:
                return False, f"Unsupported image format: {img.format}"

            # Check channel count (protection against complex images)
            if hasattr(img, "mode"):
                if img.mode not in ["L", "RGB", "RGBA", "P"]:
                    return False, f"Unsupported color mode: {img.mode}"

            logger.debug(
                f"Image validation passed: {width}x{height}, {img.format}, {img.mode}"
            )
            return True, None

    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, f"Failed to process image: {str(e)}"


def get_memory_usage() -> Dict[str, Any]:
    """
    Get memory usage information.

    Returns:
        Dict[str, Any]: Memory information
    """
    try:
        import psutil

        # System information
        memory = psutil.virtual_memory()

        # Current process information
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()

        return {
            "system_total": memory.total,
            "system_available": memory.available,
            "system_used": memory.used,
            "system_percent": memory.percent,
            "process_rss": process_memory.rss,
            "process_vms": process_memory.vms,
            "process_percent": process.memory_percent(),
        }
    except ImportError:
        logger.warning("psutil not installed, memory information unavailable")
        return {}
    except Exception as e:
        logger.error(f"Error getting memory information: {e}")
        return {}


def get_extension_from_mime(
    content_type: str, supported_formats: dict
) -> Optional[str]:
    """
    Determine file extension by MIME type considering supported formats.

    Args:
        content_type: MIME type from Content-Type header
        supported_formats: Dictionary of supported formats from settings.SUPPORTED_FORMATS

    Returns:
        Optional[str]: File extension or None if type is not supported
    """
    if not content_type:
        return None

    content_type = content_type.lower().strip()

    # Get list of supported image extensions
    supported_image_formats = supported_formats.get("images_ocr", [])

    # MIME type to extension mapping
    mime_mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/tif": "tif",
    }

    # Look for exact MIME type match among supported formats
    for mime, ext in mime_mapping.items():
        if mime in content_type and ext in supported_image_formats:
            return ext

    # If no exact match, check for partial matches
    if "jpeg" in content_type or "jpg" in content_type:
        return "jpg" if "jpg" in supported_image_formats else None
    elif "png" in content_type:
        return "png" if "png" in supported_image_formats else None
    elif "webp" in content_type:
        return "webp" if "webp" in supported_image_formats else None
    elif "gif" in content_type:
        return "gif" if "gif" in supported_image_formats else None
    elif "bmp" in content_type:
        return "bmp" if "bmp" in supported_image_formats else None
    elif "tiff" in content_type or "tif" in content_type:
        return (
            "tiff"
            if "tiff" in supported_image_formats
            else "tif" if "tif" in supported_image_formats else None
        )

    # If MIME type is not supported, return None
    return None


def decode_base64_image(base64_data: str) -> Optional[bytes]:
    """
    Decode base64 image from data URI.

    Args:
        base64_data: String in format data:image/jpeg;base64,/9j/4AAQ...

    Returns:
        Optional[bytes]: Decoded image bytes or None on error
    """
    try:
        # Check data URI format
        if not base64_data.startswith("data:image/"):
            return None

        # Extract base64 part after comma
        if "," not in base64_data:
            return None

        base64_part = base64_data.split(",", 1)[1]

        # Decode base64
        import base64

        return base64.b64decode(base64_part)

    except Exception as e:
        logger.warning(f"Error decoding base64 image: {str(e)}")
        return None


def extract_mime_from_base64_data_uri(data_uri: str) -> Optional[str]:
    """
    Extract MIME type from data URI.

    Args:
        data_uri: String in format data:image/jpeg;base64,/9j/4AAQ...

    Returns:
        Optional[str]: MIME type (e.g., 'image/jpeg') or None on error
    """
    try:
        if not data_uri.startswith("data:"):
            return None

        # Extract part before semicolon
        if ";" not in data_uri:
            return None

        mime_part = data_uri.split(";")[0]
        mime_type = mime_part.replace("data:", "")

        return mime_type if mime_type.startswith("image/") else None

    except Exception as e:
        logger.warning(f"Error extracting MIME type from data URI: {str(e)}")
        return None
