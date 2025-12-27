# Text Extraction API for RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

API for extracting text from files in various formats, designed for creating vector embeddings in RAG (Retrieval-Augmented Generation) systems.

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/GoGoButters/extract-text.git
cd extract-text
cp env_example .env
```

### 2. Run with Docker

```bash
# Development (with hot reload)
docker-compose up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Verify

```bash
curl http://localhost:7555/health
```

---

## API Endpoints

### `GET /health` — Health Check
```bash
curl http://localhost:7555/health
```

### `GET /v1/supported-formats` — Supported Formats
```bash
curl http://localhost:7555/v1/supported-formats
```

### `POST /v1/extract/file` — Extract Text from File
```bash
curl -X POST -F "file=@document.pdf" http://localhost:7555/v1/extract/file
```

### `POST /v1/extract/base64` — Extract from Base64
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "encoded_base64_file": "SGVsbG8gV29ybGQh",
    "filename": "test.txt"
  }' \
  http://localhost:7555/v1/extract/base64
```

### `POST /v1/extract/url` — Extract from Web Page or File URL
```bash
# Web page
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/page"}' \
  http://localhost:7555/v1/extract/url

# File by URL
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}' \
  http://localhost:7555/v1/extract/url

# With advanced options
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/spa-app",
    "extraction_options": {
      "enable_javascript": true,
      "js_render_timeout": 15,
      "process_images": true
    }
  }' \
  http://localhost:7555/v1/extract/url
```

---

## Supported Formats

| Category | Formats |
|----------|---------|
| **Documents** | PDF, DOCX, DOC, ODT, RTF |
| **Presentations** | PPTX, PPT |
| **Spreadsheets** | XLSX, XLS, ODS, CSV |
| **Images (OCR)** | JPG, JPEG, PNG, TIFF, BMP, GIF, WebP |
| **Archives** | ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ |
| **Source Code** | Python, JavaScript, TypeScript, Java, C/C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, SQL, Shell, and more |
| **Configuration** | JSON, YAML, XML, TOML, INI, and more |
| **Web** | HTML, CSS, Markdown |
| **Email** | EML, MSG |
| **E-books** | EPUB |

---

## Environment Variables

### Basic Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `7555` | API port |
| `DEBUG` | `false` | Debug mode |
| `WORKERS` | `1` | Number of uvicorn workers |
| `OCR_LANGUAGES` | `rus+eng` | Languages for OCR |

### Memory Limits (IMPORTANT!)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MEMORY_MB` | `2048` | Maximum container memory (MB) |
| `UVICORN_LIMIT_MAX_REQUESTS` | `1000` | Restart worker after N requests |
| `ENABLE_RESOURCE_LIMITS` | `true` | Enable resource limits for subprocesses |
| `MAX_SUBPROCESS_MEMORY` | `1073741824` | Subprocess memory limit (1GB) |
| `MAX_LIBREOFFICE_MEMORY` | `1610612736` | LibreOffice memory limit (1.5GB) |
| `MAX_TESSERACT_MEMORY` | `536870912` | Tesseract memory limit (512MB) |

### File Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE` | `20971520` | Maximum file size (20MB) |
| `MAX_ARCHIVE_SIZE` | `20971520` | Maximum archive size (20MB) |
| `MAX_EXTRACTED_SIZE` | `104857600` | Maximum extracted content size (100MB) |
| `MAX_ARCHIVE_NESTING` | `3` | Maximum archive nesting depth |
| `PROCESSING_TIMEOUT_SECONDS` | `300` | Processing timeout (seconds) |

### Web Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_JAVASCRIPT` | `false` | Enable JavaScript rendering (Playwright) |
| `WEB_PAGE_TIMEOUT` | `30` | Page load timeout (sec) |
| `JS_RENDER_TIMEOUT` | `10` | JS rendering timeout (sec) |
| `WEB_PAGE_DELAY` | `3` | Delay after JS load (sec) |
| `ENABLE_LAZY_LOADING_WAIT` | `true` | Auto-scroll for lazy loading |
| `MAX_SCROLL_ATTEMPTS` | `3` | Maximum scroll attempts |
| `MAX_IMAGES_PER_PAGE` | `20` | Maximum images for OCR per page |
| `MIN_IMAGE_SIZE_FOR_OCR` | `22500` | Minimum image size for OCR (pixels) |
| `ENABLE_BASE64_IMAGES` | `true` | Process base64 images |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `BLOCKED_IP_RANGES` | `127.0.0.0/8,...` | Blocked IP ranges (SSRF protection) |
| `BLOCKED_HOSTNAMES` | `localhost,...` | Blocked hostnames |

---

## Response Examples

### Successful Extraction
```json
{
  "status": "success",
  "filename": "document.pdf",
  "count": 1,
  "files": [
    {
      "filename": "document.pdf",
      "path": "document.pdf",
      "size": 1024000,
      "type": "pdf",
      "text": "Extracted text..."
    }
  ]
}
```

### Error
```json
{
  "status": "error",
  "filename": "broken.pdf",
  "message": "File is corrupted or format is not supported."
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad request |
| `413` | File too large |
| `415` | Unsupported format |
| `422` | Corrupted file |
| `504` | Processing timeout |

---

## Interactive Documentation

- **Swagger UI**: http://localhost:7555/docs
- **ReDoc**: http://localhost:7555/redoc

---

## License

MIT License
