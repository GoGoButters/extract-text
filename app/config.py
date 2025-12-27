"""Конфигурация приложения."""

import os
from typing import List


class Settings:
    """Настройки приложения."""

    # Basic settings
    VERSION: str = "1.10.8"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API settings
    API_PORT: int = int(os.getenv("API_PORT", "7555"))

    # ===========================================
    # ALL SIZE/MEMORY SETTINGS ARE IN MEGABYTES
    # ===========================================
    
    # File processing settings (MB, converted to bytes internally)
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))  # 20 MB
    MAX_FILE_SIZE: int = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    PROCESSING_TIMEOUT_SECONDS: int = int(
        os.getenv("PROCESSING_TIMEOUT_SECONDS", "300")
    )

    # Child process resource management settings
    # Maximum memory for child processes (MB)
    MAX_SUBPROCESS_MEMORY_MB: int = int(os.getenv("MAX_SUBPROCESS_MEMORY_MB", "1024"))  # 1 GB
    MAX_SUBPROCESS_MEMORY: int = MAX_SUBPROCESS_MEMORY_MB * 1024 * 1024  # Convert to bytes

    # Maximum memory for LibreOffice (MB)
    MAX_LIBREOFFICE_MEMORY_MB: int = int(os.getenv("MAX_LIBREOFFICE_MEMORY_MB", "1536"))  # 1.5 GB
    MAX_LIBREOFFICE_MEMORY: int = MAX_LIBREOFFICE_MEMORY_MB * 1024 * 1024  # Convert to bytes

    # Maximum memory for Tesseract (MB)
    MAX_TESSERACT_MEMORY_MB: int = int(os.getenv("MAX_TESSERACT_MEMORY_MB", "512"))  # 512 MB
    MAX_TESSERACT_MEMORY: int = MAX_TESSERACT_MEMORY_MB * 1024 * 1024  # Convert to bytes

    # Maximum resolution for OCR images (pixels)
    MAX_OCR_IMAGE_PIXELS: int = int(
        os.getenv("MAX_OCR_IMAGE_PIXELS", str(50 * 1024 * 1024))
    )  # 50 MP

    # Enable/disable resource limits
    ENABLE_RESOURCE_LIMITS: bool = (
        os.getenv("ENABLE_RESOURCE_LIMITS", "true").lower() == "true"
    )

    # Maximum container memory (MB)
    # Used in docker-compose.yml for mem_limit
    MAX_MEMORY_MB: int = int(os.getenv("MAX_MEMORY_MB", "2048"))

    # Request limit for uvicorn worker restart
    # 0 = no limit
    UVICORN_LIMIT_MAX_REQUESTS: int = int(
        os.getenv("UVICORN_LIMIT_MAX_REQUESTS", "1000")
    )

    # OCR settings
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "rus+eng")

    # Performance settings
    WORKERS: int = int(os.getenv("WORKERS", "1"))

    # Archive settings (MB, converted to bytes internally)
    MAX_ARCHIVE_SIZE_MB: int = int(os.getenv("MAX_ARCHIVE_SIZE_MB", "20"))  # 20 MB
    MAX_ARCHIVE_SIZE: int = MAX_ARCHIVE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    MAX_EXTRACTED_SIZE_MB: int = int(os.getenv("MAX_EXTRACTED_SIZE_MB", "100"))  # 100 MB
    MAX_EXTRACTED_SIZE: int = MAX_EXTRACTED_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    MAX_ARCHIVE_NESTING: int = int(os.getenv("MAX_ARCHIVE_NESTING", "3"))

    # Настройки веб-экстрактора (v1.10.0)
    MIN_IMAGE_SIZE_FOR_OCR: int = int(
        os.getenv("MIN_IMAGE_SIZE_FOR_OCR", "22500")
    )  # 150x150 пикселей
    MAX_IMAGES_PER_PAGE: int = int(os.getenv("MAX_IMAGES_PER_PAGE", "20"))
    WEB_PAGE_TIMEOUT: int = int(os.getenv("WEB_PAGE_TIMEOUT", "30"))  # секунды
    IMAGE_DOWNLOAD_TIMEOUT: int = int(
        os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "15")
    )  # секунды
    DEFAULT_USER_AGENT: str = os.getenv("DEFAULT_USER_AGENT", "Text Extraction Bot 1.0")
    ENABLE_JAVASCRIPT: bool = os.getenv("ENABLE_JAVASCRIPT", "false").lower() == "true"

    # Новые настройки для определения типа контента и скачивания файлов (v1.10.3)
    HEAD_REQUEST_TIMEOUT: int = int(
        os.getenv("HEAD_REQUEST_TIMEOUT", "10")
    )  # таймаут HEAD запроса
    FILE_DOWNLOAD_TIMEOUT: int = int(
        os.getenv("FILE_DOWNLOAD_TIMEOUT", "60")
    )  # таймаут скачивания файла

    # Новые настройки веб-экстрактора (v1.10.1)
    ENABLE_BASE64_IMAGES: bool = (
        os.getenv("ENABLE_BASE64_IMAGES", "true").lower() == "true"
    )
    WEB_PAGE_DELAY: int = int(
        os.getenv("WEB_PAGE_DELAY", "3")
    )  # секунды задержки после загрузки JS
    ENABLE_LAZY_LOADING_WAIT: bool = (
        os.getenv("ENABLE_LAZY_LOADING_WAIT", "true").lower() == "true"
    )
    JS_RENDER_TIMEOUT: int = int(
        os.getenv("JS_RENDER_TIMEOUT", "10")
    )  # отдельный таймаут для JS-рендеринга
    MAX_SCROLL_ATTEMPTS: int = int(
        os.getenv("MAX_SCROLL_ATTEMPTS", "3")
    )  # защита от бесконечного скролла

    # Заблокированные IP-диапазоны для защиты от SSRF
    BLOCKED_IP_RANGES: str = os.getenv(
        "BLOCKED_IP_RANGES",
        "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,::1/128,fe80::/10",
    )

    # Заблокированные хосты (включая Docker и loopback)
    BLOCKED_HOSTNAMES: str = os.getenv(
        "BLOCKED_HOSTNAMES", "localhost,host.docker.internal,ip6-localhost,ip6-loopback"
    )

    # Поддерживаемые форматы
    SUPPORTED_FORMATS = {
        "images_ocr": ["jpg", "jpeg", "png", "tiff", "tif", "bmp", "gif", "webp"],
        "documents": ["doc", "docx", "pdf", "rtf", "odt"],
        "spreadsheets": ["csv", "xls", "xlsx", "ods"],
        "presentations": ["pptx", "ppt"],
        "structured_data": ["json", "xml", "yaml", "yml"],
        "source_code": [
            # Python
            "py",
            "pyx",
            "pyi",
            "pyw",
            # JavaScript/TypeScript
            "js",
            "jsx",
            "ts",
            "tsx",
            "mjs",
            "cjs",
            # Java
            "java",
            "jav",
            # C/C++
            "c",
            "cpp",
            "cxx",
            "cc",
            "c++",
            "h",
            "hpp",
            "hxx",
            "h++",
            # C#
            "cs",
            "csx",
            # PHP
            "php",
            "php3",
            "php4",
            "php5",
            "phtml",
            # Ruby
            "rb",
            "rbw",
            "rake",
            "gemspec",
            # Go
            "go",
            "mod",
            "sum",
            # Rust
            "rs",
            "rlib",
            # Swift
            "swift",
            # Kotlin
            "kt",
            "kts",
            # Scala
            "scala",
            "sc",
            # R
            "r",
            "rmd",
            # SQL
            "sql",
            "ddl",
            "dml",
            # Shell/Bash
            "sh",
            "bash",
            "zsh",
            "fish",
            "ksh",
            "csh",
            "tcsh",
            # PowerShell
            "ps1",
            "psm1",
            "psd1",
            # Perl
            "pl",
            "pm",
            "pod",
            "t",
            # Lua
            "lua",
            # 1C and OneScript
            "bsl",
            "os",
            # Configuration files
            "ini",
            "cfg",
            "conf",
            "config",
            "toml",
            "properties",
            # Web
            "css",
            "scss",
            "sass",
            "less",
            "styl",
            # Markup
            "tex",
            "latex",
            "rst",
            "adoc",
            "asciidoc",
            # Data formats
            "jsonl",
            "ndjson",
            "jsonc",
            # Docker
            "dockerfile",
            "containerfile",
            # Makefile
            "makefile",
            "mk",
            "mak",
            # Git
            "gitignore",
            "gitattributes",
            "gitmodules",
        ],
        "other": ["txt", "html", "htm", "md", "markdown", "epub", "eml", "msg"],
        "archives": [
            "zip",
            "rar",
            "7z",
            "tar",
            "gz",
            "bz2",
            "xz",
            "tgz",
            "tbz2",
            "txz",
            "tar.gz",
            "tar.bz2",
            "tar.xz",
        ],
    }

    MIME_TO_EXTENSION = {
        "application/pdf": "pdf",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-powerpoint": "ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/zip": "zip",
        "application/x-rar-compressed": "rar",
        "application/x-7z-compressed": "7z",
        "application/x-tar": "tar",
        "application/gzip": "gz",
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "text/plain": "txt",
        "text/html": "html",
        "text/csv": "csv",
        "application/json": "json",
        "application/xml": "xml",
        "text/xml": "xml",
    }

    @property
    def all_supported_extensions(self) -> List[str]:
        """Все поддерживаемые расширения файлов."""
        extensions = []
        for format_group in self.SUPPORTED_FORMATS.values():
            extensions.extend(format_group)
        return extensions


settings = Settings()
