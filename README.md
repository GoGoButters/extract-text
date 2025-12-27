# Text Extraction API for RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

API для извлечения текста из файлов различных форматов, предназначенный для создания векторных представлений (embeddings) в системах RAG.

## Quick Start

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd extract-text-main
cp env_example .env
```

### 2. Запуск в Docker

```bash
# Разработка (с hot reload)
docker-compose up

# Продакшен
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. Проверка работы

```bash
curl http://localhost:7555/health
```

---

## API Endpoints

### `GET /health` — Проверка состояния
```bash
curl http://localhost:7555/health
```

### `GET /v1/supported-formats` — Поддерживаемые форматы
```bash
curl http://localhost:7555/v1/supported-formats
```

### `POST /v1/extract/file` — Извлечение текста из файла
```bash
curl -X POST -F "file=@document.pdf" http://localhost:7555/v1/extract/file
```

### `POST /v1/extract/base64` — Извлечение из base64
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "encoded_base64_file": "0J/RgNC40LLQtdGCIQ==",
    "filename": "test.txt"
  }' \
  http://localhost:7555/v1/extract/base64
```

### `POST /v1/extract/url` — Извлечение с веб-страницы или по URL файла
```bash
# Веб-страница
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/page"}' \
  http://localhost:7555/v1/extract/url

# Файл по URL
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}' \
  http://localhost:7555/v1/extract/url

# С расширенными настройками
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

## Поддерживаемые форматы

| Категория | Форматы |
|-----------|---------|
| **Документы** | PDF, DOCX, DOC, ODT, RTF |
| **Презентации** | PPTX, PPT |
| **Таблицы** | XLSX, XLS, ODS, CSV |
| **Изображения (OCR)** | JPG, JPEG, PNG, TIFF, BMP, GIF, WebP |
| **Архивы** | ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ |
| **Исходный код** | Python, JavaScript, TypeScript, Java, C/C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, SQL, Shell, и др. |
| **Конфигурации** | JSON, YAML, XML, TOML, INI, и др. |
| **Веб** | HTML, CSS, Markdown |
| **Email** | EML, MSG |
| **Книги** | EPUB |

---

## Переменные окружения

### Основные настройки

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `API_PORT` | `7555` | Порт API |
| `DEBUG` | `false` | Режим отладки |
| `WORKERS` | `1` | Количество workers uvicorn |
| `OCR_LANGUAGES` | `rus+eng` | Языки для OCR |

### Ограничения памяти (ВАЖНО!)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MAX_MEMORY_MB` | `2048` | Максимальный объём памяти контейнера (MB) |
| `UVICORN_LIMIT_MAX_REQUESTS` | `1000` | Перезапуск worker после N запросов |
| `ENABLE_RESOURCE_LIMITS` | `true` | Включить ограничения ресурсов для подпроцессов |
| `MAX_SUBPROCESS_MEMORY` | `1073741824` | Лимит памяти для подпроцессов (1GB) |
| `MAX_LIBREOFFICE_MEMORY` | `1610612736` | Лимит памяти для LibreOffice (1.5GB) |
| `MAX_TESSERACT_MEMORY` | `536870912` | Лимит памяти для Tesseract (512MB) |

### Обработка файлов

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MAX_FILE_SIZE` | `20971520` | Максимальный размер файла (20MB) |
| `MAX_ARCHIVE_SIZE` | `20971520` | Максимальный размер архива (20MB) |
| `MAX_EXTRACTED_SIZE` | `104857600` | Максимальный размер распакованного содержимого (100MB) |
| `MAX_ARCHIVE_NESTING` | `3` | Максимальная глубина вложенности архивов |
| `PROCESSING_TIMEOUT_SECONDS` | `300` | Таймаут обработки (секунды) |

### Веб-экстракция

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ENABLE_JAVASCRIPT` | `false` | Включить JavaScript рендеринг (Playwright) |
| `WEB_PAGE_TIMEOUT` | `30` | Таймаут загрузки страницы (сек) |
| `JS_RENDER_TIMEOUT` | `10` | Таймаут JS-рендеринга (сек) |
| `WEB_PAGE_DELAY` | `3` | Задержка после загрузки JS (сек) |
| `ENABLE_LAZY_LOADING_WAIT` | `true` | Автоскролл для lazy loading |
| `MAX_SCROLL_ATTEMPTS` | `3` | Максимальное количество попыток скролла |
| `MAX_IMAGES_PER_PAGE` | `20` | Максимум изображений для OCR на странице |
| `MIN_IMAGE_SIZE_FOR_OCR` | `22500` | Минимальный размер изображения для OCR (пиксели) |
| `ENABLE_BASE64_IMAGES` | `true` | Обрабатывать base64 изображения |

### Безопасность

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `BLOCKED_IP_RANGES` | `127.0.0.0/8,...` | Заблокированные IP-диапазоны (защита от SSRF) |
| `BLOCKED_HOSTNAMES` | `localhost,...` | Заблокированные хосты |

---

## Примеры ответов

### Успешное извлечение
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
      "text": "Извлеченный текст..."
    }
  ]
}
```

### Ошибка
```json
{
  "status": "error",
  "filename": "broken.pdf",
  "message": "Файл поврежден или формат не поддерживается."
}
```

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| `400` | Некорректный запрос |
| `413` | Файл слишком большой |
| `415` | Неподдерживаемый формат |
| `422` | Поврежденный файл |
| `504` | Превышен таймаут обработки |

---

## Интерактивная документация

- **Swagger UI**: http://localhost:7555/docs
- **ReDoc**: http://localhost:7555/redoc

---

## Лицензия

MIT License
