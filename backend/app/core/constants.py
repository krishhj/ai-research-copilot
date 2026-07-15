"""
Application-wide Constants
"""

from pathlib import Path

# Project information
APP_NAME = "AI Research Copilot"
APP_VERSION = "0.1.0"

# Project Directories
# backend/
BASE_DIR = Path(__file__).resolve().parents[2]

# backend/data/
DATA_DIR = BASE_DIR / "data"

# backend/logs/
LOG_DIR = BASE_DIR / "logs"

# backend/data/database/
DATABASE_DIR = DATA_DIR / "database"

# backend/data/vector_store/
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# backend/data/cache/
CACHE_DIR = DATA_DIR / "cache"

# backend/data/exports/
EXPORT_DIR = DATA_DIR / "exports"

# Application Constants

DEFAULT_ENCODING = "utf-8"
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = 100
