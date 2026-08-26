import os
import sys

# Paths setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PHASE1_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase1"))
PHASE2_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase2"))
PHASE3_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Phase3"))

# Add Phase 1, Phase 2, and Phase 3 to sys.path
for p in [PHASE1_PATH, PHASE2_PATH, PHASE3_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Database Configuration
# Default to SQLite local file, override with PostgreSQL DATABASE_URL if set in env
raw_db_url = os.environ.get("DATABASE_URL")
if raw_db_url:
    DATABASE_URL = raw_db_url
else:
    # On Windows, SQLite connection URL needs three slashes followed by absolute path
    db_path = os.path.join(BASE_DIR, "sepsis_prediction.db").replace("\\", "/")
    DATABASE_URL = f"sqlite:///{db_path}"

# API settings
API_TITLE = "Sepsis Prediction Backend API"
API_VERSION = "1.0.0"
API_PORT = int(os.environ.get("PORT", 8000))
API_HOST = "127.0.0.1"

# Dynamically import and merge Phase 3 and Phase 1 configuration variables
import importlib.util

# 1. Load Phase 3 config variables
p3_config_path = os.path.join(PHASE3_PATH, "config.py")
spec = importlib.util.spec_from_file_location("p3_config", p3_config_path)
p3_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3_config)
for key, value in p3_config.__dict__.items():
    if not key.startswith("__") and key not in globals():
        globals()[key] = value

# 2. Load Phase 1 config variables
p1_config_path = os.path.join(PHASE1_PATH, "config.py")
spec2 = importlib.util.spec_from_file_location("p1_config", p1_config_path)
p1_config = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(p1_config)
for key, value in p1_config.__dict__.items():
    if not key.startswith("__") and key not in globals():
        globals()[key] = value
