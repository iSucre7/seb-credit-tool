from dotenv import load_dotenv
import os

load_dotenv()

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Models
MODELS = {
    "extractor": "claude-haiku-4-5-20251001",
    "analyst":   "claude-sonnet-4-6",
    "advisory":  "claude-sonnet-4-6",
    "writer":    "claude-sonnet-4-6"
}

# Accounting standards
class Standard:
    IFRS16     = "IFRS16"
    IFRS_PRE16 = "IFRS_PRE16"
    US_GAAP    = "US_GAAP"

DEFAULT_STANDARD = Standard.IFRS16

# Paths
BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "companies")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
METHODOLOGY_DIR = os.path.join(BASE_DIR, "data", "methodologies")

# Model settings
FORECAST_YEARS    = 3
HISTORICAL_YEARS  = 2
MAX_ITERATIONS    = 20
CONVERGENCE_LIMIT = 0.1