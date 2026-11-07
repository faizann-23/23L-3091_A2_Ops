"""Test configuration. Runs before any test module is imported."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # MODEL_PATH is relative to the repository root

# Use an isolated throw-away SQLite file; never touch a developer's real database.
# (load_dotenv() does not override variables that are already set.)
os.environ["DB_BACKEND"] = "sqlite"
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_visionguard.db")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ["MAX_FILE_SIZE_MB"] = "5"
