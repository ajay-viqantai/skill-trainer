"""Central paths and settings for the training project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
RAW_DIR = DATA / "raw"              # put your PDF / DOCX resumes and JDs here
TEXT_DIR = DATA / "text"            # extracted plain text (one .txt per file)
MANIFEST = DATA / "manifest.csv"    # extraction report
MENTIONS_FILE = DATA / "mentions" / "mentions.jsonl"
LLM_LABELS_FILE = DATA / "labels" / "llm_labels.jsonl"
HUMAN_LABELS_FILE = DATA / "labels" / "human_labels.jsonl"
REVIEW_DIR = DATA / "review"
DISCOVERY_DIR = DATA / "discovery"
EXPERIENCE_DIR = DATA / "experience"

LIBRARY_FILE = ROOT / "library" / "skills_library.json"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

# Ollama (self-hosted LLM, used only for labeling in this project)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# Characters of context kept on each side of a mention
CONTEXT_CHARS = 80

# Files with less text than this are probably scanned images
MIN_TEXT_CHARS = 200