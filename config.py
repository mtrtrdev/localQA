import os
from pathlib import Path
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# 環境変数が設定されているか確認
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY が設定されていません。.envファイルを確認してください。")

# Base Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = Path("index")  # 相対パスに変更
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, INDEX_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Embedding Configuration
CHUNK_SIZE = 500  # より小さなチャンクサイズ
CHUNK_OVERLAP = 100  # 適度なオーバーラップ
MAX_CHUNKS_PER_QUERY = 5  # 1回のクエリで取得する最大チャンク数
CONTEXT_WINDOW = 2  # 前後のチャンクを考慮する数

# Supported File Types
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".txt": "Text",
    ".docx": "Word",
}

# Vector Store Configuration
VECTOR_STORE_TYPE = "FAISS"  # ChromaDBからFAISSに変更
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"  # 追加

# LLM Configuration
LLM_TYPE = "OpenAI"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# UI Configuration
APP_TITLE = "LocalQA"
APP_DESCRIPTION = "ローカルの文書をベースに質問応答を行うシステムです。" 