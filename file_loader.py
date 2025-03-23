import fitz
import docx
from pathlib import Path
from typing import List, Dict
from logger import setup_logger
from config import SUPPORTED_EXTENSIONS

logger = setup_logger(__name__)

class DocumentLoader:
    """各種ドキュメントを読み込むためのクラス"""
    
    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """PDFファイルを読み込み、テキストを抽出"""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"PDFの読み込みに失敗: {file_path}, エラー: {e}")
            raise

    @staticmethod
    def load_txt(file_path: Path) -> str:
        """テキストファイルを読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"テキストファイルの読み込みに失敗: {file_path}, エラー: {e}")
            raise

    @staticmethod
    def load_docx(file_path: Path) -> str:
        """Wordファイルを読み込む"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Wordファイルの読み込みに失敗: {file_path}, エラー: {e}")
            raise

    def load_document(self, file_path: Path) -> Dict[str, str]:
        """ドキュメントを読み込み、メタデータとテキストを返す"""
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"未対応のファイル形式です: {extension}")

        try:
            if extension == ".pdf":
                text = self.load_pdf(file_path)
            elif extension == ".txt":
                text = self.load_txt(file_path)
            elif extension == ".docx":
                text = self.load_docx(file_path)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": SUPPORTED_EXTENSIONS[extension],
                "text": text
            }

        except Exception as e:
            logger.error(f"ドキュメントの読み込みに失敗: {file_path}, エラー: {e}")
            raise 