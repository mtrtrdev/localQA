from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from logger import setup_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP, CONTEXT_WINDOW

logger = setup_logger(__name__)

class DocumentEmbedder:
    """ドキュメントのエンベッディングを行うクラス"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings()

    def split_text(self, text: str, metadata: Dict) -> List[Dict]:
        """テキストをチャンクに分割"""
        try:
            texts = self.text_splitter.split_text(text)
            documents = []
            for i, text_chunk in enumerate(texts):
                # 前後のチャンクのインデックスを計算
                prev_chunks = list(range(max(0, i - CONTEXT_WINDOW), i))
                next_chunks = list(range(i + 1, min(len(texts), i + CONTEXT_WINDOW + 1)))
                
                doc = {
                    "text": text_chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "chunk_total": len(texts),
                        "prev_chunks": prev_chunks,
                        "next_chunks": next_chunks,
                        "file_chunk_index": i
                    }
                }
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"テキスト分割に失敗: {e}")
            raise

    async def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """ドキュメントのエンベッディングを生成"""
        try:
            texts = [doc["text"] for doc in documents]
            embeddings = await self.embeddings.aembed_documents(texts)
            
            for doc, embedding in zip(documents, embeddings):
                doc["embedding"] = embedding
            
            return documents
        except Exception as e:
            logger.error(f"エンベッディング生成に失敗: {e}")
            raise

    async def process_document(self, document: Dict) -> List[Dict]:
        """ドキュメントの処理を行う（分割とエンベッディング）"""
        try:
            metadata = {
                "file_path": document["file_path"],
                "file_name": document["file_name"],
                "file_type": document["file_type"]
            }
            split_docs = self.split_text(document["text"], metadata)
            embedded_docs = await self.embed_documents(split_docs)
            return embedded_docs
        except Exception as e:
            logger.error(f"ドキュメント処理に失敗: {e}")
            raise 