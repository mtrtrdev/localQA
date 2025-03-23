import faiss
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set
from logger import setup_logger
from config import INDEX_DIR, FAISS_INDEX_PATH, MAX_CHUNKS_PER_QUERY, CONTEXT_WINDOW
import hashlib
import os
import urllib.parse

logger = setup_logger(__name__)

class VectorStore:
    """FAISSを使用したベクトルストア"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        # URLエンコードを使用して安全なファイル名を生成
        safe_name = urllib.parse.quote(collection_name, safe='')
        
        # カレントディレクトリからの相対パスを使用
        self.index_dir = Path.cwd() / "index" / safe_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 絶対パスを文字列として使用（日本語パスを避ける）
        self.index_path = str(self.index_dir.absolute() / "index.faiss")
        self.metadata_path = str(self.index_dir.absolute() / "metadata.pkl")
        self.name_path = self.index_dir / "name.txt"

        # オリジナルの名前を保存
        if not self.name_path.exists():
            with open(self.name_path, 'w', encoding='utf-8') as f:
                f.write(collection_name)

        if Path(self.index_path).exists() and Path(self.metadata_path).exists():
            self._load_index()
        else:
            self._create_index()

    def _create_index(self):
        """新しいインデックスを作成"""
        self.dimension = 1536  # OpenAIのada-002モデルの次元数
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_list = []
        self._save_index()

    def _load_index(self):
        """既存のインデックスを読み込み"""
        try:
            # 相対パスの文字列を使用
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata_list = pickle.load(f)
        except Exception as e:
            logger.error(f"インデックスの読み込みに失敗: {e}")
            raise

    def _save_index(self):
        """インデックスを保存"""
        try:
            # 相対パスの文字列を使用
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_list, f)
        except Exception as e:
            logger.error(f"インデックスの保存に失敗: {e}")
            raise

    def add_documents(self, documents: List[Dict]) -> None:
        """ドキュメントをベクトルストアに追加"""
        try:
            embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)
            self.index.add(embeddings)
            
            for doc in documents:
                metadata = {
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
                self.metadata_list.append(metadata)
            
            self._save_index()
            logger.info(f"{len(documents)}個のドキュメントを追加しました")
        except Exception as e:
            logger.error(f"ドキュメント追加に失敗: {e}")
            raise

    def _get_related_chunks(self, chunk_indices: List[int]) -> Set[int]:
        """関連するチャンクのインデックスを取得"""
        related_indices = set()
        for idx in chunk_indices:
            if 0 <= idx < len(self.metadata_list):
                metadata = self.metadata_list[idx]["metadata"]
                # 前後のチャンクを追加
                related_indices.update(metadata.get("prev_chunks", []))
                related_indices.update(metadata.get("next_chunks", []))
                related_indices.add(idx)
        return related_indices

    def search(self, query_embedding: List[float], n_results: int = MAX_CHUNKS_PER_QUERY) -> List[Dict]:
        """類似度検索を実行"""
        try:
            query_np = np.array([query_embedding], dtype=np.float32)
            # より多くの結果を取得して関連チャンクを考慮
            distances, indices = self.index.search(query_np, n_results * 2)
            
            # 最初のn_results個の結果を取得
            initial_results = []
            for i, idx in enumerate(indices[0][:n_results]):
                if idx != -1:  # FAISSの無効なインデックスをチェック
                    initial_results.append(idx)
            
            # 関連するチャンクを取得
            related_indices = self._get_related_chunks(initial_results)
            
            # 結果を整理（関連チャンクを含む）
            results = []
            seen_indices = set()
            
            # 最初の結果を追加
            for idx in initial_results:
                if idx not in seen_indices:
                    result = {
                        "text": self.metadata_list[idx]["text"],
                        "metadata": self.metadata_list[idx]["metadata"],
                        "distance": float(distances[0][list(indices[0]).index(idx)])
                    }
                    results.append(result)
                    seen_indices.add(idx)
            
            # 関連チャンクを追加
            for idx in related_indices:
                if idx not in seen_indices and idx in initial_results:
                    result = {
                        "text": self.metadata_list[idx]["text"],
                        "metadata": self.metadata_list[idx]["metadata"],
                        "distance": float(distances[0][list(indices[0]).index(idx)])
                    }
                    results.append(result)
                    seen_indices.add(idx)
            
            return results
        except Exception as e:
            logger.error(f"検索に失敗: {e}")
            raise

    def get_collection_info(self) -> Dict:
        """コレクション情報を取得"""
        try:
            return {
                "name": self.collection_name,  # オリジナルの名前を使用
                "document_count": len(self.metadata_list),
                "created_at": datetime.fromtimestamp(
                    self.index_dir.stat().st_ctime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "status": "利用可"
            }
        except Exception as e:
            logger.error(f"コレクション情報の取得に失敗: {e}")
            return {
                "name": self.collection_name,
                "document_count": 0,
                "created_at": "不明",
                "status": "破損"
            }

    def delete_collection(self) -> None:
        """コレクションを削除"""
        try:
            import shutil
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            logger.info(f"コレクション {self.collection_name} を削除しました")
        except Exception as e:
            logger.error(f"コレクション削除に失敗: {e}")
            raise

    def clear_collection(self) -> None:
        """コレクションの内容をクリア"""
        try:
            self._create_index()  # 新しい空のインデックスを作成
            self._save_index()    # 空のインデックスを保存
            logger.info(f"コレクション {self.collection_name} をクリアしました")
        except Exception as e:
            logger.error(f"コレクションのクリアに失敗: {e}")
            raise

    def analyze_collection(self) -> Dict:
        """コレクションの内容を分析"""
        try:
            # ファイルごとの統計情報
            file_stats = {}
            chunk_lengths = []  # チャンクの長さを記録
            file_chunk_counts = []  # ファイルごとのチャンク数を記録
            
            for doc in self.metadata_list:
                file_name = doc["metadata"]["file_name"]
                chunk_length = len(doc["text"])
                chunk_lengths.append(chunk_length)
                
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        "chunk_count": 0,
                        "total_length": 0,
                        "file_type": doc["metadata"]["file_type"]
                    }
                file_stats[file_name]["chunk_count"] += 1
                file_stats[file_name]["total_length"] += chunk_length
                file_chunk_counts.append(file_stats[file_name]["chunk_count"])

            # 全体的な統計情報
            total_chunks = len(self.metadata_list)
            total_files = len(file_stats)
            total_length = sum(stat["total_length"] for stat in file_stats.values())
            avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0
            avg_length_per_chunk = total_length / total_chunks if total_chunks > 0 else 0

            # ファイルタイプごとの集計
            file_types = {}
            for stat in file_stats.values():
                file_type = stat["file_type"]
                if file_type not in file_types:
                    file_types[file_type] = 0
                file_types[file_type] += 1

            # 分布統計
            distribution_stats = {
                "chunk_lengths": {
                    "min": min(chunk_lengths) if chunk_lengths else 0,
                    "max": max(chunk_lengths) if chunk_lengths else 0,
                    "mean": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    "median": sorted(chunk_lengths)[len(chunk_lengths)//2] if chunk_lengths else 0,
                    "values": chunk_lengths
                },
                "file_chunk_counts": {
                    "min": min(file_chunk_counts) if file_chunk_counts else 0,
                    "max": max(file_chunk_counts) if file_chunk_counts else 0,
                    "mean": sum(file_chunk_counts) / len(file_chunk_counts) if file_chunk_counts else 0,
                    "median": sorted(file_chunk_counts)[len(file_chunk_counts)//2] if file_chunk_counts else 0,
                    "values": file_chunk_counts
                }
            }

            return {
                "total_chunks": total_chunks,
                "total_files": total_files,
                "total_length": total_length,
                "avg_chunks_per_file": round(avg_chunks_per_file, 2),
                "avg_length_per_chunk": round(avg_length_per_chunk, 2),
                "file_types": file_types,
                "file_stats": file_stats,
                "distribution_stats": distribution_stats
            }
        except Exception as e:
            logger.error(f"コレクション分析に失敗: {e}")
            raise 