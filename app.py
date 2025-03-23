import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict
import tempfile
from file_loader import DocumentLoader
from embedder import DocumentEmbedder
from vector_store import VectorStore
from qa_engine import QAEngine
from config import APP_TITLE, APP_DESCRIPTION, SUPPORTED_EXTENSIONS
from logger import setup_logger
import urllib.parse
import time

logger = setup_logger(__name__)

def initialize_app():
    """アプリケーションの初期化"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔍",
        layout="wide"
    )
    
    # セッション状態の初期化
    if 'current_db' not in st.session_state:
        st.session_state.current_db = None
    if 'qa_engine' not in st.session_state:
        st.session_state.qa_engine = QAEngine()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "qa"  # デフォルトは質問回答

def get_original_db_name(encoded_name: str) -> str:
    """URLエンコードされたデータベース名から元の名前を取得"""
    try:
        return urllib.parse.unquote(encoded_name)
    except:
        return encoded_name

def list_databases() -> List[Dict]:
    """利用可能なデータベースの一覧を取得"""
    dbs = []
    for db_path in Path("index").glob("*"):
        if db_path.is_dir():
            try:
                # 元のデータベース名を取得
                name_file = db_path / "name.txt"
                if name_file.exists():
                    with open(name_file, 'r', encoding='utf-8') as f:
                        original_name = f.read().strip()
                else:
                    original_name = get_original_db_name(db_path.name)
                
                vs = VectorStore(original_name)
                dbs.append(vs.get_collection_info())
            except Exception as e:
                logger.error(f"データベース情報の取得に失敗: {e}")
                continue
    return dbs

def create_database():
    """新しいデータベースを作成"""
    with st.form("create_db"):
        db_name = st.text_input("データベース名")
        submitted = st.form_submit_button("作成")
        if submitted and db_name:
            try:
                # 既存のデータベース名との重複チェック
                existing_dbs = list_databases()
                if any(db["name"] == db_name for db in existing_dbs):
                    st.error(f"データベース '{db_name}' は既に存在します")
                    return
                
                VectorStore(db_name)
                st.success(f"データベース '{db_name}' を作成しました")
                st.rerun()
            except Exception as e:
                st.error(f"データベース作成に失敗: {e}")

async def process_uploaded_files(files, vector_store):
    """アップロードされたファイルを処理"""
    loader = DocumentLoader()
    embedder = DocumentEmbedder()
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = Path(tmp.name)
        
        try:
            # ファイル読み込み
            document = loader.load_document(tmp_path)
            
            # エンベッディング処理
            embedded_docs = await embedder.process_document(document)
            
            # ベクトルストアに保存
            vector_store.add_documents(embedded_docs)
            
            st.success(f"ファイル '{file.name}' を処理しました")
        except Exception as e:
            st.error(f"ファイル '{file.name}' の処理に失敗: {e}")
        finally:
            tmp_path.unlink()  # 一時ファイルを削除

def database_management_page():
    """データベース管理ページ"""
    st.title("データベース管理")
    
    # 新規データベース作成（最上部に移動）
    st.markdown("### 📁 新規データベース作成")
    create_database()
    
    st.markdown("---")
    
    # データベース一覧と選択
    dbs = list_databases()
    if dbs:
        # データベース選択用のプルダウン
        selected_db = st.selectbox(
            "操作するデータベースを選択",
            options=[db["name"] for db in dbs],
            index=None if st.session_state.current_db is None 
                  else [db["name"] for db in dbs].index(st.session_state.current_db)
        )
        
        if selected_db:
            st.session_state.current_db = selected_db
            
            # 選択中のデータベースの操作
            st.markdown("---")
            st.subheader(f"データベース: {selected_db}")
            
            # データベース分析
            st.markdown("### 📊 データベース分析")
            try:
                vs = VectorStore(selected_db)
                analysis = vs.analyze_collection()
                
                # 基本統計情報
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総チャンク数", analysis["total_chunks"])
                with col2:
                    st.metric("総ファイル数", analysis["total_files"])
                with col3:
                    st.metric("総文字数", analysis["total_length"])
                
                # 平均値
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ファイルあたりの平均チャンク数", analysis["avg_chunks_per_file"])
                with col2:
                    st.metric("チャンクあたりの平均文字数", analysis["avg_length_per_chunk"])
                
                # ファイルタイプ分布
                st.markdown("#### ファイルタイプ分布")
                file_types_df = pd.DataFrame(
                    list(analysis["file_types"].items()),
                    columns=["ファイルタイプ", "件数"]
                )
                st.bar_chart(file_types_df.set_index("ファイルタイプ"))
                
                # チャンク長の分布
                st.markdown("#### チャンク長の分布")
                chunk_lengths_df = pd.DataFrame({
                    "チャンク長": analysis["distribution_stats"]["chunk_lengths"]["values"]
                })
                st.bar_chart(chunk_lengths_df)
                
                # チャンク長の統計
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最小チャンク長", analysis["distribution_stats"]["chunk_lengths"]["min"])
                with col2:
                    st.metric("最大チャンク長", analysis["distribution_stats"]["chunk_lengths"]["max"])
                with col3:
                    st.metric("平均チャンク長", round(analysis["distribution_stats"]["chunk_lengths"]["mean"], 2))
                with col4:
                    st.metric("中央値チャンク長", analysis["distribution_stats"]["chunk_lengths"]["median"])
                
                # ファイルごとのチャンク数分布
                st.markdown("#### ファイルごとのチャンク数分布")
                file_chunks_df = pd.DataFrame({
                    "チャンク数": analysis["distribution_stats"]["file_chunk_counts"]["values"]
                })
                st.bar_chart(file_chunks_df)

                # ファイル詳細
                st.markdown("#### ファイル詳細")
                file_stats_data = []
                for file_name, stats in analysis["file_stats"].items():
                    file_stats_data.append({
                        "ファイル名": file_name,
                        "チャンク数": stats["chunk_count"],
                        "総文字数": stats["total_length"],
                        "ファイルタイプ": stats["file_type"]
                    })
                st.dataframe(pd.DataFrame(file_stats_data))
                
            except Exception as e:
                st.error(f"データベース分析に失敗: {e}")
            
            # ファイルアップロード
            st.markdown("### 📤 ファイルアップロード")
            files = st.file_uploader(
                "ファイルを選択",
                type=[ext[1:] for ext in SUPPORTED_EXTENSIONS.keys()],
                accept_multiple_files=True
            )
            if files:
                vector_store = VectorStore(st.session_state.current_db)
                asyncio.run(process_uploaded_files(files, vector_store))
            
            # データベース削除
            st.markdown("### ⚙️ データベース操作")
            if st.button("データベースを削除", type="secondary"):
                try:
                    vs = VectorStore(selected_db)
                    vs.delete_collection()
                    if st.session_state.current_db == selected_db:
                        st.session_state.current_db = None
                    st.success(f"データベース '{selected_db}' を削除しました")
                    st.rerun()
                except Exception as e:
                    st.error(f"削除に失敗: {e}")
    else:
        st.info("登録されているデータベースはありません。新規作成してください。")

async def process_question(question: str):
    """質問処理"""
    try:
        # 質問のエンベッディングを取得
        query_embedding = await st.session_state.qa_engine.get_query_embedding(question)
        
        # 類似文書を検索
        vector_store = VectorStore(st.session_state.current_db)
        relevant_docs = vector_store.search(query_embedding)
        
        # コンテキストを整形
        context = st.session_state.qa_engine.format_context(relevant_docs)
        
        # 回答を生成
        answer = await st.session_state.qa_engine.generate_answer(question, context)
        
        # 類似質問を生成
        similar_questions = st.session_state.qa_engine.suggest_similar_questions(
            question, context
        )
        
        return answer, relevant_docs, similar_questions
    except Exception as e:
        logger.error(f"質問処理に失敗: {e}")
        raise

def qa_interface():
    """質問応答インターフェース"""
    st.title("質問応答")
    
    # データベース選択
    dbs = list_databases()
    if dbs:
        selected_db = st.selectbox(
            "データベースを選択",
            options=[db["name"] for db in dbs],
            index=None if st.session_state.current_db is None 
                  else [db["name"] for db in dbs].index(st.session_state.current_db)
        )
        if selected_db:
            st.session_state.current_db = selected_db
            
            # セッション状態に質問を保存
            if 'current_question' not in st.session_state:
                st.session_state.current_question = ""
            
            # 質問入力フォーム
            question = st.text_input(
                "質問を入力してください",
                value=st.session_state.current_question,
                key="question_input"
            )
            
            if question:
                with st.spinner("回答を生成中..."):
                    try:
                        answer, sources, similar_questions = asyncio.run(
                            process_question(question)
                        )
                        
                        # 回答を表示
                        st.markdown("### 回答")
                        st.write(answer)
                        
                        # 参照元を表示
                        st.markdown("### 参照元")
                        for i, doc in enumerate(sources):
                            # 改行を空白に置換して1行にまとめる
                            text = doc['text'].replace('\n', ' ').strip()
                            # プレビューテキストを作成（最初の100文字）
                            preview = text[:100] + "..." if len(text) > 100 else text
                            
                            with st.expander(
                                "📄 " + doc['metadata']['file_name'] +
                                "\n\n" +  # 2つの改行を入れてスペースを確保
                                "(" + preview + ")"
                            ):
                                st.text_area(
                                    label="",
                                    value=text,
                                    height=200,
                                    disabled=True,
                                    key=f"source_text_{i}"
                                )
                        
                        # 類似質問を表示
                        st.markdown("### 関連する質問")
                        for q in similar_questions:
                            # コピー可能なテキストとして表示
                            st.markdown(f"```\n{q}\n```")
                        
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
    else:
        st.warning("先にデータベースを作成してください")

def main():
    """メイン関数"""
    initialize_app()
    
    # サイドメニュー
    with st.sidebar:
        st.title(APP_TITLE)
        st.markdown("---")
        
        # ページ選択（ラジオボタンに変更）
        selected = st.radio(
            "メニュー",
            ["💭 質問回答", "🗄️ データベース管理"],
            index=0 if st.session_state.current_page == "qa" else 1,
            horizontal=True  # 横並びに表示
        )
        
        if selected == "💭 質問回答":
            st.session_state.current_page = "qa"
        else:
            st.session_state.current_page = "db"
        
        # 現在のデータベース表示
        st.markdown("---")
        if st.session_state.current_db:
            st.info(f"現在のデータベース: {st.session_state.current_db}")
        else:
            st.warning("データベースが選択されていません")
    
    # メインコンテンツ
    if st.session_state.current_page == "qa":
        qa_interface()
    else:
        database_management_page()

if __name__ == "__main__":
    main() 