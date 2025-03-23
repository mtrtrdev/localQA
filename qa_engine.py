from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from logger import setup_logger
from config import LLM_MODEL, TEMPERATURE, MAX_TOKENS
import os

logger = setup_logger(__name__)

class QAEngine:
    """質問応答を行うエンジン"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは文書検索システムのアシスタントです。
与えられたコンテキストに基づいて、ユーザーの質問に簡潔に答えてください。
コンテキストに含まれていない情報については、「その情報はコンテキストにありません」と回答してください。
回答は日本語で行ってください。"""),
            ("user", "コンテキスト:\n{context}\n\n質問:\n{question}")
        ])

    async def get_query_embedding(self, query: str) -> List[float]:
        """質問文のエンベッディングを取得"""
        try:
            embedding = await self.embeddings.aembed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"クエリのエンベッディングに失敗: {e}")
            raise

    def format_context(self, relevant_docs: List[Dict]) -> str:
        """コンテキストを整形"""
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[文書{i}] "
            context += f"(ファイル: {doc['metadata']['file_name']}, "
            if doc['metadata'].get('page_num'):
                context += f"ページ: {doc['metadata']['page_num']}, "
            context += f"チャンク: {doc['metadata']['chunk_id'] + 1}/{doc['metadata']['chunk_total']})\n"
            context += f"{doc['text']}\n\n"
        return context.strip()

    async def generate_answer(self, question: str, context: str) -> str:
        """回答を生成"""
        try:
            chain = self.qa_prompt | self.llm
            response = await chain.ainvoke({
                "context": context,
                "question": question
            })
            return response.content
        except Exception as e:
            logger.error(f"回答生成に失敗: {e}")
            raise

    def suggest_similar_questions(self, question: str, context: str) -> List[str]:
        """類似質問を提案"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは文書検索システムのアシスタントです。
与えられたコンテキストと元の質問に基づいて、関連する3つの質問を提案してください。

以下の条件を満たす質問を生成してください：
1. コンテキストの内容に基づいていること
2. 元の質問と異なる視点からの質問であること
3. 具体的で明確な質問であること
4. 日本語で自然な表現であること

出力形式：
- 質問1
- 質問2
- 質問3"""),
                ("user", "コンテキスト:\n{context}\n\n元の質問:\n{question}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # 箇条書きを行ごとに分割して整形
            questions = []
            for line in response.content.split("\n"):
                line = line.strip()
                # 箇条書きの記号（-、・、*）を除去
                if line and (line.startswith("-") or line.startswith("・") or line.startswith("*")):
                    question = line[1:].strip()
                    if question:  # 空でない場合のみ追加
                        questions.append(question)
            
            # 最大3つまでに制限
            return questions[:3]
            
        except Exception as e:
            logger.error(f"類似質問の生成に失敗: {e}")
            logger.error(f"コンテキスト: {context[:200]}...")  # コンテキストの一部をログに記録
            logger.error(f"元の質問: {question}")  # 元の質問をログに記録
            return [] 