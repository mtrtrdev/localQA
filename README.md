# LocalQA - ローカル文書QAシステム

ローカルフォルダ内の非構造データ（PDF, TXT, DOCX等）をエンベディングし、自然言語による質問に対して内容ベースで回答するアプリケーションです。

## 🔧 主な機能

- 複数のデータベース管理（作成・選択・削除）
- 各種ファイル形式のサポート（PDF, TXT, DOCX）
- ドキュメントのチャンク分割とエンベッディング
- 自然言語による質問応答
- 関連する質問の提案
- 回答の出典表示

## 🛠 セットアップ

1. 必要なパッケージのインストール：
```bash
pip install -r requirements.txt
```

2. 環境変数の設定：
- `.env`ファイルを作成し、OpenAI APIキーを設定：
```
OPENAI_API_KEY=your_api_key_here
```

3. アプリケーションの起動：
```bash
streamlit run app.py
```

## 📦 プロジェクト構造

```
local_qa/
├── app.py              # Streamlitアプリのメイン
├── file_loader.py      # フォルダ読み込みとファイル形式別の処理
├── embedder.py         # エンベッディング処理
├── vector_store.py     # ChromaDB処理
├── qa_engine.py        # 質問→コンテキスト→回答生成
├── config.py           # パラメータ設定
├── logger.py           # ログ設定
├── logs/               # ログファイル保存先
├── data/               # アップロードされたファイル群
├── index/              # ベクトルDB格納先
├── requirements.txt
└── README.md
```

## 💡 使用方法

### データベース管理画面
1. サイドバーの「データベース管理」を選択
2. データベースの操作:
   - 新規データベースの作成
   - 既存データベースの選択（ラジオボタン）
   - データベースの削除
   - データベースの内容クリア
3. ファイルのアップロード:
   - 対応フォーマット: PDF, TXT, DOCX
   - 複数ファイルの同時アップロード可能
4. データベース分析:
   - 総チャンク数、総ファイル数、総文字数の表示
   - チャンク長の分布
   - ファイルごとのチャンク数分布
   - ファイル詳細一覧

### 質問応答画面
1. サイドバーの「質問応答」を選択
2. 使用するデータベースを選択
3. 質問を入力
4. 「質問する」ボタンをクリック
5. 回答の表示:
   - AIによる回答
   - 参照元のテキストチャンク
   - 関連する質問の提案

### データベース分析機能
- 基本統計情報:
  - 総チャンク数
  - 総ファイル数
  - 総文字数
  - 平均チャンク長
- 分布表示:
  - チャンク長の分布グラフ
  - ファイルごとのチャンク数分布
- ファイル詳細:
  - ファイル名
  - チャンク数
  - 総文字数
  - ファイルタイプ

## 🔒 注意事項

- APIキーは必ず`.env`ファイルで管理し、公開しないようにしてください
- アップロードするファイルサイズの上限は200MBです
- 大量のファイルを処理する場合は時間がかかる場合があります
- データベースの削除は元に戻せません
- チャンク分割は自動で行われます（デフォルト設定: 500文字、オーバーラップ100文字）

## 🔧 トラブルシューティング

1. アップロードエラー:
   - ファイルサイズを確認
   - 対応フォーマットであることを確認
2. 質問応答エラー:
   - データベースが選択されていることを確認
   - OpenAI APIキーが正しく設定されていることを確認
3. データベースエラー:
   - 権限の確認
   - ディスク容量の確認

## 🔧 技術仕様

- 使用言語: Python
  - Pythonは、豊富なライブラリとフレームワークを持ち、データ処理や機械学習に適した言語です。

- フレームワーク: Streamlit
  - Streamlitは、データアプリケーションを簡単に構築できるフレームワークで、インタラクティブなUIを提供します。

- ベクトルデータベース: FAISS
  - FAISSは、Facebookが開発した高速な類似検索とクラスタリングのライブラリで、大規模なベクトルデータの検索に適しています。

- 埋め込みモデル: OpenAI
  - OpenAIのモデルを使用して、テキストデータをベクトル化し、自然言語処理を行います。

- チャンク分割: 500文字（オーバーラップ100文字）
  - ドキュメントを500文字ごとに分割し、100文字のオーバーラップを持たせることで、文脈を保ちながら情報を処理します。

- マルチチャンク検索: 前後のチャンクを含めて検索
  - 質問に対する回答を生成する際、関連する複数のチャンクを同時に検索し、より正確な回答を提供します。
