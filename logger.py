import logging
import sys
from pathlib import Path
from config import LOGS_DIR

def setup_logger(name="LocalQA"):
    """アプリケーション用のロガーをセットアップします"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ファイルハンドラの設定
    log_file = LOGS_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)

    # ストリームハンドラの設定（コンソール出力用）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_format)

    # ハンドラの追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger 