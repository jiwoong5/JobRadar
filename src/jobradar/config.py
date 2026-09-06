"""프로젝트 전역 설정.

3단계까지 진행하면서 바뀌는 값은 대부분 여기에 모여 있습니다.
경로는 전부 환경변수로 덮어쓸 수 있게 해두었습니다 - 나중에 Docker로
옮길 때 코드를 고치지 않고 환경변수만 주입하면 되도록 하기 위함입니다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# src/jobradar/config.py -> parents[2] == 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("JOBRADAR_DATA_DIR") or PROJECT_ROOT / "data" / "raw")
STORAGE_DIR = Path(os.getenv("JOBRADAR_STORAGE_DIR") or PROJECT_ROOT / "storage" / "chroma")

COLLECTION_NAME = "job_postings"

# --- 모델 ---
# 임베딩: Anthropic은 임베딩 API를 제공하지 않으므로 OpenAI를 사용합니다.
EMBEDDING_MODEL = os.getenv("JOBRADAR_EMBEDDING_MODEL", "text-embedding-3-small")
# 답변 생성
CHAT_MODEL = os.getenv("JOBRADAR_CHAT_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.getenv("JOBRADAR_MAX_TOKENS", "4096"))

# --- 청킹 ---
# 공고 한 건이 1,800~2,100자 수준이므로 800자면 공고당 3~4청크가 나옵니다.
CHUNK_SIZE = int(os.getenv("JOBRADAR_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("JOBRADAR_CHUNK_OVERLAP", "120"))

# --- 검색 ---
TOP_K = int(os.getenv("JOBRADAR_TOP_K", "4"))
