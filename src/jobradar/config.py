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
# 1·2단계(벡터 단독)에서 쓰는 값입니다.
TOP_K = int(os.getenv("JOBRADAR_TOP_K", "4"))

# --- 3단계: 하이브리드 검색 ---
# "넓게 20개 건진 뒤 정밀하게 5개로 줄인다"는 2단 구성입니다.
# 싼 검색으로 재현율을 확보하고, 비싼 재순위로 정밀도를 확보합니다.
CANDIDATE_K = int(os.getenv("JOBRADAR_CANDIDATE_K", "20"))
FINAL_K = int(os.getenv("JOBRADAR_FINAL_K", "5"))

# RRF: 점수 스케일이 다른 두 랭킹을 등수만으로 합칩니다. 60은 원 논문의 관행값으로,
# 상위권의 등수 차이를 완만하게 만들어 한쪽 검색기가 독주하는 것을 막습니다.
RRF_K = int(os.getenv("JOBRADAR_RRF_K", "60"))
VECTOR_WEIGHT = float(os.getenv("JOBRADAR_VECTOR_WEIGHT", "1.0"))
BM25_WEIGHT = float(os.getenv("JOBRADAR_BM25_WEIGHT", "1.0"))

# --- 3단계: 크로스인코더 리랭킹 ---
# 한국어로 파인튜닝된 리랭커. bge-reranker-large 계열이라 첫 실행 시
# 약 1.1GB를 내려받아 캐시(~/.cache/huggingface)에 둡니다.
RERANK_MODEL = os.getenv("JOBRADAR_RERANK_MODEL", "Dongjin-kr/ko-reranker")
RERANK_ENABLED = os.getenv("JOBRADAR_RERANK", "1") not in ("0", "false", "False")
# CPU에서 한 번에 넘길 (질문, 문서) 쌍의 수.
RERANK_BATCH = int(os.getenv("JOBRADAR_RERANK_BATCH", "16"))
# 리랭커 점수(0~1) 하한. 0.0은 "거르지 않음". 무관한 문서는 0에 바싹 붙으므로
# 0.01 정도만 줘도 노이즈가 걸러지지만, 값은 모델·데이터마다 다르니 실측할 것.
RERANK_MIN_SCORE = float(os.getenv("JOBRADAR_RERANK_MIN_SCORE", "0.0"))
