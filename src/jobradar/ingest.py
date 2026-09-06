"""② 청킹 → ③ 임베딩 → 저장. 오프라인에서 한 번만 도는 단계입니다.

배우는 개념
-----------
- RecursiveCharacterTextSplitter: separators 목록을 앞에서부터 시도하며
  chunk_size 아래로 떨어질 때까지 재귀적으로 쪼갭니다. 즉 "## 제목"에서
  먼저 끊고, 그래도 크면 문단 → 줄 → 문장 → 단어 순으로 내려갑니다.
- chunk_overlap: 청크 경계에서 문맥이 잘리는 걸 막는 완충 구간입니다.
- Chroma.from_documents: 청크를 임베딩해 벡터와 함께 디스크에 저장합니다.

왜 ask 와 분리했나
------------------
임베딩은 돈과 시간이 드는 작업이라 질문마다 다시 할 이유가 없습니다.
또 3단계에서 인덱스 구성(BM25 추가)을 바꿀 때, 이 파일만 손대면 됩니다.
"""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from jobradar.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    STORAGE_DIR,
)
from jobradar.loader import load_postings


def get_embeddings() -> OpenAIEmbeddings:
    """임베딩 모델. ingest 와 ask 가 반드시 같은 것을 써야 합니다."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 공고가 "## 주요업무" 같은 섹션 구조라 헤더를 최우선 경계로 둡니다.
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    # split_documents 는 원본 Document 의 metadata 를 모든 자식 청크에 복사해 줍니다.
    return splitter.split_documents(documents)


def build_index(rebuild: bool = False) -> tuple[int, int]:
    """공고를 읽어 벡터 인덱스를 만듭니다. (문서 수, 청크 수)를 돌려줍니다."""
    documents = load_postings()
    if not documents:
        raise RuntimeError("공고 파일을 찾지 못했습니다. data/raw/*.txt 를 확인하세요.")

    chunks = split_documents(documents)

    if rebuild and STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(STORAGE_DIR),
    )
    return len(documents), len(chunks)


def load_vectorstore() -> Chroma:
    """이미 만들어진 인덱스를 엽니다 (ask 에서 사용)."""
    if not STORAGE_DIR.exists():
        raise RuntimeError(
            "벡터 인덱스가 없습니다. 먼저 `python -m jobradar ingest` 를 실행하세요."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(STORAGE_DIR),
    )
