"""① 문서 로드 - 텍스트 파일을 LangChain Document 로 바꾸는 단계.

배우는 개념
-----------
Document 는 `page_content`(임베딩될 본문)와 `metadata`(딕셔너리) 두 칸으로
이루어진 LangChain의 기본 자료구조입니다. 이 단계의 핵심은 "무엇을 본문에
넣고 무엇을 메타데이터로 뺄 것인가"를 정하는 일입니다.

여기서는 공고 파일 상단의 YAML frontmatter 를 metadata 로 승격시키고,
본문만 page_content 에 남깁니다. frontmatter 필드명은 사람인 채용정보 API
응답 필드를 본떠 두었으므로, 나중에 실제 크롤링/API 데이터로 교체할 때
이 loader 만 갈아끼우면 됩니다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from jobradar.config import DATA_DIR

# 파일 맨 앞의 ---...--- 블록을 잡아냅니다.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# 1단계에서 실제로 쓰는 메타데이터. (2단계에서 필터용 필드를 여기에 추가합니다)
_KEEP_FIELDS = (
    "site",
    "job_id",
    "url",
    "company",
    "title",
    "location",
    "job_category",
    "job_type",
    "experience",
    "education",
    "salary",
    "posted_at",
    "expires_at",
)


def _scalarize(value: Any) -> str | int | float | bool:
    """Chroma 메타데이터는 str/int/float/bool 만 허용합니다.

    YAML은 `keywords: [A, B]`를 list로, `posted_at: 2026-08-28`을 date 객체로
    파싱하기 때문에 그대로 넣으면 저장 단계에서 터집니다.
    """
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():]
    return meta, body


def load_postings() -> list[Document]:
    """data/raw/*.txt 를 읽어 Document 리스트로 돌려줍니다."""
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        # 한글 파일이므로 인코딩을 명시하지 않으면 Windows에서 cp949로 읽다 깨집니다.
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()

    documents: list[Document] = []
    for raw in raw_docs:
        front, body = _split_frontmatter(raw.page_content)

        metadata: dict[str, Any] = {
            # DirectoryLoader가 넣어준 전체 경로 대신 파일명만 남깁니다.
            "source": Path(raw.metadata.get("source", "")).name,
        }
        for field in _KEEP_FIELDS:
            if field in front:
                metadata[field] = _scalarize(front[field])

        documents.append(Document(page_content=body.strip(), metadata=metadata))

    documents.sort(key=lambda d: d.metadata["source"])
    return documents
