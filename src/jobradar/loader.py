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

import datetime
import re
from pathlib import Path
from typing import Any

import yaml

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from jobradar.config import DATA_DIR

# 파일 맨 앞의 ---...--- 블록을 잡아냅니다.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# 사람이 읽는 값. 화면 출력과 프롬프트 헤더에 쓰입니다.
_KEEP_FIELDS = (
    "site",
    "job_id",
    "url",
    "company",
    "title",
    "industry",
    "location",
    "job_category",
    "job_type",
    "experience",
    "education",
    "salary",
    "posted_at",
    "expires_at",
)

# 2단계: 기계가 거르는 값. frontmatter에 이미 들어 있는 정수 필드입니다.
_INT_FIELDS = ("experience_min", "experience_max")

# 2단계: 리스트로 저장할 필드. Chroma는 동종 스칼라 리스트를 허용하며
# `{"필드": {"$contains": "값"}}`으로 원소를 정확히 일치 검색할 수 있습니다.
_LIST_FIELDS = {
    "job_categories": "job_category",  # "웹개발, 백엔드·서버개발" -> ["웹개발", "백엔드·서버개발"]
    "keywords": "keywords",            # YAML이 이미 리스트로 파싱해 줍니다
}


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


def _listify(value: Any) -> list[str]:
    """리스트 필드를 문자열 리스트로 정규화합니다.

    `job_category`는 YAML에서 "웹개발, 백엔드·서버개발" 한 줄짜리 문자열이고
    `keywords`는 이미 리스트라, 두 경우를 모두 받아 같은 모양으로 만듭니다.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return [s for s in (str(v).strip() for v in items) if s]


def _date_to_int(value: Any) -> int:
    """날짜를 YYYYMMDD 정수로 바꿉니다.

    Chroma의 `$gte`/`$lte`는 숫자에만 걸리므로, "마감일이 오늘 이후" 같은
    비교를 하려면 정수 필드가 따로 필요합니다. YAML이 `2026-08-28`을
    date 객체로 파싱한다는 점을 그대로 이용합니다.
    """
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.year * 10000 + value.month * 100 + value.day
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(value))
    return int(match.group(1) + match.group(2) + match.group(3)) if match else 0


def _split_location(value: Any) -> tuple[str, str, bool]:
    """근무지를 (시도, 시군구, 재택여부)로 쪼갭니다.

    Chroma의 where 절에는 부분 문자열 매칭이 없습니다. "부산 > 해운대구"를
    통째로 두면 "부산"으로 거를 방법이 없으므로, 색인 시점에 미리 쪼개 둡니다.
    """
    text = str(value or "").strip()
    if "재택" in text or "원격" in text:
        return "재택", "", True
    city, _, district = text.partition(">")
    return city.strip(), district.strip(), False


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

        # --- 2단계: 여기서부터 필터 전용 파생 필드 ---
        for field in _INT_FIELDS:
            if field in front:
                metadata[field] = int(front[field])

        for target, origin in _LIST_FIELDS.items():
            metadata[target] = _listify(front.get(origin))

        city, district, is_remote = _split_location(front.get("location"))
        metadata["location_city"] = city
        metadata["location_district"] = district
        metadata["is_remote"] = is_remote

        metadata["posted_at_num"] = _date_to_int(front.get("posted_at"))
        metadata["expires_at_num"] = _date_to_int(front.get("expires_at"))

        documents.append(Document(page_content=body.strip(), metadata=metadata))

    documents.sort(key=lambda d: d.metadata["source"])
    return documents
