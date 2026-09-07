"""2단계: 사람 말 → Chroma where 절.

배우는 개념
-----------
벡터 검색은 "의미가 비슷한 것"을 찾을 뿐, "부산인 것만"처럼 조건을 **단정**하지
못합니다. 근무지·경력·마감일처럼 답이 딱 떨어지는 조건은 메타데이터 필터로
걸러야 합니다. 이것이 1단계와 2단계를 가르는 지점입니다.

사전 필터링(pre-filtering)
--------------------------
Chroma는 벡터를 훑기 **전에** where 절로 후보를 줄입니다. 검색한 뒤 파이썬에서
거르는 사후 필터링과 달리, k개를 온전히 채울 수 있습니다. 상위 4개를 뽑고 나서
부산 아닌 걸 빼면 1~2개만 남지만, 부산으로 먼저 줄이고 4개를 뽑으면 4개가 옵니다.

Chroma where 절의 한계 (실측으로 확인한 것)
-------------------------------------------
- 부분 문자열 매칭이 **없습니다**. `{"company": {"$contains": "컴앤휴먼"}}`은
  `(주)컴앤휴먼`을 찾지 못합니다.
- `$contains`는 **리스트 필드의 원소 정확 일치**로만 동작합니다.
  `job_categories`가 `["웹개발", "백엔드·서버개발"]`일 때 `$contains "백엔드"`는
  빈 결과이고, `$contains "백엔드·서버개발"`이라야 걸립니다.
- `$gte`/`$lte`는 숫자에만 걸립니다. 그래서 loader가 `expires_at_num`(YYYYMMDD)을
  따로 만들어 둡니다.

그래서 이 모듈의 역할은 하나입니다 — **어휘 해석은 파이썬이 하고, Chroma에는
정확한 값만 넘긴다.** 사용자가 "백엔드"라고 쓰면 색인에 실제로 존재하는 값
목록에서 `백엔드·서버개발`을 찾아낸 뒤, 그 정확한 값으로 where 절을 만듭니다.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any


@dataclasses.dataclass
class FilterSpec:
    """조립된 필터 하나.

    `unsatisfiable`이 중요합니다. 사용자가 "광주"로 걸렀는데 색인에 광주가 없으면
    where 절은 비게 되는데, 그대로 두면 **필터가 사라져 전체 검색이 되어 버립니다.**
    "조건에 맞는 게 없다"와 "조건 없이 아무거나"는 정반대의 답이므로, 조건이
    주어졌으나 해석에 실패한 경우를 따로 표시합니다.
    """

    where: dict[str, Any] | None = None
    described: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    unsatisfiable: bool = False

# facets 명령이 보여줄 필드. (표시 이름, 메타데이터 키, 리스트 여부)
FACET_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("회사", "company", False),
    ("직무", "job_categories", True),
    ("근무지 시도", "location_city", False),
    ("근무지 시군구", "location_district", False),
    ("업종", "industry", False),
    ("기술 키워드", "keywords", True),
)


def today_int() -> int:
    """오늘 날짜를 YYYYMMDD 정수로. loader의 `_date_to_int`와 같은 표현입니다."""
    t = datetime.date.today()
    return t.year * 10000 + t.month * 100 + t.day


def collect_facets(vectorstore: Any) -> dict[str, list[str]]:
    """색인에 실제로 존재하는 값들을 모읍니다.

    필터의 후보 어휘는 코드에 하드코딩하지 않고 **색인에서 읽어 옵니다.**
    공고가 늘거나 바뀌면 후보도 자동으로 따라오게 하기 위해서입니다.
    (문서가 많아지면 이 방식은 느려집니다. 그때는 별도 사전을 두어야 합니다.)
    """
    got = vectorstore._collection.get(include=["metadatas"])
    facets: dict[str, set[str]] = {key: set() for _, key, _ in FACET_FIELDS}

    for meta in got["metadatas"]:
        for _, key, is_list in FACET_FIELDS:
            value = meta.get(key)
            if value is None:
                continue
            if is_list:
                facets[key].update(str(v) for v in value)
            elif str(value).strip():
                facets[key].add(str(value))

    return {key: sorted(values) for key, values in facets.items()}


def resolve(user_value: str, candidates: list[str]) -> list[str]:
    """사용자가 쓴 말을 색인에 있는 정확한 값으로 바꿉니다.

    Chroma가 못 하는 부분 문자열 매칭을 여기서 대신합니다.
    정확히 일치하는 값이 있으면 그것만, 없으면 부분 일치를 전부 돌려줍니다.
    """
    needle = user_value.strip().casefold()
    if not needle:
        return []

    exact = [c for c in candidates if c.casefold() == needle]
    if exact:
        return exact
    return [c for c in candidates if needle in c.casefold()]


def build_where(
    facets: dict[str, list[str]],
    *,
    city: str | None = None,
    district: str | None = None,
    remote: bool | None = None,
    company: str | None = None,
    category: str | None = None,
    exp: int | None = None,
    open_only: bool = False,
    deadline_before: int | None = None,
) -> FilterSpec:
    """필터 조건들을 Chroma where 절 하나로 조립합니다."""
    clauses: list[dict[str, Any]] = []
    described: list[str] = []
    warnings: list[str] = []
    unsatisfiable = False

    def add_scalar(user_value: str, key: str, label: str) -> None:
        nonlocal unsatisfiable
        matched = resolve(user_value, facets.get(key, []))
        if not matched:
            warnings.append(f"{label} '{user_value}'와 일치하는 값이 색인에 없습니다.")
            unsatisfiable = True
            return
        clauses.append({key: matched[0] if len(matched) == 1 else {"$in": matched}})
        described.append(f"{label}={', '.join(matched)}")

    if city:
        add_scalar(city, "location_city", "근무지")
    if district:
        add_scalar(district, "location_district", "시군구")
    if company:
        add_scalar(company, "company", "회사")

    if category:
        matched = resolve(category, facets.get("job_categories", []))
        if not matched:
            warnings.append(f"직무 '{category}'와 일치하는 값이 색인에 없습니다.")
            unsatisfiable = True
        else:
            # 리스트 필드는 $contains(원소 정확 일치)로만 걸립니다.
            per_value = [{"job_categories": {"$contains": m}} for m in matched]
            clauses.append(per_value[0] if len(per_value) == 1 else {"$or": per_value})
            described.append(f"직무={', '.join(matched)}")

    if remote is not None:
        clauses.append({"is_remote": remote})
        described.append("재택" if remote else "재택 아님")

    if exp is not None:
        # "경력 N년인 내가 지원 가능한 공고" = 하한 <= N <= 상한
        # 신입(0~0)과 경력무관(0~99)이 exp=0에 함께 걸리는 것이 의도된 동작입니다.
        clauses.append({"$and": [
            {"experience_min": {"$lte": exp}},
            {"experience_max": {"$gte": exp}},
        ]})
        described.append(f"경력 {exp}년 지원가능")

    if open_only:
        today = today_int()
        clauses.append({"expires_at_num": {"$gte": today}})
        described.append(f"마감 전 (오늘 {today})")

    if deadline_before is not None:
        clauses.append({"expires_at_num": {"$lte": deadline_before}})
        described.append(f"마감일 {deadline_before} 이전")

    # Chroma는 최상위에 조건이 여럿이면 $and로 묶어 줘야 합니다.
    where = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})
    return FilterSpec(
        where=where, described=described, warnings=warnings, unsatisfiable=unsatisfiable
    )
