"""3단계 후반부: 크로스인코더 리랭킹.

배우는 개념
-----------
지금까지 쓴 임베딩은 **바이인코더(bi-encoder)** 입니다. 질문과 문서를 *각각 따로*
벡터로 만든 뒤 거리를 잽니다. 문서 벡터를 미리 계산해 둘 수 있어서 100만 건이든
밀리초에 끝나지만, 질문과 문서가 서로를 보지 못한 채 인코딩되므로 미묘한 관계를
놓칩니다.

**크로스인코더(cross-encoder)** 는 반대입니다. `(질문, 문서)`를 한 쌍으로 묶어
모델에 통째로 넣고 관련도 점수 하나를 뽑습니다. 어텐션이 질문의 단어와 문서의
단어를 직접 맞대 보므로 훨씬 정확합니다. 대신 **미리 계산할 수 없습니다** —
질문이 올 때마다 문서 수만큼 모델을 돌려야 합니다.

    바이인코더   문서 → 벡터 (미리)      질문 → 벡터      거리 계산      싸다
    크로스인코더  (질문, 문서) → 점수 (질문이 와야 계산 가능)            비싸다

그래서 둘을 **2단으로** 씁니다.

    싼 검색(벡터+BM25)으로 넓게 20개  →  비싼 크로스인코더로 정밀하게 5개
    ── 재현율 확보 ──                    ── 정밀도 확보 ──

20개만 다시 채점하면 되므로 비용이 감당 가능해집니다. 이것이 CLAUDE.md의
"상위 20 → 5"가 뜻하는 구조입니다.

왜 한국어 전용 모델인가
-----------------------
`cross-encoder/ms-marco-*` 계열은 영어 전용이라 한국어 질문에 거의 무력합니다.
여기서는 `Dongjin-kr/ko-reranker`(bge-reranker-large를 한국어로 파인튜닝)를 씁니다.
첫 실행 때 약 1.1GB를 내려받아 캐시에 두고, 이후에는 로컬에서 읽습니다.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langchain_core.documents import Document

from jobradar.config import RERANK_BATCH, RERANK_MIN_SCORE, RERANK_MODEL

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_reranker(model_name: str = RERANK_MODEL) -> Any:
    """크로스인코더를 한 번만 로드해 재사용합니다.

    모델 로딩은 수 초가 걸리므로 질문마다 다시 하면 안 됩니다. `lru_cache`가
    프로세스 안에서 싱글턴 역할을 합니다. (CLI는 한 번 실행하고 끝나므로 체감이
    없지만, 나중에 FastAPI를 붙이면 이 캐시가 그대로 값을 합니다.)
    """
    from sentence_transformers import CrossEncoder

    logger.info("리랭커 로딩: %s", model_name)
    return CrossEncoder(model_name, max_length=512)


def rerank(
    query: str,
    docs: list[Document],
    top_n: int,
    *,
    model_name: str = RERANK_MODEL,
    batch_size: int = RERANK_BATCH,
    min_score: float = RERANK_MIN_SCORE,
) -> list[tuple[Document, float]]:
    """(질문, 문서) 쌍마다 점수를 매겨 상위 top_n개를 돌려줍니다.

    점수는 **0~1 범위**입니다. `CrossEncoder`가 출력 라벨이 하나일 때
    `activation_fn=Sigmoid()`를 기본으로 씌우기 때문입니다(실측으로 확인).
    로짓 -2.67 -> 0.065, -8.94 -> 0.00013 처럼 무관한 문서는 0에 바싹 붙습니다.

    `min_score`를 주면 그 아래를 버립니다. 기본값 0.0은 "버리지 않음"이며,
    **임계값은 모델과 데이터에 종속되므로 반드시 실측해서 정해야 합니다.**
    """
    if not docs:
        return []

    model = get_reranker(model_name)
    # 크로스인코더에 넣는 문서 텍스트는 검색 때와 같은 표현을 씁니다.
    # 본문만 넣으면 회사명·기술 키워드가 빠져 판단 근거가 줄어듭니다.
    from jobradar.retrieval import searchable_text

    pairs = [(query, searchable_text(doc)) for doc in docs]
    scores = model.predict(pairs, batch_size=batch_size)

    ranked = sorted(zip(docs, (float(s) for s in scores)), key=lambda p: p[1], reverse=True)
    if min_score > 0:
        ranked = [pair for pair in ranked if pair[1] >= min_score]
    return ranked[:top_n]
