"""3단계 전반부: BM25 키워드 검색 + 벡터 검색을 RRF로 융합.

배우는 개념
-----------
1·2단계의 검색은 임베딩 하나였습니다. 그래서 "MS-SQL 다루는 자리"를 물으면
MS-SQL이 없는 공고가 상위에 섞였습니다 — 임베딩은 "데이터베이스 관련 이야기"라는
뭉뚱그린 의미로 가까운 것을 끌어올 뿐, **정확한 문자열의 유무를 구분하지 못하기**
때문입니다.

BM25는 정확히 반대입니다. 의미는 전혀 모르지만 "MS-SQL"이라는 토큰이 실제로
몇 번 나왔는지를 셉니다. 게다가 IDF 가중치 덕에 **희귀한 단어일수록 강하게**
반영합니다("및", "그리고"는 값이 싸고 "MS-SQL"은 비쌉니다).

    score(D,Q) = Σ IDF(qᵢ) · f(qᵢ,D)·(k₁+1) / (f(qᵢ,D) + k₁·(1-b + b·|D|/avgdl))

두 검색기의 약점이 서로 반대라서 합치면 이득입니다. 문제는 **점수 스케일이 다르다**는
것입니다. 벡터는 제곱 L2 거리(낮을수록 좋음), BM25는 무한대로 열린 양수(높을수록 좋음)라
그냥 더할 수 없습니다. 그래서 점수를 버리고 **등수만 가지고** 합칩니다 — RRF입니다.

    RRF(d) = Σᵢ weightᵢ / (rrf_k + rankᵢ(d))      보통 rrf_k = 60

정규화 고민이 사라지고, 한쪽에서 1등이지만 다른 쪽에 아예 없는 문서도
자연스럽게 살아남습니다.

2단계 필터와의 관계
-------------------
메타데이터 필터는 이 위층이 아니라 **아래층**입니다. 두 검색기 모두 필터가 적용된
같은 후보군 위에서 돕니다. 벡터 쪽은 Chroma의 where 절로, BM25 쪽은 같은 where로
문서를 먼저 가져와 그 위에 색인을 세워서요.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from jobradar.config import BM25_WEIGHT, CANDIDATE_K, FINAL_K, RRF_K, VECTOR_WEIGHT

# 영문·숫자·기호가 붙은 기술 용어(C#, .NET, MS-SQL)와 한글 덩어리를 따로 잡습니다.
# '-'는 토큰 문자에 넣지 않았습니다. "MS-SQL"을 ["ms","sql"]로 쪼개 두면
# "MS SQL"처럼 띄어 쓴 표기와도 걸리기 때문입니다.
_TOKEN_RE = re.compile(r"[A-Za-z0-9#+.]+|[가-힣]+")

# 불용어. 두 갈래로 나뉩니다.
#   (1) 질문에 섞이는 의문사·기능어: 어디, 있어, 알려줘 ...
#   (2) 채용 공고라면 거의 모두가 가진 말: 채용, 모집, 업무, 경험, 운영 ...
# 둘 다 변별력이 없는데, **문서가 10건뿐이라 IDF가 이를 걸러주지 못합니다.**
# 실측: "Python 쓰는 곳 어디야?"에서 #9(Node.js)가 1.86점으로 1위였는데
# 전부 '어디' 토큰에서 나온 점수였습니다(python은 0.71). 소규모 코퍼스에서는
# 우연히 한 문서에만 있는 기능어가 희귀한 전문용어처럼 보입니다.
#
# 코퍼스에 종속된 목록이므로, 데이터가 바뀌면 다시 살펴봐야 합니다.
_STOPWORDS = frozenset(
    """
    어디 어디야 어디에 어디서 무엇 뭐 뭔가 어떤 어떻게 누가 언제 왜
    있어 있나 있는 있을 없어 없나 하는 해줘 알려줘 알려 주는 쓰는 다루는 찾는
    곳 자리 것 등 및 그리고 또는 좀 지금 요즘 정도
    지원 채용 공고 모집 회사 업무 담당 경험 운영 관련 가능 우대 자격 요건 조건
    """.split()
)


def tokenize(text: str) -> list[str]:
    """BM25용 토크나이저. 색인과 질의에 **똑같이** 적용해야 합니다.

    한글 형태소 분석은 하지 않습니다("부산에서"와 "부산"이 다른 토큰으로 남습니다).
    형태소 분석기는 무겁고, 여기서 BM25가 맡은 일은 **정확한 기술 용어**를 잡는
    것이기 때문입니다 — 그쪽은 대부분 영문이라 공백 분리로 충분합니다.
    한국어 의미 매칭은 벡터 검색이 이미 잘합니다. 역할을 나누는 편이 낫습니다.

    (글자 2-gram으로 부분 일치를 흉내 내 봤지만 노이즈만 늘어 걷어냈습니다.
     위 _STOPWORDS 주석의 실측을 참고하세요.)
    """
    return [
        token
        for token in (raw.lower() for raw in _TOKEN_RE.findall(text or ""))
        if token not in _STOPWORDS
    ]


def searchable_text(doc: Document) -> str:
    """BM25가 실제로 색인할 문자열.

    본문만 넣으면 회사명·기술 키워드가 빠집니다. 이 값들은 metadata로 승격되면서
    본문에서 사라졌으므로(loader가 frontmatter를 잘라냅니다) 되돌려 붙여 줍니다.
    검색 엔진에서 title/tag 필드에 가중치를 주는 것과 같은 발상입니다.
    """
    meta = doc.metadata
    keywords = meta.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    parts = [
        str(meta.get("title", "")),
        str(meta.get("company", "")),
        str(meta.get("job_category", "")),
        " ".join(str(k) for k in keywords),
        doc.page_content,
    ]
    return "\n".join(p for p in parts if p)


def fetch_documents(vectorstore: Any, where: dict | None = None) -> list[Document]:
    """BM25 색인을 세울 문서를 색인에서 직접 가져옵니다.

    `data/raw`를 다시 읽지 않는 이유는 **벡터 인덱스가 진실의 원천**이기 때문입니다.
    두 검색기가 서로 다른 문서 집합을 보면 융합 결과가 어긋납니다.

    주의: 질문마다 BM25 색인을 새로 만듭니다. 문서가 10개라 무시할 만한 비용이지만,
    규모가 커지면 색인을 미리 만들어 두거나(ingest에서 pickle) 전문 검색 엔진으로
    옮겨야 합니다.
    """
    got = vectorstore._collection.get(
        where=where or None, include=["documents", "metadatas"]
    )
    return [
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(got["documents"], got["metadatas"])
    ]


def bm25_search(docs: list[Document], query: str, k: int) -> list[Document]:
    """BM25 상위 k개. 문서가 없으면 빈 리스트."""
    if not docs:
        return []
    corpus = [tokenize(searchable_text(d)) for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    # 점수 0은 질의어가 하나도 안 나온 문서입니다. 순위에 올릴 이유가 없습니다.
    return [doc for doc, score in ranked[:k] if score > 0]


def _key(doc: Document) -> str:
    """융합할 때 같은 문서인지 판별할 키. 청크 단위라 본문이 곧 신원입니다."""
    return f"{doc.metadata.get('source', '')}::{hash(doc.page_content)}"


def rrf_fuse(
    rankings: list[tuple[list[Document], float]], rrf_k: int = RRF_K
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion. 점수는 버리고 등수만 씁니다.

    rankings 는 (문서목록, 가중치) 쌍의 목록입니다. 목록의 순서가 곧 등수입니다.
    """
    totals: dict[str, float] = {}
    seen: dict[str, Document] = {}

    for docs, weight in rankings:
        for rank, doc in enumerate(docs, start=1):
            key = _key(doc)
            seen.setdefault(key, doc)
            totals[key] = totals.get(key, 0.0) + weight / (rrf_k + rank)

    return sorted(
        ((seen[key], score) for key, score in totals.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


class HybridRetriever(BaseRetriever):
    """벡터 검색과 BM25를 RRF로 합치는 Retriever.

    Retriever 인터페이스(str -> list[Document])를 그대로 지키므로, 체인의 나머지
    (프롬프트·생성·출처 출력)는 한 줄도 바뀌지 않습니다. 1단계에서 이 경계를
    지켜 둔 것이 여기서 값을 합니다.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    where: dict | None = None
    candidate_k: int = CANDIDATE_K
    final_k: int = FINAL_K
    rrf_k: int = RRF_K
    vector_weight: float = VECTOR_WEIGHT
    bm25_weight: float = BM25_WEIGHT
    trace: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        vector_hits = self.vectorstore.similarity_search(
            query, k=self.candidate_k, filter=self.where or None
        )
        bm25_hits = bm25_search(
            fetch_documents(self.vectorstore, self.where), query, self.candidate_k
        )

        fused = rrf_fuse(
            [(vector_hits, self.vector_weight), (bm25_hits, self.bm25_weight)],
            rrf_k=self.rrf_k,
        )

        # 어느 검색기가 무엇을 건졌는지 남겨 둡니다. 하이브리드의 효과는
        # 최종 목록만 봐서는 안 보이고, 두 갈래를 비교해야 드러납니다.
        self.trace.clear()
        self.trace.update(
            vector=[_key(d) for d in vector_hits],
            bm25=[_key(d) for d in bm25_hits],
            fused=[(_key(d), s) for d, s in fused],
        )
        return [doc for doc, _ in fused[: self.final_k]]
