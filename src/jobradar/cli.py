"""커맨드라인 진입점.

    python -m jobradar ingest [--rebuild]
    python -m jobradar facets
    python -m jobradar ask "Python 쓰는 곳 어디야?" [-k 4] [--show-chunks]
    python -m jobradar ask "신입 뽑아?" --city 부산 --exp 0 --open-only [--show-filter]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from jobradar.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    FINAL_K,
    STORAGE_DIR,
    TOP_K,
)


def _require_keys(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print(f"[에러] 환경변수가 없습니다: {', '.join(missing)}", file=sys.stderr)
        print("       .env.example 을 .env 로 복사하고 키를 채우세요.", file=sys.stderr)
        raise SystemExit(1)


def cmd_ingest(args: argparse.Namespace) -> int:
    _require_keys("OPENAI_API_KEY")
    from jobradar.ingest import build_index

    print(f"공고 디렉터리 : {DATA_DIR}")
    print(f"청킹 설정     : chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    n_docs, n_chunks = build_index(rebuild=args.rebuild)
    print(f"공고 {n_docs}건 -> 청크 {n_chunks}개 임베딩 완료")
    print(f"저장 위치     : {STORAGE_DIR}")
    return 0


def cmd_facets(args: argparse.Namespace) -> int:
    """필터에 쓸 수 있는 값들을 색인에서 읽어 보여줍니다."""
    _require_keys("OPENAI_API_KEY")
    from jobradar.filters import FACET_FIELDS, collect_facets
    from jobradar.ingest import load_vectorstore

    facets = collect_facets(load_vectorstore())
    for label, key, _ in FACET_FIELDS:
        values = facets.get(key, [])
        print(f"\n{label} ({key}) — {len(values)}개")
        print("-" * 70)
        for value in values:
            print(f"  {value}")
    print()
    return 0


def _parse_date_arg(text: str | None) -> int | None:
    """YYYY-MM-DD 를 YYYYMMDD 정수로. loader의 표현과 맞춥니다."""
    if not text:
        return None
    parts = text.strip().split("-")
    if len(parts) != 3:
        raise SystemExit(f"[에러] 날짜 형식은 YYYY-MM-DD 여야 합니다: {text}")
    return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])


def _print_ranking(
    question: str, where: dict | None, k: int | None,
    rerank: bool, min_score: float | None,
) -> None:
    """검색 단계별 순위를 나란히 보여줍니다.

    하이브리드와 리랭킹의 효과는 최종 목록만 봐서는 보이지 않습니다.
    어느 검색기가 무엇을 건졌고 각 단계가 순위를 어떻게 바꿨는지 비교해야 드러납니다.
    """
    from jobradar.config import CANDIDATE_K
    from jobradar.ingest import load_vectorstore
    from jobradar.retrieval import HybridRetriever, bm25_search, fetch_documents

    vs = load_vectorstore()
    vector_hits = vs.similarity_search(question, k=CANDIDATE_K, filter=where or None)
    bm25_hits = bm25_search(fetch_documents(vs, where), question, CANDIDATE_K)

    kwargs = {} if min_score is None else {"min_score": min_score}
    retriever = HybridRetriever(
        vectorstore=vs, where=where, final_k=k or FINAL_K,
        rerank_enabled=rerank, **kwargs,
    )
    final = retriever.invoke(question)
    by_key = {}
    for doc in vector_hits + bm25_hits + final:
        by_key[f"{doc.metadata.get('source', '')}::{hash(doc.page_content)}"] = doc
    fused_keys = [key for key, _ in retriever.trace.get("fused", [])]
    rerank_scores = dict(retriever.trace.get("reranked", []))

    def label(doc) -> str:
        return doc.metadata.get("source", "?").replace("job_", "#").replace(".txt", "")[:24]

    stage = "리랭킹(교차)" if rerank else "RRF 상위"
    print("\n" + "-" * 78)
    print(f"검색 단계 비교 (후보 {CANDIDATE_K} → 최종 {len(final)})")
    print("-" * 78)
    print(f"{'벡터':<26}{'BM25':<26}{'RRF 융합':<26}{stage}")
    for i in range(max(len(final), 3)):
        a = label(vector_hits[i]) if i < len(vector_hits) else "-"
        b = label(bm25_hits[i]) if i < len(bm25_hits) else "-"
        c = label(by_key[fused_keys[i]]) if i < len(fused_keys) else "-"
        if i < len(final):
            key = f"{final[i].metadata.get('source', '')}::{hash(final[i].page_content)}"
            score = rerank_scores.get(key)
            d = label(final[i]) + (f" {score:+.2f}" if score is not None else "")
        else:
            d = "-"
        print(f"{i + 1}.{a:<24}{i + 1}.{b:<24}{i + 1}.{c:<24}{i + 1}.{d}")
    if not bm25_hits:
        print("  * BM25 매칭 0건 — 질의어가 본문에 없어 벡터 검색만으로 결정됨")
    if rerank:
        print("  * 리랭킹 점수는 0~1 (CrossEncoder가 Sigmoid를 씌움). 무관한 문서는 0에 붙음")


def cmd_ask(args: argparse.Namespace) -> int:
    _require_keys("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    from jobradar.chain import ask
    from jobradar.filters import build_where, collect_facets
    from jobradar.ingest import load_vectorstore

    # 2단계: 사람 말을 색인에 실제로 있는 값으로 해석해 where 절을 만듭니다.
    spec = build_where(
        collect_facets(load_vectorstore()),
        city=args.city,
        district=args.district,
        remote=args.remote,
        company=args.company,
        category=args.category,
        exp=args.exp,
        open_only=args.open_only,
        deadline_before=_parse_date_arg(args.deadline_before),
    )

    for line in spec.warnings:
        print(f"[경고] {line}", file=sys.stderr)

    # 조건을 줬는데 해석이 안 됐다면 필터 없이 검색해서는 안 됩니다.
    # 그대로 두면 "광주 공고 없음"이 "아무 공고나 4건"으로 바뀝니다.
    if spec.unsatisfiable:
        print("조건에 맞는 공고가 없습니다. 사용 가능한 값은 다음으로 확인하세요:")
        print("  python -m jobradar facets")
        return 1

    if spec.described:
        print(f"필터: {' / '.join(spec.described)}")
    if args.show_filter:
        print(f"where: {json.dumps(spec.where, ensure_ascii=False)}")

    if args.show_ranking and args.hybrid:
        _print_ranking(args.question, spec.where, args.k, args.rerank, args.min_score)

    result = ask(
        args.question, k=args.k, where=spec.where,
        hybrid=args.hybrid, rerank=args.rerank, min_score=args.min_score,
    )

    print("\n" + "=" * 70)
    print(f"Q. {args.question}")
    print("=" * 70)
    print(result["answer"])

    # 출처는 LLM 의 말이 아니라 검색 결과에서 직접 뽑아 출력합니다.
    print("\n" + "-" * 70)
    print(f"출처 (검색된 청크 {len(result['context'])}개)")
    print("-" * 70)
    if not result["context"]:
        print("  조건에 맞는 공고가 없습니다. 필터를 넓혀 보세요.")
        print("  (사용 가능한 값: python -m jobradar facets)")
        return 0

    for i, doc in enumerate(result["context"], start=1):
        meta = doc.metadata
        print(
            f"[{i}] {meta.get('company', '?')} | {meta.get('title', '?')}\n"
            f"     {meta.get('location', '?')} | {meta.get('experience', '?')} | "
            f"마감 {meta.get('expires_at', '?')}\n"
            f"     파일: {meta.get('source', '?')}  |  {meta.get('url', '')}"
        )
        if args.show_chunks:
            preview = doc.page_content.replace("\n", " ")[:160]
            print(f"     내용: {preview}...")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jobradar", description="채용 공고 검색 어시스턴트 (LangChain RAG)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="공고를 청킹·임베딩해 벡터 인덱스 생성")
    p_ingest.add_argument(
        "--rebuild", action="store_true", help="기존 인덱스를 지우고 새로 만듭니다"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_facets = sub.add_parser("facets", help="필터에 쓸 수 있는 값 목록 보기")
    p_facets.set_defaults(func=cmd_facets)

    p_ask = sub.add_parser("ask", help="질문하고 출처와 함께 답변 받기")
    p_ask.add_argument("question", help="질문 (예: \"Python 쓰는 곳 어디야?\")")
    # 기본값을 None으로 두어야 config가 결정합니다.
    # 하이브리드면 FINAL_K(5), 벡터 단독이면 TOP_K(4).
    p_ask.add_argument(
        "-k", type=int, default=None,
        help=f"프롬프트에 넣을 최종 청크 수 (하이브리드 {FINAL_K}, 벡터단독 {TOP_K})",
    )
    p_ask.add_argument("--show-chunks", action="store_true", help="청크 본문 미리보기 출력")

    # --- 2단계: 메타데이터 필터 (회사 / 직무 / 근무지 / 마감일 / 경력) ---
    f = p_ask.add_argument_group("메타데이터 필터")
    f.add_argument("--city", help="근무지 시도 (예: 부산)")
    f.add_argument("--district", help="근무지 시군구 (예: 해운대구)")
    f.add_argument("--company", help="회사명 일부 (예: 컴앤휴먼)")
    f.add_argument("--category", help="직무 일부 (예: 백엔드). 값 목록은 facets 참고")
    f.add_argument(
        "--exp", type=int, metavar="N",
        help="내 경력 N년으로 지원 가능한 공고만 (신입은 0). 하한<=N<=상한",
    )
    f.add_argument("--open-only", action="store_true", help="오늘 기준 마감 전인 공고만")
    f.add_argument("--deadline-before", metavar="YYYY-MM-DD", help="이 날짜까지 마감인 공고만")
    remote = f.add_mutually_exclusive_group()
    remote.add_argument("--remote", dest="remote", action="store_true", default=None,
                        help="재택 공고만")
    remote.add_argument("--onsite", dest="remote", action="store_false",
                        help="재택이 아닌 공고만")
    f.add_argument("--show-filter", action="store_true", help="생성된 where 절을 출력")

    # --- 3단계: 하이브리드 검색 ---
    h = p_ask.add_argument_group("검색 방식")
    h.add_argument(
        "--no-hybrid", dest="hybrid", action="store_false", default=True,
        help="BM25를 끄고 벡터 검색만 사용 (1·2단계와 동일)",
    )
    h.add_argument(
        "--no-rerank", dest="rerank", action="store_false", default=True,
        help="크로스인코더 리랭킹을 끄고 RRF 순위를 그대로 사용",
    )
    h.add_argument(
        "--min-score", type=float, default=None, metavar="X",
        help="리랭커 점수(0~1)가 X 미만인 문서는 버림 (기본: 안 버림)",
    )
    h.add_argument(
        "--show-ranking", action="store_true",
        help="벡터/BM25/RRF/리랭킹 각 단계의 순위를 비교 출력",
    )

    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
