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

from jobradar.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, STORAGE_DIR, TOP_K


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

    result = ask(args.question, k=args.k, where=spec.where)

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
    p_ask.add_argument("-k", type=int, default=TOP_K, help=f"검색할 청크 수 (기본 {TOP_K})")
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

    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
