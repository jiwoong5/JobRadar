"""커맨드라인 진입점.

    python -m jobradar ingest [--rebuild]
    python -m jobradar ask "Python 쓰는 곳 어디야?" [-k 4] [--show-chunks]
"""

from __future__ import annotations

import argparse
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


def cmd_ask(args: argparse.Namespace) -> int:
    _require_keys("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    from jobradar.chain import ask

    result = ask(args.question, k=args.k)

    print("\n" + "=" * 70)
    print(f"Q. {args.question}")
    print("=" * 70)
    print(result["answer"])

    # 출처는 LLM 의 말이 아니라 검색 결과에서 직접 뽑아 출력합니다.
    print("\n" + "-" * 70)
    print("출처 (검색된 청크)")
    print("-" * 70)
    for i, doc in enumerate(result["context"], start=1):
        meta = doc.metadata
        print(
            f"[{i}] {meta.get('company', '?')} | {meta.get('title', '?')}\n"
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

    p_ask = sub.add_parser("ask", help="질문하고 출처와 함께 답변 받기")
    p_ask.add_argument("question", help="질문 (예: \"Python 쓰는 곳 어디야?\")")
    p_ask.add_argument("-k", type=int, default=TOP_K, help=f"검색할 청크 수 (기본 {TOP_K})")
    p_ask.add_argument("--show-chunks", action="store_true", help="청크 본문 미리보기 출력")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
