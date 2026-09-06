"""④ 검색 → ⑤ 프롬프트 → ⑥ 생성 → ⑦ LCEL 조립.

배우는 개념
-----------
- Retriever 도 Runnable 입니다. 문자열을 넣으면 Document 리스트가 나옵니다.
- RunnableParallel: 입력 하나를 여러 갈래로 동시에 흘려보냅니다.
  여기서는 질문 문자열이 (a) retriever 로, (b) 그대로 통과해서 두 갈래로 갑니다.
- RunnablePassthrough.assign: 지금까지의 dict 를 유지한 채 키를 하나 더 붙입니다.
  덕분에 답변과 함께 '어떤 청크를 봤는지'가 최종 출력에 그대로 남습니다.
  이게 출처 표시의 핵심입니다 - LLM 이 지어낼 수 없는 부분이니까요.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from jobradar.config import CHAT_MODEL, MAX_TOKENS, TOP_K
from jobradar.ingest import load_vectorstore

SYSTEM_PROMPT = """당신은 채용 공고 검색 어시스턴트입니다.

반드시 지킬 규칙:
1. 아래 <공고> 블록 안의 내용만 근거로 답하십시오. 블록에 없는 내용은 절대
   추측하거나 일반 상식으로 보충하지 마십시오.
2. 근거를 찾을 수 없으면 다른 말을 덧붙이지 말고 정확히 이렇게 답하십시오:
   "제공된 공고에서 확인할 수 없습니다."
3. 회사나 공고를 언급할 때는 문장 끝에 [1], [2] 형태로 출처 번호를 붙이십시오.
4. 한국어로, 불릿 위주로 간결하게 답하십시오.

<공고>
{context}
</공고>"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def format_docs(docs: list[Document]) -> str:
    """검색된 청크를 번호가 붙은 텍스트로 만듭니다.

    각 청크 앞에 회사/직무/근무지를 한 줄 붙이는 게 중요합니다. 청크가
    "## 우대사항" 한복판에서 잘리면 본문만으로는 어느 회사인지 알 수 없어,
    LLM 이 출처를 제대로 말할 수 없기 때문입니다.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('company', '(회사미상)')} | "
            f"{meta.get('title', '(제목미상)')} | "
            f"근무지: {meta.get('location', '?')} | "
            f"경력: {meta.get('experience', '?')} | "
            f"마감: {meta.get('expires_at', '?')}"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def get_llm() -> ChatAnthropic:
    # 주의: Claude Opus 5 는 temperature / top_p 를 받지 않습니다 (400 에러).
    # LangChain 예제에서 흔히 보이는 temperature=0 을 습관적으로 넣으면 터집니다.
    return ChatAnthropic(model=CHAT_MODEL, max_tokens=MAX_TOKENS)


def build_chain(k: int = TOP_K) -> Runnable:
    """질문(str) -> {question, context, answer} 를 돌려주는 RAG 체인."""
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": k})

    # 검색된 Document 를 프롬프트 변수로 변환하는 작은 체인.
    # dict 리터럴은 LCEL 에서 자동으로 RunnableParallel 로 바뀝니다.
    answer_chain = (
        {
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )

    return RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer_chain)


def ask(question: str, k: int = TOP_K) -> dict:
    return build_chain(k=k).invoke(question)
