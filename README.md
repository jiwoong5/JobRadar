# JobRadar

채용 공고에게 질문하면, **출처와 함께** 답해주는 RAG 어시스턴트.
LangChain 개념을 단계별로 익히려고 만드는 학습 프로젝트입니다.

```
$ python -m jobradar ask "Python 쓰는 곳 어디야?"

Q. Python 쓰는 곳 어디야?
======================================================================
- (주)컴앤휴먼 — Django/FastAPI 기반 백엔드 개발, 신입 지원 가능 [1]
- (주)데이터스트림랩스 — Python ETL 파이프라인, 경력 2~4년 [3]

----------------------------------------------------------------------
출처 (검색된 청크)
----------------------------------------------------------------------
[1] (주)컴앤휴먼 | [부산] Python 백엔드 개발자 (신입/경력)
     파일: job_01_comnhuman_python_backend.txt | https://...
```

## 빠른 시작

```powershell
# 1) 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) 설치 (src 레이아웃이라 -e 로 설치해야 python -m jobradar 가 동작)
pip install -r requirements.txt
pip install -e .

# 3) API 키
Copy-Item .env.example .env
notepad .env      # ANTHROPIC_API_KEY, OPENAI_API_KEY 채우기

# 4) 인덱스 생성 (한 번만)
python -m jobradar ingest

# 5) 질문
python -m jobradar ask "Python 쓰는 곳 어디야?"
python -m jobradar ask "부산에서 신입으로 갈 수 있는 데 있어?" --show-chunks
```

## 파이프라인

```
data/raw/*.txt
   │ ① Load        DirectoryLoader + TextLoader     → loader.py
   │                배울 것: Document = page_content + metadata
   │ ② Split       RecursiveCharacterTextSplitter   → ingest.py
   │                배울 것: chunk_size / overlap / separators
   │ ③ Embed+Store OpenAIEmbeddings + Chroma        → ingest.py
   ▼                배울 것: 벡터스토어, persist
[Chroma DB]                                    ← 여기까지 `ingest` (오프라인)
   │ ④ Retrieve    as_retriever(k=4)                → chain.py
   │                배울 것: Retriever 도 Runnable 이다
   │ ⑤ Prompt      ChatPromptTemplate               → chain.py
   │ ⑥ Generate    ChatAnthropic + StrOutputParser  → chain.py
   ▼ ⑦ 조립        LCEL 파이프 `|`                   → chain.py
답변 + 출처                                     ← 여기까지 `ask` (온라인)
```

## 출처는 어떻게 보장되나

두 층으로 나눠 놨습니다.

1. **코드 층 (신뢰 가능)** — `RunnablePassthrough.assign` 으로 검색된 `Document`
   리스트를 최종 출력에 그대로 남깁니다. 회사명·파일명은 검색 결과에서 직접
   꺼내 출력하므로 **LLM이 지어낼 수 없습니다.**
2. **프롬프트 층 (가독성)** — 컨텍스트를 `[1] 회사명 | 직무 | 근무지` 헤더가
   붙은 형태로 넣고, 문장 끝에 `[1]` 인용을 달게 시킵니다.

## 데이터 형식

`data/raw/*.txt` = YAML frontmatter + 마크다운 본문. 필드명은 **사람인 채용정보 API
응답을 본떠** 두었습니다. 나중에 사람인/잡코리아 크롤링 데이터로 바꿀 때
`loader.py` 하나만 갈아끼우면 되도록 한 설계입니다.

```yaml
---
site: saramin
company: (주)컴앤휴먼
title: "[부산] Python 백엔드 개발자 (신입/경력)"
location: 부산 > 해운대구
experience: 신입
expires_at: 2026-10-15
---
## 주요업무
...
```

## 로드맵

- [x] **1단계** 청킹 + 임베딩 + 출처 표시 Q&A
- [ ] **2단계** 메타데이터 필터 검색 (`부산 + 신입`)
- [ ] **3단계** BM25 하이브리드 + 크로스인코더 리랭킹
- [ ] **4단계** PDF / HTML / JSON 멀티 소스

## 삽질 기록

- **`temperature=0` 넣으면 터진다.** Claude Opus 5는 `temperature`/`top_p`를
  받지 않습니다 (400). LangChain 예제엔 거의 항상 들어 있어서 그대로 베끼면 막힙니다.
- **Chroma 메타데이터에 list를 못 넣는다.** YAML이 `keywords: [A, B]`를 list로,
  `posted_at: 2026-08-28`을 date 객체로 파싱해서 그대로 넣으면 저장에서 실패합니다.
- **`TextLoader(encoding="utf-8")` 필수.** Windows 기본이 cp949라 한글이 깨집니다.
- **frontmatter 키를 `source:`로 쓰면 안 된다.** `TextLoader`가 `metadata["source"]`에
  파일 경로를 넣어서 덮어씁니다. `site:`로 이름을 바꿨습니다.
