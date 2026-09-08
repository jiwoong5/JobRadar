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

# 6) 메타데이터 필터 (2단계)
python -m jobradar facets                       # 필터에 쓸 수 있는 값 목록
python -m jobradar ask "어떤 회사가 있어?" --city 부산 --exp 0
python -m jobradar ask "곧 마감되는 공고는?" --deadline-before 2026-09-20
python -m jobradar ask "요약해줘" --category 백엔드 --onsite --show-filter
```

### 필터 옵션

| 옵션 | 의미 | 예 |
|---|---|---|
| `--city` / `--district` | 근무지 시도 / 시군구 | `--city 부산` |
| `--remote` / `--onsite` | 재택 여부 | `--remote` |
| `--company` | 회사명 (일부만 써도 됨) | `--company 컴앤휴먼` |
| `--category` | 직무 (일부만 써도 됨) | `--category 백엔드` |
| `--exp N` | 경력 N년으로 지원 가능한 공고 | `--exp 0` (신입) |
| `--open-only` | 오늘 기준 마감 전 | |
| `--deadline-before` | 이 날짜까지 마감 | `--deadline-before 2026-09-20` |
| `--show-filter` | 생성된 Chroma `where` 절 출력 | |

### 검색 방식 (3단계)

기본은 **하이브리드 + 리랭킹**입니다. 단계를 꺼 가며 비교할 수 있습니다.

```bash
python -m jobradar ask "MS-SQL 다루는 자리 있어?" --show-ranking   # 단계별 순위 비교
python -m jobradar ask "..." --no-rerank      # RRF까지만 (크로스인코더 끄기)
python -m jobradar ask "..." --no-hybrid      # 벡터 검색만 (1·2단계와 동일)
```

```
벡터 검색 20건 ─┐
                ├─ RRF 융합 ─→ 상위 20 ─→ 크로스인코더 ─→ 최종 5건 ─→ LLM
BM25 검색 20건 ─┘
   재현율 확보                              정밀도 확보
```

첫 실행 때 리랭커 모델(약 1.1GB)을 내려받아 `~/.cache/huggingface`에 캐시합니다.

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
- [x] **2단계** 메타데이터 필터 검색 (`부산 + 신입`)
- [x] **3단계** BM25 하이브리드 + 크로스인코더 리랭킹
- [ ] **3.5단계** 청킹 재조정 (`chunk_size=800`이라 한 번도 쪼개지지 않음)
- [ ] **4단계** PDF / HTML / JSON 멀티 소스

## 삽질 기록

- **`temperature=0` 넣으면 터진다.** Claude Opus 5는 `temperature`/`top_p`를
  받지 않습니다 (400). LangChain 예제엔 거의 항상 들어 있어서 그대로 베끼면 막힙니다.
- **Chroma 메타데이터 타입 제약은 버전에 따라 다르다.** chromadb 1.5.9에서 직접
  확인해 보니 `["A", "B"]` 같은 **동종 리스트는 통과**하고(그래서 2단계에서
  `job_categories`를 리스트로 저장합니다), 실제로 막히는 건 `date` 객체와 `None`입니다.
  YAML이 `posted_at: 2026-08-28`을 date로 파싱하므로 `_scalarize()`는 여전히 필요합니다.
- **Chroma `where` 절에는 부분 문자열 매칭이 없다.** `$contains`는 리스트 필드의
  **원소 정확 일치**로만 동작합니다 — `["웹개발", "백엔드·서버개발"]`에 `$contains "백엔드"`는
  0건입니다. 그래서 "백엔드" → "백엔드·서버개발" 해석은 `filters.py`가 파이썬에서 하고,
  Chroma에는 정확한 값만 넘깁니다. `$gte`/`$lte`도 숫자 전용이라 `expires_at_num`(YYYYMMDD)을
  따로 만듭니다.
- **`TextLoader(encoding="utf-8")` 필수.** Windows 기본이 cp949라 한글이 깨집니다.
- **frontmatter 키를 `source:`로 쓰면 안 된다.** `TextLoader`가 `metadata["source"]`에
  파일 경로를 넣어서 덮어씁니다. `site:`로 이름을 바꿨습니다.
