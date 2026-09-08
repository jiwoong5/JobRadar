# JobRadar — 채용 공고 검색 어시스턴트

LangChain 개념 학습용 RAG 프로젝트. 소스는 GitHub, 학습 내용은 블로그로 관리.

## 단계별 로드맵

| 단계 | 목표 | 상태 |
|---|---|---|
| 1 | 텍스트 공고 10개 → 청킹 → 임베딩 → 질문하면 관련 청크로 답변 + **출처 표시** | 완료 |
| 2 | 메타데이터(회사/직무/근무지/마감일/경력)를 붙여 **필터 검색** | 완료 |
| 3 | BM25 하이브리드 + 크로스인코더 리랭킹 (상위 20 → 5) | 완료 |
| 3.5 | **청킹 재조정** — 아래 "미뤄둔 문제" 참고 | 예정 |
| 4 | PDF/HTML/JSON 멀티 소스 (컴앤휴먼 요구사항) | 예정 |

### 미뤄둔 문제 (3.5단계)

**청킹이 한 번도 동작한 적이 없다.** 실측 본문 길이가 610~799자인데 `chunk_size=800`이라
스플리터가 단 한 번도 쪼개지 않았다. 공고 10건 → 청크 10건, 즉 `chunk_overlap`도
`separators`의 `"\n## "` 우선순위도 결과에 아무 영향을 주지 못했다.

이게 3단계에도 영향을 준다. 청크가 10개뿐이라 **"상위 20 → 5"의 20 단계가 no-op**이다
(후보 20 = 코퍼스 전체). 리랭킹은 순서를 바꿀 뿐 걸러낼 것이 없다.

`chunk_size=350`으로 낮추면 31청크가 되어 퍼널이 비로소 의미를 갖는다(실측:
800→10, 500→20, 400→24, 350→31, 300→37, 200→61이며 200에서는 7자짜리 파편이 생긴다).
3단계를 끝낸 뒤 손댄다.

**각 단계의 범위를 넘지 말 것.** 단계 구분 자체가 학습 장치다.

## 확정된 기술 선택과 이유

| 항목 | 선택 | 이유 |
|---|---|---|
| 답변 LLM | `claude-opus-5` (`langchain-anthropic`) | 한국어 품질 |
| 임베딩 | OpenAI `text-embedding-3-small` | **Anthropic은 임베딩 API가 없음**. 설치 가볍고(torch 불필요) 한국어 충분 |
| 벡터스토어 | Chroma | 2단계 메타데이터 필터(`where=`)를 네이티브 지원. FAISS는 그게 없어 갈아엎어야 함 |
| 청킹 | `RecursiveCharacterTextSplitter` 800/120 | 공고 1건 ≈ 1,800~2,100자 → 공고당 3~4청크 (**실측은 610~799자. 3.5단계 참고**) |
| 키워드 검색 | `rank_bm25` (BM25Okapi) | 8.6KB 순수 파이썬. Chroma 내장 FTS5도 후보였으나 토크나이저를 직접 통제하려고 선택 |
| 리랭커 | `Dongjin-kr/ko-reranker` | bge-reranker-large를 한국어로 파인튜닝. `ms-marco` 계열은 영어 전용이라 한국어에 무력 |

키 2개 필요: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## 알려진 함정

- **Claude Opus 5는 `temperature`/`top_p`를 받지 않는다 (400 에러).** LangChain 예제에 흔한
  `temperature=0`을 습관적으로 넣으면 터진다. `chain.py:get_llm()` 참고.
- **Chroma 메타데이터 타입 규칙은 버전을 타므로 실측할 것.** chromadb 1.5.9 기준으로
  동종 리스트(`["A", "B"]`)는 **허용**되고, 거부되는 건 `date` 객체와 `None`이다.
  YAML이 `posted_at: 2026-08-28`을 date로 파싱하므로 `loader.py:_scalarize()`는 여전히 필요하다.
  2단계의 `job_categories`/`keywords`는 리스트 그대로 저장한다.
- **Chroma `where` 절에 부분 문자열 매칭이 없다.** `$contains`는 리스트 필드의 원소
  정확 일치로만 걸리고(`$contains "백엔드"`는 `"백엔드·서버개발"`을 못 찾는다), 문자열
  필드에는 아예 안 걸린다. `$gte`/`$lte`는 숫자 전용이다. 그래서 색인 시점에
  `location_city`/`location_district`/`expires_at_num`을 미리 쪼개 두고, 어휘 해석은
  `filters.py`가 파이썬에서 한다.
- **frontmatter 키는 `source:`가 아니라 `site:`.** `TextLoader`가 `metadata["source"]`에
  파일 경로를 넣기 때문에 이름이 충돌한다.
- **`TextLoader`에 `encoding="utf-8"` 필수.** Windows에서 생략하면 cp949로 읽다 한글이 깨진다.
- **소규모 코퍼스에서 BM25는 기능어에 낚인다.** 문서가 10건뿐이라 우연히 한 문서에만
  있는 조사·의문사의 IDF가 전문용어를 압도한다. 실측: "Python 쓰는 곳 어디야?"에서
  #9(Node.js)가 1.86점 1위였는데 전부 '어디' 토큰 점수였다(python은 0.71).
  `retrieval.py`의 불용어 + 한 글자 한글 제거로 막는다.
- **`CrossEncoder`는 라벨이 1개면 Sigmoid를 기본 적용한다.** 즉 `predict()` 결과는
  raw logit이 아니라 **0~1**이다(로짓 -2.67 → 0.065). 임계값을 쓸 거라면 이 사실을
  전제로 실측해서 정할 것.

## 데이터 규약

`data/raw/*.txt` = **YAML frontmatter + 마크다운 본문**. frontmatter 필드명은
사람인 채용정보 API 응답(`company.detail.name`, `position.location.name`,
`position.experience-level.name` 등)을 본떴다. 나중에 사람인/잡코리아 크롤링·API로
교체할 때 **`loader.py`만 갈아끼우면 되도록** 설계한 것이다.

샘플 10개는 2~3단계 검증까지 계산해서 배치했다:

| # | 근무지 | 직무 | 경력 | 심어둔 의도 |
|---|---|---|---|---|
| 1 | 부산 해운대 | Python 백엔드 | 신입 | 1단계 정답 |
| 2 | 서울 강남 | Java/Spring | 경력 3~5년 | Python 아님 (오답 유인) |
| 3 | 부산 수영 | **MS-SQL** DBA | 경력무관 | 3단계 BM25 정확키워드 |
| 4 | 판교 | React 프론트 | 신입 | — |
| 5 | 서울 마포 | 데이터엔지니어(Python/Spark) | 경력 2~4년 | 1단계 정답 2 |
| 6 | 부산 부산진 | 웹 퍼블리셔 | 신입 | 2단계 "부산+신입" |
| 7 | 대전 유성 | C#/.NET + **MS-SQL** | 경력 5년↑ | 3단계 리랭킹 변별력 |
| 8 | 서울 송파 | DevOps(AWS/K8s) | 경력 3년↑ | — |
| 9 | 재택(전국) | Node.js | 경력무관 | 근무지 엣지케이스 |
| 10 | 부산 강서 | QA | 신입, 마감임박 | 2단계 마감일 필터 |

## 1단계 검증 기준 (DoD)

| 질문 | 기대 |
|---|---|
| "Python 쓰는 곳 어디야?" | #1, #5 (#9는 Node라 걸리면 안 됨) + 출처 |
| "부산에서 일할 수 있는 데 있어?" | #1, #3, #6, #10 중 다수 |
| "연봉 1억 주는 데 있어?" | "제공된 공고에서 확인할 수 없습니다." — **환각 금지** |

세 번째가 가장 중요하다. RAG의 절반은 "모르면 모른다고 하기"다.

## 모듈 경계

```
loader.py   ① 로드 + frontmatter → metadata 승격 + 필터용 파생 필드
ingest.py   ② 청킹 ③ 임베딩·저장  (오프라인, 1회성)
filters.py  사람 말 → Chroma where 절  (2단계에서 추가)
retrieval.py BM25 + 벡터 → RRF 융합, HybridRetriever  (3단계에서 추가)
rerank.py   크로스인코더 재순위  (3단계에서 추가)
chain.py    ④ 검색 ⑤ 프롬프트 ⑥ 생성 ⑦ LCEL 조립  (온라인, 질문마다)
config.py   단계가 바뀌며 변하는 값은 전부 여기
```

**ingest / ask 분리가 핵심 설계다.** 임베딩은 돈과 시간이 드니 질문마다 다시 할 이유가
없고, 3단계에서 인덱스 구성을 바꿀 때 `ingest.py`만 손대면 된다. 나중에 이 경계가
그대로 컨테이너 경계가 된다 (ingest = 일회성 Job, ask = 서비스).

## 배포 / Docker 방침

**3단계까지는 로컬 venv.** 지금 Docker를 얹으면 학습 루프("chunk_size 바꿔 다시 돌리기")에
마찰만 는다. `torch`가 들어오는 3단계, 또는 FastAPI를 붙이는 시점에 컨테이너화한다.
그때를 위해 지금 지키는 습관:

1. 경로는 `pathlib` + 환경변수 (`JOBRADAR_DATA_DIR`, `JOBRADAR_STORAGE_DIR`)
2. 키는 `.env` / 환경변수로만
3. `requirements.txt` 버전 핀 고정
4. 진입점은 `python -m jobradar` (그대로 `CMD`가 됨)
5. ingest / ask 분리

**프로젝트를 OneDrive 안에 두지 말 것** — `.git` 손상 위험, Chroma sqlite 재동기화,
Docker bind mount 충돌. 그래서 `C:\dev\JobRadar`에 있다.
