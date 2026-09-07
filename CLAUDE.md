# JobRadar — 채용 공고 검색 어시스턴트

LangChain 개념 학습용 RAG 프로젝트. 소스는 GitHub, 학습 내용은 블로그로 관리.

## 단계별 로드맵

| 단계 | 목표 | 상태 |
|---|---|---|
| 1 | 텍스트 공고 10개 → 청킹 → 임베딩 → 질문하면 관련 청크로 답변 + **출처 표시** | 완료 |
| 2 | 메타데이터(회사/직무/근무지/마감일/경력)를 붙여 **필터 검색** | 완료 |
| 3 | BM25 하이브리드 + 크로스인코더 리랭킹 (상위 20 → 5) | 예정 |
| 4 | PDF/HTML/JSON 멀티 소스 (컴앤휴먼 요구사항) | 예정 |

**각 단계의 범위를 넘지 말 것.** 단계 구분 자체가 학습 장치다.

## 확정된 기술 선택과 이유

| 항목 | 선택 | 이유 |
|---|---|---|
| 답변 LLM | `claude-opus-5` (`langchain-anthropic`) | 한국어 품질 |
| 임베딩 | OpenAI `text-embedding-3-small` | **Anthropic은 임베딩 API가 없음**. 설치 가볍고(torch 불필요) 한국어 충분 |
| 벡터스토어 | Chroma | 2단계 메타데이터 필터(`where=`)를 네이티브 지원. FAISS는 그게 없어 갈아엎어야 함 |
| 청킹 | `RecursiveCharacterTextSplitter` 800/120 | 공고 1건 ≈ 1,800~2,100자 → 공고당 3~4청크 |

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
