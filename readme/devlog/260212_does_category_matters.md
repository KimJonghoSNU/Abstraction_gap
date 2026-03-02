# 2026-02-12 방향 점검: Category-Rewrite-Retrieval Loop

## 목적
- 지금 파이프라인의 큰 흐름이 맞는지 점검한다.
- 특히 아래 두 전이(transition)에서 성능 저하/환각(hallucination) 원인을 분리해서 본다.
    - `retrieval -> category selection`
    - `category selection -> rewrite`

## 현재 큰 흐름 (합의 초안)
- `tree 구축` (sanity check 필요)
- `retrieval -> category -> rewrite -> retrieval -> category -> rewrite -> retrieval`
- 핵심 우려:
    - category 선택이 evidence를 제대로 반영하는가?
    - rewrite가 category를 받았을 때 근거 기반으로 확장되는가, 아니면 label echo/환각으로 흐르는가?

## 질문에 대한 답
- 네, 지금 우리가 문제 삼아야 하는 지점은 정확히 아래 2개가 맞다.
    - 1) `retrieval -> category selection`에서 무엇을 선택 근거로 쓰고, 얼마나 보수적으로 선택할지
    - 2) `category selection -> rewrite`에서 category를 어떻게 프롬프트에 주입해야 실제 retrieval 성능 향상으로 이어질지

## Interface 1: retrieval -> category selection

### 실패 모드
- 상위 retrieval 문서가 노이즈일 때 category가 연쇄 오염된다.
- 문서 내용보다 표면 토픽 단어를 따라가서 "종류"가 아니라 "주제" 기준으로 분류된다.
- 애매한 경우에도 억지로 category를 고정해서 다음 rewrite를 좁혀버린다.

### 권장 설계
- multi-label 유지 + confidence 기반 선택.
- `abstain(보류)` 허용:
    - 증거가 약하면 category 수를 줄이거나 다음 iter에서 재평가.
- category 점수는 최소 2신호 결합:
    - retriever similarity 신호
    - evidence snippet과의 정합성 신호(LLM 또는 embedding 기반)

### 운영 규칙 (초안)
- `max_categories`는 유지하되, confidence 낮으면 실제 선택 개수는 더 적게.
- explore 시:
    - 기존 top category 일부 유지(keep)
    - 신규 category 제한적으로 추가(add)
- exploit 시:
    - low-support category drop 가능, 단 한 번에 과도하게 제거하지 않음

## Interface 2: category selection -> rewrite

### 실패 모드
- category key만 반복하고 value가 비어 있거나 추상적 문장만 생성.
- corpus category와 prompt example category가 충돌해서 모델이 헷갈림.
- pre-planner 힌트(키워드/카테고리)가 "근거"가 아니라 "주제 확장"으로만 사용됨.

### 권장 설계
- rewrite 출력 제약을 강화:
    - 각 category value는 "검색 가능한 구체 문장"이어야 함
    - category 라벨 그대로 복붙 금지
- pre-planner 신호는 분리:
    - `Selected_Categories`는 routing prior
    - `Branch_Keywords`는 lexical anchor
- iter 0는 broad prompt(v1), iter>=1는 taxonomy prompt(v5) 전략 유지 가능

## Category-Rewrite 사이 Hallucination 점검 항목
- `Grounding Check`:
    - rewrite value가 retrieval evidence 또는 pre-planner keyword와 연결되는가
- `Label Echo Check`:
    - value가 category 이름 재진술인지(정보량 부족) 측정
- `Support Delta Check`:
    - rewrite 이후 next retrieval에서 category support가 증가했는가
- `Drift Check`:
    - iter가 진행될수록 category가 쿼리 의도에서 멀어지는가

## Tree Sanity Check (필수)
- leaf path -> doc_id 매핑 일관성 확인
- long-doc chunk 병합 규칙 검증
- subset별 depth/branch 분포 점검:
    - 특정 subset에서 과도한 편향 branch가 있는지 확인

## 지금 시점의 핵심 결정 포인트
- Decision A: category selector 정책
    - Option 1: 보수형(precision 우선, abstain 적극 허용)
        - 장점: rewrite 오염 감소
        - 단점: recall 손실 가능
    - Option 2: 공격형(recall 우선, 항상 다수 선택)
        - 장점: 놓치는 근거 감소
        - 단점: 노이즈 증가로 rewrite 흔들림

- Decision B: pre-planner reference source
    - Option 1: `website_title`
        - 장점: 의미 정보가 있어 키워드 품질이 높을 가능성
        - 단점: 표면 토픽 유도 위험
    - Option 2: `doc_id`
        - 장점: 데이터 정합성/재현성 높음
        - 단점: 의미 빈약으로 키워드 품질 저하 가능

## 다음 실험 최소 단위 (제안)
- 공통 고정:
    - 동일 subset, 동일 seed, 동일 retrieval 설정
- 비교축 1:
    - `website_title` vs `doc_id` pre-planner
- 비교축 2:
    - selector 보수형 vs 공격형
- 필수 로그:
    - iter별 selected_categories, category_support_scores
    - rewrite value groundedness 지표
    - next-iter nDCG/recall 변화량

## Gold-Category Oracle Analysis (new)
- 질문:
    - "카테고리를 잘 고르면 실제 retrieval 성능이 오르는가?"
- 설계:
    - `gold docs -> LLM category 분류 -> category-only rewrite -> retrieval`
    - 비교 baseline:
        - original query
        - (optional) category 없는 control rewrite
- 구현:
    - `scripts/analyze_gold_category_oracle_rewrite.py`
    - 출력:
        - `per_sample.jsonl` (query별 selected category, rewrite, metric)
        - `summary.json` (mean nDCG delta, win/tie/loss)
- 해석 포인트:
    - oracle이 original/control 대비 유의미한 +delta를 내면, "category selection quality"가 병목이라는 가설을 지지.
    - oracle 이득이 작으면, category 자체보다 rewrite prompt/compose 방식이 병목일 가능성이 큼.

## 인터뷰 질문 (정렬용)
- Q1. 현재 목표는 평균 성능 향상인가, 아니면 실패 케이스 안정화인가?
- Q2. false positive category와 false negative category 중 어떤 오류를 더 크게 볼지?
- Q3. `abstain`을 공식 허용할지?
- Q4. 이번 라운드의 1차 성공 기준을 무엇으로 둘지? (예: biology 기준 nDCG@10 +X, 또는 hallucination rate -Y)
