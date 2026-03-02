# Reasoning-Intensive Retrieval via Adaptive Granularity Navigation

> **Project Status:** Active (Phase 4: Agentic Framework Implementation)  
> **Target Benchmark:** BRIGHT (Reasoning-Intensive Retrieval)  
> **Key Concept:** Reasoning Derailment, Logic Poisoning, Knowledge Granularity, Agentic Search

## 1. Introduction & Problem Definition

기존의 RAG(Retrieval-Augmented Generation) 시스템은 단순 사실 검색(Factoid)에는 효과적이나, 복잡한 추론이 필요한 **Reasoning-Intensive Task**에서는 한계를 보임.

### **The Reasoning Gap**
사용자의 질문(Query)과 정답 문서(Gold Doc) 사이에 **지식의 층위(Abstraction Level)**가 다른 현상.
* **Query:** "벌들의 움직임을 측정하는 가장 좋은 방법은?" (Concrete Phenomenon)
* **Gold Doc:** "푸아송 분포(Poisson Distribution)의 확률 모델..." (Abstract Theory)
* **Issue:** 기존 Retriever는 의미적 유사도(Semantic Similarity)에 기반하므로, '벌'에 대한 문서만 찾고 '푸아송'을 찾지 못함.


## 1.1. Problem Definition

Reasoning‑Intensive Retrieval(예: BRIGHT)에서는 질의가 종종 **구체(현상/사례)** 로 표현되지만, 정답 문서는 **추상(원리/모델/메커니즘)** 수준에 존재한다. 이때 단순 semantic similarity 기반 top‑K는 gold(또는 gold로 가는 bridge concept)를 놓치기 쉽고, 그 결과 LLM의 추론이 쉽게 흔들린다. 우리는 이 현상을 **abstraction gap**으로 부른다.
Direction: Query rewriting w/ abstraction to write principles & evidence that support the answer
{“Theory”: …, “Entity”: …, “Example”: …, “Other”: … }

### 1.1 RQs

Challenge / RQs
- RQ 1) LLM may generate abstract hints, but often fails to ground them to corpus-existing principle-level concepts
- RQ 2) Multiple evidence paths can be valid, but some plausible evidence may not exist in the corpus

#### example
Query: How best to count bees entering and leaving a hive to measure hive activity?
Retrieved documents ← ‘bees’ match
Monitor the hive environment, …
Caring for a colony isn't a set-and-forget task;  …
Gold document: Poisson-Distribution ← theory/principle
RQ 1) query rewriting에서, “수학적으로 계산해야 한다.” 같은 generic principle은 생성하지만 poisson-distribution 같은 corpus 내 실제 principle concept까지 연결하지 못하는 경우. corpus에 존재하는 실제 개념까지 grounding해야 함
RQ 2) query rewriting에서, 복수 evidence를 탐색해야 하지만 “양봉업자 커뮤니티 Q&A”처럼 그럴듯한 evidence가 corpus에 없을 수도 있음. 따라서 Corpus에 존재하지 않는 abstraction category/evidence는 생성-선택하면 안됨

RQ 1): Corpus tree의 depth와 corpus-grounded principle alignment와 관련있음
RQ 2): Corpus tree의 width와 corpus-grounded constraint와 관련있음


## 현재 상황

online: retrieve →  category -> rewrite -> retrieval -> category -> rewrite -> retrieval (category-rewrite 사이의 hallucination있는지도 확인 필요)

지금 성능 잘나오는 거: retrieve -> rewrite -> retrieval  -> rewrite -> retrieval (category: human이 임의로 theory, entity,example로 제공)
    필요한 거: 왜 theory entity example이어야 하냐?

Iterative rewrite은 많음. 여기서 발생하는 문제가 뭘까?
theory를 생성하는 게 internal knowledge로 생성할 때 해결하는 법 -> tree depth based corpus feedback
- tree traversal 잘 하는 학습? / 어떻게 tree 구축할까?
multiple evidence: beam search

“좋은 tree: 검색에 도움이 되는 tree. tree위에서 검색하는 거랑 end2end 강화학습”

평가 방법: branch level잘 select하는 정확도 vs LLM internal knowledge 정확도. 이걸 우리는 학습 데이터 만들 수 있고,...
학습데이터 어떻게 만들지 고민

## 고민에 대한 답 (v1)

### 1) tree generation에서 어떤 프롬프트를 써야 할까?

문제:
- 상위 level로 갈수록 요약 기반 정보 손실이 생기고, retrieval에 필요한 신호가 사라질 수 있음.

답:
- "요약문 품질"보다 "retrieval용 구조 신호"를 우선한다.
- 즉, tree 생성 프롬프트는 topic 요약이 아니라 역할 기반 category path를 안정적으로 뽑는 데 집중한다.
- 권장 출력 단위:
    - node-level mapping (어떤 leaf가 어떤 상위 노드에 배정되는지)

권장 프롬프트 설계 원칙:
- category-first: topic naming보다 answer-support role 분류를 우선.
- corpus-grounded: 코퍼스에 없는 개념/카테고리 생성 금지.
- multi-path 허용: 한 문서가 복수 역할을 가질 수 있게 alternate path 허용.
- id-aware: passage id/doc id를 명시해 매핑 안정화.

### 2) 학습 전략 (2-stage, v1)

현재 결정은 2-stage로 간다.
1. Stage A (메인라인): Teacher graph/tree를 고정하고, query rewriter + tree traversal policy를 학습한다.
2. Stage B (부가실험): summary 기반 cluster model(= graph builder)을 학습/증류한다.

왜 이렇게 가는가:
- joint co-train은 credit assignment가 불안정하다. (tree 오류와 policy 오류가 섞여 원인 분리가 어려움)
- retrieval 품질 개선이 현재 1순위이므로, 먼저 online policy를 안정화하는 것이 맞다.
- graph builder 학습은 별도 오프라인 objective로 풀고, 나중에 교체 실험으로 붙이는 게 안전하다.

### 3) Cluster 학습 상세 (난관 대응)

가정:
- 각 long document의 5-level summary는 ground truth로 본다.
- cluster 학습의 본질은 "프롬프트 문장 생성"이 아니라 "summary_id -> cluster_id partition" 구조 예측이다.

#### 3.1 학습 타깃 (Graph Builder)
- 입력: `{summary_id, summary_text(level i), count, parent_path context}`
- 출력:
    - `K`개 cluster (2 <= K <= M)
    - 각 `summary_id`의 cluster assignment (multi-assigned 허용)
    - cluster `name/description` (부가 출력; 핵심은 assignment)

#### 3.2 라벨 생성 (Teacher supervision)
1. 큰 모델(teacher)로 동일 입력에 대해 cluster를 여러 번 샘플링한다. (seed/temperature 다양화)
2. 유효성 필터를 통과한 결과만 남긴다.
    - 모든 summary_id가 최소 1회 이상 할당
    - K 범위 제약 만족
    - JSON 구조 유효
3. 후보 cluster들에 점수를 매겨 최종 pseudo-label을 선택한다.
    - 구조 점수: coverage, 중복/누락, cluster balance
    - retrieval proxy 점수: oracle query-doc 기준 branch recall, depth hit
4. 최종적으로 `(input set, assignment, cluster metadata)` 데이터셋을 만든다.

#### 3.3 모델 학습 방식
- 현 단계는 SFT 1단계만 수행한다.
- 학습 샘플 단위는 "한 parent node의 partition 문제 1개"로 둔다.
- 입력은 `summary_id + summary_text + count + parent context`로 고정한다.
- 출력은 `summary_id -> cluster_id` 매핑 JSON으로 고정한다.
- distill, preference/ranking 학습은 이 단계에서 하지 않는다.

#### 3.4 핵심 loss 설계 (중요)
- `L_assign`: summary별 cluster assignment CE
- 현 단계 총합: `L = L_assign`
- CE를 계산하기 위해 학습 데이터는 아래 조건을 만족하는 샘플만 사용한다.
1. 모든 `summary_id`가 정확히 1개의 `cluster_id`를 가짐
2. 누락 `summary_id`가 없음
3. JSON 파싱이 성공함
- 위 조건을 만족하지 않는 샘플은 학습셋에서 제외한다.
- `L_pair`, `L_balance` 등 추가 loss는 현 단계에서 사용하지 않는다.

#### 3.5 정보 보존 우선 규칙 (현 단계)
- 목표는 cluster 균형보다 정보 손실 최소화다.
- graph 저장/추론 단계에서는 multi-assigned를 허용한다.
- cluster 크기 불균형은 현 단계에서 제약하지 않는다.
- 학습 단계에서는 CE 계산 가능성을 위해 single-assigned 샘플만 사용한다.
- 데이터 생성 로그에 아래를 반드시 기록한다.
1. 전체 샘플 수
2. 필터 후 학습 가능 샘플 수
3. 제외 사유별 개수(파싱 실패/중복 할당/누락)

#### 3.6 평가 지표
- Intrinsic:
    - assignment accuracy (held-out pseudo-label 기준)
    - parse success rate
    - 학습 가능 샘플 비율 (filter pass rate)
- Extrinsic:
    - 최종 retrieval nDCG@10
    - Recall@100
    - gold ancestor hit

핵심 포인트:
- "cluster_prompt_template를 잘 말하게"가 목표가 아니다.
- "retrieval에 유리한 partition을 안정적으로 만들게"가 목표다.

### 4) Query Rewriter + Traversal 학습 (Stage A 메인라인)

쓸 수 있는 데이터:
- tongsearchqr (query rewriting 학습): 문제: 지금 gold label은 corpus에 없어서 학습을 위해서는 corpus tree 학습용으로 재구축할 필요 있음.
- BRIGHT oracle trace 기반 branch supervision

학습 신호:
- rewrite quality (gold/bridge concept recall)
- traversal action supervision (올바른 branch 선택)
### 5) 지금 당장 필요한 실험

1. Teacher tree 고정 baseline 확정 (retrieval main score)
2. Rewriter + traversal 정책 학습/튜닝 (Stage A)
3. Cluster 학습용 pseudo-label 생성 파이프라인 구축 (Stage B 준비)
4. Qwen3-4B graph builder distill 소규모 pilot 1회

### 6) 현재 결정

- 논문 메인 메시지: retrieval quality 개선 (Stage A 중심)
- graph builder 학습: 부가실험 트랙(Stage B)으로 분리
- cluster 학습 난관은 "assignment supervision + corpus-grounded 제약"으로 명시적으로 다룬다

### 7) Distill 이후 RL 계획 (Traversal 정확도 향상)

전제:
- distill 이후에는 tree traversal policy에 대해 RL을 적용한다.
- 목표는 "특정 tree를 더 자주 선택"이 아니라 "retrieval 성능을 올리는 traversal"이다.

#### 7.1 보상 정의
- terminal reward:
    - `R = a * nDCG@10`
- tree 교체 판단용 reward:
    - 같은 query, 같은 budget, 같은 policy 조건에서 `ΔR = R_new_tree - R_old_tree`
    - `ΔR > 0`이면 positive, `ΔR <= 0`이면 negative
- 사용하지 않을 신호:
    - "새 tree가 선택된 횟수/확률 증가" 자체는 reward로 사용하지 않는다.

#### 7.2 학습 절차 (교대학습)
1. tree 고정, traversal policy RL 업데이트
2. policy 고정, candidate tree 평가 (`ΔR` 계산)
3. `ΔR`가 유의미하게 양수인 tree만 채택
4. 1~3 반복

#### 7.3 안정화 규칙
- 학습/평가에서 query set을 고정해 분산을 줄인다.
- tree 비교 시 random seed와 budget을 동일하게 맞춘다.
- 작은 개선(노이즈 수준)은 채택하지 않고 임계값 이상만 채택한다.

#### 7.4 1차 실험 체크리스트
1. RL 없이 distill policy baseline 기록
2. RL 1회 업데이트 후 metric 변화 기록
3. candidate tree 2~3개에 대해 `ΔR` 비교
4. 채택/비채택 결과와 실패 케이스 로그 저장
