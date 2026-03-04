# 2026-02-26 Tree Traversal RL 계획

- 목표: abstraction gap을 해결하자
- 1) LLM의 지식 한계 때문에 구체적으로 원하는 내용을 못만든다, 2) corpus에 없는 근거를 만들면 안됨.
Tree를 쓰자고 LG와 합의봄: 1) tree depth 2) tree width

지금 해결하고자 하는 문제: tree의 정보를 corpus feedback으로 사용할 수 있을까?
어떤 내용으로 rewrite하면 좋을지 줄 수 있을까? 지금은 abstraction category: theory/entity/example 고정해서 쓰고 있음. 근데 각 branch (=하위 cluster의 요약)은 코퍼스에 어떤 내용이 들어있는지 요약해서 알려주는 역할임. 여기서 뽑은 정보로 대체할 수 있을까?
첫 iter: depth1에서의 abstraction category중 3-5개 선별해서 rewrite, ...

- traversal하는 모델을 학습하면?
실험: LATTICE 선행연구와 동일하게 *retrieval 없이* tree traversal로, 매 step마다 subtree 하나씩 내려감

(실험 0: 이 이야기는 안해도 됨)
현재 tree가 내용별 cluster라서 문제 -> 유형별 cluster로 재배치하자
- 내용별 cluster tree reproduce 실패 -> 유형별 cluster까지 확장 아직 안함. 기존 LATTICE에서 제공해주는 tree상에서 실험

실험 1. category 보여주고, category 

실험 1-1: rewrite + reranking을 학습 -> 실패
preference learning DPO
reasonembed data: training (q,d) pair 제공. 
각 branch에서 gold subbranch를 포함하게 고른다면 pos, 아니면 neg (original query를 받고 reasoning하는 데 기존의 프롬프트를 넣었음)
QLORA (4bit 적용)
- 성능 안나옴
- tree 만 써서 위에서부터 내려가는 구조: 작은모델 잘 못함. 큰모델 -> 42점까지 오르는데, tree traversak vs retriever + query rewriter
@results/BRIGHT/ndcg_summaryS=baseline1_tree_only.csv

GRPO
- OOM으로 실패

실험 1-2. 프롬프트 자체는 효과 있나?
- qwen3-4b
@results/BRIGHT/ndcg_summaryS=baseline1_tree_only.csv
프롬프트 자체가 약하긴함



----------
discussion

원하는 전체 파이프라인:
tree구축 -> category -> 답이 있을 거 같은 카테고리 3-5개 골라서 rewrite -> retrieval -> category -> rewrite -> retrieval (category-rewrite 사이의 hallucination있는지도 확인 필요)

정해야 할 사항: 
1. training 때 궁극적으로 얻고자 하는 게 카테고리 매핑인 건지? 어떤 걸 학습할지 reward 구체화
2. "답이 있을 거 같은 카테고리 3-5개 골라서 rewrite" 는 이전 retrieval 결과에 따라 결정하는지? 아니면 모델이 카테고리를 고르는건지? (top 100으로 MRR 값이 가장 높은 카테고리 x개 보여주기?) [회의 때 나온 얘기: 미리 구축해놨던 corpus tree에 얼마만큼 Hit되는가 보고 카테고리를 선별하기. (tree정보가 더 활용되면 좋겠음) tree위에서 corpus feedback을 잘 얻을 수 있지 않을까?]
3. retrieval pool: 선택된 branch 밑에서만? 아니면 다른 브랜치들도 포함?: 다른 브랜치들도 포함하는 게 좋겠음. clustering이 제대로 이뤄지지 않았을 거라는 가정


또하나 발견: 30b로 query rewrite하는 건 성능 향상 없음. 30b로 graph traversal하는 건 성능 향상 있음. 즉 어떤 branch로 들어갈지 고르는 건 학습해서 좋은 모델 만들기에 효과가 있을 것이다.

# 2026-03-03 Lab Meeting

## 0) 이번 주 한 줄 요약
- 목표: **abstraction gap**을 줄이기 위해 tree 정보를 활용한 evidence retrieval 파이프라인 가능성 검증
- 핵심 결론: **tree-only traversal을 바로 학습으로 강화하는 전략은 아직 근거 부족**. 이번 주 결과는 "학습 실패"라기보다 "문제 설정/비교군/인프라가 섞여 있어 결론이 약함"이 더 정확함.

## 1) 우리가 풀려는 문제 (명확화)
- 질문: "LLM이 답 자체를 생성하기 어렵더라도, **어떤 근거 문서가 필요한지(카테고리/방향)**는 판단할 수 있는가?" & 어떤 걸 학습해야 하는가?
- 이번 주 실험 초점:
    1. tree-only traversal로 retrieval을 대체할 수 있는지
    2. traversal 정책을 DPO/GRPO로 학습하면 성능이 오르는지
    3. 프롬프트 자체 개선이 유의미한지

## 2) 실험/가설/결과

### 실험 A. DPO 기반 branch 선택 학습
- 가설: branch 선택을 preference learning으로 학습하면 tree-only 성능이 오른다.
- 설정:
    - 데이터: reason-embed 기반 (각 step에서 gold sub-branch 포함 여부로 pos/neg 구성)
    - 학습: QLoRA 4bit + DPO
- 결과 (`results/BRIGHT/ndcg_summaryS=baseline1_tree_only.csv`):
    - Base(30B): overall **42.03**
    - prompt 교체 실험 row overall: **34.21**
    - Base(4B): overall **34.69**
    - DPO merged: overall **35.16**
- 해석:
    - 프롬프트 자체는 성능 향상 없음 (34.69 -> 34.21)
    - 그래도 DPO 학습이 어떤 branch가 좋은 branch인지 신호를 제공해줌 (34.21 -> 35.16)
    - 단, 이것만으로 "RL/선호학습이 무효" 결론은 불가 (보상 설계/데이터 구성/학습 안정성 요인 분리 안 됨).

### 실험 B. GRPO
- 가설: online reward 최적화(또는 준-online)가 branch 정책에 유리하다.
- 결과: **학습 완료 실패 (OOM + vLLM 통신/포트 충돌 이슈)**.
- 해석:
    - 성능 결론 없음.
    - 이번 주 시점에서 GRPO 관련 주장은 "미완료"로 보고해야 함.

## 3) 이번 주 결과에서 **말해도 되는 주장 / 아직 말하면 안 되는 주장**

### 말해도 되는 주장
- tree-only setting에서 prompt 교체만으로는 성능 개선이 없었음 (34.69 -> 34.21).
- DPO는 현재 설정에서 prompt 대비는 개선(34.21 -> 35.16)됐지만, base(30B, 42.03)에는 아직 큰 격차가 있음.
- 따라서 다음 단계는 tree-only 강화가 아니라, retrieval과 결합한 category policy 검증으로 넘어가는 게 타당함.

### 아직 말하면 안 되는 주장
- "RL은 효과 없다"
- "프롬프트는 의미 없다"
- "tree-only가 retrieval+rewrite보다 본질적으로 나쁘다"
    - 이유: 실험 비교 조건(데이터/예산/실패 원인)이 완전히 정렬되지 않음.

## 4) 논리적으로 약했던 부분과 보완한 프레이밍
- 약점 1: 목표는 retrieval-결합 파이프라인인데, 실험은 tree-only 중심이라 질문-실험 정렬이 약했음.
    - 보완: 다음 주는 "tree를 retrieval 제어 신호로 쓰는 실험"으로 복귀.
- 약점 2: 학습 실패 원인(보상/데이터/인프라) 분리가 없어서 결론이 과도했음.
    - 보완: "성능 결론"과 "실행 실패"를 분리 보고.
- 약점 3: discussion이 질문 나열 형태라 의사결정이 안 남았음.
    - 보완: 아래 3개 결정을 회의에서 확정.

## 5) 회의에서 확정할 결정 3개
1. **주 실험 축**: **B안 확정**  
    - retrieval + category policy 결합으로 전환

2. **학습 목표(reward)**: **step-wise gold-branch nDCG**  
    - category selector에서 각 step의 branch 품질을 직접 최적화

3. **카테고리 선택 규칙**: **retrieval feedback 기반 선택**  
    - support score/top-k evidence를 기반으로 카테고리 선택

## 6) 다음 주 실행 계획 (발표용 액션)
- P0: GRPO 인프라 안정화(포트/통신/메모리) 완료 여부 먼저 체크
- P1: retrieval feedback 기반 category selector 구현 및 로그 저장
    - 의도: 선택 근거를 남겨 policy 품질(step-wise nDCG)과 최종 retrieval 성능을 함께 분석
- P2: step-wise gold-branch nDCG를 reward로 사용해 category selector 학습/비학습 비교
- P3: 실험 보고 형식 통일: 각 실험마다 `가설 1줄 / 설정 1줄 / 수치 1줄 / 결론 1줄`

## 7) 현재 결론 (잠정)
- 이번 주는 "tree traversal 학습이 잘 된다"를 보인 주가 아니라,
- **어떤 결론을 아직 내리면 안 되는지**가 명확해진 주.
- 다음 주 목표는 tree를 단독 대체재가 아니라 **retrieval 의사결정 신호**로 검증하는 것.
- 구체적으로는 **retrieval + category policy(B안)**, **step-wise gold-branch nDCG reward**, **retrieval feedback 기반 선택 규칙**으로 고정한다.
