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

---

## 2. Phenomenon Analysis: Reasoning Derailment

### **Hypothesis**
Reasoning Task에서 검색된 **오답 문서(Hard Negative)**는 단순한 노이즈가 아니라, LLM의 추론 과정을 망가뜨리는 **"Logic Poisoning"**을 유발한다.

### **Key Findings (MS MARCO vs. BRIGHT)**
* **MS MARCO (Standard IR):** 주제가 다른 **Random Negative**에 취약함. (Keyword Distraction)
* **BRIGHT (Reasoning IR):** 주제는 같지만 논리가 틀린 **Hard Negative**에 치명적임. (Logic Poisoning)
    * *Evidence:* Hard Negative를 보여줬을 때 생성된 쿼리와 오답 문서 간의 Cosine Similarity가 비정상적으로 높음 (56.1).
    * *Implication:* LLM이 문맥 순응성(Sycophancy)으로 인해 오답 논리를 진실로 수용하여 **Reasoning Derailment(추론 탈선)** 발생.

---

## 3. Methodological Evolution (Failed Attempts & Pivots)

초기에는 Negative Document를 단순한 "제약 조건"으로 활용하려 했으나 실패함.

### **Phase 1: Naive Constraints (Fail)**
* **Methods:**
    1.  **Prompting:** "이 문서는 쓰지 마" (Zero-shot/Few-shot) -> 효과 미미.
    2.  **Training (RL):** Negative Reward 추가 -> 학습 불안정, OOM.
    3.  **Post-processing:** Vector Subtraction -> Semantic Noise 제거 실패.
* **Lesson:** LLM은 Gold와 Hard Negative를 구분하지 못하므로(Indistinguishability), 단순 제약은 무의미함.

### **Phase 2: The "No Docs" Crisis**
* **Result:** 문서를 아예 안 보여주고(No Docs) LLM 지식만으로 짠 쿼리가 문서를 보여준 것보다 성능이 더 높게 나옴 (34.6 vs 33.6).
* **Insight:** "어설픈 문서를 보여주느니 안 보여주는 게 낫다." -> 하지만 No Docs는 Hallucination 위험이 있음.
* **Pivot:** 문서를 보여주되, **내용(Content)의 바이어스를 제거하고 "구조적 신호(Aspect/Type)"만 활용**해야 한다.

### **Phase 3: Aspect-based Negative Constraints (Success)**
* **Method:** 문서의 본문을 숨기고 "이 문서가 어떤 측면(Why/How/Definition)을 다루는지"만 요약해서 보여줌.
* **Result:** 34.3 달성 (No Docs에 근접).
* **Current Best:** No Docs 결과물을 **Aspect-guided Prompt**로 정제(Refine)했을 때 **35.0 달성**.

---

## 4. Theoretical Framework: Epistemic Taxonomy

단순한 "Answer Type" 분류를 넘어, 지식의 **입도(Granularity)**와 **성격(Epistemic Nature)**에 따른 분류 체계 정립.
- 예시. **CAUTION**: 이건 잘못된 정보일 수 있음. 굉장히 coarse하게 작성한 내용. 

| Type | Definition | Example (Bee Query) | Role in Search |
| :--- | :--- | :--- | :--- |
| **Type 1 (Entity/Phenomenon)** | Grounding된 구체적 실체, 현상, 사실 | "벌통 관찰 일지", "CCTV 설치법" | **Initial Entry Point** |
| **Type 2 (Theory/Mechanism)** | 현상을 지배하는 추상적 원리, 모델 | "푸아송 분포", "확률 밀도 함수" | **Goal of Abstraction** |
| **Type 3 (Example/Analogy)** | 동일한 원리가 적용된 타 사례 | "교통량 측정 모델", "대기 행렬" | **Supporting Evidence** |

**Research Goal:** 쿼리와 문서 간의 **Epistemic Type Mismatch**를 해결하기 위해, 에이전트가 이 계층을 수직적으로 탐색(Vertical Navigation)하도록 함.

> **Note:** 위 Type들은 “항상 존재하는 고정 카테고리”가 아니라, 쿼리/데이터셋에 따라 유효성이 달라지는 **휴리스틱 예시**임. 따라서 목표는 (사람이 미리 레벨을 정의하는 것 자체보다) **corpus 구조/분포를 반영해 유효한 추상 레벨로 이동하는 탐색 메커니즘**을 만드는 것. LATTICE 실험은 이 메커니즘을 “전역 앵커링(flat) + 지역 확장(traversal)” 관점에서 검증하려는 시도다.

### From fixed levels → corpus-induced levels (LATTICE로 “레벨 설계”를 흡수할 수 있나?)
이전 실험의 agentic search는 (example/entity/theory 같은) **고정 레벨 프롬프트**로 탐색을 안내했다. 이는:
- 레벨이 데이터셋/쿼리마다 항상 성립하지 않으며
- 특정 서브카테고리에서는 “해당 레벨로는 bridge가 생성되지 않는” brittleness가 존재한다.

LATTICE류의 hierarchical corpus는 문서 클러스터의 **중간 요약 노드(branch)** 를 제공한다. 이를 활용하면 “고정 레벨”을 직접 맞추기보다:
- **branch 요약 자체를 query rewriting의 컨텍스트로 사용**하여(= corpus가 제공하는 중간 개념으로 rewrite),
- query마다 다른 “유효한 추상 단위(= corpus-induced level)”를 **동적으로 발견**할 수 있다.

즉, LATTICE는 fixed level을 “대체”하기보다는, fixed level이 하려던 역할(bridge concept 제공/수직 이동)을 **문서 그래프(요약 노드)에서 유도**하도록 만들어 레벨 설계 부담을 줄이는 도구가 될 수 있다.

---

## 5. Proposed Method: Agentic Granularity Navigation

**"Explore(탐색) vs. Exploit(활용)"** 전략을 수행하는 에이전트 프레임워크.

### **Module 1: The Router (Diagnosis)**
이전 라운드의 가설(Hypothesis)이 실제 검색 결과(Evidence)에 존재하는지 검증.
* **Input:** `Previous Generated Docs` (Hypotheses) + `Retrieved Docs` (Evidence)
* **Decision per Type:**
    * **EXPLOIT:** 가설이 증거에 있음. (성공)
    * **EXPLORE:** 가설이 증거에 없음. (실패 - Corpus에 해당 내용 없음)
    * **PRUNE:** 노이즈.

### **Module 2: The Executor (Refinement & Pivot)**
Router의 결정에 따라 쿼리를 재조립.
* **For EXPLOIT:** **Anchor(닻)**로 사용. 기존 키워드를 유지하고 심화함.
* **For EXPLORE:** **Negative Constraint(오답 노트)**로 사용.
    * *Logic:* "이 Entity/Direction을 찾으려 했으나 실패했다(Failed Path). 따라서 이것은 Corpus에 없다고 가정하고, **새로운 가설(Alternative Hypothesis)**로 Pivot하라."

### **Implementation Detail**
* **Output Format:** JSON Structured Output (`Plan` field for CoT, `Answer_documents` for content).
* **Flow:** Round 0 (Generate Initial Hypotheses) -> Retrieval -> Router (Check) -> Executor (Refine/Pivot) -> Next Retrieval.

### Interaction Hypothesis (Query rewriting ↔ Graph traversal)
abstraction gap이 큰 경우, “어떤 bridge/추상이 필요한지”는 초기에 고정된 query로는 드러나지 않는다. 따라서:
- traversal이 매 단계 관측하는 **경로/요약(ancestor branch desc)**, **실패 신호(stagnation)** 를 rewriting의 입력으로 넣고
- rewriting 출력이 다시 traversal의 **게이팅/확장(child scoring)** 에 영향을 주는
**닫힌 루프**가 필요하다.

---

## 6. Current Status & TODOs

### **Current Experiments**
- [x] **Fragility Test:** BRIGHT가 Hard Negative에 취약함을 입증 (Similiarity Analysis).
- [x] **No Docs vs. Docs:** 문서의 편향(Bias)이 성능을 저하시킴을 확인.
- [x] **Prompt Engineering:** JSON 구조화 및 Plan(CoT) 강제를 통해 Instruction Following 성능 확보.
- [ ] **Flat Retrieval → Gate → Traversal (LATTICE 기반):**
    * **Idea:** leaf/branch 구분 없이 전 노드를 flat retriever로 먼저 검색하고, 상위 branch를 gate로 삼아 LATTICE traversal을 수행.
    * **왜 LATTICE인가 (정당화):** 기존 실험은 “predefined level(Type/Aspect)”을 사람이 설계하고 agent가 그 레벨을 따라 explore/exploit 했음. 그런데 효과 없는 서브카테고리가 존재한다는 건 **(i) 레벨 설계 실패** 또는 **(ii) 탐색이 특정 지역에 갇히는 문제(locality)** 또는 둘 다일 수 있음. LATTICE는 문서 계층(요약 branch + leaf)과 **탐색의 지역성(현재 노드 주변 확장)**이 구조적으로 존재하므로, “abstraction gap = 표현/층위 mismatch”와 “abstraction gap = 탐색 경로의 지역성 제약”을 **분리/검증**하기 좋은 테스트베드임.
    * **의미 (abstraction gap과의 연결):** abstraction gap을 “필요한 추상/브릿지 정보가 다른 semantic region에 있고, 계층 탐색이 그 region에 도달하지 못하는 현상”으로 재정의하면, flat→gate는 **teleport(전역 앵커링)** 역할을 해서 지역성 제약을 완화하는지 보여줌.
    * **핵심 설계:** flat에서 직접 잡힌 leaf는 **버리지 않고 보관**, traversal 결과 leaf들과 **RRF로 fusion**.
    * **Ablation (주장 가능성 확보):**
        1) `Original traversal` vs 2) `Flat-only` vs 3) `Flat→Gate→Traversal` (cost는 동일한 beam=2, iter=20 유지).
        - (2)와 (3)의 차이는 “gate로 만든 **local expansion의 질**”을, (1)과 (3)의 차이는 “전역 앵커링(flat)으로 **locality 제약을 풀었는지**”를 보여줌.
    * **해석:**
        - (3)이 nDCG/Recall을 유의미하게 오름: 실패 원인의 상당 부분이 **탐색 지역성 제약(local expansion bias)** 에서 왔다는 근거.
        - flat 단계에서 gold leaf의 **ancestor branch hit / gate hit**가 (1) 대비 증가함: “원래 traversal이 도달하지 못하던 올바른 branch region을 flat이 찾아줬다”는 근거 → vertical navigation의 필요성과, abstraction gap의 “검색-탐색” 측면을 강화.
- [ ] **Interaction: Evidence-conditioned query rewriting during traversal**
    * **Idea:** flat→gate로 “어디로 들어갈지”를 잡은 뒤, traversal이 관측한 branch 요약/경로 신호를 사용해 query를 **동적으로 업데이트**하고, 그 query로 다음 iter의 child slate scoring/선택을 수행.
    * **목표:** “전역 앵커링으로 locality를 푸는 것”을 넘어서, “도달한 영역 내부에서 필요한 abstraction/bridge를 점진적으로 정제”하여 추가 이득이 있는지 확인.
    * **Ablation:** (C0) original traversal, (C1) flat→gate 고정 query, (C2) gate-conditioned rewrite 1회, (C3) stagnation-triggered rewrite(예산 제한), (C4) beam-conditioned rewrite + score fusion.
    * **해석:** C1까지만 오르면 locality 병목이 주원인, C2–C4까지 추가로 오르면 “동적 브릿지 생성(표현 업데이트)”도 gap의 핵심 원인.
- [x] **Agent Pipeline Construction:** Router-Executor 루프 구현 및 성능 평가.

### **Future Work**
1.  **Policy Learning:**
    * 현재의 Prompt-based Router/Executor를 경량화하기 위해, **Successful Trajectory**를 수집하여 작은 모델(Policy Network)로 Distillation.
    * *Approach:* Oracle(Gold Doc 존재 여부)을 이용해 Action Label(`EXPLORE` vs `EXPLOIT`) 자동 생성 후 학습.
2.  **Benchmark Expansion:** BRIGHT 외 다른 Reasoning Task(HotpotQA 등)에서의 일반화 성능 검증.

---

## 7. How to Run (Prompt Structure)

### **Router Prompt**
```python
# Analyzes overlap between Hypotheses and Evidence
Input: Prev_Docs (Hypotheses), Retrieved_Docs (Evidence)
Output JSON: {
  "Actions": {
    "Theory": "EXPLOIT",
    "Entity": "EXPLORE",
    ...
  }
}
```
### ** Executor Prompt**
# Generates next query based on Router's decision
Input: Router_Actions, Prev_Docs
Instruction:
  - If EXPLOIT: Keep and refine.
  - If EXPLORE: Treat as "Failed Path". Pivot to new hypothesis.
Output JSON: {
  "Plan": "Step-by-step reasoning...",
  "Possible_Answer_Docs": { "Theory": "...", "Entity": "..." }
}
