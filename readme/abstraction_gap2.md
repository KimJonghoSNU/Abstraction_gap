# Abstraction Gap in Reasoning-Intensive Retrieval: Feedback Locality and Corpus-Induced Abstraction Units

> 목적: `abstraction_gap.md`의 **Problem definition ↔ Method**를 논문 서술처럼 “같은 축”에 맞춰 정렬한 버전.  
> 핵심: abstraction gap을 (A) **bridge 표현 부족** + (B) **top‑K→rewrite 피드백이 만드는 locality(경로 의존성)** 로 정의하고, 이를 완화하는 방법을 **corpus‑induced abstraction units(요약/중간 노드) + 전역 앵커링(flat) + (선택적) gated navigation**으로 제시한다.

---

## 0. 나의 이전 연구 REPAIR와 비교
- 큰 방향은 
    - 기존 연구에서 plan = query에 대한 답이 뭘지 작성하는 것
    - 지금은 abstraction 을 해야한다 = query에 대한 답이 있을 때 어떤 원리/근거에 기반해서 작성된 답인지 찾는 게 핵심이다

- 변경점
    - query plan -> query abstraction 으로 바뀌고
    - corpus도 비슷한 내용끼리 연결한 (수평적) graph로 표현하는 게 아니라 abstraction한 (계층적) tree 구조로 표현하는 방법 시도하고 있습니다

## 1. Problem Definition

Reasoning‑Intensive Retrieval(예: BRIGHT)에서는 질의가 종종 **구체(현상/사례)** 로 표현되지만, 정답 문서는 **추상(원리/모델/메커니즘)** 수준에 존재한다. 이때 단순 semantic similarity 기반 top‑K는 gold(또는 gold로 가는 bridge concept)를 놓치기 쉽고, 그 결과 LLM의 추론이 쉽게 흔들린다. 우리는 이 현상을 **abstraction gap**으로 부른다.

### 1.1 Two coupled failure modes of abstraction gap

**(A) Bridge representation bottleneck (표현/브릿지 부족)**  
- 초기 질의 표현이 gold의 추상 개념과 직접 연결되지 않는다.  
- 따라서 retriever가 “gold로 가는 중간 개념(bridge)”을 top‑K에 올리지 못한다.  
- 관측되지 않은 개념은 이후 어떤 reasoning/rewriting에도 사용될 수 없으므로, 실패가 구조적으로 고착된다.

**(B) Feedback locality (top‑K → query rewrite가 만드는 경로 의존성)**  
- agentic search는 보통 `Retrieve(top‑K) → LLM으로 요약/답/키워드 생성 → Query rewrite → Retrieve ...` 형태의 피드백 루프를 가진다.  
- 이때 rewrite의 입력이 항상 “직전 top‑K”에 의해 제한되므로, 초기에 특정 semantic region(대개 표면적으로 비슷한 region)에 진입하면 이후 rewrite가 그 region을 **self‑reinforcing** 하여 다른 region으로 점프하기 어렵다.  
- 즉, 여기서의 locality는 “계층 탐색을 했기 때문”이 아니라, **top‑K 기반 관측 자체가 만드는 locality**다.

> 연결성: (A)로 인해 올바른 bridge가 top‑K에 안 뜨면, (B)의 피드백 루프는 잘못된 region을 계속 강화한다. 두 병목은 독립이 아니라 **서로를 증폭**한다.

### 1.2 Hard negative → logic poisoning as a special case of (A)+(B)

Reasoning setting에서는 “주제는 비슷하지만 논리가 틀린” hard negative가 LLM의 추론을 오염시키는 **logic poisoning**을 유발한다.

이 현상은 (A)(B)와 유기적으로 연결된다.
- (A) bridge가 없을 때, hard negative는 gold의 대체 근거처럼 보이며 “잘못된 추상 방향”을 제공한다.
- (B) 피드백 루프에서 hard negative는 단지 reasoning만 망치는 게 아니라, **rewrite를 오염**시켜 다음 retrieval까지 같은 region으로 끌고 간다(= “초기에 잘못 진입하면…”의 강화판).

따라서 abstraction gap을 해결하려면 “더 많은 문서”가 아니라, **관측(top‑K)과 이동(다음 query/다음 region)을 강건하게 만드는 메커니즘**이 필요하다.


좀 더 engineering 관점에서 서술한다면
- query -> answer -> 그 answer를 내기 위해 쓰인 (핵심) 근거 문서들 (theory / entity / example) . query rewriting은 이런 레벨로 맞춰줘야 함
- 근데 정확히 그 문서 detail까지 뱉기는 어려울 것. 문서 level에 대해서도 그러면 abstraction 시킨 정보들로 보관해줘야 함.

---

## 2. Prior Attempts and the pivot

### 2.1 Baseline agentic search (일반적 형태)
우리가 사용해온 agentic search의 핵심은 다음 중 하나다.
- top‑K 문서를 주고 LLM이 **답/요약**을 작성하게 한 뒤, 그 텍스트를 다음 retrieval query로 사용
- top‑K에서 **키워드/핵심 표현**을 뽑아 query rewrite에 반영

이 방식은 (B) 피드백 locality를 본질적으로 가진다(입력이 top‑K로 제한되기 때문).

<!-- ### 2.2 Fixed-level prompts (우리 아이디어)와 한계
저번 주까지의 시도는 example/entity/theory 같은 **고정 레벨(level)을 프롬프트로 명시**하고, 레벨별 explore/exploit을 수행하는 방식이었다.
- 장점: “어떤 추상 방향으로 갈지”를 시스템이 강제로 만들어냄
- 한계: 레벨 taxonomy는 쿼리/도메인에 따라 유효성이 달라서, 효과 없는 서브카테고리가 남는다
- 한계: “추상 방향”은 만들어도, **그 방향에 대응하는 retrieval anchor**가 없으면 rewrite가 문서와 접합되지 않는다
- 한계: top‑K에 묶인 rewrite loop 안에서 레벨만 바꾸면 **잘못된 region의 locality**를 깨지 못한다

결론적으로, 다음 단계의 질문은:
> “레벨을 사람이 미리 정하는 대신, **corpus 정보를 사용해 쿼리마다 유효한 abstraction 단위를 동적으로 만들고, 그 단위가 rewrite의 anchor로 작동하도록 만들 수 있을까?**” 현재 결과: 이거 망함. 다른 방향 탐색 -->

---

## 3. Corpus-induced abstraction units (and why interaction matters)

LATTICE류 방법은 코퍼스에서 문서 클러스터의 **중간 요약 노드(branch)** 를 생성한다. 
동시에 피드백 locality(B)를 깨기 위한 **외부 개념 핸들(concept handle)** 로 기능할 수 있다.
- top‑K 문서만 보던 루프에, 그래프를 따라서 다른 문서들을 추가해서
- 잘못된 region을 강화하는 rewrite를 완화하고, 새로운 region으로 점프할 기회를 만든다.
대략적인 아이디어
2. Your Solution: "Corpus-Guided Exploration via Tree"선생님의 아이디어(Tree를 이용해 안 보이던 요약 정보를 줌)는 이 '장님'에게 **"도서관 지도(Map)"**나 **"책 목록(Catalog)"**을 쥐여주는 것과 같습니다.
이것이 왜 핵심인지 3단계로 연결됩니다.
- Observability (가시성 확보):기존: "벌 세는 법 찾아줘." (상상 속의 쿼리)Ours: "여기 [생물학], [통계학], [농업] 섹션이 있어. 어디를 볼래?" (Tree Branch 제공)
- Adaptive Steering (방향 조정):Rewriter가 Tree의 Branch Summary를 보고, "아, 내가 찾는 건 [생물학]이 아니라 [통계학] 섹션에 있을 확률이 높겠구나"라고 판단.$\rightarrow$ Blind Projection이 아니라 Targeted Projection이 됨.
- Expansion (범위 확장):현재 방향(Local)에서는 안 보이던 문서들을, Tree 구조를 통해 **강제로 시야에 넣음(Exposure)**으로써 검색 범위를 넓힘.



<!-- 중요한 점은:
- 이 요약은 **질의 없이 코퍼스만 보고 만들어진** 표현이다(= query‑dependent가 아님).

따라서 핵심은 “branch 요약이 곧 레벨”이 아니라,
> 검색 과정에서 **어떤 branch 요약을 관측/선택해 rewriting에 넣고**, 그 결과가 다음 이동을 어떻게 바꾸는지(= 상호작용)다.

이 관점에서 branch 요약은 fixed level의 “대체물”이라기보다, **fixed‑level prompt가 만든 추상 방향을 문서 쪽에서 정착시키는 retrieval anchor**다.  
동시에 피드백 locality(B)를 깨기 위한 **외부 개념 핸들(concept handle)** 로 기능할 수 있다:
- top‑K 문서만 보던 루프에, **코퍼스가 제공하는 중간 개념**을 주입하여
- 잘못된 region을 강화하는 rewrite를 완화하고, 새로운 region으로 점프할 기회를 만든다. -->

---

<!-- ## 4. Method:anchoring → gate → (optional) navigation

### 4.1 Global anchoring by flat retrieval over all nodes
leaf/branch를 구분하지 않고 “모든 노드”를 dense retriever로 한 번에 검색한다.
- 목적: (A)의 bridge 관측 실패를 줄이고, (B)의 피드백 locality를 깨기 위한 **전역 앵커 후보**를 확보

### 4.2 Gate induction
flat top‑K에서 선택된 노드들을 “들어갈 region”으로 정규화한다.
- branch hit: 그대로 gate 후보
- leaf hit: leaf의 ancestor branch로 승격(규칙은 ablation)
- 결과 gate는 허용 prefix 집합(`allowed_prefixes`)으로 표현

### 4.3 Local expansion inside the gate (선택적)
gate 내부에서만 local expansion을 수행한다(예: 기존 LATTICE traversal).  
여기서 “계층 탐색”은 연구의 major direction이라기보다, **gate로 정한 region 내부에서의 exploit** 구현체다.

### 4.4 Robust final ranking
flat에서 직접 잡힌 leaf와 gate 내부 탐색 결과 leaf를 함께 사용하고, 랭크‑레벨 fusion(RRF 등)으로 결합한다. -->

---

## 5. Interaction design (TODO): Query rewriting ↔ corpus units ↔ retrieval loop

이 연구가 궁극적으로 겨냥하는 것은 다음의 닫힌 루프다.

1) flat/top‑K로 얻은 후보 + 선택된 branch 요약(코퍼스 단위)을 관측  
2) 그 컨텍스트로 query rewrite를 생성(bridge/추상 이동을 반영)  
3) rewrite로 다음 retrieval(또는 gate/확장 정책)을 업데이트  
4) 반복하며 “필요한 abstraction”을 점진적으로 정렬
prompt예시
    ```
    "round3_action_v1": (
        "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
        "Goal:\n"
        "- Decide whether to stay local (EXPLOIT) or change direction (EXPLORE).\n"
        "- If evidence supports the current hypothesis (previous rewrite), choose EXPLOIT.\n"
        "- If evidence is missing, contradictory, or points elsewhere, choose EXPLORE.\n\n"
        "Task:\n"
        "- Write a short Plan that follows these steps:\n"
        "  1. Identify the user's intent and answer type.\n"
        "  2. Abstraction: infer which academic terms, theories, models, or canonical methods would be cited in a correct answer.\n"
        "  3. Verification: compare the evidence against your current hypothesis.\n"
        "     - If supported -> EXPLOIT.\n"
        "     - If missing/contradictory -> EXPLORE and treat the previous rewrite as a negative constraint.\n"
        "  4. Ensure every generated document is abstract evidence (NOT surface-level restatements).\n"
        "- Action-specific behavior:\n"
        "  - EXPLOIT: Refine the current hypothesis and **use key terms from evidence as anchors**, while staying abstract.\n"
        "  - EXPLORE: Pivot to a new abstract hypothesis (different theory/framework/entity name), avoiding the failed direction.\n"
        "- Produce 3-5 distinct Possible_Answer_Docs that could serve as evidence for the assumed correct answer.\n"
        "- Do NOT require lexical overlap with the original query; prioritize abstract evidence.\n"
        "- Evidence forms may include (not exhaustive): theory/mechanism, entity/fact, analogy/example, method/metric, canonical reference.\n"
        "Output JSON only:\n"
        "{\n"
        "  \"action\": \"exploit\",\n"
        "  \"Plan\": \"short reasoning\",\n"
        "  \"Possible_Answer_Docs\": {\n"
        "    \"Theory\": \"...\",\n"
        "    \"Entity\": \"...\",\n"
        "    \"Example\": \"...\",\n"
        "    \"Other\": \"...\"\n"
        "  }\n"
        "}\n\n"
        "Original Query:\n{original_query}\n\n"
        "Previous Rewritten Query:\n{previous_rewrite}\n\n"
        "Leaf Evidence:\n{leaf_descs}\n\n"
        "Branch Context:\n{branch_descs}\n"
    ),
    ```

<!-- 비용 제어를 위해 rewrite는 정체 시에만 호출한다. (호출 threshold 정해야 함) -->
Gate 내부 탐색(Exploit)만으로 충분한지(Router 판단), 아니면 Gate 자체를 다시 설정해야 하는지(Rewrite = Explore)를 결정한다.

---

## 6. What we can claim (aligned to the problem)

### 6.1 Empirical result we already have (True)
`Flat → Traversal`이 baseline 대비 **Recall을 증가**시킨다.

이 결과로부터 최소한 다음을 주장할 수 있다.
- top‑K 기반 피드백 루프가 만드는 locality(B)에서 벗어나기 위해, **전역 앵커(flat) + region gate**가 유효하다.  
  (초기 관측/진입이 바뀌면 이후 탐색이 달라진다.)

### 6.2 Next claim to validate (TODO. not fixed. still on a discussion)
- 코퍼스가 제공하는 중간 요약 노드를 rewriting 컨텍스트로 주입할 때 (B) locality와 hard negative 오염을 완화하는 방법이 필요. 
그런데 현재 상태: psychology같은 경우 중간 요약 노드가 오히려 성능 떨어짐

언제 rewrite해야할까? 
- stagnation: 이전 라운드 대비 top-K 노드들의 prefix 다양성이 거의 안 변함
- low gate confidence: gate 후보들의 score gap이 작음(1등~3등 비슷)
- poison signal: top-K leaf 간 상호 모순/불일치가 높거나, 질문과의 “논리적 entailment”가 낮음(간단히는 LLM judge 1회)


## discussion
-(MT) 고려해야 할 사항
1. gold document가 유일한가? 다양한 abstraction aspect가 있을 수 있다. apsect ambiguity 제거가 필요하다
	- e.g. 벌들의 움직임을 관측하는 방법?
		- 지금 정답: 푸아송 분포. 가능한 다른 정답들: 양봉 커뮤니티 게시글...
	- 그래서 setting도 중요하게 작용할 것: web인지, domain-focused corpus인지. corpus가 web이면 corpus feedback으로 어떤 aspect가 정답인지 찾는 것은 불가능하기 때문. 이런 경우에는 user persona... 등에 의존해야 함.
		- -> (JH) 일단 domain-focused corpus에 집중하기로. document environment와 상호작용하면서 abstraction aspect를 줄여나갈 수 있는 세팅.
2. LLM이 주어진 정보가 쿼리에 대한 답으로 충분한지, 충분하지 않은지 internal knowledge로 알 수 있다면, retrieval 자체가 필요 없는 상황 아니냐?
	- (MT) LLM이 직접 답을 생성하는 건 못할 수 있음. 세부적인 지식들을 모델이 다 알지는 못함.
	- (MT) 하지만 지식을 abstraction -> abstraction 카테고리별로 지식이 나눠져 있다면, 카테고리별로 yes/no 판단하는 능력은 모델이 갖고 있다고 전제할 수 있음.
		- = 카테고리 레벨로 navigate하는건 llm이 모든 세부적인 지식 없이 할 수 있지 않을까




## 메모
- flat retrieval을 한다는 아이디어: RAPTOR (ICLR2024)에 이미 있음 이 자체는 contribution이 아님
- 
