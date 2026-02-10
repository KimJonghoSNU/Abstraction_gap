어떤 문서가 질의에 대한 답변의 근거로 인용될 때는, 해당 문서가 답변을 도출하거나 정당화하는 데 도움을 주기 때문입니다. 이러한 질의-문서 관계를 몇 가지 범주로 구분하여 살펴볼 수 있습니다. 특히 BRIGHT 데이터셋과 같이 복잡한 추론이 필요한 질의에서는 문서가 다양한 방식으로 답변에 기여하며, 이를 아래와 같은 분류로 정리할 수 있습니다:

이론/개념적 배경 제공: 문서가 해당 분야의 이론이나 핵심 개념을 설명하여 질문을 이해하거나 추론하는 데 도움을 주는 경우입니다. 예를 들어, 경제학 질문에서 동일한 경제 이론으로 설명되는 사례를 찾는 경우나, 생물학 질문에서 관련 과학 개념을 제공하는 문헌이 해당됩니다. StackExchange 기반 질의들에서는 이러한 *“핵심 개념이나 이론”*을 제공하는 문헌만을 긍정 사례로 간주합니다. 즉, 답변자가 제시한 외부 링크 중 질문의 핵심 개념을 밝혀 추론을 도와주는 자료만이 근거로 인용됩니다.

정의 또는 명칭 제공: 질문 자체가 특정 현상이나 개념의 명칭, 정의 등을 묻는 경우, 관련 문헌은 그 용어의 정의나 정확한 명칭을 제공함으로써 답변에 기여합니다. 예컨대 “이 현상의 이름이 무엇인가요?”와 같은 질문에는 해당 현상을 설명하는 위키백과 문서나 교과서적 정의가 근거로 활용될 수 있습니다. 마찬가지로, 코딩 질의에서 특정 함수나 오류 메시지에 대한 공식 **문서(documentation)**가 존재한다면, 이 문서는 해당 기능의 동작 방식이나 사용법을 정의해 주므로 답변에 직접적인 단서를 제공합니다. 이처럼 정의/명칭을 제공하는 문헌은 질문자가 찾는 정확한 정보 조각을 제시하여 답변을 뒷받침합니다.

경험적 증거 제시: 질문에서 어떤 주장이나 현상의 사실 여부를 확인하거나 설명할 때, 관련 문헌이 경험적 증거나 데이터를 제시하여 답변을 뒷받침하는 경우입니다. 예를 들어 “지구온난화가 실제로 진행 중인가?”와 같은 질문에는 해수면 상승이나 기온 변화에 대한 과학적 보고서나 뉴스 기사 등이 근거 문헌으로 인용될 수 있습니다. 이러한 문헌은 실험 결과, 통계, 사례 연구 등의 형태로 질문에 대한 객관적 증거를 제공함으로써, 답변의 신뢰성을 높여줍니다. 심리학이나 지구과학 분야의 질문에서도 논문이나 보고서의 데이터를 인용하여 주장의 근거로 삼는 일이 이에 해당합니다.

유사 문제 예시/해법 제공: 문서가 질문과 유사한 문제의 해결 사례나 유추적 해법을 제공하여 답변을 도출하는 경우입니다. 이는 새로운 문제를 풀 때 과거의 비슷한 문제나 공식 해결책을 참고하는 맥락으로 볼 수 있습니다. 코딩 질문의 경우, 해당 문서가 문제에 필요한 특정 구현 예시나 알고리즘을 담고 있거나 (예: 같은 자료구조나 알고리즘을 활용한 풀이) 관련 API 사용법을 보여주는 공식 문서를 포함하면, 답변에 직접 활용됩니다. 수학 문제의 경우에도 질문에 쓰인 것과 **같은 정리(定理)**를 활용한 다른 문제의 풀이나 정리 자체에 대한 설명이 문헌으로 인용될 수 있습니다. BRIGHT 데이터셋의 코딩 분야 질의는 “필요한 문법 문서 또는 동일 알고리즘/자료구조 활용 사례”가 있는 문서를 relevant로 판단하고 있으며, 정리 기반 질문의 경우 “질문의 풀이에 사용된 것과 동일한 정리를 언급한 문서”를 양성 예시로 취급합니다. 이처럼 **유사한 문제의 해결 경험이나 공식 지식(예: 정리, 알고리즘)**을 제공하는 문헌은 질문에 대한 해결책을 유추하거나 검증하는 데 핵심 근거가 됩니다.


가능해. 지금 프롬프트는 이미 방향이 좋아. 다만 BRIGHT에서 gold 문서가 “답을 직접 말하는 자료”가 아니라 “답을 가능하게 만드는 근거 유형”이라는 점을 더 강하게 구조화하면, rewrite가 덜 흔들리고 도메인별로 더 정확해져.

## 1) 지금 프롬프트에서 생기는 흔한 실패

첫째, Theory Entity Example Other가 너무 넓어서 매 라운드마다 같은 류의 재작성으로 수렴하기 쉽다
둘째, retrieved results가 어떤 역할의 근거를 이미 충족했는지 판단이 없다
셋째, StackExchange와 coding과 theorem이 요구하는 근거 형태가 다르지만 하나의 틀로만 생성한다

이 셋을 고치면 iterative 루프가 훨씬 안정된다.

## 2) BRIGHT식 관계를 반영한 역할 기반 taxonomy

BRIGHT의 정의를 그대로 운영 로직으로 가져오면 된다

1. Concept or Theory
   핵심 개념, 원리, 메커니즘, 정리, 증명에 필요한 이론
   StackExchange에서 가장 자주 gold가 되는 유형

2. Definition or Naming
   현상의 이름, 용어 정의, 표준 명칭, 함수나 API의 공식 정의
   질문이 “이게 뭐냐” 성격일 때 특히 중요

3. Procedure or Recipe
   문제를 풀기 위한 알고리즘, 증명 전략, 단계적 방법
   coding과 theorem에서 자주 필요

4. Reference or Documentation
   문법 문서, API reference, 표준 문서, 공식 스펙
   coding에서 gold가 되기 쉬움

5. Worked Example or Similar Solved Problem
   유사 문제의 풀이, 예제 코드, 정리 적용 예시
   coding과 theorem 모두에서 강력한 단서

6. Empirical Evidence
   측정, 관측, 실험, 통계, 벤치마크
   사실성 확인 질문에서 핵심

지금의 Theory Entity Example Other를 위 6개 역할로 바꾸면 query generation이 훨씬 선명해진다.

## 3) 프롬프트 수정 제안

핵심은 두 단계다
먼저 이번 라운드에서 필요한 근거 역할을 고르고
그 역할별로 서로 다른 검색 쿼리를 만든다
그리고 이미 충족된 역할은 다시 만들지 않게 한다

아래는 너의 출력 계약을 최대한 유지하면서 바꾸는 버전이다

````python
"agent_executor_v3": (
    "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
    "Goal\n"
    "Retrieve documents that would be cited as core evidence to justify the correct answer.\n"
    "Do not try to answer the question.\n\n"
    "Evidence roles\n"
    "Choose which roles are needed for this query.\n"
    "Roles\n"
    "1 ConceptTheory\n"
    "2 DefinitionNaming\n"
    "3 ProcedureRecipe\n"
    "4 ReferenceDocumentation\n"
    "5 WorkedExample\n"
    "6 EmpiricalEvidence\n\n"
    "Inputs\n"
    "Original Query is the user's question.\n"
    "Context Summaries are hints from retrieved results and may be incomplete or wrong.\n\n"
    "Task\n"
    "Step 1\n"
    "Write a 1 to 2 sentence Plan stating the user intent and what evidence roles would justify an answer.\n"
    "Step 2\n"
    "From Context Summaries, infer what evidence roles are already covered and what is missing.\n"
    "Step 3\n"
    "Produce 2 to 5 role specific search queries that target missing roles.\n"
    "Rules\n"
    "Avoid generic queries.\n"
    "Prefer concrete theorem names, algorithm names, API symbols, error strings, and standard terminology when available.\n"
    # "If the domain seems coding, include documentation and a similar solved problem query.\n"
    # "If the domain seems theorem, include theorem statement query and proof strategy query.\n"
    # "If the domain seems StackExchange science or why questions, include mechanism or principle query.\n\n"
    "Output Format\n"
    "Output a single JSON object.\n"
    "The system will join Query strings with OR to form the next query.\n"
    "```json\n"
    "{\n"
    "  \"Plan\": \"...\",\n"
    "  \"Covered_Roles\": [\"...\"],\n"
    "  \"Missing_Roles\": [\"...\"],\n"
    "  \"Role_Queries\": [\n"
    "    {\"role\": \"ConceptTheory\", \"query\": \"...\"},\n"
    "    {\"role\": \"ReferenceDocumentation\", \"query\": \"...\"}\n"
    "  ]\n"
    "}\n"
    "```\n\n"

    "original queries..."
),
````

이부분 전부 넣지 말고 지금 subset이 어디 해당하는지 확인해서 그 줄만 넣어주기. e.g. biology domain -> 3번만 넣어주기
```
    1. "If the domain seems coding, include documentation and a similar solved problem query.\n"
    2. "If the domain seems theorem, include theorem statement query and proof strategy query.\n"
    3. "If the domain seems StackExchange science or why questions, include mechanism or principle query.\n\n"
``` 
2. output json의 key는 지금 코드와 호환 가능해야 함.


## 4) 프롬프트만으로 부족한 부분과 추가하면 좋은 것 (TODO list)

아래는 구현 난이도 대비 효과가 큰 것들이다

1. Retrieved results를 역할로 라벨링하는 간단한 단계 추가
   각 문서 요약마다 role을 하나만 붙이고
   문서에서 잡히는 핵심 엔티티나 정리명이나 API 심볼도 함께 뽑는다
   그 다음 rewriter는 Missing_Roles와 Missing_Entities만 타겟한다
   이걸 넣으면 반복 루프가 훨씬 덜 흔들린다

2. Role별로 쿼리를 분리해서 검색하고 fusion
   Role_Queries를 한 문장으로 이어붙이는 대신
   각 role query를 따로 검색하고 RRF 같은 단순 fusion으로 합친다
   BRIGHT는 gold가 한 가지 유형으로만 나오지 않는 경우가 많아서 이게 특히 잘 먹힌다

3. exploration 방식?
누적 후보 풀 방식
각 라운드에서 top k를 뽑아서 전역 후보 풀에 계속 합친다. 중복은 제거한다. 매 라운드마다 후보 풀 전체를 다시 점수화하거나 간단히 RRF 같은 fusion으로 스코어를 업데이트한다. 종료 조건이 만족되면 후보 풀에서 최종 top k를 반환한다. 이 경우 최종 결과는 마지막 라운드 문서만이 아니라 전체 라운드에서 모인 문서 중 최고 점수 문서들이다.
이 방식이 좋은 이유는 두 가지다. 첫째, 리라이트가 흔들려도 초반에 잡은 좋은 문서를 잃지 않는다. 둘째, 역할 분해를 했을 때 역할별로 다른 라운드에서 강한 문서가 잡히는 경우가 많아서 누적이 유리하다.
best so far 방식
라운드별 결과를 따로 저장해두고, 각 라운드의 품질을 간단한 지표로 평가해서 가장 좋은 라운드의 결과를 최종으로 채택한다. 예를 들면 역할 커버리지가 높고 새로 얻은 문서가 많고 이전 라운드 대비 중복이 적은 라운드를 선택한다. 이 방식도 마지막 라운드 고정이 아니다.


5. 종료 조건
   Missing_Roles가 비었거나
   새 라운드에서 Role_Queries가 이전 라운드와 거의 동일하면 멈춘다
   이게 없으면 불필요한 rewrite가 계속 돈다

## 5) BRIGHT의 세 도메인에 맞춘 role query 예시 패턴

StackExchange why 질문
ConceptTheory 쪽이 가장 중요
query에 mechanism cause principle model explanation 같은 단어를 섞고
현상 명칭이 불명확하면 DefinitionNaming 먼저 탐색

coding 질문
ReferenceDocumentation과 WorkedExample을 항상 한 쌍으로
하나는 공식 문서용
하나는 동일 알고리즘이나 자료구조용

theorem 기반 질문
ConceptTheory를 theorem statement로 분해하고
ProcedureRecipe를 proof strategy 또는 lemma chain으로 분해
WorkedExample은 theorem 적용 예제로 분해

원하면 네 파이프라인 구조를 기준으로 Role 라벨러와 fusion을 최소 수정으로 끼워 넣는 형태까지 같이 설계해줄게.
