# 260208 Lab Meeting Slide Storyline (compressed)

## Slide 1. RQ and Why this line of work
- Key message: Reasoning-intensive retrieval의 병목은 "답 생성"보다 "근거 문서 탐색"이다.
- Evidence:
    - Query-gold evidence abstraction gap 존재.
    - 모델이 필요한 category를 알아도 corpus에 없거나 너무 추상적이면 실패.
- Takeaway: 핵심 RQ는 "abstraction을 retrieval gain으로 어떻게 변환할 것인가?".

## Slide 2. Abstraction is necessary (quantitative)
- Key message: 추상화 없는 rewrite는 miss 탈출 능력이 약하다.
- Evidence (iter0->1, thinkqe vs action_v1, 5 subsets):
    - `MissToHit@10`: `+13.49pp`
    - `Delta_RegionHitL2@10_givenPreMiss`: `+13.19pp`
    - `Hit@10`: `+2.38pp`
    - `nDCG@10`: `+5.75`
    - trade-off: `HarmRate@10_givenGood0`: `+3.44pp`
- Takeaway: abstraction은 필요하지만 안정성 trade-off가 남는다.

## Slide 3. Tree feedback attempt: opportunity exists, conversion is weak
- Key message: branch 신호 기회는 있었지만 현재 branch->leaf 정책(v2~v6)은 이득 연결이 약했다.
- Evidence:
    - `Opportunity@10_givenFlatLeafMiss`: `9.45%`
    - `ConversionByGraphOn@10_givenOpportunity`: `23.33%`
    - `Delta_Hit@10 (graph_on - off)`: `-1.17pp`
    - `Delta_nDCG@10 (graph_on - off)`: `-0.46`
- Takeaway: "tree가 무용"이 아니라 "현재 tree 구성/정책이 약함"으로 해석하고, 다음은 category 기준 tree 재설계로 전환.

## Slide 4. Prompt family comparison (what currently works)
- Key message: free rewrite보다 구조화된 action/agent prompting이 우세.
- Evidence (avg nDCG@10 max):
    - `none + action_v1 = 44.02`
    - `none + agent_executor_v1 = 44.57`
    - `none + free_rewrite_v2 = 43.04`
    - `v2 + action_v1 = 44.59`
- Takeaway: category/control 구조가 중요.

## Slide 6. Round4 pivot: LLM generation + tool-based controller
- Key message: 생성은 LLM, explore/exploit 선택은 score-rule controller로 분리.
- Evidence:
    - Rule A margin gate + exploit lock + worst-one-drop.
    - force_drop_one(biology): 상승 케이스 존재(샘플 기준 73.8%에서 >=1회 상승), 단 tie가 큼.
- Takeaway: 이 결과는 "가능성 evidence"까지만 주장하고, 평균 우월성 결론은 보류한다.

## Slide 7. Decisions and next 2 weeks
- Key message: 이번 2주는 "무엇이 안 되는지"를 제거했고, 다음은 controller quality 검증이다.
- Keep:
    - category policy / score-based controller
    - tree 재설계 방향: 내용 유사도 기준이 아니라 category 기준 그룹화
- Deprioritize:
    - summary/history 중심 접근
- Next actions:
    1. `force_full` vs `force_drop_one` controlled A/B (biology/psychology/earth_science)
    2. exploit gain의 조건부 패턴 분석 (query 유형별)
    3. Intro 문장 고정: "Abstraction is necessary but not sufficient; controller policy determines retrieval gains."

## Questions to resolve before final slide deck
- Q1. 기존 tree 실패(v2~v6)와 새 tree 재설계(category grouping)의 연결 문장을 어떻게 고정할까?
    - 제안 답: "기존 실패는 신호 부재가 아니라 구성 기준의 미스매치였고, 그래서 category-conditioned tree로 전환한다."
- Q2. Slide 2의 비교(thinkqe vs action_v1)와 Slide 4의 비교(action/agent/free rewrite)는 실험 조건이 완전히 동일한가?
    - 동일하지 않으면 "직접비교"가 아니라 "방향성 evidence"로 표현을 낮춰야 함.
- Q3. Main RQ를 "category discovery"로 둘지 "controller quality"로 둘지 최종 1개로 잠글 필요가 있음.
    - 현재 문서는 controller quality 중심으로 정렬됨.

## Locked for this meeting
- force_drop_one 결과는 "가능성 evidence"까지만 주장한다.


## Appendix 1. Retrieved-context feedback (history/summary) result
- Key message: history/summary 주입은 baseline 대비 실효 이득이 없었다.
- Evidence:
    - baseline avg: `44.02`
    - `summary=on, history=off`: `41.94` (delta `-2.08`)
    - `summary=on, history=on`: `41.96` (delta `-2.06`)
    - history on/off 차이 자체: `+0.02` (near neutral)
- Takeaway: LLM salience 선택(요약/기억 주입)만으로는 병목 해결이 어렵다.
