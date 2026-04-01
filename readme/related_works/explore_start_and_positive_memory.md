# Related Works: Explore Start Selection and Positive-Memory Management

이 메모는 아래 두 질문만 기준으로 정리한다.

1. explore할 때 어디서부터 explore할지 어떻게 정하는가?
2. 이전 iteration에서 나온 좋은 정보, 즉 positive reward / promising discovery를 어떻게 저장하고 관리하는가?

비교 대상:

- Query Decomposition for RAG: Balancing Exploration-Exploitation
- Go-Explore
- MCTS / UCT
- Dual-Scale World Memory for LLM Agents Towards Hard-Exploration Problems (GLoW)

중요한 제한:

- `Query Decomposition for RAG: Balancing Exploration-Exploitation`는 이 환경에서 primary source로 abstract 수준만 안정적으로 확인되었다.
- 그래서 이 논문은 bandit framing과 memory abstraction까지만 적고, 세부 bandit variant는 과장하지 않는다.

## One-line comparison

| Work | Explore starts from | Positive information is stored as | Why this matters for our setting |
| --- | --- | --- | --- |
| Query Decomposition for RAG | sub-query / arm | per-subquery utility belief | branch가 아니라 query-arm level controller에 가깝다 |
| Go-Explore | archive에 저장된 promising state / cell | archive of cells + return trajectories | "좋은 frontier를 기억하고 그 지점으로 돌아간다"는 철학이 가장 직접적이다 |
| MCTS / UCT | current search tree의 frontier node | node visit counts + value statistics | positive memory가 언어형 메모리가 아니라 search statistics다 |
| GLoW | state archive 중 global frontier와 alignment가 높은 state | dual-scale memory: trajectory frontier + local multi-path reflections | global frontier memory와 local retry memory를 분리한다 |

## 1. Query Decomposition for RAG: Balancing Exploration-Exploitation

Source:

- arXiv abstract: <https://arxiv.org/abs/2510.18633>

핵심 framing:

- complex query를 여러 sub-query로 decomposition한다.
- retrieval은 "문서를 하나씩 가져오면서 어떤 sub-query가 더 유망한지 belief를 업데이트하는" exploitation-exploration problem으로 본다.
- abstract 기준으로, 논문은 이 과정을 bandit learning으로 본다.

### Explore를 어디서부터 시작하나

- explore start의 기본 단위는 `branch state`가 아니라 `sub-query arm`이다.
- 즉 "어느 state로 돌아갈까?"가 아니라, "다음 문서는 어느 sub-query에서 뽑을까?"를 고른다.
- 우리 setting으로 옮기면:
    - tree branch selection보다
    - `rewrite candidate / hypothesis arm` selection에 더 가까운 아이디어다.

### Positive rewards를 어떻게 저장하나

- primary source abstract만 기준으로 보면, memory는 `per-subquery utility belief` 형태다.
- 즉 좋은 정보를 trajectory로 저장한다기보다:
    - 어떤 sub-query가 유용했는지
    - 지금까지 관측한 document utility가 어땠는지
  를 arm statistics로 축적한다.

### 우리 쪽에 주는 시사점

- 장점:
    - explore/exploit을 명시적으로 분리한다.
    - "다음에 어디서 retrieve할지"를 bandit controller로 만들기 쉽다.
- 약점:
    - branch state를 기억하고 그 지점으로 돌아가는 memory는 약하다.
    - hard exploration에서 필요한 "frontier 기억"과는 거리가 있다.

내 해석:

- 이 논문은 `explore start selection`을 state-space가 아니라 `query-space`에서 푼다.
- 그래서 round6의 ended-reseat 같은 branch-controller 문제를 직접 해결해주진 않지만, `which hypothesis to query next`에는 참고가 된다.

## 2. Go-Explore

Sources:

- arXiv: <https://arxiv.org/abs/1901.10995>
- ar5iv HTML: <https://ar5iv.labs.arxiv.org/html/1901.10995>

핵심 아이디어:

- hard-exploration의 핵심 실패 원인을 `detachment`와 `derailment`로 본다.
- 즉 agent가 좋은 frontier를 발견하고도, 그 frontier를 잊거나 다시 정확히 돌아가지 못해서 탐색이 끊긴다고 본다.

### Explore를 어디서부터 시작하나

- explore start는 `archive 안의 promising cell/state`다.
- 중요한 건:
    - 현재 위치에서 조금 더 가는 게 아니라
    - archive에서 고른 좋은 지점으로 `return`한 다음
    - 거기서 다시 explore한다는 점이다.

원문에서 중요한 포인트:

- promising cell은 visit count, novelty, 새 cell discovery 기여도 같은 heuristic으로 weighting된다.
- 선택 확률은 이 weight로 정해지지만, 모든 cell은 원칙적으로 남겨둔다.
- 즉 "frontier가 유망했던 state"를 버리지 않고 archive에 유지한다.

### Positive rewards를 어떻게 저장하나

- memory는 `archive` 자체다.
- archive에 각 cell/state와 그 cell로 돌아가는 trajectory가 저장된다.
- 중요한 건 positive reward를 scalar score 하나로만 저장하지 않고:
    - 어느 지점이 promising했는지
    - 그 지점으로 어떻게 되돌아가는지
  까지 같이 저장한다는 점이다.

### 우리 쪽에 주는 시사점

- 장점:
    - "좋은 frontier를 발견했으면 그걸 잊지 말고 다시 거기서 시작하라"는 메시지가 아주 명확하다.
    - round6에서 reseat 이후 retrieval state가 새 branch를 따라가지 못하는 문제와 직접 맞닿아 있다.
- 약점:
    - deterministic resettable simulator 가정이 강하다.
    - text retrieval/tree retrieval에서는 exact return이 어려워서, archive를 그대로 옮기기보다는 `return proxy`가 필요하다.

내 해석:

- Go-Explore가 네 현재 문제에 주는 가장 직접적인 교훈은:
    - explore를 random하게 하든 score-based로 하든,
    - `좋았던 frontier state를 explicit memory로 보관하고 거기로 복귀하는 메커니즘`이 없으면 탐색이 흔들린다는 점이다.

## 3. MCTS / UCT

Sources:

- Kocsis and Szepesvari, "Bandit Based Monte-Carlo Planning": <https://proceedings.mlr.press/v22/kocsis02a.html>
- Browne et al., "A Survey of Monte Carlo Tree Search Methods": <https://repository.essex.ac.uk/4117/1/MCTS-Survey.pdf>

중요한 점:

- `MCTS`는 너무 넓다.
- 그래서 여기서는 generic MCTS/UCT template로 본다.

### Explore를 어디서부터 시작하나

- explore start는 `current search tree frontier node`다.
- selection 단계에서 tree policy가 exploration/exploitation 균형을 잡으며 다음 urgent node를 선택한다.
- 즉 Go-Explore처럼 archive 밖 state로 teleport하지 않고, 현재 tree 안에서 고른다.

### Positive rewards를 어떻게 저장하나

- memory는 `node statistics`다.
- 대표적으로:
    - visit count `N`
    - value / reward estimate `Q`
- rollout 결과가 backpropagation으로 ancestor까지 올라가며 누적된다.

이 점이 중요하다:

- MCTS의 positive memory는 natural language note가 아니다.
- `which branch looked good`를 tree statistics로 압축해서 저장한다.

### 우리 쪽에 주는 시사점

- 장점:
    - current frontier 안에서 exploration/exploitation을 제어하는 가장 정석적인 틀이다.
    - branch selection score를 `instant retrieval score` 하나에만 맡기지 않고, cumulative statistics와 합칠 근거를 준다.
- 약점:
    - hard reset / explicit archived frontier memory는 약하다.
    - sparse reward에서는 rollout quality가 안 좋으면 tree statistics도 쉽게 흔들린다.

내 해석:

- round6에 가장 직접적으로 가져올 수 있는 건:
    - ended-reseat branch를 고를 때
    - `current retrieval score`만 보지 말고
    - `visit / success / future gain` 같은 cumulative branch statistics를 저장하는 것이다.

## 4. Dual-Scale World Memory for LLM Agents Towards Hard-Exploration Problems (GLoW)

Sources:

- OpenReview: <https://openreview.net/forum?id=bH5uHIVtTe>
- accessible paper text mirror used for line-level inspection: <https://www.researchgate.net/publication/395969391_Dual-Scale_World_Models_for_LLM_Agents_Towards_Hard-Exploration_Problems>

핵심 아이디어:

- hard-exploration을 위해 memory를 `global`과 `local` 두 scale로 나눈다.
- global 쪽은 `trajectory frontier of high-value discoveries`
- local 쪽은 같은 state에서 여러 path를 굴려 advantage-like progress signal을 만든다.

### Explore를 어디서부터 시작하나

- 시작점은 `state archive`에서 고른다.
- 그 selection은 random이 아니라:
    - frontier에서 뽑은 high-value pattern
    - archived state가 그 pattern과 얼마나 align되는지
  를 LLM이 평가해서 결정한다.

즉 이 논문은:

- Go-Explore처럼 archive를 쓰되,
- 어떤 state가 다음 explore start가 될지는
  `achieved value + potential value`를 모두 본다.

이게 중요한 이유:

- 단순히 "이전에 reward가 컸던 상태"만 고르는 게 아니라
- `bottleneck states with high future potential`도 선택 대상으로 둔다.

### Positive rewards를 어떻게 저장하나

- memory가 두 층이다.

1. `Global world memory`
    - high-value trajectory frontier
    - state archive
    - frontier에서 LLM이 읽어낸 achieved/potential value pattern

2. `Local world memory`
    - same starting state에서 multiple trajectories를 굴린 뒤
    - Multi-path Advantage Reflection으로 progress signal을 만든다.

즉 좋은 정보를 저장하는 방식이:

- "좋은 trajectory frontier"
- "같은 state에서의 local trial-and-error 결과"
으로 분리되어 있다.

### 우리 쪽에 주는 시사점

- 장점:
    - 네가 지금 고민하는 두 질문을 가장 직접적으로 같이 다룬다.
    - `어디서부터 explore할지`와 `좋은 정보를 어떻게 memory로 남길지`를 한 프레임 안에 넣는다.
- 약점:
    - LLM-based value analysis가 많이 들어가서 시스템이 무거워진다.
    - memory quality가 LLM analysis 품질에 크게 의존한다.

내 해석:

- GLoW가 가장 좋은 reference인 이유는:
    - global frontier memory와
    - local state-specific learning
  을 분리했기 때문이다.

- 너의 round6 맥락으로 옮기면, 최소한 아래 둘을 분리해야 한다.
    - `global memory`: 어느 branch / path family가 promising했는지
    - `local memory`: 같은 branch 아래에서 어떤 rewrite / retrieval update가 실제로 먹혔는지

## Synthesis for our project

네 질문을 기준으로 보면, 네 현재 문제는 사실 둘을 섞어서 다루고 있다.

- `explore를 어디서부터 시작할지`
- `좋은 결과를 어떻게 기억할지`

그런데 related works를 보면 이 둘을 분리하는 쪽이 더 정교하다.

### A. Explore start selection

- Query Decomposition:
    - sub-query arm 선택
- Go-Explore:
    - archive에서 promising state 복귀
- MCTS:
    - current tree frontier node selection
- GLoW:
    - archive state selection with achieved + potential value

우리 쪽에서 가장 직접적인 선택지는 두 가지다.

1. `MCTS-style`
    - 현재 tree 안에서만 고른다.
    - 장점: 구조 일관성
    - 단점: hard exploration에서는 detached frontier를 다시 못 잡을 수 있다

2. `Go-Explore / GLoW-style`
    - archived promising branch/state로 돌아간다.
    - 장점: lost frontier 재진입이 가능하다
    - 단점: return target memory를 따로 관리해야 한다

내 판단:

- 네 현재 round6 문제는 MCTS보다 Go-Explore / GLoW 쪽 reference가 더 중요하다.
- 이유는 지금 핵심 failure가 "현재 local frontier만 잘 따라가느냐"보다
  `좋은 frontier를 잃고 다시 못 붙는 것`에 더 가깝기 때문이다.

### B. Positive-memory management

related works가 보여주는 핵심 차이는 이거다.

- Query Decomposition:
    - positive memory = arm utility belief
- MCTS:
    - positive memory = cumulative search statistics
- Go-Explore:
    - positive memory = promising state archive + return path
- GLoW:
    - positive memory = global frontier + local advantage reflection

우리 쪽에서 지금 부족한 건:

- 좋은 문서가 나왔다, 좋은 branch가 나왔다, 정도의 local signal은 있는데
- `그걸 다음 explore start selection에 어떻게 반영할지`가 explicit memory object로 정리되어 있지 않다.

즉 더 필요한 건 단순한 cumulative pool보다:

- `promising branch archive`
- `branch family별 achieved / potential score`
- `그 branch로 돌아갈 수 있게 하는 return representation`

## What I would borrow first

우선순위를 정하면 이렇다.

1. Go-Explore에서:
    - promising frontier를 archive로 명시적으로 저장
    - explore는 그 archive에서 다시 시작

2. GLoW에서:
    - global memory와 local memory를 분리
    - achieved value와 potential value를 같이 저장

3. MCTS에서:
    - branch-level cumulative statistics를 유지
    - current retrieval score 하나로만 branch를 고르지 않기

4. Query Decomposition paper에서:
    - hypothesis / rewrite candidate를 arm으로 보고
    - 어떤 query branch를 더 pull할지 bandit처럼 제어

## Bottom line

짧게 말하면:

- Go-Explore는 `좋은 frontier를 잊지 말고 거기로 돌아가라`
- MCTS는 `현재 tree 안에서 exploration/exploitation을 statistics로 제어하라`
- GLoW는 `global frontier memory와 local learning memory를 분리하라`
- Query Decomposition for RAG는 `query-space 자체를 bandit controller로 볼 수 있다`

네 현재 문제의 핵심에 가장 가까운 건:

- `Go-Explore + GLoW`

이유는 둘 다 결국

- "어디서부터 다시 explore할지"
- "좋았던 정보를 어떻게 memory로 보관해서 다음 selection에 반영할지"

를 explicit하게 설계하기 때문이다.


중간 결론: 
retriever score 단독 proxy는 약하다
matched_count, best_rank, branch reappearance count도 단독 proxy로는 약하다
```
Target:

next-step Δ nDCG@10
max_{t+1:t+3} Δ nDCG@10
Result:

score_mean correlation with next-step gain: 0.0002
score_mean correlation with max-3-step gain: 0.0201
matched_sum correlation with next-step gain: -0.0135
inv_best_rank_mean correlation with max-3-step gain: 0.0326
So:

raw retriever score is basically useless as a reward proxy
even simple branch support counts are nearly useless
I also tested post-action branch reappearance signals at t+1:

how many next-step pre_hit / active_eval docs fall under the reopened branch
both count and rank-mass versions
Those were also weak:

pre10_hits vs max-3-step gain: 0.051
pre100_mass vs max-3-step gain: 0.025
```