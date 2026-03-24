# 2026-03-24 How to improve more

## Question

`results/BRIGHT/ndcg_iter_summaryembed.csv`를 보면

- `round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat`

의 성능이 iter `4-5`에서 크게 흔들린다.

궁금한 점은 두 가지였다.

- 이 하락이 특정 subset 하나의 문제인가?
- 아니면 `ended_reseat` transition 자체의 구조적 문제인가?

## Target run

분석 대상:

- `MaxBS=10`
- `RMP=reason-embed-qwen3-8b-0928`
- `round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat`

주의:

- `biology`에는 `_emr` 변형 run이 하나 더 있어서, 이건 제외하고 봤다.
- summary row와 맞는 canonical run만 기준으로 사용했다.

## Overall trajectory

`results/BRIGHT/ndcg_iter_summaryembed.csv` 기준:

- `36.02, 35.94, 35.95, 33.72, 31.62, 35.97, 36.19, 36.13, 36.04, 35.91`

즉 실제 dip은:

- iter `2 -> 3`
    - `35.95 -> 33.72`
- iter `3 -> 4`
    - `33.72 -> 31.62`

이고, iter `5`에서 다시 회복한다.

## What changes structurally at iter 3-4

동일 run에서 `all_eval_sample_dicts.pkl`을 직접 집계한 결과:

| iter | nDCG mean | ended_beam_count mean | reseat_rate | BranchHit@B mean | SelectedDepth mean |
|---|---:|---:|---:|---:|---:|
| 2 | 35.95 | 0.10 | 0.06 | 82.85 | 2.99 |
| 3 | 33.72 | 3.36 | 0.82 | 43.70 | 3.76 |
| 4 | 31.62 | 8.44 | 0.99 | 17.45 | 3.68 |
| 5 | 35.97 | 7.07 | 1.00 | 12.00 | 3.65 |

핵심 해석:

- iter `3`에서 reseat가 본격적으로 시작된다.
- iter `4`에서는 거의 full-beam reseat 상태가 된다.
- 그 순간 `BranchHit@B`가 `43.70 -> 17.45`로 급락한다.

즉 dip의 직접 원인은:

- 많은 query에서 동시에
- `ended beam`이 거의 beam 전체로 커지고
- reseat가 거의 모든 sample에 적용되면서
- selected branch의 gold coverage가 무너지는 것

이다.

## Is it one problematic subset?

아니다.

iter `3 -> 4` drop이 큰 subset은 여러 개다:

- `sustainable_living`: `-15.14`
- `psychology`: `-12.79`
- `stackoverflow`: `-11.41`
- `economics`: `-8.45`
- `leetcode`: `-5.65`
- `theoremqa_questions`: `-1.68`

즉 overall dip은:

- 특정 dataset 하나가 무너져서 생긴 게 아니라
- 중간 이상 weight를 가진 여러 subset이 동시에 떨어져서 생긴다

반례도 있다:

- `theoremqa_theorems`는 iter `3 -> 4`에서 오히려 `+22.74` 회복한다.

그래서 평균만 보면 원인이 흐려질 수 있지만, main dip을 만든 쪽은 위 5-6개 subset이다.

## Why does iter 5 recover?

이 부분이 중요하다.

iter `4 -> 5`에서 overall nDCG는 회복한다:

- `31.62 -> 35.97`

하지만 `BranchHit@B`는 회복하지 않는다:

- `17.45 -> 12.00`

이건 의미가 분명하다.

- iter `5` 회복은 branch controller가 다시 gold branch를 잘 고르기 때문이 아니다.
- 더 가능성 높은 원인은:
    - cumulative reached-leaf memory
    - rewrite/retrieve loop
    - reranking side signal

즉 retrieval state는 회복되지만, branch state는 계속 약하다.

## Concrete failure pattern

`theoremqa_questions`에서 개별 query를 보면 아주 극단적인 케이스가 반복된다.

예:

- iter `2`: `nDCG=100`, `BranchHit@B=100`, `ended_beam_count=0`
- iter `3`: `nDCG=100`, `BranchHit@B=100`, `ended_beam_count=3`, `reseat=1`
- iter `4`: `nDCG=0`, `BranchHit@B=0`, `ended_beam_count=10`, `reseat=1`
- iter `5`: `nDCG=100`, `BranchHit@B=0`, `ended_beam_count=8`, `reseat=1`

이 패턴은 말 그대로:

- reseat가 조금 들어가서 흔들리는 게 아니라
- `full-beam reseat shock`이 오면서 branch region을 통째로 잃는 것

에 가깝다.

## Current conclusion

지금 ended_reseat의 약점은

- partial reseat 자체보다
- iter `3-4` 근처에 오는 **near-full reseat transition**

에 더 가깝다.

즉 지금 method는:

- local exploit state에서
- 거의 beam 전체를 reseat하는 단계로 넘어갈 때
- branch-state continuity를 보존하지 못한다.

## What to try next

현재 결과만 놓고 보면, 다음 개선 후보는 이 세 가지가 맞다.

### 1. Full-reseat guard

iter 한 번에 거의 모든 beam을 reseat하지 않도록 막는다.

예:

- reseat upper bound를 `k < beam_size`로 두기
- `ended_beam_count`가 커도 한 iter에 commit하는 reseat slot 수를 제한하기

장점:

- 지금 보이는 iter `4` shock를 직접 줄이는 방향이다.

약점:

- exploit 쪽에 너무 오래 머물 수 있다.

### 2. Transition-state carryover

reseat 직후 한 step은 이전 high-confidence retrieval state를 더 강하게 보존한다.

예:

- previous iter top docs를 더 강하게 bank/fuse
- reseat 직후 1 step은 eval/rewrite evidence를 hybrid하게 유지

장점:

- branch state가 바뀌더라도 retrieval state가 급격히 무너지지 않게 할 수 있다.

약점:

- transition이 부드러워지는 대신, explore signal이 약해질 수 있다.

### 3. Evidence-aligned reseat

reseated branch가 다음 rewrite evidence에 반드시 들어가게 강제한다.

예:

- reseated branch descendant evidence를 최소 quota로 포함
- next-step pre-hit/evidence가 old state에만 묶이지 않도록 제한

장점:

- branch state와 retrieval/evidence state mismatch를 직접 겨냥한다.

약점:

- 잘못 고른 reseat branch를 더 강하게 증폭시킬 위험도 있다.

## My current read

지금은 `1 + 3` 조합이 제일 맞다.

- `full reseat`를 완화하고
- reseat된 branch가 다음 evidence state에 최소한 연결되게 만들어야 한다

이 두 가지를 같이 하지 않으면,

- branch state는 바뀌는데
- retrieval/evidence state는 그 변화를 제대로 따라가지 못하는 문제

가 계속 남을 가능성이 높다.
