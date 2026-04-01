# 2026-03-30 Reseat Query Proposal + Bandit Plan

## Question

`frontiercum_qstate`에서는 `random reseat`가 `score reseat`보다 더 낫다. 따라서 archived branch return을 current-query score-only로 고르면 안 된다.

현재 목표는 두 가지다.

- `random`을 이기는 reseat policy를 만들기
- offline supervised reward 없이 online document-level signal만 사용하기

비교 기준은 반드시 공통 8개 subset만 쓴다.

- `biology`
- `earth_science`
- `economics`
- `pony`
- `psychology`
- `robotics`
- `stackoverflow`
- `sustainable_living`

이번 비교에서는 아래 subset은 제외한다.

- `leetcode`
- `aops`
- `theoremqa_theorems`
- `theoremqa_questions`

## What is already established

아래 사실만 현재 고정 사실로 둔다.

- `frontiercum_qstate_random`가 공통 8개 subset에서 `frontiercum_qstate` score reseat보다 `avg_ndcg_end`, `avg_ndcg_max`가 더 낫다.
- `mean retrieval score` 단독 correlation은 delayed gain과 매우 약하다.
- 따라서 `mean_score`를 final decision rule로 바로 쓰는 것은 misaligned일 가능성이 높다.

여기서 잘못된 comparison axis는 다시 쓰지 않는다.

- old `ended_reseat` vs `frontiercum_qstate`를 improvement 근거로 사용하지 않는다.

## Design decision

핵심 제안은 간단하다.

- reseat이 발생하면 unexplored archived branches를 바로 점수로 고르지 않는다.
- 먼저 `current qstate + archived frontier summary`를 넣어 explore용 query proposal 하나를 만든다.
- 그 proposal query로 candidate branches를 다시 scoring한다.
- branch 선택은 raw score sorting이 아니라 Gaussian UCB로 한다.

명시적으로 유지하는 것은 아래와 같다.

- `path relevance`는 그대로 둔다.
- 새 정책은 tree 내부 traversal이 아니라 archived branch reseat layer에만 적용한다.

## Why this is different from path relevance

`path relevance`와 이번 reseat bandit은 같은 점수 체계가 아니다.

- `path relevance`
    - current tree 안에서 local relevance를 parent relevance에 누적하는 exploit score다.
    - archived branch memory가 없다.
    - uncertainty 항이 없다.
- `score-only reseat`
    - archived branch를 current query 아래 mean score로 즉시 정렬한다.
    - branch history가 없다.
    - uncertainty 항이 없다.
- `proposal + Gaussian UCB reseat`
    - archived candidate branch를 arm으로 본다.
    - proposal query 아래의 mean retrieval score를 online reward signal로 사용한다.
    - branch 선택에 uncertainty bonus를 넣는다.

즉 `path relevance`는 tree 내부 exploit score이고, 이번 제안은 archived branch return policy다.

## Proposed v1 controller

v1은 좁게 정의한다.

- arm = candidate branch path
- trigger = `ended_beam_count > 0`
- proposal = existing `agent_executor_v1_icl2_explore` prompt 재사용
- proposal context = `current qstate + archive summary`
- archive summary = leftover expandable 후보를 current query로 1차 score한 뒤 top `rewrite_context_topk` branch `desc`
- branch scoring query = proposal query
- per-branch observed signal = proposal query 아래 `mean_score`
- selection rule = Gaussian UCB
- stats scope = query-local / sample-local
- stats update = scored된 모든 candidate branch에 대해 update

v1에서 의도적으로 하지 않는 것은 아래다.

- novelty 안 넣음
- redundancy 안 넣음
- support-gain 안 넣음
- offline oracle reward 안 씀

구현 세부 수치 튜닝값은 오늘 문서에 박지 않는다.

- `c`, `sigma0_sq` 같은 상수는 implementation 단계에서 정한다.

## Why this follows Query Decomposition for RAG

이 연결은 `bandit over arms` 수준까지만 주장한다.

- 그 논문은 sub-query arm을 bandit으로 고른다.
- 우리는 archived branch를 arm으로 둔다.
- 그 논문의 online setting처럼 document-level retrieval signal을 reward로 쓴다.
- 차이는 sub-query 대신 branch-conditioned explore direction을 고른다는 점이다.

즉 이번 제안은 Query Decomposition for RAG의 sub-query bandit을 branch reseat 문제로 옮긴 버전이다.

## Acceptance target

first target은 `frontiercum_qstate_random`를 이기는 것이다.

비교 metric은 공통 8개 subset에서 아래 두 개만 본다.

- `avg_ndcg_end`
- `avg_ndcg_max`

실패 조건도 분명히 둔다.

- random보다 못하면, bandit framing만으로는 부족한 것이다.
- 그 경우에만 novelty나 redundancy 같은 추가 signal을 다시 검토한다.

오늘 문서에는 새 결과 숫자를 추가로 만들지 않는다.

- 이미 확인된 사실과 다음 설계만 요약한다.

## Next implementation

다음 구현 작업은 세 개만 둔다.

- `run_round6.py`에 new reseat policy mode 추가
- sample-local branch bandit stats 추가
- reseat proposal query와 UCB score를 iter record에 logging

## 2026-03-30 update: soft depth-batched ended-reseat control

이번 구현은 bandit proposal 이전에, 현재 `ended_reseat`가 너무 깊은 depth로 빨리 jump하는 문제를 먼저 줄이려는 목적이다.

- 대상:
    - `src/run_round6.py`
    - copied launcher `src/bash/round6/run_round6_expandable_emr_depth_batch.sh`
- 유지:
    - retrieval pool
    - rewrite / EMR memory
    - local continuation selector
- 변경:
    - ended-reseat candidate를 depth별로 묶고, sample별 `active reseat depth`를 먼저 본다
    - 현재 active depth에서 same-depth로 이미 reseat한 endpoint는 skip한다
    - active depth가 이번 ended slots를 못 채우면 더 깊은 depth로 spill한다
    - 한 depth에서 reseat가 대략 beam-size만큼 누적되거나, 그 depth 후보가 exhausted되면 다음 depth로 넘어간다

이건 hard barrier가 아니다.

- 각 depth에서 반드시 beam size개를 고른다는 뜻은 아니다
- goal은 deep reseat를 완전히 막는 것이 아니라, early deep jump를 늦추는 것이다
