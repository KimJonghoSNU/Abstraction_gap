# 2026-03-16 Global Search: round5 legacy vs round6 expandable_pool

## 목적

이 문서를 다시 정리한 이유는 하나다.

- `round5 legacy`와 `round6 method2 expandable_pool`을 비교하면서,
- 한 번 exploit을 마쳐 tree leaf에 도달했을 때
- 다음 explore target을 어떻게 고를지를 명확히 설명하기 위함이다.

이 관점에서 보면, 중요한 질문은 metric 자체가 아니다.

- leaf hit 이후 다음 beam이 어떻게 정해지는가
- 그 전환이 local continuation인가, global jump인가
- 그 global jump를 누가 결정하는가

즉 핵심은 `score report`가 아니라 `process difference`다.

---

## round5 legacy process

round5 legacy에서는 leaf에 도달해도 별도의 explore mode로 전환되지 않는다.

동작 순서는 아래와 같다.

1. 현재 beam(`selected_before`)으로 iteration을 시작한다.
2. `sample.update(...)`가 먼저 실행된다.
3. 이 내부에서 tree 전체의 `all_expandable_paths`를 모은 뒤, `path_relevance` 기준으로 beam을 다시 앉힌다.
4. 그 다음에야 round5 selector가 `selected_before`의 direct child branch 후보를 만들고 local override를 시도한다.
5. 만약 current beam 아래에 더 이상 branch child가 없으면, selector는 override를 못 하고 `sample.update(...)`가 이미 만든 beam이 그대로 남는다.

즉 round5의 핵심은 아래 한 줄이다.

- leaf hit 이후 beam state는 global leftover frontier로 점프할 수 있다.
- 하지만 그 점프는 retrieval-driven decision이 아니라 `path_relevance` 기반의 tree update 결과다.

그리고 rewrite evidence는 그대로 이전에 도달한 leaf memory 쪽에 남아 있다.

즉 round5는:

- beam state만 global로 jump하고
- rewrite evidence는 여전히 기존 reached-leaf memory를 본다.

이걸 한 줄로 줄이면:

- `round5 = implicit path-relevance global reseat`

---

## round6 `method2_expandable_pool` process

round6 `expandable_pool`은 leaf hit 다음 iteration을 명시적인 explore step으로 다룬다.

이 mode에서는:

1. leaf-trigger가 발생하면 다음 iteration이 explore step이 된다.
2. explore step의 rewrite는 normal prompt를 그대로 쓴다.
3. 하지만 gate-desc retrieval pool을 현재 beam 아래 leaf들이 아니라, `all_expandable_paths`의 descendant leaf union으로 바꾼다.
4. branch selection 후보도 현재 beam의 direct child가 아니라 `all_expandable_paths` 전체로 넓힌다.
5. 그 후보들 중 무엇을 beam에 넣을지는 `meanscore_global` 같은 retrieval-based selector score가 결정한다.

즉 round6 `expandable_pool`은:

- beam state만 바꾸는 것이 아니라
- rewrite evidence pool도 global unresolved frontier로 전환하고
- 그 위에서 retrieval score로 다음 explore target을 고른다.

이걸 한 줄로 줄이면:

- `round6 expandable_pool = explicit retrieval-driven global explore`

---

## 핵심 차이

위 둘의 차이를 꼭 남겨야 할 문장만 쓰면 아래다.

- `round5 legacy`
    - beam state만 global leftover `expandable_paths`로 점프한다.
    - 점프 기준은 retrieval score가 아니라 `path_relevance`다.
    - rewrite evidence는 이전 reached-leaf memory에 남는다.

- `round6 method2 expandable_pool`
    - beam state뿐 아니라 rewrite evidence pool도 global unresolved frontier로 바꾼다.
    - 다음 beam은 retrieval-based selector score로 고른다.
    - 즉 global jump를 explicit explore step으로 만든다.

비판적으로 보면, 이 비교에서 정말 중요한 건 `global로 가느냐`가 아니다.

- 둘 다 결국 leaf hit 이후 global leftover frontier를 보게 된다.
- 진짜 차이는 그 transition이
    - tree의 internal relevance(`path_relevance`)로 결정되느냐
    - 아니면 retrieval signal로 결정되느냐
  이다.

그래서 이후 method 설계에서 질문은 이렇게 바뀌어야 한다.

- leaf hit 이후 global move를 할지 말지
가 아니라,
- **leaf hit 이후 next explore target을 무엇으로 score할지**
가 핵심이다.

---

## first occurrence는 언제 일어나고, 왜 늦어질 수 있나

처음에는 직관적으로 이렇게 생각하기 쉽다.

- tree max depth가 거의 비슷하면
- beam search는 거의 동시에 leaf에 닿을 것이고
- 그러면 first occurrence는 항상 비슷한 iteration에서 일어나야 한다

하지만 실제로는 더 늦게 나타날 수 있다.

이유는 round5 beam dynamics가 항상 같은 depth로 동기화되지 않기 때문이다.

- `sample.update(...)`가 먼저 실행되고
- 그 뒤 selector override가 들어가는 구조라서
- intermediate iteration에서 beam depth가 섞일 수 있다

즉 어떤 slot은 더 빨리 leaf-adjacent가 되고,
다른 slot은 더 늦게 branch child를 유지한다.

그래서 first occurrence는 단순히 tree max depth로 결정되지 않는다.

정확히는:

- `현재 beam 전체가 언제 동시에 local child branch를 잃느냐`

로 결정된다.

이 점은 중요하다. 왜냐면 이후 explore 설계에서 `leaf hit`을 sample-level event로 볼지, beam-slot-level event로 볼지를 정해야 하기 때문이다.

지금 분석이 말해주는 건 다음이다.

- 실제 transition은 beam 전체의 상태에 의해 결정된다.
- 즉 explore trigger를 생각할 때도 beam-level event로 해석하는 편이 더 자연스럽다.

---

## method implication

이 분석으로부터 남겨야 할 결론은 아래다.

1. round5 legacy도 leaf hit 이후에는 사실상 global unresolved frontier로 넘어간다.
2. 따라서 round6 `expandable_pool`의 novelty는 "global move 자체"가 아니다.
3. novelty가 있으려면, **그 global move를 retrieval signal로 더 잘 제어한다**는 쪽에 있어야 한다.
4. 그래서 이후 method 설계의 핵심 질문은:
    - leaf hit 이후 next explore target을 어떤 signal로 score할 것인가
    - rewrite evidence pool과 branch candidate scope를 어떻게 align할 것인가

결국 이 문서의 핵심은 한 줄이다.

- `round5`는 global move를 이미 하고 있었고, 차이는 그 move를 누가 어떻게 제어하느냐에 있다.

---

## `expandable_pool_freeze_terminal` ablation은 무엇을 끄는가

추가로 본 실험은 아래 ablation이다.

- `round6 ... method2_expandable_pool_freeze_terminal ...`

이 실험의 목적은 단순하다.

- `expandable_pool`이 leaf hit 이후 성능이 흔들리는 이유가
    - global unresolved frontier를 retrieval-driven으로 다시 여는 것 자체 때문인지
    - 아니면 leaf 이후 leftover `expandable_paths`로 계속 reseat되는 과정 때문인지
  분리해서 보기 위함이다.

### 코드 기준 동작

이 ablation은 아래 조건에서만 발동한다.

- 현재 `selected_before` 기준 direct non-leaf child candidate가 하나도 없을 때

즉 정확히는:

- beam의 일부 slot이 leaf-adjacent인 경우가 아니라
- **현재 beam 전체의 local child-branch union이 비는 순간**

에만 발동한다.

그 순간 코드에서는:

1. 현재 beam의 slate는 한 번 처리한다.
2. 하지만 `sample.update(...)`가 하던 기본 `all_expandable_paths` reseat는 하지 않는다.
3. 대신 `update_keep_beam(...)`만 호출해서 beam state를 그대로 유지한다.
4. 이후 iteration들에서는 traversal을 더 확장하지 않고, 마지막 beam-local pool을 고정해서 rewrite/eval만 계속 돈다.

즉 이 ablation이 끄는 것은:

- leaf 이후 남아 있는 다른 `expandable_paths` 쪽으로 다시 beam을 옮기는 과정

이다.

반대로, 이 ablation이 끄지 않는 것은:

- 현재 leaf slate를 한 번 반영하는 것
- 그 시점까지 누적된 retrieval pool 위에서 rewrite/eval을 계속 하는 것

이다.

### `expandable_pool`과의 차이

- `expandable_pool`
    - leaf-trigger 다음 explore step에서 `all_expandable_paths` descendant pool을 보고
    - retrieval-driven selector로 next beam을 다시 고른다

- `expandable_pool_freeze_terminal`
    - local child branch가 완전히 끝난 순간
    - 그 다음 leftover `expandable_paths` reseat를 아예 막는다
    - 즉 "다른 unexplored branch로 더 가보는 단계"를 제거한 ablation이다

### 결과 해석

`results/BRIGHT/ndcg_end_summaryround6.csv` 기준으로 보면,

- `expandable_pool`: overall `33.35 / 37.17` (`end / max`)
- `expandable_pool_freeze_terminal`: overall `33.52 / 37.29`

즉 freeze ablation은 전체 평균에서 거의 차이가 없다.

이건 중요한 관찰이다.

- leaf 이후 leftover `expandable_paths`로 계속 옮겨 가는 과정이
  전체 성능 하락의 유일한 원인은 아니라는 뜻이다.
- 즉 문제의 더 큰 축은
    - leaf hit 이후 어떤 global frontier evidence를 보여주고
    - 그걸로 next explore target을 retrieval-driven으로 다시 고르는 방식
  자체일 가능성이 더 크다.

비판적으로 보면, 이 ablation은 다음 결론을 준다.

- "`expandable_pool`이 안 좋은 이유는 leftover branch로 계속 이동해서"라고 단정하면 안 된다.
- freeze를 해도 평균 성능 차이가 거의 없기 때문이다.
- 따라서 이후에 더 봐야 할 것은
    - freeze 여부보다
    - leaf-trigger 다음 iteration의 rewrite evidence / selector score 설계다.

---

## 새 ablation 구현: ended beam만 retrieval score로 reseat

위 비교를 바탕으로 새 ablation을 추가했다.

목적은 명확하다.

- `round5 legacy`처럼 leaf 이후 다른 branch로 넘어가기는 하되,
- 그 선택을 `path_relevance`가 아니라 retrieval score로 다시 하게 만들고,
- 동시에 `expandable_pool`처럼 beam 전체를 갈아끼우지는 않게 하는 것

즉 이 구현은 다음 두 방식의 중간에 있다.

- `round5 legacy`
    - local child가 끊긴 뒤 global leftover branch 쪽으로 implicit jump
    - 점수는 `path_relevance`

- `round6 expandable_pool`
    - leaf-trigger 이후 whole beam을 global unresolved frontier 쪽으로 explicit explore

- 새 `ended_beam_reseat`
    - rewrite/eval은 legacy 그대로 유지
    - non-ended beam은 기존 local direct-child continuation 유지
    - ended beam slot 수만큼만 leftover expandable path를 retrieval score로 다시 골라 채움

핵심 구현은 아래다.

1. 현재 beam에서 direct non-leaf child가 없는 endpoint를 `ended beam`으로 본다.
2. 현재 iteration의 tree update는 그대로 수행한다.
3. non-ended beam 쪽 local child branch는 기존 `meanscore_global` selector로 유지한다.
4. ended beam 수가 `k`개면, local selector가 이미 고른 endpoint를 제외한 leftover expandable endpoints를 모은다.
5. 그 leftover candidate들의 descendant leaf를 retrieve하고, 동일한 selector score(`meanscore_global` 등)로 candidate를 다시 정렬한다.
6. top-`k` candidate만 ended slot replacement로 넣는다.

즉 이 모드는:

- beam 전체 explore가 아니라
- **ended slot만 retrieval-driven으로 재배치하는 local-to-global bridge ablation**

이다.

실행 스크립트는 따로 분리했다.

- `src/bash/round5/run_round6_expandable.sh`

이렇게 분리한 이유는 기존 `run_round6.sh`의 legacy / global escape / method2 실험과 섞이지 않게 하기 위해서다.
