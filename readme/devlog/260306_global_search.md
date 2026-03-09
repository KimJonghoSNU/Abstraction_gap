네 구현을 보면 rewrite context는 현재까지 도달한 leaf들의 cumulative pool에서 가져오고, 다음 step의 후보 branch도 현재 선택된 branch의 child들로 제한돼 있어. 그래서 한 번 잘못 들어간 branch는 이후 rewrite와 branch selection 둘 다 같은 지역 정보에 의해 계속 갱신될 가능성이 크다. 즉, 오류가 한 단계짜리 오류가 아니라 self-reinforcing 오류가 된다. 

method

 

method

method.md

왜 corpus grounded neighborhood 밖의 feedback이 문제인가

쉽게 말하면, rewrite는 검색 엔진에게 “이런 종류의 근거를 찾아라”라고 방향을 주는 단계야. 그런데 그 방향을 정할 때 지금 실제로 탐색 중인 코퍼스 주변과 무관한 힌트가 섞이면, 모델은 그럴듯하지만 현재 코퍼스에서 회수 불가능한 방향으로 query를 바꿔버릴 수 있어.

예를 들어 원래 질문이 어떤 현상을 설명하는 원리를 찾는 건데, 현재 선택된 branch 아래에는 그 원리를 직접 설명하는 문서가 없고 대신 표면적으로 비슷한 사례 문서들만 있다고 해보자. 이때 neighborhood 밖에서 들어온 힌트가 우연히 강하게 보이면, 모델은 “아, 이 질문은 이 사례를 더 파면 되겠구나”라고 rewrite할 수 있다. 문제는 그 순간부터 retrieval은 그 rewrite를 따라가고, branch selection도 그 retrieval score를 다시 먹기 때문에, 잘못된 추상화가 다음 iteration의 탐색 공간까지 바꿔버린다는 점이야. 그러면 abstraction gap이 줄어드는 게 아니라 오히려 drift가 커진다.

조금 더 직관적으로 말하면 이거야.

현재 search state는 “지금 내가 어느 동네를 뒤지고 있는가”를 뜻해.
feedback은 “그 동네에서 다음에 뭘 더 찾아야 하는가”를 정해줘.
그런데 feedback이 다른 동네 이야기면, rewrite는 내비게이션을 다른 목적지로 바꾸고, 실제 이동은 여전히 현재 동네 안에서만 하게 된다.
그러면 query와 pool 사이에 mismatch가 생긴다.
이 mismatch가 반복되면 retrieval quality가 떨어지고, branch도 잘못 고착된다.

네 슬라이드에서 off-branch 비율과 off-branch 문서가 feedback으로 들어왔을 때의 nDCG drop을 보겠다고 한 게 바로 이 현상을 측정하는 metric이 될 수 있어. 이 framing이 좋다. “잘못된 문서를 retrieved했다”보다 “현재 탐색 neighborhood와 불일치하는 feedback이 rewrite를 오염시켰다”가 더 정확한 문제 정의야. 

260305_LG meeting

이걸 한 문장으로 쓰면 이렇게 돼.

The problem is not merely noisy retrieval, but feedback drift, where rewriting is updated using evidence that is not grounded in the currently explored corpus neighborhood.

잘못된 branch를 골랐을 때 회복이 안 되는 문제를 어떻게 풀까

이건 완전히 풀 수는 없지만, “지역 탐색의 효율”과 “전역 복구 가능성”을 같이 두는 방식으로 많이 완화할 수 있어. 탐색 쪽에서도 greedy commitment의 약점을 줄이기 위해 best-first frontier 유지, diverse beams, exploration bonus 같은 아이디어를 쓴다. hierarchical retrieval 쪽에서도 LATTICE가 local judgment를 바로 믿지 않고 global path relevance로 보정하려고 한 이유가 비슷하다.

내가 보기엔 네 세팅에서 가장 현실적인 해법은 아래 네 가지야.

escape beam을 따로 두기

지금 beam 전부를 local child expansion에 쓰지 말고, beam 하나 정도는 항상 전역 탈출용으로 남겨.
예를 들면 beam size가 10이면 8개는 local child에서 뽑고, 2개는 root 혹은 상위 depth의 global retrieval로 다시 뽑는 거야.

이렇게 하면 main search는 local하게 가되, 한두 개 후보는 늘 “혹시 완전히 다른 가지가 맞나”를 확인해준다. diverse beam search가 greedy beam의 mode collapse를 줄이려는 논리와 비슷해.

이건 구현도 쉬워.
현재 구조에서 selector override 전에 global probe를 한 번 더 돌리고, top scoring unexplored branch를 beam 마지막 몇 칸에 넣으면 된다.

uncertainty triggered rollback을 넣기

항상 backtracking할 필요는 없어. 대신 “지금 branch가 틀렸을 가능성”이 높을 때만 rollback하면 된다.

트리거는 이런 걸 쓰면 돼.

현재 iteration에서 rewrite 전후 nDCG proxy가 안 오름
top local hits의 score margin이 작음
branch 후보들의 score가 서로 비슷함
off-branch 비율이 갑자기 증가함
iteration이 진행되는데 retrieval diversity가 줄고 coverage가 정체됨

이런 신호가 나오면 한 depth 위로 올라가서 sibling branch를 다시 열어보는 거야.
즉, local search를 유지하되 commitment를 irreversible하게 만들지 않는 거지.

open list 방식으로 frontier memory를 남기기

지금은 “현재 선택된 branch의 child”만 후보가 되니까 회복이 어렵다. 

method


여기서 한 단계만 바꾸면 훨씬 나아진다.

현재 beam만 유지하지 말고, 이전 iteration에서 점수가 괜찮았던 sibling branch 몇 개를 open list처럼 따로 저장해.
다음 iteration에서는 current children만 보지 말고, saved siblings와 current children을 같이 경쟁시키는 거야.

이건 best-first search의 핵심 직관과 가깝다. 한 경로에만 올인하지 않고, 여러 frontier 후보를 남겨두는 방식이다.

논문적으로도 좋다.
“tree constrained local search”가 아니라 “tree constrained search with recoverable frontier”라고 쓸 수 있어서, locality의 장점은 살리고 irreversibility criticism을 피할 수 있어.

rewrite는 local only로 두되, branch selection에는 tiny global signal을 섞기

나는 이게 특히 좋아 보여.

rewrite context는 계속 local pool에서만 가져와. 그래야 abstraction이 corpus grounded하게 유지된다.
대신 branch scoring에는 작은 전역 신호를 섞어.

예를 들면

branch score
= local matched leaf score

λ × global retriever prior

μ × exploration bonus for under-visited branches

이렇게 하면 rewrite drift는 막으면서도, branch selection이 완전히 local trap에 갇히지는 않는다. UCT 계열이 exploitation과 exploration을 같이 보려는 이유도 이 균형 때문이야.

네 세팅에서 가장 추천하는 버전

지금 바로 실험 가능한 건 이 조합이야.

rewrite는 지금처럼 cumulative reachable leaf pool만 사용
branch selection은 80퍼센트 local, 20퍼센트 global escape
uncertainty가 높을 때만 rollback
그리고 open list에 previous sibling 3개 정도 유지

이 조합의 장점은 네 메인 메시지를 안 깨는 거야.
tree를 corpus constraint로 쓴다는 핵심은 유지된다.
동시에 리뷰어가 “한 번 branch 틀리면 끝나는 거 아니냐”라고 물었을 때, 아니라고 답할 수 있다.
우리는 local grounding을 유지하되 recoverable search를 설계했다고 말할 수 있으니까.

논문에서 어떻게 쓰면 좋나

문제 제기는 이렇게 쓰면 된다.

A strict local pool improves grounding but introduces path dependence. Once an incorrect branch is selected, subsequent rewriting and retrieval are conditioned on a biased evidence pool, making the error self-reinforcing.

그리고 해결은 이렇게.

To mitigate irreversible commitment, we augment local tree-constrained search with a lightweight recovery mechanism that preserves a small global escape budget and reopens alternative frontiers under high uncertainty.