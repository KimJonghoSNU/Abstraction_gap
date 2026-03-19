# Intro Draft (v2)



<!-- Reasoning-intensive retrieval differs from standard semantic retrieval because the user query and the evidence-bearing documents often live at different abstraction levels. A query may describe a concrete phenomenon, while the useful support documents are theoretical mechanisms, canonical concepts, or methodological evidence. This abstraction gap is a major source of retrieval failure.

Our starting point is an objective mismatch: in many rewrite pipelines, the prompt implicitly optimizes answer drafting, not evidence discovery. However, the retrieval target is not a final answer sentence. The target is a set of documents that can justify the answer. When rewriting is answer-shaped, retrieval can remain fluent but fail to expose bridge evidence, especially under iterative top-K feedback loops.

We therefore frame rewriting as **evidence planning**: infer what evidence should exist if the answer is correct, then rewrite toward those evidence anchors. Concretely, we generate structured evidence hypotheses (e.g., theory/mechanism, entity/fact, example/analogy, other support) and use them directly for retrieval. This moves rewriting from paraphrasing the query to planning retrievable support.

Positioning against related prompts needs care. Methods such as RETRO-style reranking and prompts that ask “what information would be helpful” are directionally aligned with evidence awareness. But their primary role is often relevance estimation over existing candidates or loosely guided rewriting. Our focus is different: **query-time candidate discovery** under abstraction gap, with evidence structure explicitly controlling what gets searched next.

Empirically, our current evidence suggests that the main gain comes from evidence-planning rewrites themselves (including structured, multi-aspect hypotheses), while explicit explore/exploit control is not yet the primary driver. In other words, policy control is a promising extension, but not the core claim at this stage.
 -->
Reasoning-intensive retrieval, by design, exhibits a substantial \textbf{abstraction gap} between a user's query and the documents that contain the evidence needed to answer it. In other words, the most useful evidence does not directly match the surface form of the query. Instead, such documents contain \textit{principles, entities, or bridging evidence} that must be identified in order to answer the query. For example, a question about counting bees entering and leaving a hive may require documents discussing the ``Poisson distribution,'' a relevant principle that is nevertheless superficially unrelated to the user query.


A key action is query rewriting over multiple hops, where a model generates alternative formulations of the query 
that better match evidential documents. However, existing rewriting approaches typically treat rewriting without grounding~\cite{su2024bright,qin-etal-2025-tongsearchqr}. 
As a result, rewriting often drifts into concepts that are relevant but not actually supported by the corpus. 

Although each retrieval step provides observations of the corpus, existing iterative query rewriting strategies use them only to update the next rewrite linearly~\cite{lei-etal-2025-thinkqe,long2025diver,yao2022react}, which makes reasoning-intensive retrieval vulnerable to \textit{detachment} and \textit{derailment}~\cite{ecoffet2019go}. Under the abstraction gap, only a small fraction of explored evidence states contain information that can genuinely support the final answer, since the required evidence often lies behind multiple layers of abstraction. Consequently, search may \textit{detach} from a promising intermediate evidence state by drifting away from a useful region of the corpus when newly retrieved observations appear relevant to the query, and then \textit{derail} because linear rewriting offers no explicit way to return to that state and resume exploration from it.

Although each retrieval step provides observations, existing iterative query rewriting strategies use them only to update the next rewrite linearly~\cite{lei-etal-2025-thinkqe,long2025diver,yao2022react}, which makes reasoning-intensive retrieval vulnerable to \textit{detachment} and \textit{derailment}~\cite{ecoffet2019go}. Under the abstraction gap, the evidence needed to answer the query often lies behind multiple layers of abstraction. Consequently, many explored evidence states may appear relevant to the query, while only a small fraction actually contain information that supports the final answer. As a result, search may briefly encounter a promising intermediate evidence state, but later abandon it as retrieval surfaces new observations that appear relevant to the query. This causes \textit{detachment} from useful regions of the corpus, and because linear rewriting lacks a mechanism for revisiting previously discovered states, the search can also \textit{derail} into superficially relevant but unsupported directions.

With this view, we reinterpret reasoning-intensive retrieval through the lens of \textbf{state exploration of structured corpus}, inspired by the Go-Explore paradigm for solving derailment and detachment. In Go-Explore~\cite{ecoffet2019go}, agents \jonghoc{first systematically explore and archive diverse states, and only later optimize trajectories once promising states have been discovered.} Translating this perspective to retrieval, the system should first (1) explore diverse evidence hypotheses that may support the query, and then reformulate queries grounded on these discovered evidence. In summary, exploration should be grounded in observable corpus evidence to prevent semantic drift.

We implement this idea as a tree-constrained iterative rewriting framework that treats retrieval as exploration over a hierarchical corpus structure. Instead of rewriting the query using feedback from the entire corpus, the system progressively explores a restricted region of the corpus tree, retrieving local evidence from reachable branches and aggregating this evidence to guide rewriting. At each iteration, the model generates (1) evidence-grounded query rewrites that describe the principles or entities that would support the answer,  The rewritten query is then used to (2) expand the search to new evidence branches in the tree. By constraining both rewriting and retrieval, to evidence drawn from reachable regions, the system maintains grounded exploration while suppressing off-branch noise



네 방법에서
detachment 해결은 expandable_paths로 유망한 non-leaf state를 기억하고 다시 beam에 올리는 것
derailment 해결은 그 state를 beam에 올린 뒤, 그 subtree 기준으로 retrieval pool을 다시 만들고 그 상태에서부터 rewrite를 이어가게 하는 것

다만 여기서도 한계는 있어.
만약 archived state로 돌아간 뒤에도 previous rewrite를 너무 강하게 계속 누적하거나, cumulative pool이 옛 경로 신호를 과하게 끌고 간다면 derailment는 부분적으로만 해결된 거야. 즉 네 방법은 RL의 Go-Explore처럼 “return”은 꽤 명시적으로 구현했지만, “return 이후 rewrite가 정말 새 state에 anchor되느냐”가 derailment 해결의 마지막 조건이야. 이 부분이 약하면, archived path를 골랐어도 query는 여전히 옛 hypothesis를 따라갈 수 있어