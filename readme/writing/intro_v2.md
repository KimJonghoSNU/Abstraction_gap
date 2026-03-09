# Intro Draft (v2)

Reasoning-intensive retrieval differs from standard semantic retrieval because the user query and the evidence-bearing documents often live at different abstraction levels. A query may describe a concrete phenomenon, while the useful support documents are theoretical mechanisms, canonical concepts, or methodological evidence. This abstraction gap is a major source of retrieval failure.

Our starting point is an objective mismatch: in many rewrite pipelines, the prompt implicitly optimizes answer drafting, not evidence discovery. However, the retrieval target is not a final answer sentence. The target is a set of documents that can justify the answer. When rewriting is answer-shaped, retrieval can remain fluent but fail to expose bridge evidence, especially under iterative top-K feedback loops.

We therefore frame rewriting as **evidence planning**: infer what evidence should exist if the answer is correct, then rewrite toward those evidence anchors. Concretely, we generate structured evidence hypotheses (e.g., theory/mechanism, entity/fact, example/analogy, other support) and use them directly for retrieval. This moves rewriting from paraphrasing the query to planning retrievable support.

Positioning against related prompts needs care. Methods such as RETRO-style reranking and prompts that ask “what information would be helpful” are directionally aligned with evidence awareness. But their primary role is often relevance estimation over existing candidates or loosely guided rewriting. Our focus is different: **query-time candidate discovery** under abstraction gap, with evidence structure explicitly controlling what gets searched next.

Empirically, our current evidence suggests that the main gain comes from evidence-planning rewrites themselves (including structured, multi-aspect hypotheses), while explicit explore/exploit control is not yet the primary driver. In other words, policy control is a promising extension, but not the core claim at this stage.

