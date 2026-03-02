# 2026-02-26 Tree Traversal RL 계획

- 목표: abstraction gap을 해결하자
- 1) LLM의 지식 한계 때문에 구체적으로 원하는 내용을 못만든다, 2) corpus에 없는 근거를 만들면 안됨.

진행하려 했던 거: 1) tree depth 2) tree width

traversal하는 모델을 학습하면?
reasonembed data: training (q,d) pair 제공. 

실험:
preference learning
각 branch에서 gold subbranch를 포함하게 고른다면 pos, 아니면 neg
QLORA (4bit 적용)
