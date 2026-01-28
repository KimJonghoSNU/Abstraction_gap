# Instructions
- You have to use 4 space tab.
- Do not just follow my order but be critic and suggest better direction if possible.
- 내가 작업 목적과 배경 자료를 공유했을 때, 우리 둘이 서로 같은 페이지에 있어야 해. 나를 최대한 꼼꼼하게 인터뷰해줘. 
- Design choice: 둘 중 하나를 내가 선택해야 한다면 각각의 장단점을 설명한 다음 물어봐줘.
- 중요한 결정 implementation할 때는 코드에 주석으로 어떤 의도를 반영했는지 한 줄 달아줘

# Motivation of this project
- Reasoning-Intensive retrieval with abstraction
- LLM이 주어진 정보가 쿼리에 대한 답으로 충분한지, 충분하지 않은지 internal knowledge로 알 수 있다면, retrieval 자체가 필요 없는 상황 아니냐?
	- LLM이 직접 답을 생성하는 건 못할 수 있음. 세부적인 지식들을 모델이 다 알지는 못함.
	- 하지만 지식을 abstraction -> abstraction 카테고리별로 지식이 나눠져 있다면, 카테고리별로 yes/no 판단하는 능력은 모델이 갖고 있다고 전제할 수 있음.
		- = 카테고리 레벨로 navigate하는건 llm이 모든 세부적인 지식 없이 할 수 있지 않을까