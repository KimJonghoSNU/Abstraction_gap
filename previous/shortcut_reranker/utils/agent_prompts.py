def get_router_prompt(query, prev_possible_docs, retrieved_docs):
    return (
        "Your goal is to analyze the Possible Answer Documents (Rewritten queries) generated in the previous round and verify if they are supported by the actual Retrieved Documents.\n\n"

        "Inputs:\n"
        "1. **Possible Answers (Previous Query Parts):**\n"
        f"- Theory Level: {prev_possible_docs.get('Theory', 'N/A')}\n"
        f"- Entity Level: {prev_possible_docs.get('Entity', 'N/A')}\n"
        f"- Example Level: {prev_possible_docs.get('Example', 'N/A')}\n\n"
        f"- Other Level: {prev_possible_docs.get('Other', 'N/A')}\n\n"
        
        "2. Actual Search Results:\n"
        f"{retrieved_docs}\n\n"

        "Reasoning Steps (inside 'Reasoning' field):\n"
        "1. Precisely identify the user's seeking intent and answer type.\n"
        "2. Assess Abstraction: Does the search results contain the academic terms/theories hypothesized?.\n"
        "3. **Verification:** Compare each answers against the search results.\n"
        "   - If Evidence contains the hypothesis -> **EXPLOIT** (Success).\n"
        "   - If Evidence misses, contradicts, or shows irrelevant topics for the hypothesis -> **EXPLORE** (Failure/Gap).\n"
        "   - If Hypothesis was noise -> **PRUNE**.\n\n"

        "**Output Format:**\n"
        "Output a single JSON object:\n"
        "```json\n"
        "{\n"
        "  \"Reasoning\": \"e.g. Theory 'Poisson' was found in Doc 1. However, Entity 'Submarine' was NOT found; docs discuss 'Bees'.\",\n"
        "  \"Actions\": {\n"
        "    \"Theory\": \"EXPLOIT\",\n"
        "    \"Entity\": \"EXPLORE\",\n"
        "    \"Example\": \"EXPLORE\"\n"
        "    \"Other\": \"EXPLOIT\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        
        f"Original Query: {query}"
    )

def get_executor_prompt(query, router_actions, prev_possible_docs, retrieved_docs):
    
    # Action에 따라 원래 프롬프트의 로직을 동적으로 할당
    instructions_block = ""
    
    for level, action in router_actions.items():
        prev_content = prev_possible_docs.get(level, "")
        
        if action == 'EXPLOIT':
            instructions_block += (
                f"### {level} Level: ACTION = EXPLOIT (Refine & Deepen)\n"
                f"- **Context:** The hypothesis answer:\n{prev_content}\n was verified in the retrieved docs.\n"
                "- **Instruction:** Refine this content to be more precise. Use key terms from 'Retrieved Documents'.\n"
                "- **Goal:** Provide strict abstractive-level theory or concrete evidence necessary for the answer.\n\n"
            )
        elif action == 'EXPLORE':
            instructions_block += (
                f"### {level} Level: ACTION = EXPLORE (Pivot & New Hypothesis)\n"
                f"- **Context:** The previous hypothesis \n{prev_content}\n FAILED (not found or irrelevant).\n"
                "- **Instruction:** Identify the common 'wrong direction' or 'missing gap' in the retrieved docs regarding this level. Treat the previous hypothesis as a **Negative Constraint**.\n"
                "- **Goal:** Generate a DIVERSE, COMPLETELY NEW hypothesis. Pivot to a different interpretation, scientific framework, or entity name that avoids the failed path.\n\n"
            )
        elif action == 'PRUNE':
             instructions_block += f"### {level} Level: ACTION = PRUNE (Remove this level)\n\n"

    return (
        "Your goal is to generate the next set of 'Possible Answer Documents' (Search Queries) by refining or pivoting based on the instructions below.\n\n"
        
        "**Navigation Instructions (Per Level):**\n"
        f"{instructions_block}\n"
        
        "**Reference - Retrieved Docs (Evidence from previous round):**\n"
        f"{retrieved_docs}\n\n"

        "**Thinking Process (Execute inside 'Plan'):**\n"
        "1. Identify User Intent & Answer Type.\n"
        "2. For EXPLOIT parts: Refine the possible answer doc using key terms from 'Retrieved Docs'\n"
        "3. For EXPLORE parts: What is the alternative hypothesis?\n"
        "4. Ensure every generated document contributes directly to the answer.\n\n"

        "**Output Format:**\n"
        "Output a single JSON object. The values will be concatenated to form the next query.\n"
        "```json\n"
        "{\n"
        "  \"Plan\": \"Detailed plan applying Step 6 logic for Explore items and Refine logic for Exploit items...\",\n"
        "  \"Possible_Answer_Docs\": {\n"
        "    \"Theory\": \"...text...\",\n"
        "    \"Entity\": \"...text...\",\n"
        "    \"Example\": \"...text...\"\n"
        "    \"Other\": \"...text...\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        
        f"Original Query: {query}"
    )