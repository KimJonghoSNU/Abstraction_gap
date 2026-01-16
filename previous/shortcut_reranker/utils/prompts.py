def query_generate_user_prompt(query, docs, mode, round_idx, revised_query=None):
    if mode == 'inf-x-retriever':
        if round_idx == 0:
            user_prompt = (
                "You are given a query and the provided passages (most of which may be incorrect or irrelevant)."
                "For the input query, formulating a concise search query for dense retrieval by distilling the core intent from a complex user prompt and ignoring LLM instructions. "
                "The response should be less than 200 words,\n\n"

                f"**Input Query:**\n{query}\n"
                f"{query}\n\n"

                "Possible helpful passages: \n"
                f"{docs}"
                
                f"**Your Output:**\n"
            )
        else:
            user_prompt = (
                "You are given a query and the provided passages (most of which may be incorrect or irrelevant)."
                "For the input query, refine a concise search query for dense retrieval by distilling the core intent from a complex user prompt and ignoring LLM instructions. "
                "The response should be less than 200 words.\n\n"

                f"**Input Query:**\n{query}\n"
                f"{query}\n\n"

                "Possible helpful passages: \n"
                f"{docs}\n\n"
                
                "Prior generated query: \n"
                f"{revised_query}\n\n"
                
                f"**Your Output:**\n"
            )

    if mode == 'thinkqe':
        if round_idx == 0:
            user_prompt = (
                "Given a query and the provided passages (most of which may be incorrect or irrelevant), "
                "identify helpful information from the passages and use it to write a correct answering passage. "
                "Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages: \n"
                f"{docs}"
            )
        else:
            user_prompt = (
                "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer, "
                "identify helpful information from the passages and refine the prior answer. "
                "Ensure the output directly addresses the original query. Use your own knowledge, not just the example passages!"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages: \n"
                f"{docs}\n\n"

                "Prior generated answer/revised query: \n"
                f"{revised_query}"
            )

    elif mode == 'thinkqe_multi':
        if round_idx == 0:
            user_prompt = (
                "Given a query and the provided passages (most of which may be incorrect or irrelevant), "
                "identify helpful information from the passages and use it to write a correct answering passage. "
                "Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages:\n"
                f"{docs}"
            )
        else:
            user_prompt = (
                "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answers/revised queries, "
                "identify helpful information from the passages and refine the prior answers. "
                "Ensure the output directly addresses the original query. Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages:\n"
                f"{docs}\n\n"

                "Prior generated answers/revised queries (from multiple evidence subsets):\n"
                f"{revised_query}\n\n"
                
                "Please refine the query by leveraging strengths of the above prior queries while correcting their weaknesses."
            )
    elif mode == 'thinkqe_multi_order':
        if round_idx == 0:
            user_prompt = (
                "Given a query and the provided passages (most of which may be incorrect or irrelevant), "
                "identify helpful information from the passages and use it to write a correct answering passage. "
                "Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages:\n"
                f"{docs}"
            )
        else:
            user_prompt = (
                "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answers/revised queries, "
                "identify helpful information from the passages and refine the prior answers. "
                "Ensure the output directly addresses the original query. Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages:\n"
                f"{docs}\n\n"

                "Prior generated answers/revised queries below are ordered from more promising to less reliable:\n"
                f"{revised_query}\n\n"
                
                "Please refine the query by leveraging strengths of the above prior queries while correcting their weaknesses."
            )
    elif mode == 'thinkqe_contrast':
        if round_idx == 0:
            user_prompt = (
                "Given a query and the provided passages (most of which may be incorrect or irrelevant), "
                "identify helpful information from the passages and use it to write a correct answering passage. "
                "Use your own knowledge, not just the example passages!\n\n"

                "Query:\n"
                f"{query}\n\n"

                "Possible helpful passages:\n"
                f"{docs}"
            )
        else:
            assert isinstance(docs, tuple)
            assert isinstance(revised_query, list)
            assert len(revised_query) == 2

            q0_docs, q1_docs, q2_docs = docs
            rq1 = revised_query[0]
            rq2 = revised_query[1]

            user_prompt = (
                "Here is the relevance definition in a retrieval task: "
                # "Given a user query (robotics post) and retrieved passages, the passage is relevant to the query "
                # "if the critical concepts or theories discussed in the passage can provide references for domain experts to draft an answer to the query.\n\n"
                "Given a query (biology post) and retrieved passages, the passage is relevant to the query if the critical concepts or theories discussed in the passage can provide references for domain experts to draft an answer to the query.\n\n"

                "Based on the relevance definition, clearly justify your final relevance annotation result and annotate an integer score from a scale of 1 to 5.\n"
                "Please use the following guide:\n"
                "- 5 (Highly Relevant): The passage is directly and fully responsive to the user query, providing comprehensive, accurate, and specific information that completely addresses all aspects of the user query.\n"
                "- 4 (Relevant):  The passage is largely relevant and provides most of the information needed, but may have minor omissions, slight inaccuracies,  or not be perfectly aligned with the user query's intent.\n"
                "- 3 (Moderately Relevant): The passage has some relevance and offers partial information, but it may be incom- plete, vague, or include some irrelevant content. It provides a basic connection but lacks depth or precision.\n"
                "- 2 (Slightly Relevant): The passage has minimal relevance, with only a small portion of content tangentially related to the user query.  The majority of the passage is off-topic or provides little value.\n"
                "- 1 (Irrelevant): The passage is completely unrelated to the user query and provides no useful information. There is no discernible connection or value for answering the user query.\n\n"

                "Now, given a user query, the previous round's answers/revised queries, and the retrieved passages, "
                "your mission is to annotate relevance for each passage within <think> tags, and refine the user query based on relevance annotation within <answer> tags. "
                "Ensure the output directly addresses the user query. Use your own knowledge, not just the example passages!\n\n"

                "User query:\n"
                f"{query}\n\n"

                "Prior generated answer/revised query 1:\n"
                f"{rq1}\n\n"

                "Possible helpful passages retrieved using the revised query 1:\n"
                f"{q1_docs}\n\n"

                "Prior generated answer/revised query 2:\n"
                f"{rq2}\n\n"

                "Possible helpful passages retrieved using the revised query 2:\n"
                f"{q2_docs}\n\n"
            )
    elif mode == 'direct':
        user_prompt = (
            "Given a query, "
            "identify helpful information and use it to write a correct answering passage. "
            "Use your own knowledge!\n\n"

            "Query:\n"
            f"{query}"
        )
    elif mode == 'stepback':
        if round_idx == 0:
            user_prompt = (
                "Your goal is to analyze the candidate passages retrieved for the query, "
                "identify which of them are strictly relevant to the user's true intent, "
                "and then generate the set of answer documents accordingly. "
                "Each generated document must be strictly relevant: it should either contain the "
                "exact answer or provide abstractive-level theory, evidence, or background that is "
                "necessary for that answer.\n\n"

                "Exactly follow these steps to plan your selection and refinement, inside <think>...</think>. "
                "1. Precisely identify what information the user is seeking "
                "(e.g., why / how / what / what is this called / which part) and what type of "
                "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
                "entity, worked example).\n"
                "2. If the query is explicitly or implicitly asking 'what is this called?', "
                "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
                "3. Abstraction: infer which academic terms, scientific theories, mathematical "
                "models, canonical methods, or standard resources lie behind this question and "
                "would be cited in a strictly correct answer.\n"
                "4. Consider alternative ways the answer might be supported: a direct definition, "
                "background theory, canonical examples, or reference websites.\n"
                "5. For causal or explanatory questions (why/how), identify multiple theoretical "
                "frameworks that offer different explanations, including mainstream, alternative, "
                "and controversial perspectives if mentioned in the query.\n"
                "6. For each Candidate Passage, judge which part of the user's query does the passage addresses. Then, "
                "identify the common wrong direction (topic, answer type, or framing) suggested by "
                "the Candidate Passages, and treat this as a negative constraint. Plan how to "
                "rewrite the query to avoid that direction and instead explore a different, more "
                "plausible interpretation aligned with the user's wording.\n"
                "Ensure that every generated document contributes directly and strictly to such an "
                "answer (no loosely related or merely interesting content).\n"
                "After </think>, Output 3–5 distinct answer-document entries:\n\n"

                "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
                "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
                "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
                "that support the answer).\n"
                "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
                "alternative but correct perspectives or examples.\n\n"

                f"Query: {query}\n\n"
                f"Candidate Passages:\n"
                f"{docs}\n\n"
            )
        else:
            user_prompt = (
                "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer, "
                "Your goal is to analyze the candidate passages retrieved for the query, "
                "identify which of them are strictly relevant to the user's true intent, "
                "and then refine the set of previous answer documents accordingly. "
                "Each refined document must be strictly relevant: it should either contain the "
                "exact answer or provide abstractive-level theory, evidence, or background that is "
                "necessary for that answer.\n\n"

                "Exactly follow these steps to plan your selection and refinement, inside <think>...</think>. "
                "1. Precisely identify what information the user is seeking "
                "(e.g., why / how / what / what is this called / which part) and what type of "
                "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
                "entity, worked example).\n"
                "2. If the query is explicitly or implicitly asking 'what is this called?', "
                "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
                "3. Abstraction: infer which academic terms, scientific theories, mathematical "
                "models, canonical methods, or standard resources lie behind this question and "
                "would be cited in a strictly correct answer.\n"
                "4. Consider alternative ways the answer might be supported: a direct definition, "
                "background theory, canonical examples, or reference websites.\n"
                "5. For causal or explanatory questions (why/how), identify multiple theoretical "
                "frameworks that offer different explanations, including mainstream, alternative, "
                "and controversial perspectives if mentioned in the query.\n"
                "6. For each Candidate Passage, judge which part of the user's query does the passage addresses. Then, "
                "identify the common wrong direction (topic, answer type, or framing) suggested by "
                "the Candidate Passages, and treat this as a negative constraint. Plan how to "
                "refine the answer documents to avoid that direction and instead explore a different, more "
                "plausible interpretation aligned with the user's wording.\n"
                "Ensure that every generated document contributes directly and strictly to such an "
                "answer (no loosely related or merely interesting content).\n"
                "After </think>, Output 3–5 distinct answer-document entries:\n\n"

                "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
                "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
                "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
                "that support the answer).\n"
                "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
                "alternative but correct perspectives or examples.\n\n"

                f"Query: {query}\n\n"
                f"Candidate Passages:\n"
                f"{docs}\n\n"
                "Prior generated answer/revised query: \n"
                f"{revised_query}" 
            )

    elif mode == 'stepback_json':
        if round_idx == 0:
            user_prompt = (
                "Your goal is to analyze the candidate passages retrieved for the query, "
                "identify which of them are strictly relevant to the user's true intent, "
                "and then generate the set of answer documents accordingly. "
                "Each generated document must be strictly relevant: it should either contain the "
                "exact answer or provide abstractive-level theory, evidence, or background that is "
                "necessary for that answer.\n\n"

                "Output Format:\n"
                "You must output a single JSON object with the following keys:\n"
                "- \"Plan\": A detailed string where you analyze availability of information and plan your selection. Follow the planning steps below.\n"
                "- \"Possible_Answer_Docs\": A JSON object with keys as document types (\"Theory\", \"Entity\", \"Example\", \"Other\") and values as strings representing distinct answer documents.\n\n"

                "Planning Steps (for the \"Plan\" field):\n"
                "1. Precisely identify what information the user is seeking "
                "(e.g., why / how / what / what is this called / which part) and what type of "
                "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
                "entity, worked example).\n"
                "2. If the query is explicitly or implicitly asking 'what is this called?', "
                "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
                "3. Abstraction: infer which academic terms, scientific theories, mathematical "
                "models, canonical methods, or standard resources lie behind this question and "
                "would be cited in a strictly correct answer.\n"
                "4. Consider alternative ways the answer might be supported: a direct definition, "
                "background theory, canonical examples, or reference websites.\n"
                "5. For causal or explanatory questions (why/how), identify multiple theoretical "
                "frameworks that offer different explanations, including mainstream, alternative, "
                "and controversial perspectives if mentioned in the query.\n"
                "6. For each Candidate Passage, judge which part of the user's query does the passage addresses. Then, "
                "identify the common wrong direction (topic, answer type, or framing) suggested by "
                "the Candidate Passages, and treat this as a negative constraint. Plan how to "
                "rewrite the query to avoid that direction and instead explore a different, more "
                "plausible interpretation aligned with the user's wording.\n"
                "Ensure that every generated document contributes directly and strictly to such an "
                "answer (no loosely related or merely interesting content).\n\n"

                "Final Answer Requirements (for the \"Answer documents\" field):\n"
                "Include 3-5 distinct answer-document entries:\n"
                "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
                "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
                "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
                "that support the answer).\n"
                "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
                "alternative but correct perspectives or examples.\n\n"

                f"Query: {query}\n\n"
                f"Candidate Passages:\n"
                f"{docs}\n\n"
                "Provide the JSON output."
                "Example Output Format:\n"
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
            )
        else:
            user_prompt = (
                "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer, "
                "Your goal is to analyze the candidate passages retrieved for the query, "
                "identify which of them are strictly relevant to the user's true intent, "
                "and then refine the set of previous answer documents accordingly. "
                "Each refined document must be strictly relevant: it should either contain the "
                "exact answer or provide abstractive-level theory, evidence, or background that is "
                "necessary for that answer.\n\n"

                "Output Format:\n"
                "You must output a single JSON object with the following keys:\n"
                "- \"Plan\": A detailed string where you analyze availability of information and plan your selection. Follow the planning steps below.\n"
                "- \"Possible_Answer_Docs\": A JSON object with keys as document types (\"Theory\", \"Entity\", \"Example\", \"Other\") and values as strings representing distinct answer documents.\n\n"
                
                "Planning Steps (for the \"Plan\" field):\n"
                "1. Precisely identify what information the user is seeking "
                "(e.g., why / how / what / what is this called / which part) and what type of "
                "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
                "entity, worked example).\n"
                "2. If the query is explicitly or implicitly asking 'what is this called?', "
                "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
                "3. Abstraction: infer which academic terms, scientific theories, mathematical "
                "models, canonical methods, or standard resources lie behind this question and "
                "would be cited in a strictly correct answer.\n"
                "4. Consider alternative ways the answer might be supported: a direct definition, "
                "background theory, canonical examples, or reference websites.\n"
                "5. For causal or explanatory questions (why/how), identify multiple theoretical "
                "frameworks that offer different explanations, including mainstream, alternative, "
                "and controversial perspectives if mentioned in the query.\n"
                "6. For each Candidate Passage, judge which part of the user's query does the passage addresses. Then, "
                "identify the common wrong direction (topic, answer type, or framing) suggested by "
                "the Candidate Passages, and treat this as a negative constraint. Plan how to "
                "refine the answer documents to avoid that direction and instead explore a different, more "
                "plausible interpretation aligned with the user's wording.\n"
                "Ensure that every generated document contributes directly and strictly to such an "
                "answer (no loosely related or merely interesting content).\n\n"

                "Final Answer Requirements (for the \"Answer documents\" field):\n"
                "Include 3-5 distinct answer-document entries:\n"
                "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
                "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
                "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
                "that support the answer).\n"
                "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
                "alternative but correct perspectives or examples.\n\n"

                f"Query: {query}\n\n"
                f"Candidate Passages:\n"
                f"{docs}\n\n"
                "Prior generated answer/revised query: \n"
                f"{revised_query}\n\n" # TODO: pass this in the argument
                
                "Provide the JSON output."
                "Example Output Format:\n"
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
            )
    
    elif mode == 'smr_json':
        user_prompt = (
            '{\n'
            f'"query": {query},\n'
            '"retrieved search results": [\n'
            f'{docs}\n'
            ']\n'
            '}\n'
        )
    return user_prompt



def query_generate_system_prompt(mode):
    assert mode == 'smr_json'

    system_prompt = (
        'You are a highly intelligent artificial agent responsible for managing a search system. '
        'Your role is to refine the given query by rewriting it, thereby enhancing both recall and precision of the search results in the next step. '
        'Your output must be exactly the Query Refinement operation.\n\n'

        '## Input Format\n'
        'The input provided to you will have the following structure:\n'
        '{\n'
        '"query": "<current version of a query>",\n'
        '"retrieved search results": [\n'
        '   ("<docid>", "document contents"),\n'
        '   ("<docid>", "document contents"),\n'
        '   ...\n'
        ']\n'
        '}\n\n'

        '## Refinement Policy:\n'
        'You must choose to perform Query Refinement if any of the following are met:\n'
        'The query is ambiguous or generic.\n'
        'The retrieved search results are unsatisfactory.\n'
        'The query is short or lacks detail.\n'
        'Key domain terms are missing in the query.\n\n'

        '## Output Format\n'
        'You must refine the query by rewriting it into a clear, specific, and formal version '
        'that is better suited for retrieving relevant information from a list of documents. Output format:\n'
        '{\n'
        '"query": "<refined version of a query>",\n'
        '"user_intent_analysis": {\n'
        '   "seeking_info": "<Precisely identify what information the user is seeking (e.g., why / how / what / which part)>",\n'
        '   "required_answer_type": "<Identify the type of answer that would satisfy the user (e.g., concept name, theory, explanation, concrete entity, worked example)>"\n'
        '},'
        '"retrieval_strategy_refinement": [\n'
        '    "<Identify the common wrong direction suggested by the current retrieved passages (topic, answer type, or framing).>",\n'
        '    "<Specific strategy to refine the query to avoid the wrong direction and strictly target the identified answer components (e.g., Add terms Z and exclude topic W).>",\n'
        '    "..."\n'
        '],\n'
        '"final_refinement_rationale": "<A brief summary of why the refined query was chosen, emphasizing how it integrates the abstraction and retrieval strategy to meet the user\'s true intent.>"\n'
        '}\n'
    )
    return system_prompt