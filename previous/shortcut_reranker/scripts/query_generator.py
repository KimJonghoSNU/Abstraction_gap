import re
import os
import time
import csv
import json
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
# from ftfy import fix_text
# from copy import deepcopy
from collections import defaultdict
from typing import Union, Tuple, Dict, List
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from scripts.llm import Base_LLM
from utils.utils import jsonl_load, json_load, json_dump, pickle_load, pickle_dump, calculate_retrieval_metrics
from utils.prompts import query_generate_user_prompt, query_generate_system_prompt
from utils.agent_prompts import get_executor_prompt, get_router_prompt

task_descriptions = {
    "biology": "Given a query (biology post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "earth_science": "Given a query (earth science post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "economics": "Given a query (economics post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "psychology": "Given a query (psychology post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "robotics": "Given a query (robotics post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "stackoverflow": "Given a query (Stack Overflow post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "sustainable_living": "Given a query (sustainable living post) and a document (passage), the document is relevant to the query if the critical concepts or theories discussed in the document can provide references for domain experts to draft an answer to the query.",
    "leetcode": "Given a query (LeetCode problem) and a document (coding problem solution), the document is relevant to the query if the underlying algorithms or data structures used in the document can provide helpful insights for solving the problem in the query.",
    "pony": "Given a query (Pony coding instruction) and a document (Pony documentation passage), the document is relevant to the query if the Pony syntax described in the document is necessary for beginners with no prior knowledge of Pony to complete the coding instruction in the query.",
    "aops": "Given a query (math problem) and a document (math problem solution), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_questions": "Given a query (math problem) and a document (math problem solution), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_theorems": "Given a query (math problem) and a document (math-related passage), the document is relevant to the query if the theorem described in the document can help solve the problem in the query."
}

class QueryGenerator:
    def __init__(self, model_name_or_path, max_qlen=1024, max_dlen=1024, max_tokens=16000, 
                batch_size=8, prompt_mode='thinkqe', n_round=2, keep_doc_nums=5, **kwargs):
        self.kwargs = kwargs
        self.max_qlen = max_qlen
        self.max_dlen = max_dlen
        self.batch_size = batch_size
        self.prompt_mode = prompt_mode
        self.n_round = n_round
        self.keep_doc_nums = keep_doc_nums
        self.max_tokens = max_tokens
        self.temperature = self.kwargs.get('temperature', 0.0)
        
        self.qa_path = self.kwargs.get('qa_path', None)
        self.corpus_path = self.kwargs.get('corpus_path', None)
        self.dataset_name = os.path.basename(self.qa_path).split(".")[0].lower()

        self.cache_dir = self.kwargs.get('cache_dir', None)
        self.embedding_name_or_path = self.kwargs.get('embedding_name_or_path', None)
        self.search_api = self.load_search_api(self.embedding_name_or_path, self.corpus_path, self.cache_dir)
        self.qa_list = json_load(self.qa_path)
        # self.qa_list = self.qa_list[:4]
        self.qrels = {qa['id']:{gold_ids:1 for gold_ids in qa['gold_ids']} for qa in self.qa_list}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        self.model = Base_LLM(
            model_name_or_path, 
            temperature=self.temperature,
            max_tokens=self.max_tokens, 
            **kwargs, 
        )
        self.agent_mode = self.kwargs.get('agent_mode', '')

    def load_search_api(self, embedding_name_or_path, corpus_path, cache_dir):
        corpus = jsonl_load(corpus_path)
        doc_ids = []
        documents = []
        did2content = {}
        for doc in corpus:
            did = doc['docno']
            content = doc['content']
            if content == "":
                continue
            doc_ids.append(did)
            documents.append(content)
            did2content[did] = content
        search_api = VectorSearchInterface(self.dataset_name, cache_dir, embedding_name_or_path, doc_ids, documents, did2content)
        return search_api

    def parse_output(self, output):
        if isinstance(output, list):
            return "\n".join(self.parse_output(response) for response in output)
        output = output.split("</think>\n")[-1].strip()
        if "<answer>" in output:
            output = output.split("<answer>")[-1]
        if "</answer>" in output:
            output = output.split("</answer>")[0]
        
        if self.agent_mode != '':
            # clean json markup
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0]
            elif "```" in output:
                output = output.split("```")[1].split("```")[0]
            
            try:
                # ensure it is valid json
                import json
                _ = json.loads(output.strip())
                return output.strip()
            except:
                print(f"#> Warning: Failed to parse JSON in state_machine mode. Raw output: {output}...")
                # import pdb; pdb.set_trace()
                return output.strip()

        return output.strip()


    def truncate(self, text, length):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > length:
            text_ids = text_ids[:length]
        return self.tokenizer.decode(text_ids, skip_special_tokens=True)
    
    def transform_did2content(self, dids, offset=0):
        docs = [self.search_api.get_content(did) for did in dids]
        docs = [self.truncate(doc, self.max_dlen) for doc in docs]
        # docs = "\n".join([f"[{idx+1+offset}]. {doc}" for idx, doc in enumerate(docs)])
        text = ""
        for did, doc in zip(dids, docs):
            text += f"   ({did}, {doc})\n"
        return text

    def build_prompt(self, query, dids, round_idx, revised_queries=None):
        messages = []
        if round_idx == 0:
            docs = self.transform_did2content(dids)
            user_prompt = query_generate_user_prompt(query, docs, self.prompt_mode, round_idx)
            user_prompt = {"role":"user", "content":user_prompt}
        else:
            assert revised_queries is not None
            docs = self.transform_did2content(dids)
            user_prompt = query_generate_user_prompt(query, docs, self.prompt_mode, round_idx, revised_query=revised_queries[-1])
            user_prompt = {"role":"user", "content":user_prompt}

        messages.append(user_prompt)

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = self.truncate(prompt, self.max_tokens)
        return prompt
    
    def retrieval(self, generated_queries=None, round_idx=-1, merge_all=False):
        retrieval_results = {}
        retrieval_scores = {}

        for qa in tqdm(self.qa_list, desc=f"Retrieval for {round_idx}-th round"):
            qid = qa['id']
            query = qa['query'].strip()
            excluded_ids = qa['excluded_ids']
            if generated_queries is not None:
                if merge_all:
                    query = query + "\n\n" + "\n".join(generated_queries[qid])
                else:
                    query = query + "\n\n" + generated_queries[qid][-1]
                query = query.strip()
            if qid == "0":
                print(f"#> retrieval query: {query}")

            qid2did2score = self.search_api.do_retrieval(qid, query, excluded_ids, num_hits=1000)
            for qid, did2score in qid2did2score.items():
                dids = [did for did, score in did2score.items()]
                retrieval_results[qid] = dids[:max(5, self.keep_doc_nums)]
                retrieval_scores[qid] = did2score
        
        retrieval_metrics = calculate_retrieval_metrics(results=retrieval_scores, qrels=self.qrels)

        if round_idx == -1:
            print(f"#> Initial retrieval: {retrieval_metrics}")
        elif merge_all == False:
            print(f"#> sub {round_idx}-th retrieval: {retrieval_metrics}")
        elif merge_all == True:
            print(f"#> {round_idx}-th retrieval: {retrieval_metrics}")

        return retrieval_metrics, retrieval_results, retrieval_scores


    # [NEW] Helper method for batch inference with JSON validation and retry
    def _generate_with_retry(self, prompts, prompt_infos, parser_func, max_retries=3):
        """
        Executes batch inference with retry logic for JSON parsing failures.
        
        Args:
            prompts: List of prompt strings.
            prompt_infos: List of metadata corresponding to prompts (e.g., qids).
            parser_func: Function that takes (output_text, qid) and returns (parsed_obj, is_valid).
            max_retries: Number of retries for failed items.
            
        Returns:
            Dict {qid: parsed_obj}
        """
        results = {}
        current_prompts = prompts
        current_infos = prompt_infos
        
        for attempt in range(max_retries + 1):
            if not current_prompts:
                break
            
            if attempt > 0:
                print(f"#> Generation attempt {attempt+1}/{max_retries+1}, batch size: {len(current_prompts)}")
            
            # Batch inference
            # Using existing batch logic but need to handle smaller batches in retries
            batch_outputs_text = []
            for batch_idx in range(0, len(current_prompts), self.batch_size):
                 batch_p = current_prompts[batch_idx : batch_idx + self.batch_size]
                 batch_out = self.model.llm_batch_inference(batch_p)
                 batch_outputs_text.extend([out['text'] for out in batch_out])
            
            next_prompts = []
            next_infos = []
            
            for qid, prompt, output_text in zip(current_infos, current_prompts, batch_outputs_text):
                parsed, is_valid = parser_func(output_text, qid)
                if is_valid:
                    results[qid] = parsed
                    self.temperature = 0.0
                else:
                    if attempt < max_retries:
                        # Append retry prompt? Or just same prompt? 
                        # Usually simple retry works for stochastic LLMs, 
                        # but adding "Ensure JSON" instruction might vary.
                        # Here assuming simple retry with temperature sampling (if enabled)
                        next_prompts.append(prompt)
                        next_infos.append(qid)
                        self.temperature = min(self.temperature + 0.1, 0.7)  # increase temperature slightly for retries
                    else:
                        print(f"#> Failed to parse after {max_retries} retries for qid {qid}. Using raw/empty.")
                        results[qid] = parsed # Return whatever we got or empty
                        self.temperature = 0.0  # reset temperature
            
            current_prompts = next_prompts
            current_infos = next_infos
            
        return results


    def generate(self):
        retrieval_metrics = []
        retrieval_results = []
        retrieval_scores = []
        generated_queries = defaultdict(list)

        # # initial retrieval
        metrics, results, scores = self.retrieval()
        retrieval_metrics.append(metrics)
        retrieval_results.append(results)
        retrieval_scores.append(scores)

        sub_metrics = []
        sub_results = []
        sub_scores = []

        ### first round
        for idx in tqdm(range(0, 5, self.keep_doc_nums), desc="Breadth"):
            # build prompt
            prompts = []
            prompt_info = []
            for qa in self.qa_list:
                qid = qa['id']
                query = qa['query']
                dids = retrieval_results[0][qid]
                dids = dids[idx : idx+self.keep_doc_nums]
                if qid == "0":
                    print(f"#> len(dids): {len(dids)}")
                    print(f"#> dids: {dids}")

                prompt = self.build_prompt(query, dids, round_idx=0)
                prompts.append(prompt)
                prompt_info.append(qid)

            # generate queries
            for batch_idx in tqdm(range(0, len(self.qa_list), self.batch_size), desc="Generation"):
                batch_prompts = prompts[batch_idx : batch_idx + self.batch_size]
                batch_prompt_info = prompt_info[batch_idx : batch_idx + self.batch_size]
                batch_outputs = self.model.llm_batch_inference(batch_prompts)

                for iidx, (qid, prompt, output) in enumerate(zip(batch_prompt_info, batch_prompts, batch_outputs)):
                    output_text = output['text']
                    parsed_output_text = self.parse_output(output_text)
                    generated_queries[qid].append(parsed_output_text)
                    if iidx == 0:
                        print(f"#> prompt: {prompt}")
                        print(f"#> output_text: {output_text}")

            # retrieval
            metrics, results, scores = self.retrieval(generated_queries=generated_queries, round_idx=0)
            sub_metrics.append(metrics)
            sub_results.append(results)
            sub_scores.append(scores)
            retrieval_metrics.append(metrics)
            retrieval_results.append(results)
            retrieval_scores.append(scores)



        if self.agent_mode:
            ## second round (Modified for 2-step Router/Executor)
            for round_num in range(1, self.n_round):
                print(f"#> Starting Round {round_num} with 2-step Logic")
                
                # --- STEP 1: ROUTER ---
                router_prompts = []
                router_qids = []
                
                for qa in self.qa_list:
                    qid = qa['id']
                    query = qa['query']
                    
                    # Get previous round's output
                    prev_out = generated_queries[qid][-1]
                    
                    # If we had JSON from previous Executor step, parse it
                    try: 
                            # Assuming previous executor output was JSON string that contains "Possible_Answer_Docs"
                            # We might need to store the parsed object in generated_queries instead of string?
                            # For now, let's try to parse:
                            data = json.loads(prev_out)
                            prev_possible_docs = data.get("Possible_Answer_Docs", {})
                    except:
                        prev_possible_docs = {"Theory": prev_out} # Fallback
                    
                    # Get retrieved docs from previous round
                    dids = sub_results[-1][qid][:self.keep_doc_nums]
                    retrieved_docs_text = self.transform_did2content(dids)
                    
                    prompt_content = get_router_prompt(query, prev_possible_docs, retrieved_docs_text)
                    # Apply chat template
                    msgs = [{"role": "user", "content": prompt_content}]
                    full_prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    full_prompt = self.truncate(full_prompt, self.max_tokens)
                    
                    router_prompts.append(full_prompt)
                    router_qids.append(qid)
                # JSON Parser for Router
                def router_parser(text, qid):
                    text = self.parse_output(text) # cleans </think> etc
                    try:
                        # Extract JSON block
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0]
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0]
                        
                        obj = json.loads(text)
                        if "Actions" in obj:
                            return obj, True
                        return obj, False # Missing key
                    except:
                        return text, False
                print("#> executing Router step...")
                router_results = self._generate_with_retry(router_prompts, router_qids, router_parser)
                
                # --- STEP 2: EXECUTOR ---
                executor_prompts = []
                executor_qids = []
                
                valid_router_results = {} # store valid ones to match later
                for qid in router_qids:
                    if qid not in router_results:
                        # If router failed effectively, what to do? Skip or use default?
                        # Let's assume default EXPLOIT? or just fail gracefully.
                        # For now, we reuse previous query if failed.
                        continue
                    router_out = router_results[qid]
                    if not isinstance(router_out, dict) or "Actions" not in router_out:
                        print(f"#> Skipping Executor for {qid} due to invalid Router output.")
                        continue
                    valid_router_results[qid] = router_out
                    actions = router_out["Actions"]
                    
                    # Re-construct context (same as Router)
                    query = next(qa['query'] for qa in self.qa_list if qa['id'] == qid)
                    prev_out = generated_queries[qid][-1]
                    if round_num == 1:
                        prev_possible_docs = {"Theory": prev_out, "Entity": "N/A", "Example": "N/A", "Other": "N/A"}
                    else:
                        try:
                            data = json.loads(prev_out)
                            prev_possible_docs = data.get("Possible_Answer_Docs", {})
                        except:
                            prev_possible_docs = {"Theory": prev_out}
                    
                    dids = sub_results[-1][qid][:self.keep_doc_nums]
                    retrieved_docs_text = self.transform_did2content(dids)
                    
                    prompt_content = get_executor_prompt(query, actions, prev_possible_docs, retrieved_docs_text)
                    
                    msgs = [{"role": "user", "content": prompt_content}]
                    full_prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    full_prompt = self.truncate(full_prompt, self.max_tokens)
                    
                    executor_prompts.append(full_prompt)
                    executor_qids.append(qid)
                # JSON Parser for Executor
                def executor_parser(text, qid):
                    text = self.parse_output(text)
                    try:
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0]
                        elif "```" in text:
                            text = text.split("```")[1].split("```")[0]
                        
                        obj = json.loads(text)
                        if "Possible_Answer_Docs" in obj:
                            # We want to return the JSON string or object? 
                            # The retrieval() method expects a string to append.
                            # But for next round recursion we might want the object.
                            # Standard approach: store the JSON string in generated_queries.
                            # And we might extract specific parts for retrieval query.
                            return json.dumps(obj), True
                        return text, False
                    except:
                        return text, False
                print("#> executing Executor step...")
                executor_results = self._generate_with_retry(executor_prompts, executor_qids, executor_parser)
                
                # Store results
                for qid in router_qids: # Iterate original QIDs to maintain order/completeness
                    if qid in executor_results:
                        # We have a valid new query (JSON string)
                        # For retrieval, we might want to flatten the "Possible_Answer_Docs"
                        # The prompt says: "The values will be concatenated to form the next query."
                        # So let's process it for the `generated_queries` list which drives retrieval logic?
                        # Actually `retrieval()` appends `generated_queries[qid][-1]`.
                        # If that's a JSON string, we should probably format it nicely or flatten it.
                        # But `get_executor_prompt` says "The values will be concatenated to form the next query"
                        # So let's flatten it here for storage.
                        
                        json_str = executor_results[qid]
                        try:
                            obj = json.loads(json_str)
                            docs_map = obj.get("Possible_Answer_Docs", {})
                            # Flatten values
                            flattened_text = "\n".join([str(v) for v in docs_map.values() if v])
                            
                            # We store the FULL JSON for next round's context,
                            json_str = json.dumps(obj)  # Store full JSON string
                            generated_queries[qid].append(json_str) 
                        except:
                            generated_queries[qid].append(json_str) # Just append text
                    else:
                        # Failed, append previous 
                        generated_queries[qid].append(generated_queries[qid][-1]) 
                # Retrieval
                # [NOTE] You might need to adjust retrieval() to handle valid JSON strings 
                # if `generated_queries` now contains JSON.
                # I will modify retrieval locally here to handle that.
                
                metrics, results, scores = self.retrieval(generated_queries=generated_queries, round_idx=round_num, merge_all=False)
                retrieval_metrics.append(metrics)
                retrieval_results.append(results)
                retrieval_scores.append(scores)
        else:
        ## second round
            # build prompt
            for round in range(1, self.n_round):
                prompts = []
                prompt_info = []
                for qa in self.qa_list:
                    qid = qa['id']
                    query = qa['query']
                    dids = sub_results[-1][qid][:self.keep_doc_nums]
                    if qid == "0":
                        print(f"#> query: {query}")
                        print(f"#> revised query: {generated_queries[qid][-1]}")
                        print(f"#> dids: {dids}")

                    prompt = self.build_prompt(query, dids, round_idx=round, revised_queries=generated_queries[qid])
                    prompts.append(prompt)
                    prompt_info.append(qid)

                # generate queries
                for batch_idx in tqdm(range(0, len(self.qa_list), self.batch_size), desc="Generation"):
                    batch_prompts = prompts[batch_idx : batch_idx + self.batch_size]
                    batch_prompt_info = prompt_info[batch_idx : batch_idx + self.batch_size]
                    batch_outputs = self.model.llm_batch_inference(batch_prompts)#, num_generation=2)

                    for iidx, (qid, prompt, output) in enumerate(zip(batch_prompt_info, batch_prompts, batch_outputs)):
                        output_text = output['text']
                        parsed_output_text = self.parse_output(output_text)
                        generated_queries[qid].append(parsed_output_text)
                        if iidx == 0:
                            print(f"#> prompt: {prompt}")
                            print(f"#> output_text: {output_text}")

                # retrieval
                metrics, results, scores = self.retrieval(generated_queries=generated_queries, round_idx=round, merge_all=True)
                retrieval_metrics.append(metrics)
                retrieval_results.append(results)
                retrieval_scores.append(scores)
        return retrieval_metrics, retrieval_results, retrieval_scores, generated_queries, sub_metrics, sub_results, sub_scores



class EmbeddingModel:
    def __init__(self, model_path, device="auto"):
        # self.model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map="auto")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.task = "Given a web search query, retrieve relevant passages that answer the query"

    def encode_query(self, query:str, max_length=8192, bs=2):
        return self.encode(self.get_detailed_instruct(self.task, query), max_length)[0]
    
    def encode_doc(self, doc, max_length=16384):
        return self.encode(doc, max_length)[0]

    def encode_docs(self, docs, max_length=16384, bs=1):
        embeddings = []
        # remove 0 length docs
        docs = [doc for doc in docs if len(doc.strip()) > 0]
        for i in trange(0, len(docs), bs):
            embeddings.append(self.encode(docs[i:i+bs], max_length))
        return np.vstack(embeddings)

    def encode(self, text:list, max_length:int=16384):
        # Tokenize the input texts
        text = [text] if isinstance(text, str) else text
        try:
            batch_dict = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict.to(self.model.device)
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        except:
            import pdb; pdb.set_trace()
        return embeddings.cpu().detach().numpy()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'


class VectorSearchInterface:
    def __init__(self, dataset_name, cache_dir, embedding_name_or_path, doc_ids:list, documents:list, did2content:dict):
        self.model = EmbeddingModel(embedding_name_or_path)
        self.cache_dir = cache_dir
        self.doc_ids = doc_ids
        self.documents = documents
        self.did2content = did2content
        self.dataset_name = dataset_name
        self.docs_emb = self.get_docs_emb(documents)

    def get_content(self, did):
        return self.did2content[did]
    
    def get_docs_emb(self, documents):
        # cache docs emb
        cache_path = os.path.join(self.cache_dir, 'doc_emb', 'diver-retriever', self.dataset_name, f"long_False") 
        os.makedirs(cache_path, exist_ok=True)
        doc_cache_file = os.path.join(cache_path, '0.npy')

        print('Encoding documents to cache:', cache_path)
        if os.path.exists(doc_cache_file):
            docs_emb = np.load(doc_cache_file, allow_pickle=True)
            print("Loaded cached doc emb from", doc_cache_file)
        else:
            with torch.inference_mode():
                docs_emb = self.model.encode_docs(documents)
            torch.cuda.empty_cache()
            np.save(doc_cache_file, docs_emb)
            print("Shape of doc emb", docs_emb.shape)
        return docs_emb

    torch.no_grad()
    def do_retrieval(self, qid, query_text, excluded_ids, num_hits=1000):
        '''
        return: dict of dict {qid: {doc_id: score}, }
        '''
        query_texts = [query_text] if isinstance(query_text, str) else query_text
        qid = [qid] if isinstance(qid, str) else qid
        
        query_emb = []
        with torch.inference_mode():
            for q in query_texts:
                query_emb.append(self.model.encode_query(q))
        query_emb = np.array(query_emb)
        torch.cuda.empty_cache()

        scores = cosine_similarity(query_emb, self.docs_emb).tolist()
        qid_doc_scores = self.get_scores(query_ids=qid, doc_ids=self.doc_ids, scores=scores, excluded_ids=excluded_ids, num_hits=num_hits)
        
        return qid_doc_scores

    def get_scores(self, query_ids, doc_ids,scores, excluded_ids, return_full_scores=False, num_hits=1000):
        assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
        assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
        emb_scores = {}
        for query_id,doc_scores in zip(query_ids,scores):
            cur_scores = {}
            # assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
            for did,s in zip(doc_ids,doc_scores):
                cur_scores[str(did)] = s
            # for did in set(excluded_ids[str(query_id)]):
            for did in excluded_ids:
                if did!="N/A":
                    cur_scores.pop(did)
            if return_full_scores:
                cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)
            else:
                cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:num_hits]
            emb_scores[str(query_id)] = {}
            for pair in cur_scores:
                emb_scores[str(query_id)][pair[0]] = pair[1]
        return emb_scores