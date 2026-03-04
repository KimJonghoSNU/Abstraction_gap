import re, os, json, hashlib
import argparse

def abbreviate_key(key: str) -> str:
    """
    Systematically abbreviate a key by:
    - Splitting on underscores
    - Taking first 3 letters of each part
    - Keeping numbers intact
    """
    parts = re.split(r'[_\-]', key)
    abbrev_parts = []
    for p in parts:
        if p.isdigit() or len(p) < 4:
            abbrev_parts.append(p.capitalize())  # keep numbers and short parts as is
        else:
            abbrev_parts.append(p[:1].capitalize())  # take first 3 letters
    return "".join(abbrev_parts)

def compress_hparam_string(hparam_str: str, chunk_size: int = 5) -> str:
    """
    Compress a hyperparameter string into a shorter form
    without hardcoded replacement rules.
    """
    parts = hparam_str.split("--")
    compressed_parts = []

    for part in parts:
        if "=" in part:
            key, val = part.split("=", 1)
            if val and val.lower() not in ['false', 'none']:
                # Intent: preserve numeric/string values (e.g., 0.035) while shortening path-like values only.
                if "/" in val or "\\" in val:
                    val = os.path.splitext(os.path.basename(val))[0]
                compressed_parts.append(f"{abbreviate_key(key)}={val}")
        else:
            # Handle flag-only parameters
            compressed_parts.append(abbreviate_key(part))

    final_str = "-".join(compressed_parts)
    final_str = re.sub(r"\.log$", "", final_str)

    # Split into subdirs in fixed-size chunks to avoid overly long paths.
    parts = [p for p in final_str.split("-") if p]
    if parts:
        chunks = ["-".join(parts[i:i + chunk_size]) for i in range(0, len(parts), chunk_size)]
        final_str = "/".join(chunks)

    return final_str

class HyperParams(argparse.Namespace):
    NO_SAVE_VARS = set([
        'dataset',
        'rerank',
        'load_existing',
        'llm_max_concurrent_calls',
        'num_threads',
        'search_with_path_relevance',
        'llm_api_timeout',
        'llm_api_max_retries',
        'llm_api_staggering_delay',
        # avoid excessively long log filenames
        'retriever_model_path',
        'node_emb_path',
        'traversal_prompt_template_path',
        'rewrite_prompt_path',
        'rewrite_cache_path',
        'round3_router_cache_path',
        'round3_v6_leaf_knn_path',
        'qe_cache_path',
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __str__(self):
        effective_items = self._effective_items_for_naming()
        return compress_hparam_string('--'.join(f'{k.lower()}={v}' for k, v in effective_items if k.lower() not in self.NO_SAVE_VARS))
        # return compress_hparam_string('--'.join(f'{k.lower()}={os.path.basename(str(v)).split(".")[0]}' for k, v in vars(self).items() if k.lower() not in self.NO_SAVE_VARS))

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, k):
        return vars(self).get(k.lower())

    def exp_hash(self, length: int = 16) -> str:
        """Stable hash over all hyperparameters."""
        def normalize(value):
            if isinstance(value, dict):
                return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            if isinstance(value, (list, tuple)):
                return [normalize(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        payload = {str(k).lower(): normalize(v) for k, v in vars(self).items()}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return digest if length is None else digest[:length]

    def _effective_items_for_naming(self):
        payload = dict(vars(self))
        # Intent: drop router-only naming args unless router prompt is enabled.
        router_enabled = bool(payload.get('round3_router_prompt_name'))
        if not router_enabled:
            for key in list(payload.keys()):
                if str(key).startswith('round3_router_'):
                    payload.pop(key, None)
        # Intent: drop history-topk naming arg when retrieval history injection is disabled.
        if not bool(payload.get('round3_rewrite_use_history')):
            payload.pop('round3_rewrite_history_topk', None)
        # Intent: keep path names compact when category policy is disabled.
        if str(payload.get('round3_category_policy', 'none')).lower() == 'none':
            payload.pop('round3_category_soft_keep', None)
            payload.pop('round3_category_support_topk', None)
            payload.pop('round3_category_explore_beta', None)
            payload.pop('round4_rule_name', None)
            payload.pop('round4_support_topm', None)
            payload.pop('round4_rule_a_margin_tau', None)
            payload.pop('round4_rule_b_drop_tau', None)
            payload.pop('round4_analysis_category_mode', None)
        else:
            rule_name = str(payload.get('round4_rule_name', 'rule_a')).lower()
            if rule_name != 'rule_a':
                payload.pop('round4_rule_a_margin_tau', None)
            if rule_name != 'rule_b':
                payload.pop('round4_rule_b_drop_tau', None)
            if str(payload.get('round4_analysis_category_mode', 'default')).lower() == 'default':
                payload.pop('round4_analysis_category_mode', None)
        if str(payload.get('category_fusion', 'off')).lower() == 'off':
            payload.pop('category_fusion', None)
        # Intent: keep naming stable for the default BRIGHT original-query setting.
        if str(payload.get('query_source', 'original')).lower() == 'original':
            payload.pop('query_source', None)
        # Intent: when summarized context is explicitly off, omit this flag from naming to keep off-baseline path stable.
        if str(payload.get('round3_summarized_context', 'on')).lower() == 'off':
            payload.pop('round3_summarized_context', None)
        # Intent: keep naming stable when round5 MRR pool uses default depth.
        if int(payload.get('round5_mrr_pool_k', 100) or 100) == 100:
            payload.pop('round5_mrr_pool_k', None)
        payload.pop('round5_category_fallback_on_parse_fail', None)
        payload.pop('round5_category_partial_ok', None)
        round5_mode = str(payload.get('round5_mode', 'legacy')).lower()
        # Intent: keep historical round5 path names stable when running the default legacy mode.
        if round5_mode == 'legacy':
            payload.pop('round5_mode', None)
            payload.pop('round5_category_k', None)
            payload.pop('round5_category_generator_prompt_name', None)
            payload.pop('round5_category_rewrite_prompt_name', None)
            payload.pop('round5_category_history_scope', None)
            payload.pop('round5_category_drift_trigger', None)
        else:
            if int(payload.get('round5_category_k', 3) or 3) == 3:
                payload.pop('round5_category_k', None)
            if str(payload.get('round5_category_generator_prompt_name', 'round5_category_generator_v1')).lower() == 'round5_category_generator_v1':
                payload.pop('round5_category_generator_prompt_name', None)
            if str(payload.get('round5_category_rewrite_prompt_name', 'round5_agent_executor_category_v1')).lower() == 'round5_agent_executor_category_v1':
                payload.pop('round5_category_rewrite_prompt_name', None)
            if str(payload.get('round5_category_history_scope', 'full')).lower() == 'full':
                payload.pop('round5_category_history_scope', None)
            if str(payload.get('round5_category_drift_trigger', 'leaf_cluster')).lower() == 'leaf_cluster':
                payload.pop('round5_category_drift_trigger', None)
        # Intent: keep historical run path naming stable for default pre-planner reference source.
        if str(payload.get('round4_preplanner_reference_mode', 'website_title')).lower() == 'website_title':
            payload.pop('round4_preplanner_reference_mode', None)
        return payload.items()
    
    @classmethod
    def from_args(cls, args=None):
        """Parse command line arguments and return HyperParams instance"""
        parser = argparse.ArgumentParser(description='Hyperparameters')
        
        # Add common hyperparameters here
        parser.add_argument('--dataset', type=str, default='BRIGHT')
        parser.add_argument('--subset', type=str, required=True, help='Subset of data to use')
        parser.add_argument(
            '--query_source',
            type=str,
            default='original',
            choices=['original', 'gpt4'],
            help='Query source for BRIGHT examples: original=examples/query, gpt4=gpt4_reason/gpt4_query',
        )
        parser.add_argument('--tree_version', type=str, required=True, help='Version of the tree structure to use')
        parser.add_argument('--traversal_prompt_version', type=int, default=5)
        parser.add_argument(
            '--traversal_prompt_template_path',
            type=str,
            default=None,
            help=(
                'Optional traversal prompt template file. '
                'If set, overrides built-in prompts.py traversal templates. '
                'Supported placeholders: {{QUERY}}, {{CANDIDATES}}, {{RELEVANCE_DEFINITION}}, {{PROMPT_ID}}'
            ),
        )
        parser.add_argument('--reasoning_in_traversal_prompt', type=int, default=-1)
        parser.add_argument('--max_query_char_len', type=int, default=None)
        parser.add_argument('--max_doc_desc_char_len', type=int, default=None)
        parser.add_argument('--max_prompt_proto_size', type=int, default=None)
        parser.add_argument('--search_with_path_relevance', type=bool, default=True)
        parser.add_argument('--num_leaf_calib', type=int, default=10)
        parser.add_argument('--pl_tau', type=float, default=5.0)
        parser.add_argument('--relevance_chain_factor', type=float, default=0.5)
        parser.add_argument('--llm_api_backend', type=str, default='genai')
        parser.add_argument('--llm', type=str, default='gemini-2.5-flash')
        parser.add_argument('--llm_max_concurrent_calls', type=int, default=20)
        parser.add_argument('--llm_api_timeout', type=int, default=120)
        parser.add_argument('--llm_api_max_retries', type=int, default=4)
        parser.add_argument('--llm_api_staggering_delay', type=float, default=0.1)
        parser.add_argument('--num_iters', type=int, default=20)
        parser.add_argument('--num_eval_samples', type=int, default=1_000)
        parser.add_argument('--max_beam_size', type=int, default=2)
        parser.add_argument('--rerank', default=False, action='store_true')
        parser.add_argument('--load_existing', default=False, action='store_true') 
        parser.add_argument('--num_threads', type=int, default=os.cpu_count())
        parser.add_argument('--suffix', type=str, default='')

        # Flat retrieval -> gated traversal (optional)
        parser.add_argument('--flat_then_tree', default=False, action='store_true')
        parser.add_argument('--retriever_model_path', type=str, default=None, help='Local path to Diver-Retriever-4B (or compatible) model')
        parser.add_argument('--node_emb_path', type=str, default=None, help='Path to precomputed node embeddings (.npy) aligned with node_registry order')
        parser.add_argument('--flat_topk', type=int, default=200, help='Top-K nodes to retrieve from the flattened node set')
        parser.add_argument('--gate_branches_topb', type=int, default=10, help='Number of branch gates to keep')
        parser.add_argument('--seed_from_flat_gates', default=False, action='store_true', help='Seed traversal beam from flat retrieval gate paths at iter 0')
        parser.add_argument('--pre_flat_rewrite', default=False, action='store_true', help='Run rewrite after an initial flat retrieval and re-run flat retrieval with that rewrite; replaces QE for that run')
        parser.add_argument(
            '--pre_flat_rewrite_source',
            type=str,
            default='branch',
            choices=['branch', 'leaf', 'all'],
            help='Initial rewrite context from flat retrieval: branch=non-leaf nodes only, leaf=leaf nodes only, all=all nodes',
        )
        # parser.add_argument('--qe_prompt_path', type=str, default=None, help='Optional prompt file for query expansion (used when QE cache misses)')
        parser.add_argument('--qe_prompt_name', type=str, default=None, help='Optional built-in QE prompt name (used when QE cache misses); QE runs before flat retrieval and expands the original query')
        parser.add_argument('--qe_cache_path', type=str, default=None, help='Optional JSONL cache for query expansion results') 
        parser.add_argument('--qe_force_refresh', default=False, action='store_true', help='Ignore QE cache and regenerate expansions')
        parser.add_argument('--rewrite_prompt_name', type=str, default=None, help='Optional built-in rewrite prompt name (used when cache misses); rewrite runs at start and/or during traversal, after flat retrieval')
        parser.add_argument('--rewrite_prompt_path', type=str, default=None, help='Optional prompt file for rewrite (used when cache misses)')
        parser.add_argument('--rewrite_cache_path', type=str, default=None, help='Optional JSONL cache for rewrite results')
        parser.add_argument('--rewrite_force_refresh', default=False, action='store_true', help='Ignore rewrite cache and regenerate rewrites')
        parser.add_argument('--rewrite_mode', type=str, default='concat', choices=['concat', 'replace'], help='How to combine rewrite with the traversal query')
        parser.add_argument('--rewrite_every', type=int, default=1, help='Rewrite every N iterations when rewrite is enabled')
        parser.add_argument('--rewrite_context_topk', type=int, default=5, help='Max number of summaries to include in rewrite context')
        parser.add_argument(
            '--rewrite_context_source',
            type=str,
            default='mixed',
            choices=['flat', 'slate', 'fused', 'mixed', 'leafslate'],
            help=(
                "Context source for rewrite: "
                "flat=topk descriptions from flat retrieval; "
                "slate=topk descriptions from current traversal slates; "
                "fused=RRF fusion of flat-retrieval leaf ranks and traversal leaf ranks; "
                "mixed=RRF fusion of fused (if available) and slate ranks; "
                "leafslate=traversal leaf hits (if any) + branch nodes from slates."
            ),
        )
        parser.add_argument('--rewrite_at_start', default=False, action='store_true', help='Run rewrite once before the first traversal iteration')
        parser.add_argument('--leaf_only_retrieval', default=False, action='store_true', help='Restrict flat retrieval to leaf nodes only (used in run_leaf_rank.py)')
        parser.add_argument('--leaf_no_initial_rewrite', default=False, action='store_true',
                            help='Skip initial rewrite in run_leaf_rank.py and start rewrite after first retrieval iteration')
        parser.add_argument('--use_retriever_traversal', default=False, action='store_true',
                            help='If set, score traversal slates with retriever embeddings instead of LLM prompts')

        # Round 3 (run_round3.py)
        parser.add_argument('--round3_anchor_topk', type=int, default=None, help='Top-K for anchor flat retrieval (defaults to flat_topk)')
        parser.add_argument('--round3_local_topk', type=int, default=None, help='Top-K for local (B_active descendants) retrieval (defaults to flat_topk)')
        parser.add_argument('--round3_global_topk', type=int, default=10, help='Top-K for global leaf retrieval')
        parser.add_argument(
            '--round5_mrr_pool_k',
            type=int,
            default=100,
            help='Top-K local leaf retrieval size used to compute Highest-MRR sub-branch in run_round5.py',
        )
        parser.add_argument(
            '--round5_mode',
            type=str,
            default='legacy',
            choices=['legacy', 'category'],
            help=(
                'run_round5.py rewrite mode: '
                'legacy=single-pass agent_executor_v1; '
                'category=generate open-set categories then category-bound rewrite'
            ),
        )
        parser.add_argument(
            '--round5_category_k',
            type=int,
            default=3,
            help='Number of categories to generate per iteration when --round5_mode=category',
        )
        parser.add_argument(
            '--round5_category_generator_prompt_name',
            type=str,
            default='round5_category_generator_v1',
            help='Prompt template name for category generation in run_round5.py',
        )
        parser.add_argument(
            '--round5_category_rewrite_prompt_name',
            type=str,
            default='round5_agent_executor_category_v1',
            help='Prompt template name for category-bound rewrite in run_round5.py',
        )
        parser.add_argument(
            '--round5_category_history_scope',
            type=str,
            default='full',
            choices=['full', 'none'],
            help='Category history context for generator prompt: full|none',
        )
        parser.add_argument(
            '--round5_category_drift_trigger',
            type=str,
            default='leaf_cluster',
            choices=['leaf_cluster', 'none'],
            help='When to inject anti-drift reminder into category generation: leaf_cluster|none',
        )
        parser.add_argument('--round3_rrf_k', type=int, default=60, help='RRF k for fusing local/global ranked lists')
        parser.add_argument('--round3_rewrite_context', type=str, default='leaf', choices=['leaf', 'leaf_branch'],
                            help='Rewrite context evidence: leaf=leaf-only; leaf_branch=leaf evidence + branch context')
        parser.add_argument('--round3_rewrite_once', default=False, action='store_true', help='Rewrite only at iter 0 in round3')
        parser.add_argument('--round3_explore_mode', type=str, default='replace', choices=['replace', 'concat'],
                            help='How to combine rewrite with the traversal query in round3')
        parser.add_argument('--round3_rewrite_use_history', default=False, action='store_true',
                            help='Prepend retrieval history (doc IDs only) to rewrite prompts in run_round3_1.py')
        parser.add_argument('--round3_rewrite_history_topk', type=int, default=10,
                            help='Top-K retrieved leaf doc IDs per previous iteration in rewrite history')
        parser.add_argument(
            '--round3_summarized_context',
            type=str,
            default='on',
            choices=['on', 'off'],
            help='Use summarized rewrite context and history evidence: on|off',
        )
        parser.add_argument(
            '--round3_category_policy',
            type=str,
            default='none',
            choices=['none', 'soft'],
            help=(
                'Category-level query policy in run_round3_1.py: '
                'none=disable; '
                'soft=focus the best supported categories based on per-category actions '
                '(use --round3_category_soft_keep=1 for hard behavior)'
            ),
        )
        parser.add_argument(
            '--round3_category_soft_keep',
            type=int,
            default=2,
            help='When --round3_category_policy=soft, keep up to this many categories (including the primary one)',
        )
        parser.add_argument(
            '--round3_category_support_topk',
            type=int,
            default=10,
            help='Top-K leaf evidence descriptions used to score category support (embedding-only selector)',
        )
        parser.add_argument(
            '--round3_category_explore_beta',
            type=float,
            default=0.1,
            help='Penalty weight for category similarity to previous selected categories during explore',
        )
        parser.add_argument(
            '--round4_rule_name',
            type=str,
            default='rule_a',
            choices=['rule_a', 'rule_b', 'rule_c'],
            help='Round4 category decision rule: rule_a=margin gate, rule_b=counterfactual drop risk, rule_c=per-category best-rewrite concat',
        )
        parser.add_argument(
            '--round4_support_topm',
            type=int,
            default=10,
            help='Top-M evidence similarities averaged for category support score in run_round4.py',
        )
        parser.add_argument(
            '--round4_rule_a_margin_tau',
            type=float,
            default=0.02,
            help='Rule A exploit threshold: exploit when (top1_support - top2_support) >= tau',
        )
        parser.add_argument(
            '--round4_rule_b_drop_tau',
            type=float,
            default=0.01,
            
            help='Rule B exploit threshold: exploit when min relative drop risk <= tau',
        )
        parser.add_argument(
            '--round4_analysis_category_mode',
            type=str,
            default='default',
            choices=['default', 'force_full', 'force_drop_one'],
            help=(
                'Round4 analysis-only category override in run_round4.py: '
                'default=use rule decision; '
                'force_full=always keep all categories; '
                'force_drop_one=always drop one category when possible'
            ),
        )
        parser.add_argument(
            '--category_fusion',
            type=str,
            default='off',
            choices=['off', 'category_query_mean'],
            help=(
                'Anchor top-k fusion in run_round4.py: '
                'off=disable; '
                'category_query_mean=rerank anchor top-10 by mean of per-category support and base query score'
            ),
        )
        parser.add_argument(
            '--round4_iter0_prompt_name',
            type=str,
            default=None,
            help=(
                'Optional rewrite prompt name used only at iter 0 in run_round4.py; '
                'iter>=1 continues using --rewrite_prompt_name (or --rewrite_prompt_path/default).'
            ),
        )
        parser.add_argument(
            '--round4_preplanner_reference_mode',
            type=str,
            default='website_title',
            choices=['website_title', 'doc_id'],
            help=(
                'Reference pool used by run_round4_1 pre-planner: '
                'website_title=use cleaned website titles; '
                'doc_id=use raw document IDs from document_categories_category_assign_v2.jsonl'
            ),
        )
        parser.add_argument(
            '--round3_action_oracle',
            type=str,
            default='none',
            choices=['none', 'select', 'rerank', 'ndcg', 'rrf'],
            help=(
                'Action-oracle mode in run_round3_oracle.py: '
                'select=LLM picks explore vs exploit; '
                'rerank=LLM reranks explore+exploit top-10; '
                'ndcg=choose by gold nDCG (oracle upper bound); '
                'rrf=RRF fuse explore/exploit results; '
                'none=disable'
            ),
        )
        parser.add_argument('--round3_router_prompt_name', type=str, default=None,
                            help='Optional router prompt name for split explore/exploit decision in run_round3_1.py')
        parser.add_argument('--round3_router_cache_path', type=str, default=None,
                            help='Optional JSONL cache for router action results in run_round3_1.py')
        parser.add_argument('--round3_router_force_refresh', default=False, action='store_true',
                            help='Ignore router cache and regenerate router actions')
        parser.add_argument('--round3_router_context', type=str, default='leaf',
                            choices=['leaf', 'leaf_branch', 'leaf_branch_depth1'],
                            help=(
                                'Router context evidence: '
                                'leaf=leaf-only; '
                                'leaf_branch=leaf evidence + non-overlap branch context; '
                                'leaf_branch_depth1=leaf evidence + non-overlap branch context + depth-1 branches'
                            ))
        parser.add_argument(
            '--round3_anchor_local_rank',
            type=str,
            default='none',
            choices=['none', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
            help=(
                'Local ranking override from anchor order: '
                'none=default local ranking; '
                'v1=replace anchor branches with top descendant leaf for local ranking; '
                'v2=use v1 behavior and also use those leaves as rewrite context; '
                'v3=replace branches by greedy best-child descent to a leaf; '
                'v4=replace branches by best path-average leaf (branch→leaf mean score); '
                'v5=leaf-only anchor candidates (top-20) then LLM rerank to choose top-10; '
                'v6=leaf-only anchor top-10 + prefix-diverse extra top-10 by max seed-leaf cosine, then LLM rerank'
            ),
        )
        parser.add_argument(
            '--round3_v6_leaf_knn_path',
            type=str,
            default=None,
            help='Optional .npz file with precomputed leaf->topK leaf neighbors for faster v6 expansion',
        )
        
        # Parse arguments
        parsed_args = parser.parse_args(args.split() if args else None)
        
        # Create HyperParams instance from parsed arguments
        return cls(**vars(parsed_args))
    
    def add_param(self, key, value):
        """Add a parameter dynamically"""
        setattr(self, key, value)
