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
                # val = val.replace('/', '__')
                val = os.path.basename(val).split(".")[0]
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
        'rewrite_prompt_path',
        'rewrite_cache_path',
        # 'qe_cache_path',
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __str__(self):
        return compress_hparam_string('--'.join(f'{k.lower()}={v}' for k, v in vars(self).items() if k.lower() not in self.NO_SAVE_VARS))
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
    
    @classmethod
    def from_args(cls, args=None):
        """Parse command line arguments and return HyperParams instance"""
        parser = argparse.ArgumentParser(description='Hyperparameters')
        
        # Add common hyperparameters here
        parser.add_argument('--dataset', type=str, default='BRIGHT')
        parser.add_argument('--subset', type=str, required=True, help='Subset of data to use')
        parser.add_argument('--tree_version', type=str, required=True, help='Version of the tree structure to use')
        parser.add_argument('--traversal_prompt_version', type=int, default=5)
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

        # Round 3 (run_round3.py)
        parser.add_argument('--round3_anchor_topk', type=int, default=None, help='Top-K for anchor flat retrieval (defaults to flat_topk)')
        parser.add_argument('--round3_local_topk', type=int, default=None, help='Top-K for local (B_active descendants) retrieval (defaults to flat_topk)')
        parser.add_argument('--round3_global_topk', type=int, default=10, help='Top-K for global leaf retrieval')
        parser.add_argument('--round3_rrf_k', type=int, default=60, help='RRF k for fusing local/global ranked lists')
        parser.add_argument('--round3_rewrite_context', type=str, default='leaf', choices=['leaf', 'leaf_branch'],
                            help='Rewrite context evidence: leaf=leaf-only; leaf_branch=leaf evidence + branch context')
        parser.add_argument('--round3_rewrite_once', default=False, action='store_true', help='Rewrite only at iter 0 in round3')
        
        # Parse arguments
        parsed_args = parser.parse_args(args.split() if args else None)
        
        # Create HyperParams instance from parsed arguments
        return cls(**vars(parsed_args))
    
    def add_param(self, key, value):
        """Add a parameter dynamically"""
        setattr(self, key, value)
