import argparse


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='/data/models/Llama-3.2-1B-Instruct')
    parser.add_argument('--peft_model_name', type=str, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_qlen', type=int, default=1024)
    parser.add_argument('--max_dlen', type=int, default=1024)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--snowflake',action='store_true')
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--llm_dtype', type=str, default='auto')
    parser.add_argument('--debug', action='store_true')
    return parser


def add_task_specific_args(parser, task):
    if task == "generate_query":
        parser.add_argument('--embedding_name_or_path', type=str, default='/data/models/Llama-3.2-1B-Instruct')
        parser.add_argument('--qa_path', type=str, default='../data/QA_Datasets/bright/aops.json')
        parser.add_argument('--corpus_path', type=str, default=None)
        parser.add_argument('--cache_dir', type=str, default=None)
        parser.add_argument('--prompt_mode', type=str, default='reasoning_and_ranking')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--keep_doc_nums', type=int, default=5)
        parser.add_argument('--n_round', type=int, default=2)
        parser.add_argument('--agent_mode', type=str, default='',
                            choices=['route_exec', ''])

    else:
        raise ValueError(f"Unsupported task: {task}")

    return parser

def parse_args():
    base_parser = get_base_parser()
    base_args, remaining_args = base_parser.parse_known_args()

    full_parser = get_base_parser()
    full_parser = add_task_specific_args(full_parser, base_args.task)

    args = full_parser.parse_args()
    return args
