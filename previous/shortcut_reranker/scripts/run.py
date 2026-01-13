import re
import os
import sys
import csv
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Tuple, Dict, List
from collections import defaultdict, Counter

from scripts.parser import parse_args
from utils.utils import json_load, json_dump

### Generating query
def generate_query(args):
    from scripts.query_generator import QueryGenerator
    print(f"#> generate_query args")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    dataset_name = os.path.basename(args.qa_path).split(".")[0].lower()
    output_path = "/".join(args.qa_path.split("/")[:-1])
    output_path = os.path.join(output_path, f'{args.prompt_mode}_round{args.n_round}')
    if args.agent_mode:
        output_path += f"_{args.agent_mode}"
    output_path = os.path.join(output_path, args.model_name_or_path.split("/")[-1].lower())
    os.makedirs(output_path, exist_ok=True)
    print(f"#> output_path: {output_path}")

    query_generator = QueryGenerator(**args.__dict__)
    retrieval_metrics, retrieval_results, retrieval_scores, generated_queries, sub_metrics, sub_results, sub_scores = query_generator.generate()

    json_dump(os.path.join(output_path, f'{dataset_name}_sub_metrics.json'), sub_metrics)
    json_dump(os.path.join(output_path, f'{dataset_name}_sub_results.json'), sub_results)
    json_dump(os.path.join(output_path, f'{dataset_name}_sub_scores.json'), sub_scores)

    json_dump(os.path.join(output_path, f'{dataset_name}_retrieval_metrics.json'), retrieval_metrics)
    json_dump(os.path.join(output_path, f'{dataset_name}_retrieval_results.json'), retrieval_results)
    json_dump(os.path.join(output_path, f'{dataset_name}_retrieval_scores.json'), retrieval_scores)
    json_dump(os.path.join(output_path, f'{dataset_name}_generated_queries.json'), generated_queries)
    print(f"Saving {output_path} done.")


def main():
    args = parse_args()

    if args.task == 'generate_query':
        generate_query(args)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
