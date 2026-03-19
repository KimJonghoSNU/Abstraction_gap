tasks


Using HuggingFace Transformers
```
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def tokenize_texts(tokenizer, texts, max_length: int, device: str):
    batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8)
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    return batch_dict


task = 'Given a Math problem, retrieve relevant theorems that help answer the problem.'
queries = [
    # taken from BRIGHT TheoT dataset, qid: examples-TheoremQA_wenhuchen/eigen_value1.json
    "Imagine you have a magical box that transforms any object you put inside it, where the object is represented by the column vector x = (x_1, x_2). The box's transformation can be represented by the matrix A = [[5, 4], [1, 2]], so when given an object x, the box outputs the new object Ax. On some special objects, this new object is just a constant multiple of the original object, λx = (λx_1, λx_2). Find both possible values of λ where this occurs — note that these are the box's eigenvalues.",
    # taken from BRIGHT TheoT dataset, qid: examples-TheoremQA_maxku/ipnetwork13-hammingdist.json
    "Imagine you're comparing three digital images that are extremely simplified down to a grid of 5 pixels each, represented by either black (0) or white (1) pixels. The images are as follows: Image A: 00000, Image B: 10101, and Image C: 01010. By counting the number of pixels that differ between each pair of images, find the smallest number of differing pixels."
]
queries = [get_detailed_instruct(task, q) for q in queries]
documents = [
    # taken from BRIGHT TheoT dataset, docid: 2723
    "\\begin{definition}[Definition:Eigenvector/Linear Operator]\nLet $K$ be a field.\nLet $V$ be a vector space over $K$. \nLet $A : V \\to V$ be a linear operator.\nLet $\\lambda \\in K$ be an eigenvalue of $A$.\nA non-zero vector $v \\in V$ is an '''eigenvector corresponding to $\\lambda$''' {{iff}}:\n:$v \\in \\map \\ker {A - \\lambda I}$\nwhere: \n:$I : V \\to V$ is the identity mapping on $V$\n:$\\map \\ker {A - \\lambda I}$ denotes the kernel of $A - \\lambda I$.\nThat is, {{iff}}: \n:$A v = \\lambda v$\n\\end{definition}",
    # taken from BRIGHT TheoT dataset, docid: 14101
    "\\section{Error Correction Capability of Linear Code}\nTags: Linear Codes\n\n\\begin{theorem}\nLet $C$ be a linear code.\nLet $C$ have a minimum distance $d$.\nThen $C$ corrects $e$ transmission errors for all $e$ such that $2 e + 1 \\le d$.\n\\end{theorem}\n\n\\begin{proof}\nLet $C$ be a linear code whose master code is $V$.\nLet $c \\in C$ be a transmitted codeword.\nLet $v$ be the received word from $c$.\nBy definition, $v$ is an element of $V$.\nLet $v$ have a distance $e$ from $c$, where $2 e + 1 \\le d$.\nThus there have been $e$ transmission errors.\n{{AimForCont}} $c_1$ is a codeword of $C$, distinct from $c$, such that $\\map d {v, c_1} \\le e$.\nThen:\n{{begin-eqn}}\n{{eqn | l = \\map d {c, c_1}\n      | o = \\le\n      | r = \\map d {c, v} + \\map d {v, c_1}\n      | c = \n}}\n{{eqn | o = \\le\n      | r = e + e\n      | c = \n}}\n{{eqn | o = <\n      | r = d\n      | c = \n}}\n{{end-eqn}}\nSo $c_1$ has a distance from $c$ less than $d$.\nBut $C$ has a minimum distance $d$.\nThus $c_1$ cannot be a codeword of $C$.\nFrom this contradiction it follows that there is no codeword of $C$ closer to $v$ than $c$.\nHence there is a unique codeword of $C$ which has the smallest distance from $v$.\nHence it can be understood that $C$ has corrected the transmission errors of $v$.\n{{Qed}}\n\\end{proof}\n\n"
]

tokenizer = AutoTokenizer.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928")
model = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928")
model.eval()

device = "cuda:0"   # set device to "cuda:0" for testing on a single GPU
model.to(device)
model.half()

max_length = 512
# Tokenize the input texts
query_batch_dict = tokenize_texts(tokenizer, queries, max_length, device)
doc_batch_dict = tokenize_texts(tokenizer, documents, max_length, device)

with torch.no_grad():
    query_outputs = model(**query_batch_dict)
    query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    
    doc_outputs = model(**doc_batch_dict)
    doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
scores = (query_embeddings @ doc_embeddings.T) * 100
print(scores.cpu().tolist())
```


```
if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# full datasets
dataset_names="biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems"

model_args="\
    --embedder_name_or_path hanhainebula/reason-embed-qwen3-8b-0928 \
    --embedder_model_class decoder-only-base \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --pooling_method last_token \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_batch_size 8 \
    --embedder_query_max_length 8192 \
    --embedder_passage_max_length 8192 \
"

split_list=("examples")

for split in "${split_list[@]}"; do
    eval_args="\
        --task_type short \
        --use_special_instructions True \
        --eval_name bright_short \
        --dataset_dir ./bright_short/data \
        --dataset_names $dataset_names \
        --splits $split \
        --corpus_embd_save_dir ./bright_short/corpus_embd \
        --output_dir ./bright_short/search_results/$split \
        --search_top_k 2000 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 10 100 \
        --eval_output_method markdown \
        --eval_output_path ./bright_short/eval_results_$split.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_100 \
    "

    cmd="python -m FlagEmbedding.evaluation.bright \
        $eval_args \
        $model_args \
    "

    echo $cmd
    eval $cmd

done
```