import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import sys

RERANKER_MODEL_PATH = os.path.expanduser(sys.argv[1])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

max_length = 1024  # 可以根据您的GPU显存调整
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH, padding_side='left')
rerank_model = AutoModelForCausalLM.from_pretrained(RERANKER_MODEL_PATH).to(DEVICE).eval()
token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")
token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")
prefix_tokens = rerank_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = rerank_tokenizer.encode(suffix, add_special_tokens=False)

def format_instruction(instruction, query, doc):
    """构建符合模型要求的输入格式。"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n"


def process_inputs(pairs):
    """对格式化的文本对进行分词和填充。"""
    inputs = rerank_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = rerank_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(rerank_model.device)
    return inputs


@torch.no_grad()
def compute_scores(pairs):
    """计算所有文本对的相关性分数。"""
    inputs = process_inputs(pairs)

    # 获取最后一个token的logits
    last_token_logits = rerank_model(** inputs).logits[:, -1, :]

    # 提取 "yes" 和 "no" 两个词的logit值
    true_logits = last_token_logits[:, token_true_id]
    false_logits = last_token_logits[:, token_false_id]

    # 将它们组合起来并通过softmax计算概率
    scores_tensor = torch.stack([false_logits, true_logits], dim=1)
    probabilities = torch.nn.functional.softmax(scores_tensor, dim=1)

    # 我们只关心 "yes" 的概率，并将其作为最终分数
    yes_probabilities = probabilities[:, 1].cpu().tolist()
    return yes_probabilities


def main():
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    # 改为写入JSONL格式，每次写入一行
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        for d in tqdm(data):
            query = d['query']
            pos = d['pos']
            recall = d['recall']

            pairs = [format_instruction(task, query, doc) for doc in recall]
            scores = compute_scores(pairs)
            results = sorted(list(zip(scores, recall)), key=lambda x: x[0], reverse=True)
            sorted_docs = [item[1] for item in results]
            sorted_scores = [item[0] for item in results]
            pos_score_list = []
            if len(pos) > 0:
                pairs_pos = [format_instruction(task, query, pos_item) for pos_item in pos]
                scores_pos = compute_scores(pairs_pos)
                score_pos_list = list(zip(scores_pos, pos))
            else:
                score_pos_list = []

            # 每个对象单独写入一行
            result_obj = {
                'query': query,
                "pos": pos,
                "pos_score": score_pos_list,
                "rank": sorted_docs,
                "score":sorted_scores
            }
            f.write(json.dumps(result_obj, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
