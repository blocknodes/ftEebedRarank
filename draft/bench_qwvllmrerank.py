# Requires vllm>=0.8.5
import logging
from typing import Dict, Optional, List

import json
import logging
from tqdm import tqdm
import torch
import sys

from transformers import AutoTokenizer, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt


with open(sys.argv[2], "r", encoding='utf-8') as f:
    data = [json.loads(l) for l in f]

##############
def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages =  tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        token_count = len(outputs[i].outputs[0].token_ids)
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores

#把下面改成if __ == main的形式
if __name__ == "__main__":

    number_of_gpu = torch.cuda.device_count()
    model= sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = LLM(model=model, tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=0.8)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length=8192
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    ############


    task = 'Given a web search query, retrieve relevant passages that answer the query'
    task = '给一个用户的问题，找到最能够回答这个问题的片段，注意：一定要能够回答这个问题'


    final_data = []
    for d in tqdm(data):
        query = d['query']
        pos = d['pos']
        recall = d['recall'][:10]
        compute_list = []
        queries = []
        documents = []

        for doc in recall:
            queries.append(query)
            documents.append(doc)
        pairs = list(zip(queries, documents))
        inputs = process_inputs(pairs, task, max_length-len(suffix_tokens), suffix_tokens)
        #pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]
        #inputs = process_inputs(pairs)
        score = compute_logits(model, inputs, sampling_params, true_token, false_token)
        #score = reranker.compute_score(compute_list)
        # sort doc by score
        sorted_results = []
        for i, doc in enumerate(recall):
            sorted_results.append((doc, score[i]))
        # 按照得分降序排序
        sorted_results.sort(key=lambda x: x[1], reverse=True)

        # 如果需要，可以将排序后的结果保存到一个新的列表中
        sorted_docs = [item[0] for item in sorted_results]
        scores = [item[1] for item in sorted_results]

        final_data.append({'query': query, "pos": pos, "rank":sorted_docs,"score":scores})

    with open(f"{sys.argv[3]}.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)


    destroy_model_parallel()