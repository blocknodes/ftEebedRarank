import json
from tqdm import tqdm
import sys
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

with open(sys.argv[2], "r", encoding='utf-8') as f:
    data = [json.loads(l) for l in f]
    #data = json.load(f)





def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores
model= sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model).eval()

# We recommensd enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

#reranker = FlagReranker('/workspace/FlagEmbedding/examples/finetune/reranker/encoder_only/test_encoder_only_base_bge-reranker-m3-0426/checkpoint-2300', use_fp16=True)
#reranker = FlagReranker(sys.argv[1], use_fp16=True)
#reranker = FlagReranker('/data/bge-reranker-v2-m3/', use_fp16=True)

final_data = []
i =0
for d in tqdm(data):
    query = d['query']
    pos = d['pos']
    recall = d['recall'][:10]
    compute_list = []
    queries = []
    documents = []

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    for doc in recall:
        queries.append(query)
        documents.append(doc)
    pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]
    inputs = process_inputs(pairs)
    score = compute_logits(inputs)
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
    print(sorted_results)

    final_data.append({'query': query, "pos": pos, "rank":sorted_docs, "score":scores})
    if i==16:
        break

with open("./bge_reranker_merge_0616.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)
