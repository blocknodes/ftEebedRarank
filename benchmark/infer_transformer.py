# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import sys

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
    #logits = model(**inputs).logits
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

model_path=sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = ["<keyword>海尔</keyword>\n海尔洗衣机的水位调节如何设置",
    "<keyword>海尔</keyword>\n海尔洗衣机的水位调节如何设置",
    #"在WD15-Y301iS洗碗机洗涤过程中，为什么不能开门或接触排水管的热水？",
    #"在WD15-Y301iS洗碗机洗涤过程中，为什么不能开门或接触排水管的热水？",
]

documents = [
    "海信洗衣机水位调节可以通过控制面板上的“水位”选项进行设置，用户可根据衣物数量选择合适的水位高度。",
    "海尔洗衣机水位调节可以通过控制面板上的“水位”选项进行设置，用户可根据衣物数量选择合适的水位高度。",
    #"洗碗机-WF16-C507iMax说明书.md\n- 洗涤过程中请勿开门，因为洗涤水、热气、洗碗机内均为高温，可能导致烫伤（若想中途添加餐具请先按暂停/启动键）。洗涤过程中请勿接触排水管排出的热水。\n",
    #"洗碗机-WD15-Y301iS-3.23.md\n- 洗涤过程中请勿开门，因为洗涤水、热气、洗碗机内均为高温，可能导致烫伤（若想中途添加餐具请先按暂停/启动按钮）。洗涤过程中请勿接触排水管排出的热水。",
    #"洗碗机-WF16-C507iMax说明书.md\n- 洗涤过程中请勿开门，因为洗涤水、热气、洗碗机内均为高温，可能导致烫伤（若想中途添加餐具请先按暂停/启动键）。洗涤过程中请勿接触排水管排出的热水。\n",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
#inputs['attention_mask'][0][-4:]=0
scores = compute_logits(inputs)


print("scores: ", scores)