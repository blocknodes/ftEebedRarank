# Requires vllm>=0.8.5
import json
import logging
import math
import sys
import gc
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt


def setup_logging() -> None:
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_input_data(file_path: str) -> List[Dict]:
    """
    加载输入数据

    Args:
        file_path: 数据文件路径

    Returns:
        解析后的JSON数据列表
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logging.error(f"加载数据文件失败: {e}")
        sys.exit(1)


def format_instruction(instruction: str, query: str, doc: str) -> List[Dict[str, str]]:
    """
    格式化指令、查询和文档为对话格式

    Args:
        instruction: 任务指令
        query: 用户查询
        doc: 待判断的文档

    Returns:
        格式化后的对话列表
    """
    return [
        {
            "role": "system",
            "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        },
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
        }
    ]


def process_inputs(
    pairs: List[Tuple[str, str]],
    instruction: str,
    max_length: int,
    suffix_tokens: List[int],
    tokenizer: AutoTokenizer
) -> List[TokensPrompt]:
    """
    处理输入数据，转换为模型可接受的格式

    Args:
        pairs: (查询, 文档)对列表
        instruction: 任务指令
        max_length: 最大长度限制
        suffix_tokens: 后缀 tokens
        tokenizer: 分词器

    Returns:
        处理后的模型输入
    """
    # 格式化所有指令
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]

    # 应用聊天模板并分词
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False
    )

    # 截断并添加后缀
    processed_messages = [
        msg[:max_length] + suffix_tokens for msg in tokenized_messages
    ]

    # 转换为 TokensPrompt 对象
    return [TokensPrompt(prompt_token_ids=msg) for msg in processed_messages]


def compute_relevance_scores(
    model: LLM,
    inputs: List[TokensPrompt],
    sampling_params: SamplingParams,
    true_token: int,
    false_token: int
) -> List[float]:
    """
    计算文档相关性分数

    Args:
        model: vllm LLM模型
        inputs: 处理后的输入
        sampling_params: 采样参数
        true_token: "yes"对应的token ID
        false_token: "no"对应的token ID

    Returns:
        每个文档的相关性分数列表
    """
    try:
        outputs = model.generate(inputs, sampling_params, use_tqdm=False)
        scores = []

        for output in outputs:
            # 获取最后一个token的logprobs
            final_logits = output.outputs[0].logprobs[-1]

            # 获取"yes"和"no"的log概率
            true_logit = final_logits[true_token].logprob if true_token in final_logits else -10
            false_logit = final_logits[false_token].logprob if false_token in final_logits else -10

            # 计算概率并归一化
            true_prob = math.exp(true_logit)
            false_prob = math.exp(false_logit)
            score = true_prob / (true_prob + false_prob)

            scores.append(score)

        return scores
    except Exception as e:
        logging.error(f"计算相关性分数失败: {e}")
        return []


def process_query(
    query_data: Dict,
    task_instruction: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    true_token: int,
    false_token: int,
    max_length: int,
    suffix_tokens: List[int]
) -> Dict:
    """
    处理单个查询，对召回的文档进行重排序

    Args:
        query_data: 包含查询和召回文档的数据
        task_instruction: 任务指令
        model: vllm LLM模型
        tokenizer: 分词器
        sampling_params: 采样参数
        true_token: "yes"对应的token ID
        false_token: "no"对应的token ID
        max_length: 最大长度限制
        suffix_tokens: 后缀 tokens

    Returns:
        包含重排序结果的数据
    """
    query = query_data['query']
    pos = query_data['pos']
    recall_docs = query_data['recall'][:10]  # 取前10个召回文档

    # 准备查询-文档对
    pairs = list(zip([query] * len(recall_docs), recall_docs))

    # 处理输入
    inputs = process_inputs(
        pairs,
        task_instruction,
        max_length - len(suffix_tokens),
        suffix_tokens,
        tokenizer
    )

    # 计算相关性分数
    scores = compute_relevance_scores(
        model, inputs, sampling_params, true_token, false_token
    )

    # 按分数排序
    sorted_results = sorted(
        zip(recall_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 提取排序后的文档和分数
    sorted_docs = [item[0] for item in sorted_results]
    sorted_scores = [item[1] for item in sorted_results]

    return {
        'query': query,
        "pos": pos,
        "rank": sorted_docs,
        "score": sorted_scores
    }


def main():
    """主函数：加载模型、处理数据并保存结果"""
    setup_logging()

    if len(sys.argv) != 4:
        logging.error("用法: python script.py <模型路径> <输入文件> <输出文件前缀>")
        sys.exit(1)

    # 解析命令行参数
    model_path = sys.argv[1]
    input_file = sys.argv[2]
    output_prefix = sys.argv[3]

    # 加载数据
    logging.info(f"加载数据: {input_file}")
    data = load_input_data(input_file)

    # 配置模型和分词器
    logging.info(f"加载模型: {model_path}")
    number_of_gpus = torch.cuda.device_count()
    logging.info(f"使用GPU数量: {number_of_gpus}")

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化LLM模型
    model = LLM(
        model=model_path,
        tensor_parallel_size=number_of_gpus,
        max_model_len=10000,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8
    )

    # 配置推理参数
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length = 8192
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # 获取"yes"和"no"的token ID
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    # 任务指令
    #task_instruction = '给一个用户的问题，找到最能够回答这个问题的片段，注意：一定要能够回答这个问题'
    task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    # 处理所有查询并写入两种JSONL文件
    logging.info("开始处理查询...")

    # 定义两个输出文件路径
    standard_output = f"{output_prefix}.jsonl"       # 标准JSONL（无缩进）
    readable_output = f"{output_prefix}_readable.jsonl"  # 易读JSONL（带缩进）

    # 同时打开两个文件
    with open(standard_output, "w", encoding="utf-8") as std_f, \
         open(readable_output, "w", encoding="utf-8") as read_f:

        for item in tqdm(data, desc="处理进度"):
            result = process_query(
                item,
                task_instruction,
                model,
                tokenizer,
                sampling_params,
                true_token,
                false_token,
                max_length,
                suffix_tokens
            )

            # 写入标准JSONL（无缩进，适合程序处理）
            json.dump(result, std_f, ensure_ascii=False)
            std_f.write("\n")

            # 写入易读JSONL（带缩进，适合人工查看）
            json.dump(result, read_f, ensure_ascii=False, indent=2)
            read_f.write("\n")

    logging.info(f"标准JSONL保存到: {standard_output}")
    logging.info(f"易读JSONL保存到: {readable_output}")

    # 清理资源
    destroy_model_parallel()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("处理完成")


if __name__ == "__main__":
    main()
