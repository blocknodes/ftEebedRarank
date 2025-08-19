from FlagEmbedding import FlagReranker
import json
from tqdm import tqdm
import sys


def compute_scores(reranker, query_document_pairs):
    """
    使用reranker计算查询与文档对的分数

    参数:
        reranker: FlagReranker实例
        query_document_pairs: 查询与文档对的列表，格式为[[query1, doc1], [query2, doc2], ...]

    返回:
        分数列表，与输入对一一对应
    """
    return reranker.compute_score(query_document_pairs)


def process_data(reranker, data):
    """
    处理数据：对每个查询的召回文档进行重排序并记录分数

    参数:
        reranker: FlagReranker实例
        data: 输入数据列表，每个元素包含query、pos和recall字段

    返回:
        处理后的结果列表，每个元素包含query、pos、排序后的文档及对应分数
    """
    processed_results = []

    for item in tqdm(data, desc="处理数据"):
        query = item['query']
        positive_docs = item['pos']
        # 取前10个召回文档
        recall_docs = item['recall'][:10]

        # 准备计算列表：[查询, 文档]对
        compute_list = [[query, doc] for doc in recall_docs]

        # 计算分数
        scores = compute_scores(reranker, compute_list)

        # 结合文档和分数，并按分数降序排序
        scored_docs = list(zip(recall_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的文档
        sorted_docs = [doc for doc, _ in scored_docs]
        sorted_scores = [score for _, score in scored_docs]

        # 添加到结果列表，包含文档和对应的分数
        processed_results.append({
            'query': query,
            'pos': positive_docs,
            'rank': sorted_docs,
            'scores': sorted_scores  # 新增：记录每个排序后文档的分数
        })

    return processed_results


def save_as_standard_jsonl(data, output_path):
    """保存为标准JSONL格式（无缩进，适合程序处理）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def save_as_readable_jsonl(data, output_path):
    """保存为带缩进的易读JSONL格式（适合人工查看）"""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            # 使用indent参数增加缩进，确保中文正常显示
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.write('\n\n')  # 增加空行分隔，进一步提高可读性


def main():
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python reranker_processor.py <reranker模型路径> <输入JSONL文件路径>")
        print("示例: python reranker_processor.py ./model ./input_data.jsonl")
        sys.exit(1)

    # 解析命令行参数
    model_path = sys.argv[1]
    input_file_path = sys.argv[2]
    standard_output_path = "./bge_reranker_standard.jsonl"  # 标准JSONL，用于程序处理
    readable_output_path = "./bge_reranker_readable.jsonl"  # 易读JSONL，用于人工查看

    # 加载输入数据
    print(f"加载数据: {input_file_path}")
    with open(input_file_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 初始化reranker
    print(f"加载模型: {model_path}")
    reranker = FlagReranker(model_path, use_fp16=True)

    # 处理数据
    print("开始处理数据...")
    processed_data = process_data(reranker, data)

    # 保存两份结果
    print(f"保存标准JSONL到: {standard_output_path}")
    save_as_standard_jsonl(processed_data, standard_output_path)

    print(f"保存易读JSONL到: {readable_output_path}")
    save_as_readable_jsonl(processed_data, readable_output_path)

    print("处理完成! 已生成两份输出文件")


if __name__ == "__main__":
    main()

