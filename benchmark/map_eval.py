import json
from typing import List, Dict, Any, Union
import sys
def calculate_precision_at_k(relevant_docs: List[str], retrieved_docs: List[Dict[str, Any]], k: int) -> float:
    """
    计算位置k的精确率

    Args:
        relevant_docs: 相关文档列表
        retrieved_docs: 检索到的文档列表
        k: 位置k

    Returns:
        位置k的精确率
    """
    if k <= 0 or k > len(retrieved_docs):
        return 0.0

    # 计算前k个文档中相关文档的数量
    relevant_count = 0
    for i in range(k):
        doc_text = retrieved_docs[i]
        for rel_doc in relevant_docs:
            if len(rel_doc)==0:
                raise
        if any(rel_doc in doc_text for rel_doc in relevant_docs):
            relevant_count += 1

    # 精确率 = 相关文档数 / k
    return relevant_count / k

def calculate_average_precision(relevant_docs: List[str], retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    计算平均精确率(AP)

    Args:
        relevant_docs: 相关文档列表
        retrieved_docs: 检索到的文档列表

    Returns:
        平均精确率
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0

    ap_sum = 0.0
    relevant_count = 0

    # 对每个位置计算精确率
    for i in range(len(retrieved_docs)):
        doc_text = retrieved_docs[i]
        is_relevant = any(rel_doc in doc_text for rel_doc in relevant_docs)

        # 如果当前文档相关，则计算该位置的精确率并累加
        if is_relevant:
            relevant_count += 1
            precision_at_i = calculate_precision_at_k(relevant_docs, retrieved_docs, i + 1)
            ap_sum += precision_at_i

    # 如果没有找到相关文档，则AP为0
    if relevant_count == 0:
        return 0.0

    # AP = 所有相关文档位置的精确率之和 / 相关文档总数
    return ap_sum / relevant_count

def after_first_newline(s):
    parts = s.split('\n', 1)
    return parts[1] if len(parts) > 1 else ""

def calculate_map(data: List[Dict[str, Any]], score_threshold: float = 0.0) -> float:
    """
    计算MAP(平均精确率均值)

    Args:
        data: 包含查询和检索结果的数据，其中scores单独存放
        score_threshold: 分数阈值，低于此阈值的条目将被过滤掉

    Returns:
        MAP值
    """
    if not data:
        return 0.0

    ap_sum = 0.0

    # 对每个查询计算AP
    for query_data in data:
        relevant_docs = query_data.get("pos", [])
        retrieved_docs = query_data.get("rank", [])
        # 获取与检索文档对应的分数列表
        scores = query_data.get("score", [])

        # 确保文档和分数数量匹配
        if len(retrieved_docs) != len(scores):
            print(f"警告: 检索文档数量与分数数量不匹配，已跳过该查询")
            continue

        # 处理相关文档
        relevant_docs_new = []
        for pos in relevant_docs:
            relevant_docs_new.append(after_first_newline(pos) + '\n')

        # 处理检索文档并应用阈值过滤
        retrieved_docs_new = []
        for doc, score in zip(retrieved_docs, scores):
            # 检查分数是否高于阈值
            if score >= score_threshold:
                if isinstance(doc, dict):
                    retrieved_docs_new.append(after_first_newline(doc.get("text", "")))
                else:
                    retrieved_docs_new.append(after_first_newline(str(doc)))

        ap = calculate_average_precision(relevant_docs_new, retrieved_docs_new)
        ap_sum += ap
    print(ap_sum)
    # MAP = 所有查询的AP之和 / 查询总数
    return ap_sum / len(data) if data else 0.0

def main():
    # 检查命令行参数
    if len(sys.argv) not in [2, 3]:
        print("用法: python map_calculator.py <jsonl_file> [score_threshold]")
        print("  其中score_threshold为可选参数，默认值为0.0")
        return

    # 解析阈值参数
    score_threshold = 0.0
    if len(sys.argv) == 3:
        try:
            score_threshold = float(sys.argv[2])
        except ValueError:
            print("错误: 阈值必须是一个数字")
            return

    # 从JSONL文件加载数据（逐行读取）
    try:
        data = []
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # 解析每行的JSON对象
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析错误 - {e}，已跳过该行")
                    continue

        if not data:
            print("没有有效的数据可处理")
            return

    except FileNotFoundError:
        print(f"错误: 文件 '{sys.argv[1]}' 不存在")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 计算MAP
    map_value = calculate_map(data, score_threshold)
    print(f"阈值为 {score_threshold} 时的MAP值: {map_value:.4f}")

if __name__ == "__main__":
    main()
