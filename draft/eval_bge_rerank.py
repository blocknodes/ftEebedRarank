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

def calculate_map(data: List[Dict[str, Any]]) -> float:
    """
    计算MAP(平均精确率均值)

    Args:
        data: 包含查询和检索结果的数据

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
        relevant_docs_new = []
        retrieved_docs_new = []
        for pos in relevant_docs:
            relevant_docs_new.append(after_first_newline(pos)+'\n')

        for doc in retrieved_docs:
            retrieved_docs_new.append(after_first_newline(doc))


        #relevant_docs = [line for doc in relevant_docs for line in doc.split('\n')[1:]]
        #retrieved_docs = [line for doc in retrieved_docs for line in doc.split('\n')[1:-1]]

        ap = calculate_average_precision(relevant_docs_new, retrieved_docs_new)
        ap_sum += ap
    print(ap_sum)
    # MAP = 所有查询的AP之和 / 查询总数
    return ap_sum / len(data)

def main():
    # 从文件加载数据
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("文件不存在，请提供包含查询和排序结果的JSON文件")
        return
    except json.JSONDecodeError:
        print("JSON格式错误，请检查文件格式")
        return

    # 计算MAP
    map_value = calculate_map(data)
    print(f"MAP值: {map_value:.4f}")

if __name__ == "__main__":
    main()