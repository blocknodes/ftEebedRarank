import json
import argparse
import os
from pathlib import Path

def after_first_newline(s):
    parts = s.split('\n', 1)
    return parts[1] if len(parts) > 1 else ""

def evaluate_retrieval_performance(data, output_dir, score_threshold=0.0):
    """评估召回内容的 Top1、Top3 命中率以及 MRR，支持过滤低于阈值的结果"""
    top1_hits = 0
    top3_hits = 0
    mrr_sum = 0
    total_queries = 0
    filtered_count = 0  # 统计被过滤掉的结果数量

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 定义输出文件路径 - 标准格式和可读性强的格式
    goodcase_std_path = os.path.join(output_dir, "goodcase.jsonl")
    goodcase_readable_path = os.path.join(output_dir, "goodcase_readable.jsonl")
    badcase_std_path = os.path.join(output_dir, "badcase.jsonl")
    badcase_readable_path = os.path.join(output_dir, "badcase_readable.jsonl")
    result_path = os.path.join(output_dir, "evaluation_result.json")

    # 清空之前的结果文件
    for path in [goodcase_std_path, goodcase_readable_path,
                 badcase_std_path, badcase_readable_path, result_path]:
        if os.path.exists(path):
            os.remove(path)

    for d in data:
        query = d.get("query", "")
        relevant_docs = d.get("pos", [])  # 正确答案列表
        retrieved_docs = d.get("rank", [])  # 召回结果列表
        doc_scores = d.get("score", [])    # 召回结果对应的分数列表

        # 过滤掉分数低于阈值的召回结果，同时保留分数信息
        filtered_recalls = []
        filtered_scores = []
        if doc_scores and len(doc_scores) == len(retrieved_docs):
            for doc, score in zip(retrieved_docs, doc_scores):
                if score >= score_threshold:
                    filtered_recalls.append(doc)
                    filtered_scores.append(score)
                else:
                    filtered_count += 1
        else:
            # 如果没有分数信息，使用所有召回结果
            filtered_recalls = retrieved_docs
            filtered_scores = [None] * len(retrieved_docs)

        pos = []
        recall_with_scores = []  # 包含分数的召回结果（仅保留此信息）

        for doc in relevant_docs:
            pos.append(after_first_newline(doc)+'\n')

        # 处理召回结果，仅生成带分数的版本
        for doc, score in zip(filtered_recalls, filtered_scores):
            doc_content = after_first_newline(doc)
            recall_with_scores.append({
                "content": doc_content,
                "score": score
            })

        if not query or not pos:
            continue

        total_queries += 1

        # 计算 Top1 命中
        top1_hit = 0
        pos_texts = [p.replace(' ', '') for p in pos]
        if recall_with_scores:  # 检查召回结果是否为空
            first_recall = recall_with_scores[0]["content"].replace(' ', '')
            if first_recall in pos_texts:
                top1_hit = 1
        top1_hits += top1_hit

        # 计算 Top3 命中和 MRR
        top3_hit = 0
        min_rank = float('inf')
        # 只检查前3个召回结果
        for rank, item in enumerate(recall_with_scores[:3], 1):
            item_clean = item["content"].replace(' ', '')
            if item_clean in pos_texts:
                top3_hit = 1
                if rank < min_rank:
                    min_rank = rank

        # 准备要写入的案例数据（只包含带分数的召回结果）
        case_item = {
            "query": query,
            "pos": pos,
            "recall_with_scores": recall_with_scores[:10],  # 仅保留带分数的召回结果
            "top1_hit": top1_hit,
            "top3_hit": top3_hit
        }

        if top3_hit:
            top3_hits += 1
            mrr_sum += 1.0 / min_rank

            # 记录好案例 - 标准格式
            with open(goodcase_std_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(case_item, ensure_ascii=False) + '\n')

            # 记录好案例 - 可读性强的格式
            with open(goodcase_readable_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(case_item, ensure_ascii=False, indent=2) + '\n')
        else:
            # 记录坏案例 - 标准格式
            with open(badcase_std_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(case_item, ensure_ascii=False) + '\n')

            # 记录坏案例 - 可读性强的格式
            with open(badcase_readable_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(case_item, ensure_ascii=False, indent=2) + '\n')

    # 计算指标
    top1_hit_rate = top1_hits / total_queries if total_queries > 0 else 0
    top3_hit_rate = top3_hits / total_queries if total_queries > 0 else 0
    mrr = mrr_sum / total_queries if total_queries > 0 else 0

    # 打印评估结果
    print(f"\n评估结果:")
    print(f"总查询数: {total_queries}")
    print(f"被过滤的低分成数量: {filtered_count}")
    print(f"Top1 命中率: {top1_hit_rate:.4f} ({top1_hits}/{total_queries})")
    print(f"Top3 命中率: {top3_hit_rate:.4f} ({top3_hits}/{total_queries})")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"使用的分数阈值: {score_threshold}")

    # 保存评估结果到文件
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": total_queries,
            "filtered_count": filtered_count,
            "score_threshold": score_threshold,
            "top1_hit_rate": top1_hit_rate,
            "top3_hit_rate": top3_hit_rate,
            "mrr": mrr,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits
        }, f, ensure_ascii=False, indent=2)

    return top1_hit_rate, top3_hit_rate, mrr


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估检索性能，支持过滤低于阈值的结果')
    # 将参数定义为位置参数，不需要--前缀
    parser.add_argument('input', help='输入的JSONL文件路径')
    parser.add_argument('output', help='输出目录路径')
    parser.add_argument('--threshold', type=float, default=0.0,
                      help='分数阈值，低于此阈值的召回结果将被过滤(默认: 0.0)')
    args = parser.parse_args()

    # 读取输入文件
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        print(f"成功读取输入文件: {args.input}，共 {len(data)} 条记录")
        print(f"使用的分数阈值: {args.threshold}")
    except Exception as e:
        print(f"读取输入文件失败: {str(e)}")
        return

    # 执行评估
    evaluate_retrieval_performance(data, args.output, args.threshold)


if __name__ == "__main__":
    main()
