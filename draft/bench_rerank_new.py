import json


# with open (r"D:\2025\知识库建设--检索--精排\验证集\results\bge-rerank\bge-m3_reranker_0721_5_5_val2_new.jsonl", "r", encoding="utf-8") as f:
# with open (r"D:\2025\知识库建设--检索--精排\验证集\results\bge-emb\bge_m3_emb_0721_5_5_val2_new.jsonl", "r", encoding="utf-8") as f:
with open (r"D:\2025\知识库建设--检索--精排\验证集\results\0730测试\rerank\bge-m3_emb-0721-qwen3-0.6b-reranker_base_val2_new.jsonl", "r", encoding="utf-8") as f:
# with open (r"D:\2025\知识库建设--检索--精排\验证集\验证结果\bge_m3_embedding_0708_5_5_new_val.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f.readlines()]
    # data = json.load(f)



def evaluate_retrieval_performance(data):
    """评估召回内容的 Top1、Top10 命中率以及 MRR"""
    top1_hits = 0
    top10_hits = 0
    mrr_sum = 0
    total_queries = 0

    for d in data:
        bad_item = {}
        good_item = {}
        query = d.get("query", "")
        pos = d.get("pos", "") # 正确答案
        print(pos)
        recall = d.get("recall", "")
        # print(pos)
        # print(recall)
        if not query or not pos:
            continue

        print(f"评估查询: {query}")
        total_queries += 1

        # 提取结果文本
        result_texts = recall

        # 计算 Top1 命中
        flag_top1 = 0
        for pos_item in pos:
            # if recall[0].replace(' ','') == (pos_item+'\n').replace(' ',''):
            if recall[0].replace(' ','') == (pos_item).replace(' ',''):
                flag_top1 = 1
                break
        top1_hits += flag_top1
        # if result_texts and result_texts[0] in pos:
        #     top1_hits += 1

        # 计算 Top10 命中
        flag_top10 = 0
        rank_list = []
        for pos_item in pos:
            # for recall_item in recall[:10]:
            for recall_item in recall[:3]:  #统计top3的准确率
                # if (pos_item + '\n').replace(' ', '') == recall_item.replace(' ', ''):
                if (pos_item).replace(' ', '') == recall_item.replace(' ', ''):
                    flag_top10 = 1
                    break
            rank_list.append(recall.index(recall_item) + 1)
        if flag_top10:
            top10_hits += 1
            rank = min(rank_list)
            mrr_sum += 1.0 / rank
            good_item["query"] = query
            good_item["pos"] = pos
            good_item["recall"] = recall[:10]
            with open(r'D:\2025\知识库建设--检索--精排\验证集\results\0730测试\分析结果\goodcase_qwen3-0.6b-rerank-base_val2_top3.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(good_item, ensure_ascii=False) + '\n')
        if flag_top10 == 0:
            bad_item["query"] = query
            bad_item["pos"] = pos
            bad_item["recall"] = recall[:10]
            with open(r'D:\2025\知识库建设--检索--精排\验证集\results\0730测试\分析结果\badcase_qwen3-0.6b-rerank-base_val2_top3.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(bad_item, ensure_ascii=False) + '\n')
    # 计算指标
    top1_hit_rate = top1_hits / total_queries if total_queries > 0 else 0
    top10_hit_rate = top10_hits / total_queries if total_queries > 0 else 0
    mrr = mrr_sum / total_queries if total_queries > 0 else 0

    print(f"\n评估结果:")
    print(f"总查询数: {total_queries}")
    print(f"Top1 命中率: {top1_hit_rate:.4f} ({top1_hits}/{total_queries})")
    print(f"Top3 命中率: {top10_hit_rate:.4f} ({top10_hits}/{total_queries})")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")

    # # 保存评估结果到文件
    # with open("/opt/ml/input/data_cxy/liuxx/知识库/eval/result/retrieval_evaluation_0708.json", "w", encoding="utf-8") as f:
    #     json.dump({
    #         "total_queries": total_queries,
    #         "top1_hit_rate": top1_hit_rate,
    #         "top10_hit_rate": top10_hit_rate,
    #         "mrr": mrr,
    #         "top1_hits": top1_hits,
    #         "top10_hits": top10_hits
    #     }, f, ensure_ascii=False, indent=2)

    return top1_hit_rate, top10_hit_rate, mrr


a, b, c = evaluate_retrieval_performance(data)
print(a)
print(b)
print(c)

