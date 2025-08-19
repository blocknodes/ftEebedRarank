from FlagEmbedding import FlagReranker
import json
from tqdm import tqdm
import sys



with open(sys.argv[2], "r", encoding='utf-8') as f:
    data = [json.loads(l) for l in f]
    #data = json.load(f)
def compute_score(reranker, compute_list):
    return reranker.compute_score(compute_list)

if __name__ == "__main__":

    #reranker = FlagReranker('/workspace/FlagEmbedding/examples/finetune/reranker/encoder_only/test_encoder_only_base_bge-reranker-m3-0426/checkpoint-2300', use_fp16=True)
    reranker = FlagReranker(sys.argv[1], use_fp16=True)
    #reranker = FlagReranker('/data/bge-reranker-v2-m3/', use_fp16=True)
    #if __name__ == "__main__":
    final_data = []
    for d in tqdm(data):
        query = d['query']
        pos = d['pos']
        recall = d['recall'][:10]
        compute_list = []
        for doc in recall:
            compute_list.append([query, doc])
        #score = reranker.compute_score(compute_list)
        score = compute_score(reranker, compute_list)
        # sort doc by score
        sorted_results = []
        for i, doc in enumerate(recall):
            sorted_results.append((doc, score[i]))
        # 按照得分降序排序
        sorted_results.sort(key=lambda x: x[1], reverse=True)

        # 如果需要，可以将排序后的结果保存到一个新的列表中
        sorted_docs = [item[0] for item in sorted_results]

        final_data.append({'query': query, "pos": pos, "rank":sorted_docs})

    with open("./bge_reranker_merge_0616.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
