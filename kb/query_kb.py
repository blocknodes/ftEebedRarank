import sys
import json
import os
import requests
from time import sleep
from datetime import datetime
from tqdm import tqdm  # 用于显示进度条
import argparse  # 新增：导入argparse模块

# 配置信息集中管理
CONFIG = {
    "es": {
        "url": "http://10.18.219.171:9200",
        "indices": {
            "block_data": "block-data-prod-000001",
            "segment_embedding": "segment-embedding-data-000001",
            "sentence_embedding": "sentence-embedding-data-000001"
        },
        "auth": ("elastic", "8ktbepQdRJVWjw@B"),
        # 动态生成当前时间戳（毫秒）
        "current_timestamp": int(datetime.now().timestamp() * 1000)
    },
    "rerank_api": {
        "url": "http://10.19.98.208:4123/rerank",
        #"url": "https://inner-apisix-test.hisense.com/hiaii/rerank?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0",
        "headers": {
            "Content-Type": "application/json",
            "Cookie": "BIGipServerPOOL_OCP_JUCLOUD_DEV80=!+tLUVeluJWXzlZLVZekhhPIyzDN0Vem6oMaHLCwK6cswdpAa2lBxosUP75seeZQfBYHlqA8nc+MiuYY="
        }
    },
    "vectorize_api": {
        "url": "https://inner-apisix-test.hisense.com/hiai/vectorize?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0",
        "headers": {
            "Content-Type": "application/json",
            "Cookie": "BIGipServerPOOL_OCP_JUCLOUD_DEV80=!kszdw5vSDwVhwpfVZekhhPIyzDN0VUEESYyjq+3xAIqTfo1X+BHgT2qyf7IFzzS3EKQEtuascy75His="
        }
    },
    "query_rewrite_api": {
        "url": "http://10.18.217.60:31975/v1/biz/completion",
        "headers": {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
    }
}


def merge_sorted_lists(list1, list2):
    """
    合并两个按'relevance_score'键降序排列的字典列表

    参数:
        list1: 第一个按'relevance_score'降序排列的字典列表
        list2: 第二个按'relevance_score'降序排列的字典列表

    返回:
        合并后的按'relevance_score'降序排列的字典列表
    """
    merged = []
    i = j = 0

    # 双指针遍历两个列表，比较并合并
    while i < len(list1) and j < len(list2):
        if list1[i]['relevance_score'] >= list2[j]['relevance_score']:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    # 添加剩余元素
    merged.extend(list1[i:])
    merged.extend(list2[j:])

    return merged


def query_elasticsearch(query_vector, query, size=20):
    """
    向Elasticsearch发送查询请求，获取qna相关结果

    参数:
        query_vector: 用于KNN搜索的向量
        query: 用于匹配qna_title的查询字符串
        size: 返回结果的数量，默认20

    返回:
        包含查询结果的字典列表，如果请求失败则返回None
    """
    # Elasticsearch配置
    username = "elastic"
    password = "8ktbepQdRJVWjw@B"
    es_host = "10.18.219.171:9200"
    index_name = "qna-embedding-data-000001"

    # 构建包含认证信息的请求URL
    url = f"http://{username}:{password}@{es_host}/{index_name}/_search"

    # 获取当前时间戳（毫秒级）
    current_timestamp = int(datetime.now().timestamp() * 1000)

    # 构建请求体
    payload = {
        "_source": {"excludes": ["qna_embedding"]},
        "knn": {
            "k": 16,
            "boost": 24,
            "num_candidates": 100,
            "field": "qna_embedding",
            "query_vector": query_vector
        },
        "query": {
            "bool": {
                "filter": [{
                    "bool": {
                        "must": [
                            {"range": {"effective_time": {"lt": current_timestamp}}},
                            {"range": {"expire_time": {"gt": current_timestamp}}}
                        ]
                    }
                }],
                "must": [{
                    "match": {"qna_title": {"query": query}}
                }]
            }
        },
        "size": size
    }

    # 设置请求头
    headers = {"Content-Type": "application/json"}

    try:
        # 发送请求
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()

        # 处理返回结果
        result = []
        for hit in response_data['hits']['hits']:
            item = {
                'score': hit['_score'],
                'qna_title': hit['_source']['qna_title'],
                'qna_content': hit['_source']['qna_content']
            }
            result.append(item)

        return result
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None


def get_filename_by_block_id(target_block_id):
    """根据block_id从Elasticsearch获取文件名"""
    try:
        url = f"{CONFIG['es']['url']}/{CONFIG['es']['indices']['block_data']}/_search"
        query_body = {
            "query": {"term": {"block_id": {"value": target_block_id}}},
            "_source": ["fileName"]
        }

        response = requests.get(
            url,
            json=query_body,
            auth=CONFIG['es']['auth'],
            params={"pretty": "true"}
        )
        response.raise_for_status()
        result = response.json()

        if result["hits"]["total"]["value"] > 0:
            return result["hits"]["hits"][0]["_source"].get("fileName")
        else:
            return f"未找到block_id为 {target_block_id} 的记录"

    except requests.exceptions.RequestException as e:
        return f"查询出错: {str(e)}"


def search_segments_from_elasticsearch(query_vector, query_string):
    """从Elasticsearch查询相关片段，使用KNN和文本匹配"""
    return search_elasticsearch_generic(
        index_name=CONFIG['es']['indices']['segment_embedding'],
        vector_field="segment_embedding",
        content_field="segment_content",
        query_vector=query_vector,
        query_string=query_string,
        source_excludes=["sentence_embedding", "segment_embedding"],
        additional_fields=["ner_n_entity", "ner_v_entity", "ner_exn_entity", "block_id"]
    )


def query_es_by_segment_id(segment_id, dir_id):
    """根据segment_id和dir_id查询Elasticsearch文档"""
    try:
        url = f"{CONFIG['es']['url']}/{CONFIG['es']['indices']['segment_embedding']}/_search"
        payload = {
            "_source": {"excludes": ["sentence_embedding", "segment_embedding"]},
            "query": {
                "bool": {
                    "must": [
                        {"term": {"segment_id": {"value": segment_id}}},
                        {"term": {"dir_id": {"value": dir_id}}}
                    ]
                }
            }
        }

        response = requests.post(
            url,
            auth=CONFIG['es']['auth'],
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()

        result = []
        for hit in response.json()['hits']['hits']:
            result.append({
                'score': hit['_score'],
                'content': hit['_source']['segment_content'],
                'ner_n_entity': hit['_source']['ner_n_entity'],
                'ner_v_entity': hit['_source']['ner_v_entity'],
                'ner_exn_entity': hit['_source']['ner_exn_entity'],
                'block_id': hit['_source']['block_id']
            })
        return result

    except requests.exceptions.RequestException as e:
        print(f"查询出错: {e}", file=sys.stderr)
        return None


def perform_reranking(query, documents):
    """
    调用重排序API对文档进行重新排序并返回带分数的结果

    参数:
        query: 查询文本
        documents: 待排序的文档列表

    返回:
        带相关性分数的文档列表，按分数降序排列
        每个元素是包含"document"和"relevance_score"的字典
    """
    try:
        # 最多尝试10次
        for i in range(10):
            data = {"documents": documents, "query": query}
            response = requests.post(
                CONFIG['rerank_api']['url'],
                headers=CONFIG['rerank_api']['headers'],
                data=json.dumps(data)
            )
            response.raise_for_status()
            rerank_result = response.json()

            if rerank_result and 'data' in rerank_result and rerank_result['data']:
                # 处理重排序结果
                scores_with_index = rerank_result["data"][0]["value"]
                docs_with_scores = []

                for item in scores_with_index:
                    if not isinstance(item, dict) or 'index' not in item or 'relevance_score' not in item:
                        print(f"无效的重排序结果: {item}，跳过处理", file=sys.stderr)
                        continue

                    idx = item["index"]
                    if idx < 0 or idx >= len(documents):
                        print(f"无效的文档索引: {idx}，跳过处理", file=sys.stderr)
                        continue

                    docs_with_scores.append({
                        "document": documents[idx],
                        "relevance_score": item["relevance_score"]
                    })

                # 按相关性降序排列并返回
                return sorted(docs_with_scores, key=lambda x: x["relevance_score"], reverse=True)
            elif rerank_result and 'scores' in rerank_result:
                scores = rerank_result['scores']
                docs_with_scores = []
                for idx in range(len(scores)):
                    docs_with_scores.append({
                        "document": documents[idx],
                        "relevance_score": scores[idx]
                    })
                # 按相关性降序排列并返回
                return sorted(docs_with_scores, key=lambda x: x["relevance_score"], reverse=True)

            print(f"重排序失败，重试 {i+1}/10", file=sys.stderr)
            sleep(i)

        print("重排序多次尝试失败", file=sys.stderr)
        return None

    except requests.exceptions.RequestException as e:
        print(f"重排序请求错误: {e}", file=sys.stderr)
        return None


def search_elasticsearch(query_vector, search_query, size=20):
    """从Elasticsearch查询相关句子，使用KNN和文本匹配"""
    return search_elasticsearch_generic(
        index_name=CONFIG['es']['indices']['sentence_embedding'],
        vector_field="sentence_embedding",
        content_field="sentence_content",
        query_vector=query_vector,
        query_string=search_query,
        source_excludes=["sentence_embedding", "segment_embedding"],
        additional_fields=["tags", "segment_id", "dir_id"],
        size=size
    )


def search_elasticsearch_generic(index_name, vector_field, content_field,
                                query_vector, query_string, source_excludes,
                                additional_fields, size=20):
    """通用的Elasticsearch查询函数，减少代码重复"""
    try:
        url = f"{CONFIG['es']['url']}/{index_name}/_search"
        payload = {
            "_source": {"excludes": source_excludes},
            "knn": {
                "k": 16,
                "boost": 24,
                "num_candidates": 100,
                "field": vector_field,
                "query_vector": query_vector
            },
            "query": {
                "bool": {
                    "filter": [{
                        "bool": {
                            "must": [
                                {"range": {"effective_time": {"lt": CONFIG['es']['current_timestamp']}}},
                                {"range": {"expire_time": {"gt": CONFIG['es']['current_timestamp']}}}
                            ]
                        }
                    }],
                    "must": [{
                        "match": {content_field: {"query": query_string}}
                    }]
                }
            },
            "size": size
        }

        response = requests.post(
            url,
            auth=CONFIG['es']['auth'],
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        response_data = response.json()

        result = []
        for hit in response_data['hits']['hits']:
            item = {'score': hit['_score'], 'content': hit['_source'][content_field]}
            # 添加额外字段
            for field in additional_fields:
                item[field] = hit['_source'].get(field)
            result.append(item)

        return result

    except requests.exceptions.RequestException as e:
        print(f"Elasticsearch请求错误: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"解析响应JSON失败: {e}", file=sys.stderr)
        return None


def vectorize_text(docs):
    """将文本转换为向量"""
    try:
        data = {"documents": docs}
        response = requests.post(
            CONFIG['vectorize_api']['url'],
            headers=CONFIG['vectorize_api']['headers'],
            data=json.dumps(data)
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"向量转换请求错误: {e}", file=sys.stderr)
        return None


def query_rewrite(query):
    """优化查询语句，提取关键词"""
    prompt = """你是一个专业的搜索查询优化助手，也是一个精准、高效的关键词提取工具，需要将输入的核心关键词、重要的语义词提取出来，提取后的信息需符合搜索引擎的高效检索要求。

    提取时请严格遵循以下要求：
    1. **高度概括性**：关键词应能准确代表文本想表达的核心内容。
    2. **信息承载性强**：保留重要实体，如专业术语、产品名、型号、功能描述、技术名词等。
    3. **避免冗余和解释**：不输出无实际意义的词汇；不添加解释、注释或格式修饰，避免出现过于通用的关键词，例如：操作，功能。
    4. **语言统一**：如原文为中文，则关键词用中文；如为英文则保持英文。
    5. **提取规则**
    - 产品型号（如WF100N2Q-6）
    - 专有名词（公司/产品/法规名称）
    - 重点关注文件名称的内容，其中包含了如产品型号、品牌名等重要信息
    - 可以对关键词进行一定程度的改写，把口语化的内容变成专业化的表达
    6. **必须删除的内容**
    - 疑问词（"是否"/"怎样"）
    - 主观描述（"惊人的"/"快速"）
    - 模糊表述（"一些"/"多种"）
    - 模棱两可的数据
    - 没有主语的内容
    7. **主题提取**：
    - 根据输入内容预测可能属于的领域主题
    8. **输出格式**：仅返回一个 JSON 对象，包含名为 `"keywords"` 的数组，数组中每个元素是一个关键词或短语。必须严格按照json格式输出

    9. **输出示例1**：
    ```
    {"rewritten_query": ["关键词1", "关键词2", "关键词3"]}
    ```

    10. **输出示例2**：当输入为“怎么收费啊”
    ```
    {"rewritten_query": ["无主语"]}
    ```

    **你的任务**
    现在请处理以下用户输入：{{QUERY}}"""

    # 替换占位符
    prompt = prompt.replace('{{QUERY}}', query)

    try:
        data = {
            "sceneType": "information_extract",
            "model": "xinghai_aliyun_deepseek_v3",
            "query": prompt,
            "clientSid": "E43BC9B3AA27,1744167435980",
            "deviceId": "861003009000002000000164c9b3aa27",
            "stream": False,
            "dialogueTurnsMax": 0,
            "history": [],
            "dynamicParam": {},
            "modelAdvance": {
                "temperature": 0.7,
                "topP": 0.8,
                "maxTokens": 2048,
                "enableSearch": False
            },
            "riskRespType": "risk_words",
            "skipInputCheck": True,
            "skipOutputCheck": True,
            "forceInternetFlag": False,
            "enablePromptPrefix": False,
            "searchRagPartition": None
        }

        response = requests.post(
            CONFIG['query_rewrite_api']['url'],
            headers=CONFIG['query_rewrite_api']['headers'],
            data=json.dumps(data)
        )
        response.raise_for_status()
        result = response.json()
        message_content = result['choices'][0]['message']['content']

        # 解析JSON内容
        content_json = json.loads(message_content)
        rewritten_queries = content_json.get('rewritten_query', [])

        return " ".join(rewritten_queries)

    except requests.exceptions.RequestException as e:
        print(f"查询重写请求失败: {e}", file=sys.stderr)
    except json.JSONDecodeError:
        print("响应内容不是有效的JSON", file=sys.stderr)
        print("响应内容:", response.text, file=sys.stderr)
    return None


def process_items(items, segments, segments2file):
    """处理检索结果条目，过滤无效数据并提取segment信息"""
    if not isinstance(items, list):
        return

    for item in items:
        if not isinstance(item, dict) or 'content' not in item or 'block_id' not in item:
            print(f"无效的条目格式: {item}，跳过处理", file=sys.stderr)
            continue

        content = item['content']
        block_id = item['block_id']
        segments.add(content)

        if content not in segments2file:
            segments2file[content] = get_filename_by_block_id(block_id)


def append_to_jsonl(file_path, data):
    """
    向JSONL文件添加一行数据

    参数:
        file_path (str): JSONL文件路径
        data (dict): 要添加的数据，必须是可序列化为JSON的字典
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 以追加模式打开文件，确保中文等特殊字符正确处理
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"操作失败: {str(e)}")


def save_progress(progress_file, line_num):
    """保存当前处理进度到文件"""
    try:
        # 先写入临时文件，成功后再替换，确保原子性
        temp_file = f"{progress_file}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({"last_processed_line": line_num}, f)

        # 原子性替换文件
        os.replace(temp_file, progress_file)
    except Exception as e:
        print(f"保存进度失败: {str(e)}", file=sys.stderr)


def load_progress(progress_file, load_progress_flag):
    """从文件加载上次处理进度"""
    # 如果不加载进度，直接返回0
    if not load_progress_flag:
        return 0

    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                return progress.get("last_processed_line", 0)
    except Exception as e:
        print(f"加载进度失败，将从头开始: {str(e)}", file=sys.stderr)
    return 0


def count_total_lines(file_path):
    """计算文件的总行数，用于进度条显示"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"计算总行数失败: {str(e)}", file=sys.stderr)
        return 0


def process_single_query(query, output_dir='output', use_rewrite=True):
    """处理单个查询，返回处理是否成功"""
    # 1. 查询重写（根据参数决定是否启用）
    new_query = query  # 默认使用原始查询
    if use_rewrite:
        success = False
        for i in range(10):
            rewritten = query_rewrite(query)
            if rewritten:
                new_query = rewritten
                success = True
                break
            print(f"查询优化失败，重试 {i+1}/10", file=sys.stderr)
            sleep(i)

        if not success:
            print("查询优化多次失败，将使用原始查询继续", file=sys.stderr)
            new_query = query  # 回退到原始查询


    # 2. 文本向量化
    vectors = vectorize_text([new_query])
    if not vectors or 'data' not in vectors or not vectors['data']:
        print("向量转换失败，无法继续", file=sys.stderr)
        return False
    query_vector = vectors['data'][0]['value']

    # 3.1 QA搜索
    qa_pair = query_elasticsearch(query_vector, new_query)
    append_to_jsonl(
        f'{output_dir}/qa_recall_result.jsonl',
        {'query': query, 'query_rewritten': new_query, 'value': qa_pair}
    )

    # 3.2 QA结果重排序 - 第一次调用重排序函数
    docs = [item['qna_title'] for item in qa_pair] if qa_pair else []
    sorted_qa = perform_reranking(new_query, docs)

    if not sorted_qa:
        print("QA重排序失败，无法继续", file=sys.stderr)
        return False

    # 补充qna_content信息
    for item in sorted_qa:
        # 找到对应的qna_content
        for qa in qa_pair:
            if qa['qna_title'] == item['document']:
                item['qna_content'] = qa['qna_content']
                item['qna_title'] = item['document']  # 重命名键以保持一致性
                del item['document']  # 删除临时键
                break

    # 输出QA重排序结果
    result = {
        'query': query,
        'query_rewritten': new_query,
        'value': sorted_qa
    }
    append_to_jsonl(f'{output_dir}/rerank_qa_result.jsonl', result)

    # 4. 初始化存储结构
    segments = set()
    segments2file = {}

    # 5. 第一阶段：sentence级检索→转换为segment
    sentence_recall_result = {'query': query, 'query_rewritten': new_query, 'value': []}
    sentence_results = search_elasticsearch(query_vector, new_query)

    if sentence_results and isinstance(sentence_results, list):
        for hit in sentence_results:
            if not isinstance(hit, dict) or 'segment_id' not in hit or 'dir_id' not in hit:
                print(f"无效的sentence结果: {hit}，跳过处理", file=sys.stderr)
                continue

            segment_result = query_es_by_segment_id(hit['segment_id'], hit['dir_id'])
            if not segment_result:
                continue

            process_items(segment_result, segments, segments2file)
            item = {
                'sentence': hit['content'],
                'sentence_score': hit['score'],
                'segment': segment_result[0]['content'],
                'segment_score': segment_result[0]['score'],
                'file_name': segments2file[segment_result[0]['content']]
            }
            sentence_recall_result['value'].append(item)

    # 按照sentence_score降序排列
    sentence_recall_result['value'].sort(key=lambda x: x['sentence_score'], reverse=True)
    append_to_jsonl(f'{output_dir}/sentence_recall_result.jsonl', sentence_recall_result)

    # 6. 第二阶段：直接检索segments
    segment_recall_result = {'query': query, 'query_rewritten': new_query, 'value': []}
    direct_segment_results = search_segments_from_elasticsearch(query_vector, new_query)

    if direct_segment_results:
        process_items(direct_segment_results, segments, segments2file)
        for hit in direct_segment_results:
            item = {
                'segment': hit['content'],
                'segment_score': hit['score'],
                'file_name': segments2file.get(hit['content'], "未知文件名")
            }
            segment_recall_result['value'].append(item)

        segment_recall_result['value'].sort(key=lambda x: x['segment_score'], reverse=True)
        append_to_jsonl(f'{output_dir}/segment_recall_result.jsonl', segment_recall_result)

    # 7. 片段重排序 - 第二次调用重排序函数
    if not segments:
        print("未检索到有效片段", file=sys.stderr)
        return False

    docs = list(segments)
    sorted_docs = perform_reranking(new_query, docs)

    if not sorted_docs:
        print("片段重排序失败，无法继续", file=sys.stderr)
        return False

    # 补充文件名信息
    for item in sorted_docs:
        item['filename'] = segments2file.get(item['document'], "未知文件名")

    # 输出片段重排序结果
    result = {
        'query': query,
        'query_rewritten': new_query,
        'value': sorted_docs
    }
    append_to_jsonl(f'{output_dir}/rerank_result.jsonl', result)

    # 8. 合并结果并输出最终结果
    final_result = merge_sorted_lists(sorted_docs, sorted_qa)
    result = {
        'query': query,
        'query_rewritten': new_query,
        'value': final_result
    }
    append_to_jsonl(f'{output_dir}/final_result.jsonl', result)

    return True  # 处理成功


def main():
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description='处理查询并从Elasticsearch检索相关结果')

    # 添加互斥组：要么处理文件，要么处理单个查询
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input_file', nargs='?', help='包含查询的JSONL文件路径')
    group.add_argument('-q', '--query', help='单个查询内容')

    # 通用参数
    parser.add_argument('-o', '--output', default='output',
                      help='输出目录路径，默认为"output"')
    parser.add_argument('--no-load-progress', action='store_true',
                      help='不加载已有的进度文件，从头开始处理')
    parser.add_argument('--use-rewrite', action='store_true', default=True,
                      help='是否使用查询改写功能（默认启用）')
    parser.add_argument('--no-rewrite', action='store_false', dest='use_rewrite',
                      help='不使用查询改写功能')

    # 解析参数
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    if args.input_file is not None:
        # 文件处理模式
        jsonl_path = args.input_file
        progress_file = f"{jsonl_path}.progress"

        try:
            # 计算总行数（仅非空行）
            total_lines = count_total_lines(jsonl_path)
            if total_lines == 0:
                print("文件为空或无法读取", file=sys.stderr)
                sys.exit(1)

            # 加载上次处理进度
            start_line = load_progress(progress_file, not args.no_load_progress)
            remaining_lines = total_lines - start_line

            # 读取JSONL文件并显示进度条
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                # 移动到上次处理的位置
                for _ in range(start_line):
                    if not f.readline():  # 如果文件已读完
                        if os.path.exists(progress_file):
                            os.remove(progress_file)
                        sys.exit(0)

                # 初始化进度条
                with tqdm(total=total_lines, initial=start_line, unit='行', desc='处理进度') as pbar:
                    for line_num, line in enumerate(f, start_line + 1):
                        line = line.strip()
                        if not line:
                            # 空行也更新进度
                            save_progress(progress_file, line_num)
                            pbar.update(1)
                            continue

                        try:
                            data = json.loads(line)
                            query = data.get('query')
                            if not query:
                                print(f"第{line_num}行缺少'query'字段，跳过", file=sys.stderr)
                                save_progress(progress_file, line_num)
                                pbar.update(1)
                                continue

                            # 处理单个查询，传入use_rewrite参数
                            success = process_single_query(query, args.output, args.use_rewrite)

                            if success:
                                save_progress(progress_file, line_num)
                                pbar.update(1)
                            else:
                                print(f"第{line_num}行处理失败，将在下次运行时重试", file=sys.stderr)
                                sys.exit(1)

                        except json.JSONDecodeError:
                            print(f"第{line_num}行JSON格式错误，跳过", file=sys.stderr)
                            save_progress(progress_file, line_num)
                            pbar.update(1)

            # 所有行处理完毕，删除进度文件
            if os.path.exists(progress_file):
                os.remove(progress_file)

        except FileNotFoundError:
            print(f"文件不存在: {jsonl_path}", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"文件读取错误: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.query is not None:
        # 单查询模式
        print(f"处理查询: {args.query}")
        # 处理单个查询，传入use_rewrite参数
        success = process_single_query(args.query, args.output, args.use_rewrite)
        if success:
            print("查询处理完成")
            sys.exit(0)
        else:
            print("查询处理失败")
            sys.exit(1)


if __name__ == "__main__":
    main()
