import sys
import json
import requests
from datetime import datetime
from time import sleep

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
        # 动态生成当前时间戳（毫秒），替代硬编码值
        "current_timestamp": int(datetime.now().timestamp() * 1000)
    },
    "rerank_api": {
        "url": "https://inner-apisix-test.hisense.com/hiaii/rerank?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0",
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

def get_filename_by_block_id(target_block_id):
    """根据block_id从Elasticsearch获取文件名"""
    try:
        url = f"{CONFIG['es']['url']}/{CONFIG['es']['indices']['block_data']}/_search"
        query_body = {
            "query": {
                "term": {
                    "block_id": {"value": target_block_id}
                }
            },
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
            "_source": {
                "excludes": ["sentence_embedding", "segment_embedding"]
            },
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


def call_rerank_api(query, documents):
    """调用重排序API对文档进行重新排序"""
    try:
        data = {"documents": documents, "query": query}
        response = requests.post(
            CONFIG['rerank_api']['url'],
            headers=CONFIG['rerank_api']['headers'],
            data=json.dumps(data)
        )
        response.raise_for_status()
        return response.json()

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
                        "match": {
                            content_field: {"query": query_string}
                        }
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

    # 修复占位符替换错误
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
        # 以追加模式打开文件，确保中文等特殊字符正确处理
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将数据序列化为JSON字符串并写入，确保不添加多余空格
            json.dump(data, f, ensure_ascii=False)
            # 写入换行符，确保每条记录占一行
            f.write('\n')
        print(f"成功向{file_path}添加一行数据")
    except Exception as e:
        print(f"操作失败: {str(e)}")


# 新增：单独处理单个查询的函数（提取原有逻辑）
def process_single_query(query, output_dir='output'):
    # 1. 查询重写
    success = False
    for i in range(10):
        new_query = query_rewrite(query)
        if not new_query:
            print(f"查询优化失败，无法继续:{new_query}", file=sys.stderr)
            sleep(i)
        else:
            success = True
            break

    if not success:
        print("查询优化失败，无法继续", file=sys.stderr)
        sys.exit(1)
    # 2. 文本向量化
    vectors = vectorize_text([new_query])
    if not vectors or 'data' not in vectors or not vectors['data']:
        print("向量转换失败，无法继续", file=sys.stderr)
        sys.exit(1)
    query_vector = vectors['data'][0]['value']

    # 3. 初始化存储结构
    segments = set()
    segments2file = {}

    # 4. 第一阶段：sentence级检索→转换为segment
    sentence_recall_result={'query':query ,'query_rewritten': new_query,'value':[]}
    sentence_results = search_elasticsearch(query_vector, new_query)
    if sentence_results and isinstance(sentence_results, list):
        for hit in sentence_results:
            if not isinstance(hit, dict) or 'segment_id' not in hit or 'dir_id' not in hit:
                print(f"无效的sentence结果: {hit}，跳过处理", file=sys.stderr)
                continue

            segment_result = query_es_by_segment_id(hit['segment_id'], hit['dir_id'])

            process_items(segment_result, segments, segments2file)
            item = {'sentence': hit['content'], 'sentence_score':hit['score'] ,
                    'segment':segment_result[0]['content'],
                    'segment_score':segment_result[0]['score'],
                    'file_name': segments2file[segment_result[0]['content']]}
            sentence_recall_result['value'].append(item)
    # 按照sentence_score降序排列，如果需要按segment_score排序可以替换键
    sentence_recall_result['value'].sort(key=lambda x: x['sentence_score'], reverse=True)
    append_to_jsonl(f'{output_dir}/sentence_recall_result.jsonl', sentence_recall_result)


    # 5. 第二阶段：直接检索segments
    segment_recall_result={'query':query ,'query_rewritten': new_query,'value':[]}
    direct_segment_results = search_segments_from_elasticsearch(query_vector, new_query)
    process_items(direct_segment_results, segments, segments2file)
    for hit in direct_segment_results:
        item = {'segment':hit['content'],
                'segment_score': hit['score'],
                'file_name': segments2file[hit['content']]}
        segment_recall_result['value'].append(item)
    segment_recall_result['value'].sort(key=lambda x: x['segment_score'], reverse=True)
    append_to_jsonl(f'{output_dir}/segment_recall_result.jsonl', segment_recall_result)


    # 6. 重排序及结果整理
    if not segments:
        print("未检索到有效片段", file=sys.stderr)
        sys.exit(1)

    docs = list(segments)
    success = False
    for i in range(10):
        rerank_result = call_rerank_api(new_query, docs)
        if not rerank_result or 'data' not in rerank_result or not rerank_result['data']:
            print("重排序失败，retry", file=sys.stderr)
            sleep(i)
        else:
            success = True
            break

    if not success:
        print("重排序失败，无法继续", file=sys.stderr)
        sys.exit(1)


    scores_with_index = rerank_result["data"][0]["value"]
    docs_with_scores = []
    for item in scores_with_index:
        if not isinstance(item, dict) or 'index' not in item or 'relevance_score' not in item:
            print(f"无效的重排序结果: {item}，跳过处理", file=sys.stderr)
            continue

        idx = item["index"]
        if idx < 0 or idx >= len(docs):
            print(f"无效的文档索引: {idx}，跳过处理", file=sys.stderr)
            continue

        docs_with_scores.append({
            "document": docs[idx],
            "relevance_score": item["relevance_score"],
            "filename": segments2file.get(docs[idx], "未知文件名")
        })

    # 按相关性降序排列
    sorted_docs = sorted(docs_with_scores, key=lambda x: x["relevance_score"], reverse=True)

    # 输出结果
    result = {
        'query': query,
        'query_rewritten': new_query,
        'value': sorted_docs
    }

    append_to_jsonl(f'{output_dir}/rerank_result.jsonl', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        print("请提供JSONL文件路径", file=sys.stderr)
        sys.exit(1)

    jsonl_path = sys.argv[1]
    try:
        # 读取JSONL文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 解析JSON行，假设每行包含"query"字段
                    data = json.loads(line)
                    query = data.get('query')
                    if not query:
                        print(f"第{line_num}行缺少'query'字段，跳过", file=sys.stderr)
                        continue

                    # 处理单个查询（复用原有逻辑）
                    process_single_query(query)

                except json.JSONDecodeError:
                    print(f"第{line_num}行JSON格式错误，跳过", file=sys.stderr)

    except FileNotFoundError:
        print(f"文件不存在: {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"文件读取错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
