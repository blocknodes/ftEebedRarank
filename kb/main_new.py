import sys
import requests
import json

def get_filename_by_block_id(target_block_id):
    # 配置信息
    es_url = "http://10.18.219.171:9200"
    index_name = "block-data-prod-000001"
    username = "elastic"
    password = "8ktbepQdRJVWjw@B"
    #target_block_id = "6f4d3b9457e140cfafc5561930ddf845"  # 目标block_id

    # 构建请求
    url = f"{es_url}/{index_name}/_search"
    query_body = {
        "query": {
            "term": {
                "block_id": {
                    "value": target_block_id
                }
            }
        },
        "_source": ["fileName"]
    }

    try:
        response = requests.get(
            url,
            json=query_body,
            auth=(username, password),
            params={"pretty": "true"}
        )
        response.raise_for_status()
        result = response.json()

        # 处理结果
        if result["hits"]["total"]["value"] > 0:
            return result["hits"]["hits"][0]["_source"].get("fileName")
        else:
            return f"未找到block_id为 {target_block_id} 的记录"

    except requests.exceptions.RequestException as e:
        return f"查询出错: {str(e)}"


def search_segments_from_elasticsearch(query_vector, query_string):
    """
    发送KNN查询到Elasticsearch

    参数:
        query_vector: 用于KNN搜索的向量列表
        query_string: 用于文本匹配的查询字符串

    返回:
        响应的JSON数据，如果请求失败则返回None
    """
    # Elasticsearch配置
    es_url = "http://10.18.219.171:9200/segment-embedding-data-000001/_search"
    username = "elastic"
    password = "8ktbepQdRJVWjw@B"  # 注意解码百分号编码后的原始密码

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求体
    payload = {
        "_source": {
            "excludes": [
                "sentence_embedding",
                "segment_embedding"
            ]
        },
        "knn": {
            "k": 16,
            "boost": 24,
            "num_candidates": 100,
            "field": "segment_embedding",
            "query_vector": query_vector  # 使用传入的向量参数
        },
        "query": {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "effective_time": {
                                            "lt": 1755846988292
                                        }
                                    }
                                },
                                {
                                    "range": {
                                        "expire_time": {
                                            "gt": 1755846988292
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ],
                "must": [
                    {
                        "match": {
                            "segment_content": {
                                "query": query_string  # 使用传入的查询字符串参数
                            }
                        }
                    }
                ]
            }
        },
        "size": 20
    }

    try:
        # 发送请求
        response = requests.post(
            es_url,
            auth=(username, password),
            headers=headers,
            data=json.dumps(payload)
        )

        # 检查响应状态
        response.raise_for_status()
        response = response.json()

        ### for now, we need only scre/ner/content
        result = []
        for hit in response['hits']['hits']:
            result.append({'score':hit['_score'], 'content': hit['_source']['segment_content'],
                'ner_n_entity':hit['_source']['ner_n_entity'],
                'ner_v_entity':hit['_source']['ner_v_entity'],
                'ner_exn_entity':hit['_source']['ner_exn_entity'],
                'block_id':hit['_source']['block_id']})

        # 解析并返回响应结果
        return result

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None


def query_es_by_segment_id(segment_id, dir_id):
    """
    查询Elasticsearch中指定segment_id和dir_id的文档

    参数:
        segment_id: 要查询的segment_id
        dir_id: 目录ID，默认值为342772650690289664

    返回:
        包含查询结果的字典，如果出错则返回None
    """
    # Elasticsearch地址和认证信息
    url = "http://10.18.219.171:9200/segment-embedding-data-000001/_search"
    auth = ("elastic", "8ktbepQdRJVWjw@B")  # 注意解码后的密码

    # 构建查询体
    payload = {
        "_source": {
            "excludes": [
                "sentence_embedding",
                "segment_embedding"
            ]
        },
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "segment_id": {
                                "value": segment_id
                            }
                        }
                    },
                    {
                        "term": {
                            "dir_id": {
                                "value": dir_id
                            }
                        }
                    }
                ]
            }
        }
    }

    try:
        # 发送请求
        response = requests.post(
            url,
            auth=auth,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        # 检查响应状态
        response.raise_for_status()
        result = []

        for hit in response.json()['hits']['hits']:
            result.append({'score':hit['_score'], 'content': hit['_source']['segment_content'],
                'ner_n_entity':hit['_source']['ner_n_entity'],
                'ner_v_entity':hit['_source']['ner_v_entity'],
                'ner_exn_entity':hit['_source']['ner_exn_entity'],
                'block_id':hit['_source']['block_id']})


        print(result)
        # 返回解析后的JSON
        return result

    except requests.exceptions.RequestException as e:
        print(f"查询出错: {e}")
        return None

def call_rerank_api(query, documents):
    """
    调用rerank接口的函数

    参数:
        query (str): 查询字符串
        documents (list): 文档列表，每个元素为字符串

    返回:
        dict: 接口返回的JSON数据，如果请求失败则返回None
    """
    # 接口URL
    url = "https://inner-apisix-test.hisense.com/hiaii/rerank?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0"

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Cookie": "BIGipServerPOOL_OCP_JUCLOUD_DEV80=!+tLUVeluJWXzlZLVZekhhPIyzDN0Vem6oMaHLCwK6cswdpAa2lBxosUP75seeZQfBYHlqA8nc+MiuYY="
    }

    # 请求数据
    data = {
        "documents": documents,
        "query": query
    }

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # 检查响应状态码
        response.raise_for_status()

        # 返回解析后的JSON数据
        return response.json()

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP错误: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"连接错误: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"超时错误: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"其他错误: {err}")

    return None

def search_elasticsearch(query_vector, search_query, size=20):
    """
    发送参数化的请求到Elasticsearch

    参数:
        query_vector: 用于KNN搜索的向量列表
        search_query: 用于文本匹配的查询字符串
        size: 返回结果的数量，默认20
    """
    # Elasticsearch连接信息
    es_url = "http://10.18.219.171:9200/sentence-embedding-data-000001/_search"
    username = "elastic"
    password = "8ktbepQdRJVWjw@B"  # 注意原URL中%40是@的URL编码

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求体数据 - 使用参数化的值
    payload = {
        "_source": {
            "excludes": [
                "sentence_embdding",
                "segment_embdding"
            ]
        },
        "knn": {
            "k": 16,
            "boost": 24,
            "num_candidates": 100,
            "field": "sentence_embedding",
            "query_vector": query_vector  # 参数化的向量
        },
        "query": {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "effective_time": {
                                            "lt": 1755846988292
                                        }
                                    }
                                },
                                {
                                    "range": {
                                        "expire_time": {
                                            "gt": 1755846988292
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ],
                "must": [
                    {
                        "match": {
                            "sentence_content": {
                                "query": search_query  # 参数化的查询字符串
                            }
                        }
                    }
                ]
            }
        },
        "size": size  # 参数化的结果数量
    }

    try:
        # 发送请求
        response = requests.post(
            es_url,
            auth=(username, password),
            headers=headers,
            data=json.dumps(payload)
        )

        # 检查响应状态
        response.raise_for_status()

        response = response.json()

        ### for now, we need only scre/tag/content
        result = []
        for hit in response['hits']['hits']:
            result.append({'score':hit['_score'], 'content': hit['_source']['sentence_content'],
                'tags':hit['_source']['tags'],
                'segment_id':hit['_source']['segment_id'],
                'dir_id':hit['_source']['dir_id']})

        # 解析并返回响应结果
        return result

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"解析响应JSON失败: {e}")
        return None

def vectorize_text(docs):
    # API端点URL
    url = "https://inner-apisix-test.hisense.com/hiai/vectorize?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0"

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Cookie": "BIGipServerPOOL_OCP_JUCLOUD_DEV80=!kszdw5vSDwVhwpfVZekhhPIyzDN0VUEESYyjq+3xAIqTfo1X+BHgT2qyf7IFzzS3EKQEtuascy75His="
    }

    # 请求数据
    data = {
        "documents": docs
    }

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # 检查响应状态码
        response.raise_for_status()

        # 解析并返回响应结果
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None

def qeury_rewrite(query):
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
    {"rewritten_query": ["关键词1", "关键词2", "关键词3”]}
    ```

    10. **输出示例2**：当输入为“怎么收费啊”
    ```
    {"rewritten_query": ["无主语"]}
    ```

    **你的任务**
    现在请处理以下用户输入：{{QUERY}}"""

    prompt = prompt.replace('{QUERY}',query)


    url = "http://10.18.217.60:31975/v1/biz/completion"

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    data = {
        "sceneType": "information_extract",
        "model": "xinghai_aliyun_deepseek_v3",
        "query": prompt,  # 使用传入的query参数
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

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 检查HTTP错误状态码
        result = response.json()
        print("请求成功:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        message_content = result['choices'][0]['message']['content']

        # 解析JSON内容
        content_json = json.loads(message_content)

        # 提取rewritten_query
        rewritten_queries = content_json.get('rewritten_query', [])

        print("提取的rewritten_query结果：")
        for key in rewritten_queries:
            print(f"- {key}")

        return " ".join(rewritten_queries)
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except json.JSONDecodeError:
        print("响应内容不是有效的JSON")
        print("响应内容:", response.text)
    return None

if __name__ == "__main__":

    query = sys.argv[1]
    #### query rewrite ####
    new_query = qeury_rewrite(query)

    #### vectorize #####
    vectors = vectorize_text([new_query])

    ##### es search #####
    results = search_elasticsearch(vectors['data'][0]['value'], new_query)

    ### we only needs segments
    segments = set()

    segments2file={}

    for hit in results:
        result = query_es_by_segment_id(hit['segment_id'],hit['dir_id'])
        for item in result:
            segments.add(item['content'])
            if item['content'] not in segments2file.keys():
                segments2file[item['content']] = get_filename_by_block_id(item['block_id'])
            #blockids.add(item['block_id'])



    ##### search segments directly
    result = search_segments_from_elasticsearch(vectors['data'][0]['value'], new_query)

    for hit in result:
        segments.add(hit['content'])
        if hit['content'] not in segments2file.keys():
            segments2file[hit['content']] = get_filename_by_block_id(hit['block_id'])
        #blockids.add(item['block_id'])




    print(segments2file)
    docs = list(segments)
    result = call_rerank_api(new_query, docs)
    # 提取分数和索引信息
    scores_with_index = result["data"][0]["value"]

    # 创建一个包含文档、索引和分数的列表
    docs_with_scores = []
    for item in scores_with_index:
        idx = item["index"]
        score = item["relevance_score"]
        docs_with_scores.append({
            "document": docs[idx],
            "relevance_score": score,
            "filename":segments2file[docs[idx]]
        })

    # 按照相关性分数升序排列
    sorted_docs = sorted(docs_with_scores, key=lambda x: x["relevance_score"], reverse=True )

    result={'orig_query':query,'new_query':new_query,'retrieved_docs':sorted_docs}
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_str)
    #print(result)






