import requests
import json

def query_es_by_segment_id(segment_id, dir_id="342772650690289664"):
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

        # 返回解析后的JSON
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"查询出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    test_segment_id = "a76af54380694938bef30a322a41a491"
    result = query_es_by_segment_id(test_segment_id)

    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))

