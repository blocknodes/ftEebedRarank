import requests
import json

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

# 示例用法
if __name__ == "__main__":
    # 测试查询和文档
    test_query = "你好"
    test_docs = ["你好", "世界", "欢迎使用"]

    # 调用函数
    result = call_rerank_api(test_query, test_docs)

    # 打印结果
    if result:
        print("请求成功，响应内容：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("请求失败")
