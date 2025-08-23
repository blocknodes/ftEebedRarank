import requests
import json
def vectorize_text():
    # API端点URL
    url = "https://inner-apisix-test.hisense.com/hiai/vectorize?user_key=nrnwhmx4tkejvdptecujmlq9eclpugw0"
    
    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Cookie": "BIGipServerPOOL_OCP_JUCLOUD_DEV80=!kszdw5vSDwVhwpfVZekhhPIyzDN0VUEESYyjq+3xAIqTfo1X+BHgT2qyf7IFzzS3EKQEtuascy75His="
    }
    
    # 请求数据
    data = {
        "documents": [
            "你好"
        ]
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

if __name__ == "__main__":
    result = vectorize_text()
    if result:
        print("API响应结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
