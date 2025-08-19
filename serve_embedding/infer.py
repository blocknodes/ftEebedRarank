import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# url = "http://localhost:8080/encode_query"
url = "http://aitc.hisense.com:8080/encode_query"
headers = {"Content-Type": "application/json"}
data = {
    "query": "帮我推荐一本适合夏天全家人看的电影。"
}

def send_request():
    start = time.time()
    response = requests.post(url, json=data, headers=headers)
    elapsed = time.time() - start
    return response.status_code, response.text, elapsed

num_requests = 10  # 10 QPS

with ThreadPoolExecutor(max_workers=num_requests) as executor:
    start_time = time.time()
    futures = [executor.submit(send_request) for _ in range(num_requests)]
    results = []
    for future in as_completed(futures):
        status, text, elapsed = future.result()
        results.append((status, text, elapsed))
    end_time = time.time()

# 输出结果
for idx, (status, text, elapsed) in enumerate(results):
    print(f"Request {idx+1}: Status={status}, Time={elapsed:.4f}s")
    # print(f"Request {idx+1}: Status={status}, Time={elapsed:.4f}s")
    if elapsed > 0.1:
        print("⚠️ 该请求超过0.1秒")
    # print(f"Response: {text}\n")  # 如需看返回内容可取消注释

total_time = end_time - start_time
print(f"\n总耗时: {total_time:.4f}s, 平均QPS: {num_requests/total_time:.2f}")

if total_time <= 1.0:
    print("✅ 达到10 QPS")
else:
    print("❌ 未达到10 QPS")