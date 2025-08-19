"""
Example of using the OpenAI entrypoint's rerank API which is compatible with
the Cohere SDK: https://github.com/cohere-ai/cohere-python

run: vllm serve BAAI/bge-reranker-base
"""
import cohere

# cohere v1 client
co = cohere.Client(base_url="http://localhost:8080", api_key="sk-fake-key")
rerank_v1_result = co.rerank(
    model="qwen-reranker",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(rerank_v1_result)

