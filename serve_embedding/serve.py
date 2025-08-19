import logging
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Union
from FlagEmbedding import BGEM3FlagModel, FlagAutoModel
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# Initialize FastAPI app
app = FastAPI()
'''
# Load model only once at startup
model = FlagAutoModel.from_finetuned(
    '/opt/ml/input/data/wangc/wangc/models/bge-m3',
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True,
    devices=['cuda:0']
)
'''
model_embedding_path = '/opt/ml/input/data/wangc/wangc/models/bge-m3'
model = BGEM3FlagModel(model_embedding_path, use_fp16=True, device="cuda")

class SentencesRequest(BaseModel):
    sentences: List[str]

class QueryRequest(BaseModel):
    query: str

@app.post("/encode_corpus")
def encode_corpus(request: SentencesRequest):
    """
    Encode a list of sentences and return their dense vectors.
    """
    embeddings = model.encode_corpus(request.sentences)['dense_vecs']
    response = {
     "code": 0,
     "data": [{"value": embedding.tolist()} for embedding in embeddings]
     }
    return response

@app.post("/encode_query")
def encode_query(request: QueryRequest):
    """
    Encode a single query string and return its dense vector.
    """
    embedding = model.encode_corpus(request.query)['dense_vecs']
    return {"embedding": embedding.tolist()}
if __name__ == "__main__":
    _ = model.encode_corpus(["预热"])["dense_vecs"]
    uvicorn.run(app, host="0.0.0.0", port=8080)
    #_ = model.encode_corpus(["预热"])["dense_vecs"]
# # 调用实例
# curl -X POST "http://localhost:8000/encode_query" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "query": "什么是人工智能？"
# }'
#
# curl -X POST "http://localhost:8000/encode_corpus" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "sentences": ["你好世界", "人工智能很有趣"]
# }'

