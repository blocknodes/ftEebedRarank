import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from openai import OpenAI
import openai
import os

from fastapi import FastAPI, HTTPException, Depends
import requests
from pydantic import BaseModel, Field

app = FastAPI()

# 数据模型定义（与Java实体类对应）
class KwBaseMixRetrievalSettingVo(BaseModel):
    top_k: int = Field(default=3, ge=1, le=50)  # 最多返回结果数
    search_mode: str = "HYBRID"  # 检索模式：混合/关键词/向量
    score_threshold: float = 0.0  # 分数阈值
    search_strategy: str = "BROAD"  # 检索策略

class FilterDataConditionVo(BaseModel):
    # 实际业务中补充过滤条件（如文档类型、时间范围等）
    doc_types: Optional[List[str]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

class KwBaseMixRetrievalSearchVo(BaseModel):
    query: str  # 用户查询文本
    user_id: Optional[str] = None  # 用户ID
    org_codes: Optional[List[str]] = None  # 组织编码列表
    retrieval_setting: Optional[KwBaseMixRetrievalSettingVo] = None  # 检索配置
    metadata_condition: Optional[FilterDataConditionVo] = None  # 元数据过滤条件
    searchSource: int = 1  # 检索来源，1表示AGENT
    dirRequestList: Optional[List[Dict[str, Any]]] = None  # 目录请求列表

class MixSearchRetrievalResp(BaseModel):
    """检索结果响应模型"""
    metadata: Dict[str, Any]  # 元数据（如文档ID、来源等）
    score: float  # 相关性分数
    title: str  # 标题
    content: str  # 内容
    source_type: str  # 来源类型（qna/doc/block/llm_qna）
    create_time: str  # 创建时间

class DifyResult(BaseModel):
    """统一响应模型"""
    code: int = 0
    message: str = "success"
    data: Optional[Any] = None

    @classmethod
    def of_success(cls, data: Any):
        return cls(data=data)

    @classmethod
    def of_fail(cls, code: int, message: str):
        return cls(code=code, message=message)

# 模拟服务实现（同步调用）
class KwBaseQnaSearchService:
    @staticmethod
    def search_qna_retrieval(request: KwBaseMixRetrievalSearchVo) -> List[MixSearchRetrievalResp]:
        """模拟QNA检索"""
        print(f"执行QNA检索: {request.query}")
        # 实际场景中这里会调用真实的QNA检索服务
        return [
            MixSearchRetrievalResp(
                metadata={"doc_id": f"qna_{i+1}", "source": "faq"},
                score=0.9 - i*0.1,
                title=f"Q&A结果{i+1}: {request.query}",
                content=f"这是关于「{request.query}」的Q&A回答内容{i+1}",
                source_type="qna",
                create_time="2023-10-01T12:00:00"
            ) for i in range(3)
        ]

class KwBaseLlmQnaSearchService:
    @staticmethod
    def search_qna_retrieval(request: KwBaseMixRetrievalSearchVo) -> List[MixSearchRetrievalResp]:
        """模拟LLM增强QNA检索"""
        print(f"执行LLM QNA检索: {request.query}")
        return [
            MixSearchRetrievalResp(
                metadata={"doc_id": f"llm_qna_{i+1}", "source": "llm"},
                score=0.85 - i*0.1,
                title=f"LLM Q&A结果{i+1}: {request.query}",
                content=f"这是LLM生成的关于「{request.query}」的回答{i+1}",
                source_type="llm_qna",
                create_time="2023-10-02T14:30:00"
            ) for i in range(2)
        ]

class KwBaseDocumentBlockSearchService:
    @staticmethod
    def search_document_retrieval(request: KwBaseMixRetrievalSearchVo) -> List[MixSearchRetrievalResp]:
        """模拟文档块检索"""
        print(f"执行文档块检索: {request.query}")
        return [
            MixSearchRetrievalResp(
                metadata={"doc_id": f"block_{i+1}", "source": "document"},
                score=0.8 - i*0.1,
                title=f"文档块{i+1}: {request.query}",
                content=f"文档中关于「{request.query}」的相关片段内容{i+1}",
                source_type="block",
                create_time="2023-10-03T09:15:00"
            ) for i in range(2)
        ]

class KwBaseDocumentSearchService:
    @staticmethod
    def search_document_retrieval(request: KwBaseMixRetrievalSearchVo) -> List[MixSearchRetrievalResp]:
        """模拟文档检索"""
        print(f"执行文档检索: {request.query}")
        return [
            MixSearchRetrievalResp(
                metadata={"doc_id": f"doc_{i+1}", "source": "document"},
                score=0.75 - i*0.1,
                title=f"文档{i+1}: {request.query}",
                content=f"完整文档中关于「{request.query}」的内容摘要{i+1}",
                source_type="doc",
                create_time="2023-10-04T16:45:00"
            ) for i in range(3)
        ]

class KwBaseMixSearchService:
    @staticmethod
    def handle_sort_mode(
        request: KwBaseMixRetrievalSearchVo,
        qna_list: List[MixSearchRetrievalResp],
        doc_list: List[MixSearchRetrievalResp],
        block_list: List[MixSearchRetrievalResp],
        llm_qna_list: List[MixSearchRetrievalResp]
    ) -> List[MixSearchRetrievalResp]:
        """混合结果排序处理"""
        print("开始混合结果排序")

        # 合并所有结果
        all_results = qna_list + doc_list + block_list + llm_qna_list

        # 按分数降序排序
        sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)

        # 应用阈值过滤
        threshold = request.retrieval_setting.score_threshold if request.retrieval_setting else 0.0
        filtered_results = [r for r in sorted_results if r.score >= threshold]

        # 限制返回数量
        top_k = request.retrieval_setting.top_k if request.retrieval_setting else 3
        final_results = filtered_results[:top_k]

        print(f"排序完成，返回{len(final_results)}条结果")
        return final_results

# 权限验证依赖（同步实现）
def get_open_api_client() -> Dict[str, Any]:
    """模拟权限验证，实际应从请求头/token解析"""
    # 调试时可直接返回管理员权限，生产环境需严格验证
    return {"user_id": "debug_user", "authorities": ["ADMIN"]}

# 大模型查询改写（同步实现）
def rewrite_query(text: str, config: Dict[str, Any]) -> str:
    """调用大模型改写查询文本，增强检索效果"""
    print(f"原始查询: {text}")
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

    prompt = prompt.replace('{QUERY}',text)


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
        return result
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except json.JSONDecodeError:
        print("响应内容不是有效的JSON")
        print("响应内容:", response.text)
    return None




# 核心接口实现（同步版）
@app.post("/openapi/kbp/mix/retrieval", response_model=DifyResult)
def search_mix_retrieval(
    request: KwBaseMixRetrievalSearchVo,
    client_info: Dict[str, Any] = Depends(get_open_api_client)
):
    """同步版混合检索接口，便于调试"""
    # 记录开始时间（便于调试性能）
    start_time = datetime.now()
    print(f"\n===== 混合检索开始: {start_time} =====")
    print(f"请求参数: {json.dumps(request.dict(), ensure_ascii=False, indent=2)}")

    # 1. 参数校验
    if not request.query:
        return DifyResult.of_fail(400, "查询文本不能为空")

    # 2. 权限校验
    if request.searchSource == 1:  # AGENT来源需要权限验证
        if "ADMIN" not in client_info.get("authorities", []):
            return DifyResult.of_fail(403, "权限不足，无法执行检索")

    # 3. 初始化检索配置（默认值处理）
    if not request.retrieval_setting:
        request.retrieval_setting = KwBaseMixRetrievalSettingVo()
        print("使用默认检索配置")

    # 4. 大模型改写查询（调试时可注释此步）
    rewrite_config = {
        "model": "bge-large-zh-v1.5",
        "prompt": "请将以下查询改写为更适合检索的表达方式: ${query}",
        "device_id": "debug_device",
        "service_url": "http://localhost:8000/llm/completion"  # 实际大模型服务地址
    }
    request.query = rewrite_query(request.query, rewrite_config)

    try:
        # 5. 调用各检索服务（同步执行，便于单步调试）
        qna_results = KwBaseQnaSearchService.search_qna_retrieval(request)
        doc_results = KwBaseDocumentSearchService.search_document_retrieval(request)
        block_results = KwBaseDocumentBlockSearchService.search_document_retrieval(request)
        llm_qna_results = KwBaseLlmQnaSearchService.search_qna_retrieval(request)

        # 6. 打印各服务返回结果数（调试用）
        print(f"QNA检索结果数: {len(qna_results)}")
        print(f"文档检索结果数: {len(doc_results)}")
        print(f"文档块检索结果数: {len(block_results)}")
        print(f"LLM QNA检索结果数: {len(llm_qna_results)}")

        # 7. 混合排序
        final_results = KwBaseMixSearchService.handle_sort_mode(
            request, qna_results, doc_results, block_results, llm_qna_results
        )

        # 8. 计算耗时（调试性能）
        end_time = datetime.now()
        print(f"===== 混合检索结束: {end_time} (耗时: {end_time - start_time}) =====")

        return DifyResult.of_success(final_results)

    except Exception as e:
        # 详细异常信息，便于调试
        error_msg = f"检索处理失败: {str(e)}"
        print(f"错误: {error_msg}")
        return DifyResult.of_fail(500, error_msg)

# 启动说明：
# 1. 安装依赖：pip install fastapi uvicorn requests pydantic
# 2. 启动服务：uvicorn mix_retrieval_sync:app --reload
# 3. 调试方法：
#    - 使用Postman发送POST请求到 http://localhost:8000/openapi/kbp/mix/retrieval
#    - 请求体示例：
#      {
#        "query": "如何使用混合检索",
#        "retrieval_setting": {"top_k": 5, "score_threshold": 0.5}
#      }
#    - 查看控制台输出的详细日志
