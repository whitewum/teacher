import torch
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import asyncio
from openai import OpenAI
import httpx
from reranker import MultiCollectionSearcher, Reranker, SearchResult, format_results
import os
import time
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InferenceResult:
    """推理结果数据类"""
    answer: str
    context: Optional[str] = None
    rerank_results: Optional[List[SearchResult]] = None
    metadata: Optional[Dict[str, Any]] = None

class AsyncInferenceEngine:
    """异步推理引擎"""
    
    def __init__(
        self,
        api_key: str,
        chroma_client,
        collection_names: List[str],
        model_name: str = "qwen-long",
        initial_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        concurrent_limit: int = 3
    ):
        """初始化推理引擎
        
        Args:
            api_key: DashScope API密钥
            chroma_client: ChromaDB客户端实例
            collection_names: 要搜索的集合名称列表
            model_name: 使用的LLM模型名称
            initial_top_k: 初始检索数量，如果为None则使用环境变量
            final_top_k: 重排序后保留数量，如果为None则使用环境变量
            concurrent_limit: 并发请求限制
        """
        self.model_name = model_name
        
        # 从环境变量获取配置
        self.initial_top_k = initial_top_k or int(os.getenv("INITIAL_TOP_K", "10"))
        self.final_top_k = final_top_k or int(os.getenv("FINAL_TOP_K", "5"))
        
        # 初始化OpenAI客户端
        http_client = httpx.Client(timeout=60.0)  # 增加超时时间
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 获取每个collection的文档数量并计算最大初始召回数量
        self.collection_limits = {}
        for name in collection_names:
            try:
                collection = chroma_client.get_collection(name)
                doc_count = len(collection.get()['documents'])
                # 设置为文档数量的1/N，向上取整，但不超过initial_top_k
                max_docs = min(self.initial_top_k, -(-doc_count // 1))  # 向上取整除法
                self.collection_limits[name] = max_docs
                logger.info(f"Collection {name}: {doc_count} documents, max retrieval: {max_docs}")
            except Exception as e:
                logger.error(f"获取集合 {name} 文档数量失败: {str(e)}")
                self.collection_limits[name] = self.initial_top_k

        # 初始化搜索和重排序组件
        self.searcher = MultiCollectionSearcher(
            chroma_client, 
            collection_names,
            collection_limits=self.collection_limits  # 传递每个collection的限制
        )
        self.reranker = Reranker()
        
        # 并发控制
        self.search_semaphore = asyncio.Semaphore(concurrent_limit)
        
    async def _create_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """创建提示词
        
        Args:
            query: 用户查询
            search_results: 检索结果
            
        Returns:
            构建的提示词
        """
        context = format_results(search_results, show_scores=False)
        
        prompt = f"""你是杭电信工的教师工作手册的问答助手。
请基于以下参考信息回答用户的问题。要求：
1. 答案必须准确，与参考信息保持一致
2. 如果参考信息不足以完整回答问题，请明确指出
3. 合理组织答案结构，适当分点说明
4. 保持语言简洁，不能丢失重要细节
5. 可以直接引用原文内容，注意语言流畅
6. 如果有页码，请在答案中说明可以查阅手册的页码

参考信息：
==========
{context}
==========

用户问题：{query}

请生成解答："""

        return prompt

    async def _generate_answer(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> str:
        """生成答案
        
        Args:
            prompt: 提示词
            temperature: 采样温度
            max_tokens: 最大生成长度
            
        Returns:
            生成的答案
        """
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            raise

    async def generate_answer(
        self,
        query: str,
        return_context: bool = False
    ) -> InferenceResult:
        """生成完整答案
        
        Args:
            query: 用户查询
            return_context: 是否返回检索上下文
            
        Returns:
            推理结果对象
        """
        try:
            logger.info(f"开始召回")
            start_time = time.time()
            # 1. 混合检索
            initial_results = self.searcher.hybrid_search(query, top_k=self.initial_top_k)
            
            # 2. 重排序
            reranked_results = self.reranker.rerank(query, initial_results)[:self.final_top_k]
            logger.info(f"重排序完成, 耗时: {time.time() - start_time:.2f}秒")
            # 3. 生成提示词
            prompt = await self._create_prompt(query, reranked_results)
            logger.info(f"开始生成流式回答")
            start_time = time.time()
            # 4. 生成答案
            answer = await self._generate_answer(prompt)
            logger.info(f"生成答案完成, 耗时: {time.time() - start_time:.2f}秒")
            # 5. 构建返回结果
            result = InferenceResult(
                answer=answer,
                rerank_results=reranked_results,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name
                }
            )
            
            if return_context:
                result.context = format_results(reranked_results, show_scores=True)
                
            return result
            
        except Exception as e:
            logger.error(f"生成答案过程中出错: {str(e)}")
            raise

    async def generate_answer_stream(
        self,
        query: str,
        return_context: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式生成答案
        
        Args:
            query: 用户查询
            return_context: 是否返回检索上下文
            
        Yields:
            包含token或context的字典
        """
        try:
            # 1. 混合检索
            initial_results = self.searcher.hybrid_search(query, top_k=self.initial_top_k)
            
            # 2. 重排序
            reranked_results = self.reranker.rerank(query, initial_results)[:self.final_top_k]
            
            # 3. 生成提示词
            prompt = await self._create_prompt(query, reranked_results)
            
            # 4. 流式生成
            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True
            )
            
            # 5. 处理流式响应，添加延迟
            current_sentence = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    current_sentence += content
                    

                    await asyncio.sleep(0.02)  # 普通token的延迟
                    
                    yield {
                        "type": "token",
                        "content": content
                    }
                    
            # 6. 如果需要，发送上下文
            if return_context:
                yield {
                    "type": "context",
                    "content": format_results(reranked_results, show_scores=True)
                }
                
        except Exception as e:
            logger.error(f"流式生成答案时出错: {str(e)}")
            yield {
                "type": "error",
                "content": str(e)
            }

async def main():
    """测试推理引擎"""
    import os
    from dotenv import load_dotenv
    import chromadb
    
    # 加载环境变量
    load_dotenv()
    API_KEY = os.getenv("DASH_SCOPE_API_KEY")
    if not API_KEY:
        raise ValueError("DASH_SCOPE_API_KEY 环境变量未设置")
        
    # 初始化ChromaDB客户端
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 初始化推理引擎
    collection_names = ["p-level", "performance", "purchase", "recruit", "work-fee"]
    engine = AsyncInferenceEngine(
        api_key=API_KEY,
        chroma_client=client,
        collection_names=collection_names
    )

    # 测试流式生成
    print("\n=== 测试流式生成 ===")
    query = "劳务费有上限吗？"
    print(f"\n问题: {query}")
    print("答案: ", end="", flush=True)
    
    async for response in engine.generate_answer_stream(query, return_context=True):
        if response["type"] == "token":
            print(response["content"], end="", flush=True)
        elif response["type"] == "context":
            print(f"\n\n上下文: {response['content']}")
        elif response["type"] == "error":
            print(f"\n错误: {response['content']}")

if __name__ == "__main__":
    asyncio.run(main()) 