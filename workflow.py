import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import asyncio
from openai import OpenAI
import httpx
import time
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建调试日志记录器
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('debug.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'))
debug_logger.addHandler(file_handler)

AVAILABLE = "专业技术职务评聘，教学科研业绩成，采购管理，人才招聘，劳务费发放，"

@dataclass
class WorkflowResult:
    """工作流处理结果"""
    answer: str
    intent: str
    context: Optional[str] = None
    rewritten_query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_history: int = 5):
        self.conversations = {}
        self.max_history = max_history
        
    def add_message(self, user_id: str, role: str, content: str, intent: str = None):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        self.conversations[user_id].append({
            'role': role,
            'content': content,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversations[user_id]) > self.max_history * 2:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history * 2:]
            
    def get_history(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])

    def build_recent_history(self, user_id: str, max_turns: int = 3) -> str:
        history = self.get_history(user_id)
        if not history:
            return ""
            
        recent_history = history[-max_turns*2:]  # 获取最近的几轮对话
        formatted_history = []
        
        for msg in recent_history:
            role = "用户" if msg['role'] == "user" else "助手"
            formatted_history.append(f"{role}: {msg['content']}")
            
        return "\n".join(formatted_history)

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self, model_name, base_url, llm_client: OpenAI):
        self.model_name = model_name
        self.base_url = base_url
        self.llm_client = llm_client
        
    async def classify_intent(self, query: str, history: str = "") -> str:
        """分类用户意图"""
        prompt = f"""请判断用户的问题意图。只返回以下意图标签之一：
- HANDBOOK: 与教师工作、教学、科研、规章制度相关的查询, 具体包括{AVAILABLE}
- CHAT: 日常对话、问候、感谢等
- CLARIFICATION: 需要澄清或补充信息的问题
- OTHER: 其他意图

对话历史：
{history}

当前问题：{query}

返回意图标签："""
        if self.model_name == "deepseek/deepseek-r1-distill-qwen-14b":
            prompt += r"""\n请逐步推理，然后将回答放在 `\box{<answer>}` 中。<think>\n"""
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10 if self.model_name == "qwen-long" else 1024
            )
            intent = response.choices[0].message.content.strip().upper()
            if self.model_name == "qwen-long":
                return intent if intent in ["HANDBOOK", "CHAT", "CLARIFICATION", "OTHER"] else "OTHER"
            else: # deepseek/deepseek-r1-distill-qwen-14b
                import re
                # 先尝试匹配\box{}中的内容
                pos = intent.find("\\box{")
                if pos != -1:
                    # 找到最后一个右大括号的位置
                    end_pos = intent.rfind("}")
                    if end_pos != -1:
                        extracted_text = intent[pos + len("\\box{"):end_pos].strip()
                    else:
                        extracted_text = intent[pos + len("\\box{"):].strip()
                else:
                    extracted_text = intent
                # 查看 ["HANDBOOK", "CHAT", "CLARIFICATION", "OTHER"] 哪个可以再 extracted_text 里面被匹配到：  
                for intent in ["HANDBOOK", "CHAT", "CLARIFICATION", "OTHER"]:
                    if intent in extracted_text:
                        return intent
                return "OTHER"

        except Exception as e:
            logger.error(f"意图分类出错: {str(e)}")
            return "OTHER"

class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, model_name, base_url, llm_client: OpenAI):
        self.model_name = model_name
        self.base_url = base_url
        self.llm_client = llm_client
        
    async def rewrite_query(self, query: str, history: str) -> str:
        """改写查询"""
        prompt = f"""基于对话历史和当前问题，生成完整的查询语句。

对话历史：
========
{history}
========
当前问题：
========
{query}
========

要求：
1. 理解用户真实意图，补充必要的上下文
2. 处理代词指代（如"它"、"这个"等）
3. 保持查询的完整性和准确性
4. 如果是追问，需要合并相关上下文
5. 如果是新问题，保持原样

只需要返回改写后的查询，不要返回任何其他内容："""

        if self.model_name == "deepseek/deepseek-r1-distill-qwen-14b":
            prompt += r"""\n请逐步推理，然后将回答放在 `\box{<answer>}` 中。<think>\n"""
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt }],
                temperature=0.6,
                max_tokens=200 if self.model_name == "qwen-long" else 1024
            )
            ret = response.choices[0].message.content.strip()
            if self.model_name == "qwen-long":
                return ret
            else: # deepseek/deepseek-r1-distill-qwen-14b
                import re
                # 先尝试匹配\box{}中的内容
                pos = ret.find("\\box{")
                if pos != -1:
                    # 找到最后一个右大括号的位置
                    end_pos = ret.rfind("}")
                    if end_pos != -1:
                        extracted_text = ret[pos + len("\\box{"):end_pos].strip()
                    else:
                        extracted_text = ret[pos + len("\\box{"):].strip()
                else:
                    # 尝试匹配</think>后的内容
                    pos = ret.find("</think>")
                    if pos != -1:
                        extracted_text = ret[pos + len("</think>"):].strip()
                    else:
                        # 如果都没有匹配，则取最后一行
                        lines = ret.strip().splitlines()
                        extracted_text = lines[-1] if lines else ret
                print("\n提取结果:", extracted_text)
                return extracted_text
        except Exception as e:
            logger.error(f"查询改写出错: {str(e)}")
            return query

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, inference_engine, model_name, base_url, llm_client: OpenAI):
        self.inference_engine = inference_engine
        self.llm_client = llm_client
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_manager = ConversationManager()
        self.intent_classifier = IntentClassifier(model_name, base_url, llm_client)
        self.query_rewriter = QueryRewriter(model_name, base_url, llm_client)
        
    async def process_message(self, user_id: str, message: str) -> WorkflowResult:
        """处理用户消息"""
        start_time = datetime.now()
        request_id = f"{user_id}-{int(start_time.timestamp())}"
        
        try:
            debug_logger.info(f"[{request_id}] 开始处理新请求")
            
            # 记录用户消息
            self.conversation_manager.add_message(user_id, "user", message)
            
            # 获取对话历史
            history = self.conversation_manager.build_recent_history(user_id)
            
            # 分类意图
            intent = await self.intent_classifier.classify_intent(message, history)
            debug_logger.info(f"[{request_id}] 意图分类结果: {intent}")
            
            if intent == "HANDBOOK":
                # 改写查询
                rewritten_query = await self.query_rewriter.rewrite_query(message, history)
                debug_logger.info(f"[{request_id}] 查询改写: {rewritten_query}")
                
                # 使用推理引擎生成答案
                result = await self.inference_engine.generate_answer(rewritten_query, return_context=True)
                
                # 记录助手回复
                self.conversation_manager.add_message(user_id, "assistant", result.answer, intent="HANDBOOK")
                
                return WorkflowResult(
                    answer=result.answer,
                    intent=intent,
                    context=result.context,
                    rewritten_query=rewritten_query,
                    metadata=result.metadata
                )
                
            else:
                # 处理其他类型的消息
                response = await self._handle_non_handbook_query(message, intent)
                
                # 记录助手回复
                self.conversation_manager.add_message(user_id, "assistant", response, intent=intent)
                
                return WorkflowResult(
                    answer=response,
                    intent=intent
                )
                
        except Exception as e:
            logger.error(f"[{request_id}] 处理消息出错: {str(e)}")
            raise
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            debug_logger.info(f"[{request_id}] 请求处理完成，耗时: {duration:.2f}秒")
            
    async def process_message_stream(
        self,
        user_id: str,
        message: str,
        return_context: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理用户消息"""
        start_time = datetime.now()
        request_id = f"{user_id}-{int(start_time.timestamp())}"
        
        try:
            debug_logger.info(f"[{request_id}] 开始流式处理新请求")
            
            # 记录用户消息
            self.conversation_manager.add_message(user_id, "user", message)
            # 获取对话历史
            history = self.conversation_manager.build_recent_history(user_id)

            # 分类意图
            intent = await self.intent_classifier.classify_intent(message, history)
            debug_logger.info(f"[{request_id}] 意图分类结果: {intent}")
            
            if intent == "HANDBOOK":
                # 改写查询
                rewritten_query = await self.query_rewriter.rewrite_query(message, history)
                debug_logger.info(f"[{request_id}] 查询改写: {rewritten_query}")
                
                time_start = datetime.now()
                debug_logger.info(f"[{request_id}] 开始流式生成答案")
                # 流式生成答案
                answer_chunks = []
                async for response in self.inference_engine.generate_answer_stream(
                    rewritten_query,
                    return_context=return_context
                ):
                    if response["type"] == "token":
                        debug_logger.info(f"[{request_id}] 流式生成中，回复: {response['content']}")
                        answer_chunks.append(response["content"])
                    yield response
                debug_logger.info(f"[{request_id}] 流式生成完成，耗时: {datetime.now() - time_start}")
                
                # 记录完整的助手回复
                self.conversation_manager.add_message(
                    user_id,
                    "assistant",
                    "".join(answer_chunks),
                    intent="HANDBOOK"
                )
                
            else:
                # 处理其他类型的消息
                response = await self._handle_non_handbook_query(message, intent)
                debug_logger.info(f"[{request_id}] 处理非手册查询，回复: {response}")
                # 记录助手回复
                self.conversation_manager.add_message(user_id, "assistant", response, intent=intent)
                
                # 将普通回答转换为流式格式
                yield {
                    "type": "token",
                    "content": response
                }
                
        except Exception as e:
            logger.error(f"[{request_id}] 流式处理消息出错: {str(e)}")
            yield {
                "type": "error",
                "content": f"处理您的请求时出错: {str(e)}"
            }
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            debug_logger.info(f"[{request_id}] 流式请求处理完成，耗时: {duration:.2f}秒")
            
    async def _handle_non_handbook_query(self, query: str, intent: str) -> str:
        """处理非手册查询"""
        if intent == "CHAT":
            prompt = f"""你是一个友好的助手。请简短回应用户的问候或感谢，并提醒用户你主要负责回答教师工作制度相关的问题，具体只包括{AVAILABLE}。"""
        elif intent == "CLARIFICATION":
            prompt = f"""你是一个助手。请礼貌地请求用户提供更多信息或澄清问题，以便更好地帮助他们。"""
        else:
            prompt = f"""你是一个助手。请礼貌地告诉用户你主要负责回答教师工作制度相关的问题，并给出一些示例问题，具体只包括{AVAILABLE}。"""
            
        prompt += f"\n\n用户输入：{query}\n\n"
        if self.model_name == "deepseek/deepseek-r1-distill-qwen-14b":
            prompt += r"""\n请逐步推理，然后将回答放在 `\box{<answer>}` 中。<think>\n"""
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=150
            )
            ret = response.choices[0].message.content.strip()
            if self.model_name == "qwen-long":
                return ret
            else: # deepseek/deepseek-r1-distill-qwen-14b
                import re
                # 先尝试匹配\box{}中的内容
                pos = ret.find("\\box{")
                if pos != -1:
                    # 找到最后一个右大括号的位置
                    end_pos = ret.rfind("}")
                    if end_pos != -1:
                        extracted_text = ret[pos + len("\\box{"):end_pos].strip()
                    else:
                        extracted_text = ret[pos + len("\\box{"):].strip()
                else:
                    # 尝试匹配</think>后的内容
                    pos = ret.find("</think>")
                    if pos != -1:
                        extracted_text = ret[pos + len("</think>"):].strip()
                    else:
                        # 如果都没有匹配，则取最后一行
                        lines = ret.strip().splitlines()
                        extracted_text = lines[-1] if lines else ret
                print("\n提取结果:", extracted_text)
                return extracted_text

        except Exception as e:
            logger.error(f"处理非手册查询出错: {str(e)}")
            return "抱歉，我暂时无法回应。"

async def main():
    """测试工作流"""
    from dotenv import load_dotenv
    import os
    from inference import AsyncInferenceEngine
    import chromadb
    
    # 加载环境变量
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY 环境变量未设置")
    BASE_URL = os.getenv("BASE_URL") 
    MODEL_NAME = os.getenv("MODEL_NAME")  
    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"API_KEY: {API_KEY}")
        
    # 初始化 ChromaDB 客户端
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 初始化推理引擎
    collection_names = ["p-level", "performance", "purchase", "recruit", "work-fee"]
    inference_engine = AsyncInferenceEngine(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        chroma_client=client,
        collection_names=collection_names
    )
    
    # 初始化 OpenAI 客户端
    http_client = httpx.Client(timeout=60.0)
    llm_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client
    )
    
    # 初始化工作流管理器
    workflow = WorkflowManager(inference_engine, MODEL_NAME, BASE_URL, llm_client)
    
    # 测试用例
    test_cases = [
        #"你好",
        "哪些劳务费可以发？",
       # "怎么申请？",
        #"谢谢你的帮助"
    ]
    
    # 测试
    user_id = "test_user"
    print("\n=== 开始测试 ===")
    
    for query in test_cases:
        print(f"\n用户问题: {query}")
        
        # 测试普通处理
        result = await workflow.process_message(user_id, query)
        print(f"意图: {result.intent}")
        if result.context:
            print(f"上下文: {result.context}")
        if result.rewritten_query:
            print(f"改写查询: {result.rewritten_query}")
            
        print("\n---")
        
        # 测试流式处理
        print("流式输出: ", end="", flush=True)
        async for response in workflow.process_message_stream(user_id, query):
            if response["type"] == "token":
                print(response["content"], end="", flush=True)
            elif response["type"] == "context":
                print(f"\n上下文: {response['content']}")
            elif response["type"] == "error":
                print(f"\n错误: {response['content']}")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main()) 