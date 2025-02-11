import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import asyncio
from dotenv import load_dotenv
import httpx
from openai import OpenAI
from workflow import WorkflowManager
import chromadb
from inference import AsyncInferenceEngine
from fastapi import WebSocketDisconnect

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="教师工作问答系统",
    description="基于教师工作手册的智能问答系统",
    version="0.0.1"
)

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建模板目录
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.connections = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str):
        async with self.lock:
            self.connections[user_id] = websocket
            logger.info(f"新的WebSocket连接已建立 (user_id: {user_id})")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            user_id_to_remove = None
            for user_id, ws in self.connections.items():
                if ws == websocket:
                    user_id_to_remove = user_id
                    break
            if user_id_to_remove:
                del self.connections[user_id_to_remove]
                logger.info(f"WebSocket连接已移除 (user_id: {user_id_to_remove})")

    async def send_message(self, user_id: str, message: dict):
        """发送消息给指定用户"""
        if user_id in self.connections:
            try:
                await self.connections[user_id].send_json(message)
            except WebSocketDisconnect:
                logger.info(f"发送消息时检测到连接断开 (user_id: {user_id})")
                await self.disconnect(self.connections[user_id])
            except Exception as e:
                logger.error(f"发送消息失败 (user_id: {user_id}): {str(e)}")
                await self.disconnect(self.connections[user_id])

manager = ConnectionManager()

# 全局工作流管理器实例
workflow_manager = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化系统"""
    global workflow_manager
    
    try:
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
        
        # 初始化OpenAI客户端
        http_client = httpx.Client(timeout=60.0)
        llm_client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=http_client
        )
        
        # 初始化工作流管理器
        workflow_manager = WorkflowManager(inference_engine, MODEL_NAME, BASE_URL, llm_client)
        
        logger.info("系统初始化完成")
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """提供Web界面"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.websocket("/ws/ask")
async def websocket_ask_endpoint(websocket: WebSocket):
    """处理流式问答的WebSocket连接"""
    user_id = None
    try:
        # 先接受连接
        await websocket.accept()
        
        # 等待客户端发送初始化消息
        init_data = await websocket.receive_json()
        user_id = init_data.get('user_id')
        
        if not user_id:
            await websocket.close(code=1008, reason="No user_id provided")
            return
            
        await manager.connect(websocket, user_id)
        
        # 发送连接成功消息
        await manager.send_message(user_id, {
            "type": "status",
            "content": "✅ 系统已连接"
        })
        
        # 使用标志来跟踪连接状态
        is_connected = True
        while is_connected:
            try:
                # 接收前端发送的问题
                data = await websocket.receive_json()
                question = data.get('question')
                return_context = data.get('return_context', True)
                
                if not question:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "content": "无效的请求数据"
                    })
                    continue
                
                # 流式生成答案
                try:
                    async for response in workflow_manager.process_message_stream(
                        user_id=user_id,
                        message=question,
                        return_context=return_context
                    ):
                        await manager.send_message(user_id, response)
                        
                    # 发送完成标记
                    await manager.send_message(user_id, {
                        "type": "done",
                        "content": None
                    })
                except Exception as e:
                    logger.error(f"生成答案时出错: {str(e)}")
                    await manager.send_message(user_id, {
                        "type": "error",
                        "content": f"生成答案时出错: {str(e)}"
                    })
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket连接断开 (user_id: {user_id})")
                is_connected = False
                break
            except Exception as e:
                logger.error(f"处理消息时出错: {str(e)}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "content": f"处理消息时出错: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接初始化时断开 (user_id: {user_id})")
    except Exception as e:
        logger.error(f"WebSocket连接出错: {str(e)}")
    finally:
        if user_id:
            await manager.disconnect(websocket)
            logger.info(f"WebSocket连接清理完成 (user_id: {user_id})")

def main():
    """主函数"""
    import uvicorn
    uvicorn.run(
        "web:app",
        host="0.0.0.0",
        port=8012,
        reload=True
    )

if __name__ == "__main__":
    main() 