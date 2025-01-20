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
        self.active_connections = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

manager = ConnectionManager()

# 全局工作流管理器实例
workflow_manager = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化系统"""
    global workflow_manager
    
    try:
        # 加载环境变量
        load_dotenv()  
        API_KEY = os.getenv("DASH_SCOPE_API_KEY")
        if not API_KEY:
            raise ValueError("DASH_SCOPE_API_KEY 环境变量未设置")
        
        # 初始化 ChromaDB 客户端
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # 初始化推理引擎
        collection_names = ["p-level", "performance", "purchase", "recruit", "work-fee"]
        inference_engine = AsyncInferenceEngine(
            api_key=API_KEY,
            chroma_client=client,
            collection_names=collection_names
        )
        
        # 初始化OpenAI客户端
        http_client = httpx.Client(timeout=60.0)
        llm_client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 初始化工作流管理器
        workflow_manager = WorkflowManager(inference_engine, llm_client)
        
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
    try:
        await manager.connect(websocket)
        # 发送连接成功消息
        await websocket.send_json({
            "type": "status",
            "content": "✅ 系统已连接，可以开始提问"
        })
        
        while True:
            try:
                # 接收前端发送的问题
                data = await websocket.receive_json()
                question = data.get('question')
                user_id = data.get('user_id')
                return_context = data.get('return_context', True)
                
                if not question or not user_id:
                    await websocket.send_json({
                        "type": "error",
                        "content": "无效的请求数据"
                    })
                    continue
                
                # 发送思考状态
                await websocket.send_json({
                    "type": "thinking",
                    "content": "正在思考..."
                })
                
                # 流式生成答案
                async for response in workflow_manager.process_message_stream(
                    user_id=user_id,
                    message=question,
                    return_context=return_context
                ):
                    await websocket.send_json(response)
                    
                # 发送完成标记
                await websocket.send_json({
                    "type": "done",
                    "content": None
                })
                
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"处理问题时出错: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket连接出错: {str(e)}")
    finally:
        await manager.disconnect(websocket)
        logger.info("问答WebSocket连接已断开")

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