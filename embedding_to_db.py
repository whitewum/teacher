import os
from typing import List, Tuple, Dict
import chromadb
import numpy as np
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from modelscope.hub.snapshot_download import snapshot_download

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_name: str, persist_directory: str, collection_name: str):
        """初始化文档处理器"""
        logger.info(f"初始化文档处理器，使用模型: {model_name}")
        
        try:
            # 设备选择
            if torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("使用 MPS (Metal Performance Shaders) 加速")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("使用 CUDA 加速")
            else:
                self.device = 'cpu'
                logger.info("使用 CPU 处理")

            # 使用ModelScope下载/加载模型
            model_dir = snapshot_download(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir)
            self.model.eval()
            self.model = self.model.to(self.device)

            # 初始化ChromaDB
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB初始化完成，使用目录: {persist_directory}")

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def load_document(self, doc_path: str, abstract_path: str) -> Tuple[str, str]:
        """加载文档和其摘要"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            abstract = ""
            if os.path.exists(abstract_path):
                with open(abstract_path, 'r', encoding='utf-8') as f:
                    abstract = f.read()
            
            return content, abstract
        except Exception as e:
            logger.error(f"加载文档失败: {str(e)}")
            raise

    def split_into_chunks(self, text: str, max_length: int = 2000) -> List[str]:
        """将文本按段落切分，并控制长度"""
        try:
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if not para.strip():
                    continue
                
                if len(current_chunk) + len(para) < max_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        except Exception as e:
            logger.error(f"切分文档失败: {str(e)}")
            raise

    def create_overlap_text(self, text: str, abstract: str, prefix_len: int = 512, suffix_len: int = 512) -> str:
        """创建带有前缀后缀的重叠文本，并在前面添加摘要"""
        try:
            paragraphs = text.split('\n\n')
            
            # 获取前缀
            prefix_text = ""
            current_len = 0
            for para in paragraphs:
                if current_len + len(para) <= prefix_len:
                    prefix_text += para + "\n\n"
                    current_len += len(para)
                else:
                    break
                    
            # 获取后缀
            suffix_text = ""
            current_len = 0
            for para in reversed(paragraphs):
                if current_len + len(para) <= suffix_len:
                    suffix_text = para + "\n\n" + suffix_text
                    current_len += len(para)
                else:
                    break
            
            # 组合文本
            if abstract:
                text = f"下文摘要: {abstract}\n\n前文: {prefix_text}\n\n正文: {text}\n\n后文: {suffix_text}"
            else:
                text = f"前文: {prefix_text}\n\n正文: {text}\n\n后文: {suffix_text}"
            
            return text.strip()
        except Exception as e:
            logger.error(f"创建重叠文本失败: {str(e)}")
            raise

    def _create_embedding(self, text: str) -> List[float]:
        """使用模型创建文本嵌入向量"""
        try:
            # 对超长文本进行截断
            encoded_input = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=8192,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embedding = model_output[0][:, 0]
                sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
                
            return sentence_embedding.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"创建嵌入向量时出错: {str(e)}")
            raise

    def process_documents(self, input_dir: str):
        """处理目录下的所有文档"""
        try:
            for filename in tqdm(os.listdir(input_dir), desc="处理文件"):
                if not filename.endswith('.txt') or '-abstract' in filename:
                    continue
                    
                doc_num = filename.split('.')[0]
                doc_path = os.path.join(input_dir, filename)
                abstract_path = os.path.join(input_dir, f"{doc_num}-abstract.txt")
                
                logger.info(f"处理文件: {filename}")
                
                # 加载文档和摘要
                content, abstract = self.load_document(doc_path, abstract_path)
                
                # 切分文档
                chunks = self.split_into_chunks(content)
                
                # 处理每个chunk
                for i, chunk in enumerate(tqdm(chunks, desc="处理文档片段")):
                    # 创建重叠文本
                    overlap_text = self.create_overlap_text(chunk, abstract)
                    
                    # 生成embedding
                    embedding = self._create_embedding(overlap_text)
                    
                    # 存储到ChromaDB
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[overlap_text],
                        ids=[f"{doc_num}-{i}"],
                        metadatas=[{"source": filename, "chunk": i}]
                    )
                
                logger.info(f"完成文件 {filename} 的处理，共 {len(chunks)} 个片段")

        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='文档处理和向量化工具')
    parser.add_argument('--model', default='BAAI/bge-m3', help='使用的模型名称')
    parser.add_argument('--db-path', default='./chroma_db', help='ChromaDB存储路径')
    parser.add_argument('--collection', help='ChromaDB collection名称')
    parser.add_argument('--input-dir', help='输入文件目录')
    
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor(
            model_name=args.model,
            persist_directory=args.db_path,
            collection_name=args.collection
        )
        processor.process_documents(args.input_dir)
        logger.info("所有文档处理完成")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
