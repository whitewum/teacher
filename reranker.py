import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from modelscope.hub.snapshot_download import snapshot_download
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import chromadb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    content: str
    collection_name: str
    metadata: Dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0

class MultiCollectionSearcher:
    def __init__(self, chroma_client, collection_names: List[str], alpha: float = 0.3):
        """初始化多集合搜索器"""
        self.collections = {}
        self.bm25_indexes = {}
        self.corpus_map = {}
        self.alpha = alpha

        # 初始化embedding模型
        model_name = "BAAI/bge-m3"  # 使用与创建集合时相同的模型
        model_dir = snapshot_download(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()
        
        # 设置设备
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"搜索器使用设备: {self.device}")
        self.model = self.model.to(self.device)

        for name in collection_names:
            try:
                collection = chroma_client.get_collection(name)
                self.collections[name] = collection
                
                # 获取collection数据并构建BM25索引
                results = collection.get()
                corpus = results['documents']
                self.corpus_map[name] = corpus
                
                tokenized_corpus = [list(doc) for doc in corpus]
                self.bm25_indexes[name] = BM25Okapi(tokenized_corpus)
                
                logger.info(f"加载集合 {name}: {len(corpus)} 个文档")
            except Exception as e:
                logger.error(f"加载集合 {name} 失败: {str(e)}")

    def _create_embedding(self, text: str) -> List[float]:
        """创建文本嵌入向量"""
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

    def hybrid_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """执行混合搜索"""
        all_results = []
        tokenized_query = list(query)
        
        # 生成查询向量
        query_embedding = self._create_embedding(query)

        for collection_name in self.collections:
            # 向量搜索
            vector_results = self.collections[collection_name].query(
                query_embeddings=[query_embedding],  # 使用生成的向量
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # BM25搜索
            bm25_scores = self.bm25_indexes[collection_name].get_scores(tokenized_query)
            normalized_bm25 = self._normalize_scores(bm25_scores)

            # 合并结果
            for i, (doc, metadata, distance) in enumerate(zip(
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            )):
                vector_score = 1 - distance  # 转换distance为相似度
                bm25_score = normalized_bm25[self.corpus_map[collection_name].index(doc)]
                
                # 计算混合分数
                final_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
                
                all_results.append(SearchResult(
                    content=doc,
                    collection_name=collection_name,
                    metadata=metadata,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    final_score=final_score
                ))

        # 按混合分数排序
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        return all_results[:top_k]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数"""
        if np.max(scores) == np.min(scores):
            return scores
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """初始化重排序器"""
        model_dir = snapshot_download(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"Reranker使用设备: {self.device}")
        
        self.model = self.model.to(self.device)

    def rerank(self, query: str, results: List[SearchResult], batch_size: int = 8) -> List[SearchResult]:
        """对搜索结果重排序"""
        all_scores = []
        
        # 批量处理
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            pairs = [[query, result.content] for result in batch]
            
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=8192
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**features).logits.squeeze(-1)
                scores = torch.sigmoid(scores).cpu().numpy()
                
            all_scores.extend(scores)

        # 更新重排序分数
        for result, score in zip(results, all_scores):
            result.rerank_score = float(score)

        # 按重排序分数排序
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results

def format_results(results: List[SearchResult], show_scores: bool = False) -> str:
    """格式化搜索结果"""
    formatted = []
    
    for idx, result in enumerate(results, 1):
        text = f"\n=== 结果 {idx} ({result.collection_name}) ===\n"
        text += f"内容: {result.content}\n"
        
        if show_scores:
            text += f"重排序分数: {result.rerank_score:.4f}\n"
            text += f"混合检索分数: {result.final_score:.4f}\n"
            text += f"向量检索分数: {result.vector_score:.4f}\n"
            text += f"BM25分数: {result.bm25_score:.4f}\n"
        
        if result.metadata:
            text += f"元数据: {result.metadata}\n"
            
        formatted.append(text)
    
    return "\n".join(formatted)

def main():
    # 初始化ChromaDB客户端
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 添加命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='多集合检索和重排序工具')
    parser.add_argument('--query', type=str, default="博士毕业去信工有多少钱？", help='搜索查询')
    parser.add_argument('--first-k', type=int, default=10, help='第一阶段检索返回数量')
    parser.add_argument('--rerank-k', type=int, default=5, help='重排序后返回数量')
    parser.add_argument('--show-scores', action='store_true', help='是否显示分数')
    parser.add_argument('--alpha', type=float, default=0.3, 
                       help='向量搜索的权重 (0-1之间，默认0.3，剩余为BM25权重)')
    
    args = parser.parse_args()
    
    # 指定要搜索的collections和初始化搜索器（使用命令行传入的alpha权重）
    collection_names = ["p-level", "performance", "purchase", "recruit", "work-fee"]
    searcher = MultiCollectionSearcher(client, collection_names, alpha=args.alpha)
    reranker = Reranker()
    
    print(f"\n查询: {args.query}")
    print(f"向量搜索权重: {args.alpha:.2f}, BM25权重: {1-args.alpha:.2f}")
    
    # 第一阶段：混合检索
    initial_results = searcher.hybrid_search(args.query, top_k=args.first_k)
    # 第二阶段：重排序
    reranked_results = reranker.rerank(args.query, initial_results)
    # 显示结果
    print(format_results(reranked_results[:args.rerank_k], show_scores=args.show_scores))

if __name__ == "__main__":
    main()
