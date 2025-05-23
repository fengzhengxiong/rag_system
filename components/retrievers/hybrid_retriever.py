#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：hybrid_retriever.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:49 
'''

# components/retrievers/hybrid_retriever.py
from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever

from core.interfaces.retriever_interface import RetrieverInterface
from utils.config_models import HybridRetrieverConfig
from utils.logger import RAGLogger


class HybridRetrieverImpl(RetrieverInterface):
    def __init__(self,
                 config: HybridRetrieverConfig,
                 bm25_retriever: RetrieverInterface,  # 依赖具体的BM25实现
                 vector_retriever: RetrieverInterface,  # 依赖具体的Vector实现
                 logger: RAGLogger):
        self.config = config
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        try:
            self._retriever = EnsembleRetriever(
                retrievers=[
                    bm25_retriever.get_langchain_retriever(),
                    vector_retriever.get_langchain_retriever()
                ],
                weights=[self.config.bm25_weight, self.config.vector_weight],
                # c=... # EnsembleRetriever 的 c 参数用于RRF等融合算法，可以配置
            )
            # EnsembleRetriever 不直接接受 top_k，它依赖子 retriever 的 top_k
            # 如果需要对融合结果再取 top_k，可能需要在 invoke 后手动处理，或者使用 Langchain Expression Language 包装
            self.logger_instance.info(
                f"HybridRetriever (EnsembleRetriever) initialized. Weights: BM25={config.bm25_weight}, Vector={config.vector_weight}"
            )
        except Exception as e:
            self.logger_instance.error(f"Failed to initialize HybridRetriever: {e}", exc_info=True)
            raise

    def get_langchain_retriever(self) -> LangchainBaseRetriever:
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        self.logger_instance.debug(f"Retrieving documents with Hybrid Search for query: '{query[:50]}...'")
        try:
            # EnsembleRetriever 的结果可能是未排序或按某种默认方式排序的
            # 如果需要对最终结果应用 top_k，可能需要在这里或外部处理
            # 例如，如果 EnsembleRetriever 返回的文档数多于 hybrid_config.top_k
            # results = self._retriever.invoke(query)
            # return results[:self.config.top_k] # 简单的截断，更好的方式是基于分数（如果Ensemble提供）

            # Langchain的EnsembleRetriever似乎没有直接的top_k参数。
            # 通常它的目的是融合来自不同检索器的结果，然后由上层（如RAG链）处理。
            # 我们可以通过 invoke 获取所有结果，然后手动取 top_k（尽管这不是最高效的）
            # 或者，依赖于子检索器（BM25, Vector）已经配置了合理的 top_k，
            # EnsembleRetriever 会融合这些结果。
            # 简单的做法是直接调用 invoke，并信任 EnsembleRetriever 的融合逻辑。
            # 如果需要严格的最终top_k，可能需要自定义EnsembleRetriever的逻辑或后处理。

            # 目前直接返回 EnsembleRetriever 的结果
            return self._retriever.invoke(query)
        except Exception as e:
            self.logger_instance.error(f"Error during Hybrid retrieval: {e}", exc_info=True)
            return []

    @property
    def description(self) -> str:
        return f"HybridRetriever (Weights: BM25={self.config.bm25_weight}, Vector={self.config.vector_weight})"