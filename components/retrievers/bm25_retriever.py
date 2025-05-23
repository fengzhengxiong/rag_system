#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：bm25_retriever.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:49 
'''

# components/retrievers/bm25_retriever.py
from typing import List
import warnings
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever

from core.interfaces.retriever_interface import RetrieverInterface
from utils.config_models import BM25RetrieverConfig
from utils.logger import RAGLogger


class BM25RetrieverImpl(RetrieverInterface):
    def __init__(self, config: BM25RetrieverConfig, documents: List[Document], logger: RAGLogger):
        self.config = config
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        if not documents:
            msg = "BM25Retriever requires a non-empty list of documents for initialization."
            self.logger_instance.error(msg)
            raise ValueError(msg)

        warnings.filterwarnings("ignore", module="rank_bm25")  # 忽略 BM25 库的警告
        try:
            self._retriever = LangchainBM25Retriever.from_documents(
                documents,
                k=self.config.top_k,  # Langchain BM25Retriever 直接用 k
                bm25_params={"k1": self.config.k1, "b": self.config.b}
            )
            self.logger_instance.info(
                f"BM25Retriever initialized with {len(documents)} documents. k1={config.k1}, b={config.b}, top_k={config.top_k}")
        except Exception as e:
            self.logger_instance.error(f"Failed to initialize BM25Retriever: {e}", exc_info=True)
            raise

    def get_langchain_retriever(self) -> LangchainBaseRetriever:
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        self.logger_instance.debug(f"Retrieving documents with BM25 for query: '{query[:50]}...'")
        try:
            return self._retriever.invoke(query)  # invoke 是新版Langchain的方法
        except Exception as e:
            self.logger_instance.error(f"Error during BM25 retrieval: {e}", exc_info=True)
            return []

    @property
    def description(self) -> str:
        return f"BM25Retriever (k1={self.config.k1}, b={self.config.b}, top_k={self.config.top_k})"