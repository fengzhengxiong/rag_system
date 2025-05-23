#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：vector_retriever.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:49 
'''

# components/retrievers/vector_retriever.py
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever

from core.interfaces.retriever_interface import RetrieverInterface
from core.interfaces.vector_store_interface import VectorStoreInterface  # 依赖VS
from utils.config_models import VectorRetrieverConfig
from utils.logger import RAGLogger


class VectorRetrieverImpl(RetrieverInterface):
    def __init__(self, config: VectorRetrieverConfig, vector_store: VectorStoreInterface, logger: RAGLogger):
        self.config = config
        self.vector_store = vector_store
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        if not self.vector_store.is_initialized:
            msg = "VectorStore is not initialized. VectorRetriever cannot be created."
            self.logger_instance.error(msg)
            raise RuntimeError(msg)

        search_kwargs = {"k": self.config.top_k}
        if self.config.search_type == "similarity_score_threshold" and self.config.score_threshold is not None:
            search_kwargs["score_threshold"] = self.config.score_threshold
        # if self.config.search_type == "mmr":
        #     if self.config.mmr_fetch_k: search_kwargs["fetch_k"] = self.config.mmr_fetch_k
        #     if self.config.mmr_lambda_mult: search_kwargs["lambda_mult"] = self.config.mmr_lambda_mult

        try:
            self._retriever = self.vector_store.as_retriever(
                search_type=self.config.search_type,
                search_kwargs=search_kwargs
            )
            self.logger_instance.info(
                f"VectorRetriever initialized. Search type: {config.search_type}, Search kwargs: {search_kwargs}")
        except Exception as e:
            self.logger_instance.error(f"Failed to initialize VectorRetriever: {e}", exc_info=True)
            raise

    def get_langchain_retriever(self) -> LangchainBaseRetriever:
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        self.logger_instance.debug(f"Retrieving documents with Vector Search for query: '{query[:50]}...'")
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            self.logger_instance.error(f"Error during Vector retrieval: {e}", exc_info=True)
            return []

    @property
    def description(self) -> str:
        return f"VectorRetriever (Search: {self.config.search_type}, TopK: {self.config.top_k}, Threshold: {self.config.score_threshold})"