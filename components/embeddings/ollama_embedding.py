#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：ollama_embedding.py
@Author  ：fengzhengxiong
@Date    ：2025/5/23 15:03 
'''

# components/embeddings/ollama_embedding.py

from typing import List
from langchain_community.embeddings import OllamaEmbeddings as LangchainOllamaEmbeddings

from core.interfaces.embedding_interface import EmbeddingInterface
from utils.config_models import OllamaEmbeddingConfig
from utils.logger import RAGLogger

class OllamaEmbeddingImpl(EmbeddingInterface):
    """
    使用 Ollama 的向量嵌入模型实现。
    """
    def __init__(self, config: OllamaEmbeddingConfig, logger: RAGLogger):
        self.config = config
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        try:
            self._embeddings = LangchainOllamaEmbeddings(
                model=self.config.model_name,
                # base_url=self.config.base_url, # 如果在配置中定义了
                # 其他 OllamaEmbeddings 支持的参数...
            )
            self.logger_instance.info(
                f"Ollama Embeddings initialized successfully with model: '{self.config.model_name}'."
            )
        except Exception as e:
            self.logger_instance.error(
                f"Failed to initialize Ollama Embeddings with model '{self.config.model_name}': {e}",
                exc_info=True
            )
            raise ConnectionError(f"Could not connect or initialize Ollama embedding model '{self.config.model_name}'. Ensure Ollama server is running and the model is available. Original error: {e}")

    def get_langchain_embeddings(self) -> LangchainOllamaEmbeddings:
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.logger_instance.debug(f"Embedding {len(texts)} documents. First document: '{texts[0][:50]}...'")
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            self.logger_instance.error(f"Error during document embedding: {e}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        self.logger_instance.debug(f"Embedding query: '{text[:50]}...'")
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            self.logger_instance.error(f"Error during query embedding: {e}", exc_info=True)
            raise