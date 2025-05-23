#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：vector_store_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:45 
'''

# core/interfaces/vector_store_interface.py
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangchainVectorStore # 使用别名
from langchain_core.retrievers import BaseRetriever

from .embedding_interface import EmbeddingInterface # 导入 Embedding 接口，因为 VS 需要它

class VectorStoreInterface(ABC):
    """
    向量数据库接口。
    所有具体的向量数据库实现都应继承此类并实现其方法。
    """

    @abstractmethod
    def get_langchain_vectorstore(self) -> Optional[LangchainVectorStore]:
        """
        返回底层的 Langchain 兼容的 VectorStore 实例。
        如果向量库尚未构建或加载，可能返回 None。
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        将文档（及其嵌入）添加到向量数据库。

        Args:
            documents: Langchain Document 对象列表。
            **kwargs: 传递给底层实现的额外参数。

        Returns:
            添加的文档的 ID 列表 (如果底层库支持并返回)。
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """
        根据查询文本执行相似度搜索。

        Args:
            query: 查询文本。
            k: 返回的最相似文档数量。
            filter: 用于元数据过滤的字典 (可选)。
            **kwargs: 传递给底层实现的额外参数。

        Returns:
            最相似的 Langchain Document 对象列表。
        """
        pass

    @abstractmethod
    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """
        根据查询向量执行相似度搜索。

        Args:
            embedding: 查询向量。
            k: 返回的最相似文档数量。
            filter: 用于元数据过滤的字典 (可选)。
            **kwargs: 传递给底层实现的额外参数。

        Returns:
            最相似的 Langchain Document 对象列表。
        """
        pass


    @abstractmethod
    def as_retriever(self, search_type: Optional[str] = None, search_kwargs: Optional[dict] = None, **kwargs: Any) -> BaseRetriever:
        """
        将向量数据库转换为 Langchain 兼容的检索器 (BaseRetriever)。

        Args:
            search_type: 检索类型 (例如 "similarity", "mmr", "similarity_score_threshold")。
            search_kwargs: 传递给检索器的搜索参数 (例如 {"k": 5, "score_threshold": 0.7})。
            **kwargs: 传递给底层 as_retriever 方法的额外参数。

        Returns:
            Langchain BaseRetriever 实例。
        """
        pass

    @abstractmethod
    def load_local(self) -> bool:
        """
        尝试从本地持久化存储加载向量数据库 (如果实现支持)。
        例如，FAISS 可以从文件加载。

        Returns:
            True 如果加载成功，False 如果失败或不需要加载 (例如，内存数据库)。
        """
        pass

    @abstractmethod
    def save_local(self) -> bool:
        """
        将向量数据库保存到本地持久化存储 (如果实现支持)。

        Returns:
            True 如果保存成功，False 如果失败或不支持保存。
        """
        pass

    @abstractmethod
    def build_from_documents(self, documents: List[Document]) -> bool:
        """
        使用给定的文档列表从头构建向量数据库。

        Args:
            documents: 用于构建索引的 Langchain Document 对象列表。

        Returns:
            True 如果构建成功，False 如果失败。
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查向量数据库是否已成功初始化/加载并且可以使用。
        """
        pass

    # 可以添加 delete_documents, update_documents 等方法如果需要