#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：retriever_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:45 
'''

# core/interfaces/retriever_interface.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever

class RetrieverInterface(ABC):
    """
    检索器接口。
    负责根据查询从数据源中检索相关文档。
    """

    @abstractmethod
    def get_langchain_retriever(self) -> LangchainBaseRetriever:
        """返回底层的 Langchain 兼容的 BaseRetriever 实例。"""
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """
        根据查询检索相关文档。

        Args:
            query: 查询字符串。

        Returns:
            相关的 Langchain Document 对象列表。
        """
        pass

    # async def aretrieve(self, query: str) -> List[Document]: ... (可选的异步版本)

    @property
    @abstractmethod
    def description(self) -> str:
        """返回检索器配置的描述。"""
        pass