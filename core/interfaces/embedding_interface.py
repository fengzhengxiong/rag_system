#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：embedding_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:44 
'''

from abc import ABC, abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings as LangchainEmbeddings # 使用别名

class EmbeddingInterface(ABC):
    """
    向量嵌入模型接口。
    所有具体的 Embedding 模型实现都应继承此类并实现其方法。
    """

    @abstractmethod
    def get_langchain_embeddings(self) -> LangchainEmbeddings:
        """返回底层的 Langchain 兼容的 Embeddings 实例。"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将一组文档文本转换为向量嵌入。

        Args:
            texts: 文档文本列表。

        Returns:
            一个列表，其中每个元素是对应输入文本的向量嵌入 (浮点数列表)。
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量嵌入。

        Args:
            text: 查询文本。

        Returns:
            查询文本的向量嵌入 (浮点数列表)。
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前 Embedding 模型实例所使用的模型名称。"""
        pass

    # 可以考虑添加获取嵌入维度的方法
    # @property
    # @abstractmethod
    # def dimensions(self) -> int:
    #     pass