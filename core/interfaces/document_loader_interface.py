#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：document_loader_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:45 
'''

# core/interfaces/document_loader_interface.py
from abc import ABC, abstractmethod
from typing import List, Iterator
from langchain_core.documents import Document

class DocumentLoaderInterface(ABC):
    """
    文档加载器接口。
    负责从不同来源加载文档。
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """
        加载所有文档并一次性返回。

        Returns:
            Langchain Document 对象列表。
        """
        pass

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载文档，一次加载一个。
        这对于非常大的数据集或流式数据源很有用。

        Returns:
            一个迭代器，每次产出一个 Langchain Document 对象。
        """
        pass

    @property
    @abstractmethod
    def source_description(self) -> str:
        """
        返回一个描述文档来源的字符串 (例如，目录路径，URL模板等)。
        """
        pass
