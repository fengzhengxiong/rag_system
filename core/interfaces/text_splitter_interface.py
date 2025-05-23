#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：text_splitter_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:45 
'''


# core/interfaces/text_splitter_interface.py
from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document

class TextSplitterInterface(ABC):
    """
    文本分割器接口。
    负责将长文档分割成较小的块。
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将一组 Langchain Document 对象分割成更小的块。
        返回的仍然是 Document 对象列表，但内容是分割后的块，
        并且通常会保留或更新元数据以指示其来源。

        Args:
            documents: 原始 Langchain Document 对象列表。

        Returns:
            分割后的 Langchain Document 对象列表。
        """
        pass

    # 可以添加一个 split_text(text: str) -> List[str] 方法如果需要直接分割纯文本
    # @abstractmethod
    # def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
    #    pass

    @property
    @abstractmethod
    def description(self) -> str:
        """返回分割器配置的描述。"""
        pass