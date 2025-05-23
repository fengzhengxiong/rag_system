#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：recursive_character_splitter.py
@Author  ：fengzhengxiong
@Date    ：2025/5/23 16:23 
'''

# components/text_splitters/recursive_character_splitter.py

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 从 langchhain_text_splitters 导入
from langchain_core.documents import Document

from core.interfaces.text_splitter_interface import TextSplitterInterface
from utils.config_models import RecursiveCharacterTextSplitterConfig
from utils.logger import RAGLogger


class RecursiveCharacterTextSplitterImpl(TextSplitterInterface):
    """
    使用 Langchain 的 RecursiveCharacterTextSplitter 实现文本分割。
    """

    def __init__(self, config: RecursiveCharacterTextSplitterConfig, logger: RAGLogger):
        self.config = config
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        # length_function = len # 默认
        # if self.config.length_function_name != "len":
        #     # 这里需要一个机制来获取自定义长度函数，或者只支持 'len'
        #     self.logger_instance.warning(f"Custom length_function_name '{self.config.length_function_name}' not yet supported, using 'len'.")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            # length_function=length_function,
            # separators=self.config.separators,
            keep_separator=self.config.keep_separator,
            # add_start_index=self.config.add_start_index,
        )
        self.logger_instance.info(
            f"RecursiveCharacterTextSplitter initialized. Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            self.logger_instance.info("No documents provided for splitting.")
            return []

        self.logger_instance.info(f"Splitting {len(documents)} documents...")
        try:
            split_docs = self._splitter.split_documents(documents)
            self.logger_instance.info(f"Successfully split documents into {len(split_docs)} chunks.")
            return split_docs
        except Exception as e:
            self.logger_instance.error(f"Failed to split documents: {e}", exc_info=True)
            return []  # 或重新抛出

    @property
    def description(self) -> str:
        return f"RecursiveCharacterTextSplitter (Chunk: {self.config.chunk_size}, Overlap: {self.config.chunk_overlap})"