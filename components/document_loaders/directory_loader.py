#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：directory_loader.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:50 
'''

# components/document_loaders/directory_loader.py

from typing import List, Iterator
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from core.interfaces.document_loader_interface import DocumentLoaderInterface
from utils.config_models import DirectoryLoaderConfig  # 具体配置
from utils.logger import RAGLogger


class DirectoryLoaderImpl(DocumentLoaderInterface):
    """
    从文件目录加载文档的实现。
    封装 Langchain 的 DirectoryLoader。
    """

    def __init__(self, config: DirectoryLoaderConfig, data_dir: Path, logger: RAGLogger):
        self.config = config
        self.data_dir = data_dir  # 从 AppConfig.paths.data_dir 传入
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            msg = f"Data directory '{self.data_dir}' does not exist or is not a directory."
            self.logger_instance.error(msg)
            raise FileNotFoundError(msg)

        self.logger_instance.info(
            f"DirectoryLoader configured. Data directory: '{self.data_dir}', Glob: '{self.config.glob_pattern}'"
        )

    def _get_loader_kwargs(self) -> dict:
        """准备传递给 TextLoader (或未来其他loader) 的参数"""
        kwargs = {}
        if self.config.text_loader_autodetect_encoding:
            kwargs["autodetect_encoding"] = True
        # 如果 config 中有更多 loader_kwargs，可以在这里处理
        return kwargs

    def load(self) -> List[Document]:
        self.logger_instance.info(
            f"Loading documents from '{self.data_dir}' using glob '{self.config.glob_pattern}'...")
        try:
            # Langchain 的 DirectoryLoader 接受 loader_cls 和 loader_kwargs
            # 我们这里硬编码 TextLoader 作为示例，但可以根据配置使其更灵活
            loader = DirectoryLoader(
                path=str(self.data_dir),
                glob=self.config.glob_pattern,
                loader_cls=TextLoader,  # 示例：固定使用 TextLoader
                loader_kwargs=self._get_loader_kwargs(),
                use_multithreading=self.config.use_multithreading,
                silent_errors=self.config.silent_errors,
                # recursive=True, # DirectoryLoader 默认递归
            )
            documents = loader.load()
            self.logger_instance.info(f"Successfully loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            self.logger_instance.error(f"Failed to load documents: {e}", exc_info=True)
            return []  # 或重新抛出异常

    def lazy_load(self) -> Iterator[Document]:
        self.logger_instance.info(
            f"Lazy loading documents from '{self.data_dir}' using glob '{self.config.glob_pattern}'...")
        try:
            loader = DirectoryLoader(
                path=str(self.data_dir),
                glob=self.config.glob_pattern,
                loader_cls=TextLoader,
                loader_kwargs=self._get_loader_kwargs(),
                use_multithreading=self.config.use_multithreading,  # 注意：lazy_load 和多线程可能行为不直观
                silent_errors=self.config.silent_errors,
            )
            # DirectoryLoader 的 load_and_split() 或其迭代版本（如果Langchain提供）
            # 目前 DirectoryLoader 主要提供 load()。要实现真正的 lazy_load，
            # 可能需要遍历文件并为每个文件单独创建一个 TextLoader。
            # Langchain 的 DirectoryLoader.lazy_load() (如果存在) 是首选。
            # 检查最新Langchain版本，DirectoryLoader自身可能已经支持lazy_load()
            # 如果没有，我们可以自己实现一个简单的版本：

            # 简易惰性加载实现 (如果 DirectoryLoader 没有直接的 lazy_load)
            for file_path in self.data_dir.rglob(self.config.glob_pattern):  # rglob 进行递归搜索
                if file_path.is_file():
                    try:
                        self.logger_instance.debug(f"Lazy loading file: {file_path}")
                        # 为每个文件创建一个 TextLoader
                        # 注意：如果文件类型多样，这里需要更复杂的逻辑来选择 loader_cls
                        file_loader = TextLoader(str(file_path), **self._get_loader_kwargs())
                        # TextLoader.load() 返回 List[Document]，通常一个文件一个Document
                        for doc in file_loader.load():  # 迭代单个文件加载的文档
                            yield doc
                    except Exception as e:
                        if self.config.silent_errors:
                            self.logger_instance.warning(f"Error lazy loading file {file_path}, skipping: {e}")
                        else:
                            self.logger_instance.error(f"Error lazy loading file {file_path}: {e}", exc_info=True)
                            raise  # 或者 yield 一个错误标记的 Document

        except Exception as e:
            self.logger_instance.error(f"Failed to initiate lazy loading: {e}", exc_info=True)
            # return iter([]) # 返回一个空迭代器

    @property
    def source_description(self) -> str:
        return f"Directory: {str(self.data_dir)}, Glob: {self.config.glob_pattern}"