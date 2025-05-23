#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：faiss_vs.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:49 
'''

# components/vector_stores/faiss_vs.py

import os
from pathlib import Path
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS as LangchainFAISS  # 从 langchain_community 导入
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from core.interfaces.vector_store_interface import VectorStoreInterface
from core.interfaces.embedding_interface import EmbeddingInterface  # 需要注入 Embedding 模型
from utils.config_models import FaissVectorStoreConfig
from utils.logger import RAGLogger


class FaissVectorStoreImpl(VectorStoreInterface):
    """
    使用 FAISS 的向量数据库实现。
    """

    def __init__(self, config: FaissVectorStoreConfig, embedding_model: EmbeddingInterface, logger: RAGLogger):
        self.config = config
        self.embedding_model = embedding_model  # 必须注入 Embedding 实现
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")

        self._faiss_store: Optional[LangchainFAISS] = None
        self._persist_path = Path(self.config.persist_directory)

        # 确保 FAISS_NO_GPU 环境变量设置 (通常在 Langchain 内部处理，但显式设置更安全)
        os.environ['FAISS_NO_GPU'] = '1'

        self.logger_instance.info(
            f"FAISS Vector Store configured. Persist directory: '{self._persist_path}', Index name: '{self.config.index_name}'"
        )
        # 尝试在初始化时加载，如果存在的话
        # if not self.load_local():
        #     self.logger_instance.info("No existing FAISS index found or failed to load. Store needs to be built.")
        # else:
        #     self.logger_instance.info("Existing FAISS index loaded successfully.")
        # 更好的做法是让 AppBuilder 或 IngestionService 决定何时加载/构建

    @property
    def is_initialized(self) -> bool:
        return self._faiss_store is not None

    def get_langchain_vectorstore(self) -> Optional[LangchainFAISS]:
        return self._faiss_store

    def _get_full_index_path(self) -> Path:
        # FAISS 通常需要一个目录，而不是直接的文件名
        return self._persist_path

    def load_local(self) -> bool:
        if self._faiss_store is not None:
            self.logger_instance.debug("FAISS store already in memory, no need to load.")
            return True

        full_path = self._get_full_index_path()
        # FAISS.load_local 需要文件夹路径和 index_name (可选，但建议与保存时一致)
        # Langchain 的 FAISS.load_local 会自动寻找 folder_path / index_name.faiss 和 .pkl
        # 所以我们只需要确保文件夹路径正确，并且里面有 index_name.faiss 和 index_name.pkl
        # Langchain 的 FAISS.load_local 会处理 index_name

        # 检查必需文件是否存在
        faiss_file = full_path / f"{self.config.index_name}.faiss"
        pkl_file = full_path / f"{self.config.index_name}.pkl"

        if not (faiss_file.exists() and pkl_file.exists()):
            self.logger_instance.warning(
                f"FAISS index files ('{self.config.index_name}.faiss', '{self.config.index_name}.pkl') not found in '{full_path}'. Cannot load."
            )
            return False

        try:
            self.logger_instance.info(
                f"Attempting to load FAISS index from '{full_path}' with index_name '{self.config.index_name}'.")
            self._faiss_store = LangchainFAISS.load_local(
                folder_path=str(full_path),  # Langchain 需要字符串路径
                embeddings=self.embedding_model.get_langchain_embeddings(),
                index_name=self.config.index_name,
                allow_dangerous_deserialization=self.config.allow_dangerous_deserialization
            )
            self.logger_instance.info(f"FAISS index loaded successfully from '{full_path}'.")
            return True
        except Exception as e:
            self.logger_instance.error(f"Failed to load FAISS index from '{full_path}': {e}", exc_info=True)
            self._faiss_store = None
            return False

    def save_local(self) -> bool:
        if not self.is_initialized:
            self.logger_instance.warning("FAISS store not initialized. Cannot save.")
            return False

        full_path = self._get_full_index_path()
        full_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        try:
            assert self._faiss_store is not None  # is_initialized 应该已经检查过
            self.logger_instance.info(
                f"Saving FAISS index to '{full_path}' with index_name '{self.config.index_name}'.")
            self._faiss_store.save_local(folder_path=str(full_path), index_name=self.config.index_name)
            self.logger_instance.info(f"FAISS index saved successfully to '{full_path}'.")
            return True
        except Exception as e:
            self.logger_instance.error(f"Failed to save FAISS index to '{full_path}': {e}", exc_info=True)
            return False

    def build_from_documents(self, documents: List[Document]) -> bool:
        if not documents:
            self.logger_instance.warning("No documents provided to build FAISS index.")
            return False
        try:
            self.logger_instance.info(f"Building new FAISS index from {len(documents)} documents...")
            self._faiss_store = LangchainFAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model.get_langchain_embeddings()
            )
            self.logger_instance.info("FAISS index built successfully.")
            # 构建后通常需要保存
            if self.config.persist_directory:  # 只在配置了持久化目录时保存
                self.save_local()
            return True
        except Exception as e:
            self.logger_instance.error(f"Failed to build FAISS index from documents: {e}", exc_info=True)
            self._faiss_store = None
            return False

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        if not self.is_initialized:
            # 尝试从文档构建，如果VS为空且有文档传入
            # self.logger_instance.warning("FAISS store not initialized. Attempting to build from provided documents.")
            # if not self.build_from_documents(documents):
            #     raise RuntimeError("Failed to initialize or build FAISS store before adding documents.")
            # 更好的做法是在 IngestionService 中处理“如果不存在则构建”的逻辑
            self.logger_instance.error("FAISS store not initialized. Cannot add documents. Please load or build first.")
            raise RuntimeError("FAISS store not initialized. Cannot add documents.")

        self.logger_instance.debug(f"Adding {len(documents)} documents to FAISS store.")
        try:
            # FAISS 的 add_documents 通常返回添加文档的ID列表
            # 但 Langchain 的 FAISS 实现可能不直接返回有意义的ID，而是内部ID
            # 检查 LangchainFAISS.add_documents 的文档
            added_ids = self._faiss_store.add_documents(documents, **kwargs)
            self.logger_instance.info(f"Successfully added {len(documents)} documents.")
            # 通常在添加新文档后，如果配置了持久化，应该保存
            if self.config.persist_directory:
                self.save_local()
            return added_ids if added_ids else []  # 确保返回列表
        except Exception as e:
            self.logger_instance.error(f"Error adding documents to FAISS store: {e}", exc_info=True)
            raise

    def similarity_search(
            self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        if not self.is_initialized:
            self.logger_instance.error("FAISS store not initialized. Cannot perform similarity search.")
            return []
        self.logger_instance.debug(f"Performing similarity search for query: '{query[:50]}...' with k={k}")
        try:
            # Langchain FAISS 的 similarity_search 不直接支持 filter 参数
            # filter 通常在 ChromaDB 等其他 VS 中支持
            # 如果需要 filter，可能要用 as_retriever 并配置 retriever 的 filter
            # 或者在获取结果后手动过滤元数据
            # 对于 FAISS，kwargs 可以传递给底层的 search 方法，例如 score_threshold
            # search_kwargs = {"k": k}
            # if "score_threshold" in kwargs: # 示例处理 score_threshold
            #    search_kwargs["score_threshold"] = kwargs.pop("score_threshold")

            # return self._faiss_store.similarity_search(query, k=k, filter=filter, **kwargs)
            # 实际 Langchain FAISS 的 filter 是在 retriever 层面
            # 这里的 filter 参数是为了接口统一性，具体实现可能忽略它或抛出 NotImplementedError
            if filter:
                self.logger_instance.warning(
                    "FAISS similarity_search called with filter, but native FAISS wrapper may not support it directly. Filter will be ignored by this implementation's direct search.")

            return self._faiss_store.similarity_search(query, k=k, **kwargs)
        except Exception as e:
            self.logger_instance.error(f"Error during similarity search: {e}", exc_info=True)
            return []

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        if not self.is_initialized:
            self.logger_instance.error("FAISS store not initialized. Cannot perform similarity search by vector.")
            return []
        self.logger_instance.debug(f"Performing similarity search by vector (vector dim: {len(embedding)}) with k={k}")
        try:
            if filter:
                self.logger_instance.warning(
                    "FAISS similarity_search_by_vector called with filter, but native FAISS wrapper may not support it directly. Filter will be ignored.")
            return self._faiss_store.similarity_search_by_vector(embedding, k=k, **kwargs)
        except Exception as e:
            self.logger_instance.error(f"Error during similarity search by vector: {e}", exc_info=True)
            return []

    def as_retriever(self, search_type: Optional[str] = None, search_kwargs: Optional[dict] = None,
                     **kwargs: Any) -> BaseRetriever:
        if not self.is_initialized:
            self.logger_instance.error("FAISS store not initialized. Cannot create retriever.")
            # 可以返回一个空的或无效的 retriever，或者抛出异常
            raise RuntimeError("FAISS store not initialized. Cannot create retriever.")

        self.logger_instance.debug(f"Creating retriever. Search type: {search_type}, Search kwargs: {search_kwargs}")

        # 准备传递给 Langchain as_retriever 的参数
        retriever_kwargs = {}
        if search_type:
            retriever_kwargs['search_type'] = search_type

        final_search_kwargs = {}  # 合并传入的 search_kwargs 和 kwargs 中可能存在的 k
        if search_kwargs:
            final_search_kwargs.update(search_kwargs)
        if 'k' in kwargs and 'k' not in final_search_kwargs:  # 如果 kwargs 里有k，且 search_kwargs 里没有
            final_search_kwargs['k'] = kwargs.pop('k')
        # 如果 kwargs 里还有其他 retriever 支持的参数，也可以在这里处理

        if final_search_kwargs:
            retriever_kwargs['search_kwargs'] = final_search_kwargs

        retriever_kwargs.update(kwargs)  # 将剩余的 kwargs 也传递过去

        try:
            return self._faiss_store.as_retriever(**retriever_kwargs)
        except Exception as e:
            self.logger_instance.error(f"Error creating retriever: {e}", exc_info=True)
            raise  # 或者返回一个备用 retriever