#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：ingestion_service.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:51 
'''

# services/ingestion_service.py

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document

from core.interfaces.document_loader_interface import DocumentLoaderInterface
from core.interfaces.text_splitter_interface import TextSplitterInterface
from core.interfaces.embedding_interface import EmbeddingInterface  # 可能不需要直接用，VS会用
from core.interfaces.vector_store_interface import VectorStoreInterface
from utils.logger import RAGLogger
from utils.config_models import AppConfig  # 可能需要访问一些全局配置


class IngestionService:
    """
    负责整个数据注入流程的服务：加载、分割、嵌入（通过VS）、存储。
    """

    def __init__(
            self,
            document_loader: DocumentLoaderInterface,
            text_splitter: TextSplitterInterface,
            # embedding_model: EmbeddingInterface, # 通常注入到 VectorStore 中
            vector_store: VectorStoreInterface,
            logger: RAGLogger,
            app_config: AppConfig  # 传入 AppConfig 以便访问全局设置如路径等
    ):
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.vector_store = vector_store  # VectorStore 内部应该已经有 Embedding Model
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.app_config = app_config
        self.logger_instance.info("IngestionService initialized.")

    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        (可选) 在分割后、存入向量数据库前对文档块进行预处理。
        例如：清洗文本、添加额外元数据等。
        """
        self.logger_instance.debug(f"Preprocessing {len(documents)} document chunks (currently a NOP).")
        # 示例：可以统一添加一个 ingestion_timestamp
        # import datetime
        # now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # for doc in documents:
        #     doc.metadata["ingestion_timestamp_utc"] = now_iso
        return documents

    def ingest_data(self, force_rebuild: bool = False) -> bool:
        """
        执行完整的数据注入流程。

        Args:
            force_rebuild: 如果为 True，则无论向量数据库是否已存在，都强制从头构建。

        Returns:
            True 如果注入成功完成，False 如果发生错误。
        """
        self.logger_instance.info("Starting data ingestion process...")

        # 1. 加载原始文档
        self.logger_instance.info(f"Loading documents from: {self.document_loader.source_description}")
        try:
            # 使用 lazy_load 减少内存占用，如果文档非常多
            # raw_documents = list(self.document_loader.lazy_load())
            # 或者直接 load，如果数据量可控
            raw_documents = self.document_loader.load()
            if not raw_documents:
                self.logger_instance.warning("No documents found by the document loader. Ingestion process will stop.")
                return True  # 没有文档也算是一种“成功”（没有错误发生）
            self.logger_instance.info(f"Successfully loaded {len(raw_documents)} raw documents.")
        except Exception as e:
            self.logger_instance.error(f"Failed to load documents: {e}", exc_info=True)
            return False

        # 2. 分割文档
        self.logger_instance.info(f"Splitting {len(raw_documents)} documents using: {self.text_splitter.description}")
        try:
            document_chunks = self.text_splitter.split_documents(raw_documents)
            if not document_chunks:
                self.logger_instance.warning("Text splitter returned no chunks. Ingestion process will stop.")
                return True
            self.logger_instance.info(f"Successfully split documents into {len(document_chunks)} chunks.")
        except Exception as e:
            self.logger_instance.error(f"Failed to split documents: {e}", exc_info=True)
            return False

        # 3. (可选) 预处理分割后的块
        processed_chunks = self._preprocess_documents(document_chunks)

        # 4. 存入向量数据库
        # 首先检查 Vector Store 状态，并根据 force_rebuild 决定行为
        vs_needs_build = False
        if force_rebuild:
            self.logger_instance.info("Force rebuild is enabled. Vector store will be built from scratch.")
            vs_needs_build = True
        elif not self.vector_store.is_initialized:
            # 尝试加载，如果AppBuilder中没有加载的话（作为后备）
            if not self.vector_store.load_local():
                self.logger_instance.info(
                    "Vector store is not initialized and no local data found. It needs to be built.")
                vs_needs_build = True
            else:
                self.logger_instance.info(
                    "Vector store was not initialized in memory but loaded successfully from local disk.")
                # 即使加载成功，也可能需要添加新文档（如果数据有更新）
                # 简单起见，如果加载成功，我们下面就直接 add_documents
        else:  # is_initialized is True
            self.logger_instance.info("Vector store is already initialized in memory.")
            # 这里可以添加逻辑来检查是否需要更新，例如基于文件哈希或时间戳
            # 为了简化，我们假设如果 VS 已初始化，我们就尝试增量添加（FAISS的add_documents是增量的）
            # 如果数据源是全新的，而VS是旧的，可能需要先清空VS或force_rebuild

        try:
            if vs_needs_build:
                self.logger_instance.info(f"Building vector store from {len(processed_chunks)} chunks...")
                if not self.vector_store.build_from_documents(processed_chunks):
                    self.logger_instance.error("Failed to build vector store.")
                    return False
                self.logger_instance.info("Vector store built successfully.")
            else:
                # Vector store 已存在 (或者刚刚加载成功)，尝试添加文档
                # 注意：这假设 add_documents 是增量的。对于某些 VS，重复添加相同内容可能导致重复。
                # FAISS 的 add_documents 是增量的。
                # 更复杂的场景需要文档ID和去重逻辑。
                self.logger_instance.info(f"Adding {len(processed_chunks)} chunks to existing vector store...")
                added_ids = self.vector_store.add_documents(processed_chunks)
                self.logger_instance.info(
                    f"Successfully added/updated documents in vector store. (Returned {len(added_ids)} IDs)")

            # (可选) 验证：可以执行一个简单的搜索来确认数据已进入
            # self._verify_ingestion(processed_chunks)

        except Exception as e:
            self.logger_instance.error(f"Failed to ingest documents into vector store: {e}", exc_info=True)
            return False

        self.logger_instance.info("Data ingestion process completed successfully.")
        return True

    def _verify_ingestion(self, ingested_chunks: List[Document]):
        """(可选) 执行一个简单的验证步骤，例如搜索一个刚添加的块。"""
        if not ingested_chunks or not self.vector_store.is_initialized:
            return

        sample_chunk_content = ingested_chunks[0].page_content
        self.logger_instance.info(f"Verifying ingestion by searching for content: '{sample_chunk_content[:30]}...'")
        try:
            results = self.vector_store.similarity_search(sample_chunk_content, k=1)
            if results and results[0].page_content == sample_chunk_content:
                self.logger_instance.info("Ingestion verification successful: Found the sample chunk.")
            else:
                self.logger_instance.warning("Ingestion verification failed or found different chunk.")
        except Exception as e:
            self.logger_instance.warning(f"Error during ingestion verification: {e}")