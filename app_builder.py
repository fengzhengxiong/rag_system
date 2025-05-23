#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：app_builder.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:43 
'''

from utils.config_models import (
    AppConfig,
    OllamaLLMConfig, LLMConfigType,
    OllamaEmbeddingConfig, EmbeddingConfigType,
    FaissVectorStoreConfig, VectorStoreConfigType,
    DirectoryLoaderConfig, DocumentLoaderConfigType,
    RecursiveCharacterTextSplitterConfig, TextSplitterConfigType
)

from core.interfaces.llm_interface import LLMInterface
from core.interfaces.embedding_interface import EmbeddingInterface
from core.interfaces.vector_store_interface import VectorStoreInterface
from core.interfaces.document_loader_interface import DocumentLoaderInterface
from core.interfaces.text_splitter_interface import TextSplitterInterface
from core.interfaces.retriever_interface import RetrieverInterface

from components.retrievers.bm25_retriever import BM25RetrieverImpl
from components.retrievers.vector_retriever import VectorRetrieverImpl
from components.retrievers.hybrid_retriever import HybridRetrieverImpl

from core.prompt_manager import PromptManager, PromptConfig
from core.rag_pipeline import RAGPipeline
from services.query_service import QueryService

from components.llms.ollama_llm import OllamaLLMImpl
from components.embeddings.ollama_embedding import OllamaEmbeddingImpl
from components.vector_stores.faiss_vs import FaissVectorStoreImpl
from components.document_loaders.directory_loader import DirectoryLoaderImpl
from components.text_splitters.recursive_character_splitter import RecursiveCharacterTextSplitterImpl

from services.ingestion_service import IngestionService

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

from utils.logger import RAGLogger
from typing import Optional, List

class AppBuilder:
    def __init__(self, config: AppConfig, logger: RAGLogger):
        self.config = config
        self.logger_instance = logger  # 直接使用传入的 RAGLogger 实例
        self.log = self.logger_instance.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.log.info("AppBuilder initialized.")
        self._llm_instance: Optional[LLMInterface] = None
        self._embedding_instance: Optional[EmbeddingInterface] = None
        self._vector_store_instance: Optional[VectorStoreInterface] = None
        self._document_loader_instance: Optional[DocumentLoaderInterface] = None
        self._text_splitter_instance: Optional[TextSplitterInterface] = None
        self._ingestion_service_instance: Optional[IngestionService] = None
        self._prompt_manager_instance: Optional[PromptManager] = None
        self._retriever_instance: Optional[RetrieverInterface] = None
        self._rag_pipeline_instance: Optional[RAGPipeline] = None
        self._query_service_instance: Optional[QueryService] = None
        self._all_documents_for_bm25: Optional[List[Document]] = None

    def create_llm(self) -> LLMInterface:
        if self._llm_instance:
            self.log.debug("Returning cached LLM instance.")
            return self._llm_instance

        # llm_config 已经是 Pydantic 解析后的具体配置对象实例
        llm_config: LLMConfigType = self.config.llm

        self.log.info(f"Attempting to create LLM of type: '{llm_config.type}' with model: '{llm_config.model_name}'")

        try:
            # 根据 llm_config 的实际类型来实例化
            if isinstance(llm_config, OllamaLLMConfig):  # 使用 isinstance 检查具体配置类型
                self._llm_instance = OllamaLLMImpl(llm_config, self.logger_instance)
            # elif isinstance(llm_config, AnotherLLMConfig): # 如果有其他LLM
            #     self._llm_instance = AnotherLLMImpl(llm_config, self.logger_instance)
            else:
                # 这种情况理论上不应该发生，如果 Pydantic 配置和联合类型都正确设置的话
                # 因为 Pydantic 在解析 AppConfig 时，如果 llm 字段的 type 不匹配 Union 中的任何一个，
                # 或者数据不符合对应模型的 schema，就会在加载配置时直接报错。
                error_msg = f"Internal error: LLM configuration object is of an unexpected type: {type(llm_config).__name__}. This might indicate an issue with Pydantic model definitions or config loading."
                self.log.error(error_msg)
                raise TypeError(error_msg)  # 或者更具体的自定义异常

            self.log.info(f"LLM instance of type '{llm_config.type}' created successfully.")
            return self._llm_instance
        except ConnectionError as e:  # 从 OllamaLLMImpl 捕获连接错误
            self.log.critical(f"Fatal error creating LLM: {e}", exc_info=True)
            # 根据应用需求决定是抛出，还是返回一个备用/空实现的 LLM
            raise  # 重新抛出，让上层 (main.py) 处理
        except Exception as e:
            self.log.error(f"An unexpected error occurred while creating LLM: {e}", exc_info=True)
            raise

    def create_embedding_model(self) -> EmbeddingInterface:  # <--- 新增方法
        if self._embedding_instance:
            self.log.debug("Returning cached Embedding model instance.")
            return self._embedding_instance

        embedding_config: EmbeddingConfigType = self.config.embedding
        self.log.info(
            f"Attempting to create Embedding model of type: '{embedding_config.type}' with model: '{embedding_config.model_name}'")

        try:
            if isinstance(embedding_config, OllamaEmbeddingConfig):
                self._embedding_instance = OllamaEmbeddingImpl(embedding_config, self.logger_instance)
            # elif isinstance(embedding_config, AnotherEmbeddingConfig):
            #     self._embedding_instance = AnotherEmbeddingImpl(embedding_config, self.logger_instance)
            else:
                error_msg = f"Unsupported Embedding configuration type: {type(embedding_config).__name__}."
                self.log.error(error_msg)
                raise ValueError(error_msg)

            self.log.info(f"Embedding model instance of type '{embedding_config.type}' created successfully.")
            return self._embedding_instance
        except ConnectionError as e:  # 从 OllamaEmbeddingImpl 捕获连接错误
            self.log.critical(f"Fatal error creating Embedding model: {e}", exc_info=True)
            raise
        except Exception as e:
            self.log.error(f"An unexpected error occurred while creating Embedding model: {e}", exc_info=True)
            raise

    def create_vector_store(self) -> VectorStoreInterface:  # <--- 新增方法
        if self._vector_store_instance and self._vector_store_instance.is_initialized:
            self.log.debug("Returning cached and initialized Vector Store instance.")
            return self._vector_store_instance

        vs_config: VectorStoreConfigType = self.config.vector_store
        self.log.info(f"Attempting to create Vector Store of type: '{vs_config.type}'")

        # Vector Store 依赖 Embedding Model，所以先创建/获取 Embedding Model
        embedding_model = self.create_embedding_model()  # 这会使用缓存或重新创建

        try:
            concrete_vs_instance: Optional[VectorStoreInterface] = None
            if isinstance(vs_config, FaissVectorStoreConfig):
                concrete_vs_instance = FaissVectorStoreImpl(vs_config, embedding_model, self.logger_instance)
            # elif isinstance(vs_config, AnotherVectorStoreConfig):
            #     concrete_vs_instance = AnotherVectorStoreImpl(vs_config, embedding_model, self.logger_instance)
            else:
                error_msg = f"Unsupported Vector Store configuration type: {type(vs_config).__name__}."
                self.log.error(error_msg)
                raise ValueError(error_msg)

            # 创建实例后，尝试加载已有的数据 (如果VS支持且配置了路径)
            # 这一步也可以移到 IngestionService 中，根据具体需求决定何时加载/构建
            if concrete_vs_instance:
                self.log.info(
                    f"Vector Store instance of type '{vs_config.type}' created. Attempting to load local data...")
                if concrete_vs_instance.load_local():
                    self.log.info(f"Successfully loaded local data for Vector Store '{vs_config.type}'.")
                else:
                    self.log.info(f"No local data loaded for Vector Store '{vs_config.type}'. It may need to be built.")

            self._vector_store_instance = concrete_vs_instance  # 缓存实例
            if not self._vector_store_instance:  # 如果 concrete_vs_instance 仍然是 None
                raise RuntimeError(f"Failed to create a concrete vector store instance for type '{vs_config.type}'")

            return self._vector_store_instance
        except ConnectionError as e:  # 例如，如果VS需要连接到远程服务
            self.log.critical(f"Fatal error creating Vector Store: {e}", exc_info=True)
            raise
        except Exception as e:
            self.log.error(f"An unexpected error occurred while creating Vector Store: {e}", exc_info=True)
            raise

    def create_document_loader(self) -> DocumentLoaderInterface:  # <--- 新增方法
        if self._document_loader_instance:
            self.log.debug("Returning cached Document Loader instance.")
            return self._document_loader_instance

        loader_config: DocumentLoaderConfigType = self.config.document_loader
        self.log.info(f"Attempting to create Document Loader of type: '{loader_config.type}'")

        try:
            if isinstance(loader_config, DirectoryLoaderConfig):
                # DirectoryLoaderImpl 需要 data_dir，从 AppConfig.paths 获取
                data_dir = self.config.paths.data_dir
                self._document_loader_instance = DirectoryLoaderImpl(loader_config, data_dir, self.logger_instance)
            # elif isinstance(loader_config, WebLoaderConfig):
            #     self._document_loader_instance = WebLoaderImpl(loader_config, self.logger_instance)
            else:
                error_msg = f"Unsupported Document Loader configuration type: {type(loader_config).__name__}."
                self.log.error(error_msg)
                raise ValueError(error_msg)

            self.log.info(
                f"Document Loader instance of type '{loader_config.type}' created successfully. Source: {self._document_loader_instance.source_description}")
            return self._document_loader_instance
        except Exception as e:
            self.log.error(f"An unexpected error occurred while creating Document Loader: {e}", exc_info=True)
            raise

    def create_text_splitter(self) -> TextSplitterInterface:  # <--- 新增方法
        if self._text_splitter_instance:
            self.log.debug("Returning cached Text Splitter instance.")
            return self._text_splitter_instance

        splitter_config: TextSplitterConfigType = self.config.text_splitter
        self.log.info(f"Attempting to create Text Splitter of type: '{splitter_config.type}'")

        try:
            if isinstance(splitter_config, RecursiveCharacterTextSplitterConfig):
                self._text_splitter_instance = RecursiveCharacterTextSplitterImpl(splitter_config, self.logger_instance)
            # elif isinstance(splitter_config, CharacterTextSplitterConfig):
            #     self._text_splitter_instance = CharacterTextSplitterImpl(splitter_config, self.logger_instance)
            else:
                error_msg = f"Unsupported Text Splitter configuration type: {type(splitter_config).__name__}."
                self.log.error(error_msg)
                raise ValueError(error_msg)

            self.log.info(
                f"Text Splitter instance of type '{splitter_config.type}' created successfully. Desc: {self._text_splitter_instance.description}")
            return self._text_splitter_instance
        except Exception as e:
            self.log.error(f"An unexpected error occurred while creating Text Splitter: {e}", exc_info=True)
            raise

    def build_ingestion_service(self) -> IngestionService:  # <--- 新增方法
        if self._ingestion_service_instance:
            self.log.debug("Returning cached IngestionService instance.")
            return self._ingestion_service_instance

        self.log.info("Building IngestionService...")
        try:
            doc_loader = self.create_document_loader()
            text_splitter = self.create_text_splitter()
            # embedding_model = self.create_embedding_model() # VS 内部会使用
            vector_store = self.create_vector_store()  # VS 创建时已注入 Embedding

            self._ingestion_service_instance = IngestionService(
                document_loader=doc_loader,
                text_splitter=text_splitter,
                vector_store=vector_store,
                logger=self.logger_instance,  # 传入 RAGLogger 实例
                app_config=self.config  # 传入 AppConfig
            )
            self.log.info("IngestionService built successfully.")
            return self._ingestion_service_instance
        except Exception as e:
            self.log.error(f"Failed to build IngestionService: {e}", exc_info=True)
            raise

    def _get_all_documents_for_bm25(self) -> List[Document]:
        """辅助方法：加载并缓存所有文档，供BM25初始化使用。"""
        if self._all_documents_for_bm25 is None:
            self.log.info("Loading all documents for BM25 retriever initialization...")
            # 需要一个 DocumentLoader 实例来加载文档
            # 这里我们假设 IngestionService 之前可能已经加载过，
            # 或者我们在这里即时加载。为简单起见，我们创建一个临时的loader。
            # 更好的方式是在AppBuilder中先确保文档已通过IngestionService处理并缓存
            # 或者 IngestionService 提供一个获取所有已处理文档的方法。
            # 目前，我们直接创建一个 DocumentLoader。
            # 注意：这可能会导致重复加载，如果 IngestionService 也加载的话。
            # 理想情况下，文档加载和分割应该只发生一次。

            # 简化处理：直接使用配置创建 DocumentLoader 加载
            # 这个临时的 DocumentLoader 应该使用与 IngestionService 相同的配置
            temp_doc_loader_config = self.config.document_loader  # DocumentLoaderConfigType
            #  data_dir from app_config.paths
            data_dir_for_bm25 = self.config.paths.data_dir

            # 动态创建loader实例 (这部分逻辑与 create_document_loader 类似，可以提取为公共方法)
            if isinstance(temp_doc_loader_config, DirectoryLoaderConfig):
                loader_for_bm25 = DirectoryLoaderImpl(temp_doc_loader_config, data_dir_for_bm25, self.logger_instance)
            # elif ...其他loader类型...
            else:
                raise ValueError(f"Unsupported document loader type for BM25: {temp_doc_loader_config.type}")

            self._all_documents_for_bm25 = loader_for_bm25.load()
            if not self._all_documents_for_bm25:
                self.log.warning("BM25Retriever: No documents loaded. BM25 may not work correctly.")
                self._all_documents_for_bm25 = []  # 确保是列表
            else:
                self.log.info(f"Loaded {len(self._all_documents_for_bm25)} documents for BM25.")
        return self._all_documents_for_bm25

    def create_retriever(self) -> RetrieverInterface:
        if self._retriever_instance:
            self.log.debug("Returning cached Retriever instance.")
            return self._retriever_instance

        ret_main_config: MainRetrieverConfig = self.config.retriever
        self.log.info(f"Attempting to create Retriever with strategy: '{ret_main_config.strategy}'")

        vs_instance = self.create_vector_store()  # Vector Store 是 Vector Retriever 的依赖

        if ret_main_config.strategy == "vector":
            if not ret_main_config.vector_config:
                raise ValueError("Vector retriever strategy selected, but 'vector_config' is missing.")
            self._retriever_instance = VectorRetrieverImpl(
                ret_main_config.vector_config, vs_instance, self.logger_instance
            )
        elif ret_main_config.strategy == "bm25":
            if not ret_main_config.bm25_config:
                raise ValueError("BM25 retriever strategy selected, but 'bm25_config' is missing.")
            all_docs = self._get_all_documents_for_bm25()  # 获取所有文档用于BM25初始化
            if not all_docs:  # 如果没有文档，BM25无法初始化
                self.log.error("Cannot initialize BM25Retriever: No documents available.")
                raise RuntimeError("BM25Retriever initialization failed due to no documents.")
            self._retriever_instance = BM25RetrieverImpl(
                ret_main_config.bm25_config, all_docs, self.logger_instance
            )
        elif ret_main_config.strategy == "hybrid":
            if not ret_main_config.vector_config or not ret_main_config.bm25_config or not ret_main_config.hybrid_config:
                raise ValueError(
                    "Hybrid retriever strategy selected, but one or more required configs (vector, bm25, hybrid) are missing.")

            all_docs_hybrid = self._get_all_documents_for_bm25()
            if not all_docs_hybrid:
                self.log.error("Cannot initialize BM25 part of HybridRetriever: No documents available.")
                raise RuntimeError("HybridRetriever BM25 initialization failed due to no documents.")

            bm25_comp = BM25RetrieverImpl(ret_main_config.bm25_config, all_docs_hybrid, self.logger_instance)
            vector_comp = VectorRetrieverImpl(ret_main_config.vector_config, vs_instance, self.logger_instance)
            self._retriever_instance = HybridRetrieverImpl(
                ret_main_config.hybrid_config, bm25_comp, vector_comp, self.logger_instance
            )
        else:
            error_msg = f"Unsupported Retriever strategy: {ret_main_config.strategy}."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        self.log.info(
            f"Retriever (Strategy: {ret_main_config.strategy}) created successfully. Desc: {self._retriever_instance.description}")
        return self._retriever_instance

    def create_prompt_manager(self) -> PromptManager:
        if self._prompt_manager_instance:
            self.log.debug("Returning cached PromptManager instance.")
            return self._prompt_manager_instance

        # PromptManager 使用 AppConfig.prompts (PromptConfigModel)
        # 我们需要将 Pydantic 模型转换为 PromptManager 构造函数期望的 PromptConfig (如果它们不同)
        # 或者让 PromptManager 直接接受 PromptConfigModel
        # 为了简单，假设 PromptManager 的构造函数可以直接使用 AppConfig.prompts
        # 如果 PromptManager 内部的 PromptConfig 与 utils.config_models.PromptConfigModel 定义一致

        # 假设 PromptManager 接受 PromptConfigModel
        prompt_config_model_instance = self.config.prompts

        # 如果 PromptManager 期望的是它内部定义的 PromptConfig，需要转换：
        # internal_prompt_config = PromptConfig(qa_template_str=prompt_config_model_instance.qa_template_str)
        # self._prompt_manager_instance = PromptManager(internal_prompt_config)

        # 假设 PromptManager(config: PromptConfigModel)
        self._prompt_manager_instance = PromptManager(prompt_config_model_instance)
        self.log.info("PromptManager created successfully.")
        return self._prompt_manager_instance

    def build_rag_pipeline(self) -> RAGPipeline:  # 通常由 QueryService 内部调用或这里构建并传入
        if self._rag_pipeline_instance:
            self.log.debug("Returning cached RAGPipeline instance.")
            return self._rag_pipeline_instance

        self.log.info("Building RAGPipeline...")
        try:
            llm_instance = self.create_llm()  # 获取 LLMInterface
            retriever_instance = self.create_retriever()  # 获取 RetrieverInterface
            prompt_manager = self.create_prompt_manager()

            self._rag_pipeline_instance = RAGPipeline(
                llm=llm_instance.get_langchain_llm(),  # RAGPipeline 需要 BaseLanguageModel
                retriever=retriever_instance.get_langchain_retriever(),  # RAGPipeline 需要 BaseRetriever
                prompt_template=prompt_manager.get_qa_prompt_template(),
                rag_chain_config=self.config.rag_chain,
                memory_config=self.config.memory,  # 传入 MemoryConfig
                logger=self.logger_instance
            )
            self.log.info("RAGPipeline built successfully.")
            return self._rag_pipeline_instance
        except Exception as e:
            self.log.error(f"Failed to build RAGPipeline: {e}", exc_info=True)
            raise

    def build_query_service(self) -> QueryService:
        if self._query_service_instance:
            self.log.debug("Returning cached QueryService instance.")
            return self._query_service_instance

        self.log.info("Building QueryService...")
        try:
            rag_pipeline = self.build_rag_pipeline()  # 构建或获取 RAGPipeline

            self._query_service_instance = QueryService(
                rag_pipeline=rag_pipeline,
                logger=self.logger_instance
            )
            self.log.info("QueryService built successfully.")
            return self._query_service_instance
        except Exception as e:
            self.log.error(f"Failed to build QueryService: {e}", exc_info=True)
            raise
