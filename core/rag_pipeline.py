#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：rag_pipeline.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:44 
'''

# core/rag_pipeline.py
from typing import Dict, Any, Optional, List
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLanguageModel  # LLMInterface.get_langchain_llm() 返回这个
from langchain_core.retrievers import BaseRetriever  # RetrieverInterface.get_langchain_retriever() 返回这个
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # 或其他类型的 Memory
from langchain_core.documents import Document

from utils.logger import RAGLogger
from utils.config_models import RAGChainConfig, MemoryConfig  # 导入相关配置


class RAGPipeline:
    def __init__(
            self,
            llm: BaseLanguageModel,
            retriever: BaseRetriever,
            prompt_template: PromptTemplate,
            rag_chain_config: RAGChainConfig,
            memory_config: MemoryConfig,  # 传入 MemoryConfig
            logger: RAGLogger
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.rag_chain_config = rag_chain_config
        self.memory_config = memory_config  # 保存 MemoryConfig
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.memory: Optional[ConversationBufferMemory] = None  # 初始化 memory 为 None
        self.qa_chain: Optional[RetrievalQA] = None  # 初始化 qa_chain 为 None

        self._build_chain()
        self.logger_instance.info("RAGPipeline initialized with RetrievalQA chain.")

    def _build_memory(self) -> Optional[ConversationBufferMemory]:
        if not self.memory_config.enable:
            self.logger_instance.info("Conversation memory is disabled.")
            return None

        if self.memory_config.memory_type == "conversation_buffer":
            self.logger_instance.info(
                f"Building ConversationBufferMemory. Memory key: '{self.memory_config.memory_key}', "
                f"Input key: '{self.memory_config.input_key}', Output key: '{self.memory_config.output_key}'"
            )
            return ConversationBufferMemory(
                memory_key=self.memory_config.memory_key,
                input_key=self.memory_config.input_key,  # Important for RetrievalQA
                output_key=self.memory_config.output_key,  # Important for RetrievalQA
                return_messages=True  # 通常 RetrievalQA 的 memory 需要这个
            )
        # elif self.memory_config.memory_type == "conversation_token_buffer":
        #     return ConversationTokenBufferMemory(...)
        else:
            self.logger_instance.warning(
                f"Unsupported memory type: {self.memory_config.memory_type}. Memory will be disabled.")
            return None

    def _build_chain(self):
        self.memory = self._build_memory()  # 构建或获取 memory 实例

        chain_type_kwargs = {"prompt": self.prompt_template}
        if self.memory:  # 只有当 memory 启用时才加入到 chain_type_kwargs
            # RetrievalQA 的 memory 是直接传递给构造函数的，而不是 chain_type_kwargs
            pass
        else:  # 如果没有 memory，确保 chain_type_kwargs 不包含 memory 相关键
            # chain_type_kwargs.pop("memory", None) # 确保没有 memory 键
            pass

        self.logger_instance.info(
            f"Building RetrievalQA chain. Type: '{self.rag_chain_config.chain_type}', "
            f"Return source docs: {self.rag_chain_config.return_source_documents}"
        )

        # RetrievalQA 构造函数直接接受 memory 参数
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.rag_chain_config.chain_type,
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=self.rag_chain_config.return_source_documents,
            memory=self.memory,  # <--- 在这里传递 memory 实例
            # verbose=self.rag_chain_config.verbose, # Langchain 的 verbose 通常由全局日志或回调控制
        )

    def invoke(self, query: str, chat_history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        执行 RAG 查询。

        Args:
            query: 用户查询字符串。
            chat_history: (可选) 传入的聊天历史，如果记忆不由链内部管理。
                          对于 RetrievalQA 和 ConversationBufferMemory，通常不需要显式传入，
                          除非你想在每次调用时重置或提供外部历史。

        Returns:
            一个字典，通常包含 'result' (答案) 和 'source_documents' (如果配置返回)。
        """
        if not self.qa_chain:
            self.logger_instance.error("QA chain is not built. Cannot invoke.")
            return {"error": "QA chain not available."}

        self.logger_instance.debug(f"Invoking RAG pipeline with query: '{query[:50]}...'")

        # 构建输入字典
        # 对于有 memory 的 RetrievalQA，输入通常只需要 'query'
        # memory 会自动从 'query' 和 'result' 中提取并更新历史
        inputs = {"query": query}

        # 如果 memory 是外部管理的，或者你想在调用时提供特定的历史记录
        # (这通常不适用于与 RetrievalQA 结合的 ConversationBufferMemory，因为它自己管理)
        # if chat_history is not None and self.memory and hasattr(self.memory, 'chat_memory'):
        #     # 这里需要根据 memory 类型具体处理如何设置外部历史
        #     # 例如，对于 BufferMemory，可能是 self.memory.chat_memory.messages = chat_history
        #     self.logger_instance.debug(f"Using provided chat history with {len(chat_history)} messages.")
        #     # inputs[self.memory.memory_key] = chat_history # 如果 memory key 不同

        try:
            result = self.qa_chain.invoke(inputs)
            # self.logger_instance.debug(f"RAG pipeline invocation result: {result}")
            return result
        except Exception as e:
            self.logger_instance.error(f"Error during RAG pipeline invocation: {e}", exc_info=True)
            return {"error": str(e), "source_documents": []}

    def clear_memory(self):
        """清空对话记忆 (如果启用了记忆)。"""
        if self.memory:
            self.memory.clear()
            self.logger_instance.info("Conversation memory cleared.")
        else:
            self.logger_instance.info("Memory is not enabled, nothing to clear.")