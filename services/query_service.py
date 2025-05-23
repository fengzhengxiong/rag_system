#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：query_service.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:52 
'''

# services/query_service.py
from typing import Optional, List, Dict, Any
from core.rag_pipeline import RAGPipeline
from utils.logger import RAGLogger
from langchain_core.documents import Document
from utils.config_models import AppConfig  # QueryService可能不需要AppConfig，除非有特定配置


class QueryService:
    """
    处理用户查询并返回答案的服务。
    管理 RAGPipeline 和对话状态。
    """

    def __init__(self, rag_pipeline: RAGPipeline, logger: RAGLogger):
        self.rag_pipeline = rag_pipeline
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger_instance.info("QueryService initialized.")

    def ask(self, query: str, session_id: Optional[str] = None) -> str:  # session_id 用于未来多用户场景
        """
        向 RAG 系统提问。

        Args:
            query: 用户的问题。
            session_id: (可选) 会话ID，用于区分不同用户的对话历史 (如果记忆是会话级别的)。
                        目前我们的 ConversationBufferMemory 是实例级别的。

        Returns:
            AI生成的答案字符串。
        """
        self.logger_instance.info(
            f"Received query (Session: {session_id if session_id else 'default'}): '{query[:100]}...'")

        # TODO: 如果需要会话级记忆，这里需要根据 session_id 获取或创建对应的 RAGPipeline/Memory
        # 目前，RAGPipeline 内部的 memory 是共享的（如果启用）

        try:
            response_dict = self.rag_pipeline.invoke(query)

            if "error" in response_dict:
                self.logger_instance.error(f"Error from RAG pipeline: {response_dict['error']}")
                return f"抱歉，处理您的请求时发生错误: {response_dict['error']}"

            answer = response_dict.get("result", "抱歉，我无法回答这个问题。")

            source_documents = response_dict.get("source_documents")
            if source_documents:
                self.logger_instance.info(f"Retrieved {len(source_documents)} source documents.")
                # for i, doc in enumerate(source_documents):
                #     self.logger_instance.debug(f"  Source {i+1} ({doc.metadata.get('source', 'N/A')}): '{doc.page_content[:50]}...'")
            else:
                self.logger_instance.info("No source documents were returned or found.")

            return str(answer)  # 确保是字符串
        except Exception as e:
            self.logger_instance.error(f"Unexpected error during 'ask' operation: {e}", exc_info=True)
            return "抱歉，系统遇到未知错误，请稍后再试。"

    def clear_conversation_history(self, session_id: Optional[str] = None):
        """清空当前会话的对话历史。"""
        # TODO: 如果是会话级记忆，根据 session_id 操作
        self.logger_instance.info(
            f"Clearing conversation history (Session: {session_id if session_id else 'default'}).")
        self.rag_pipeline.clear_memory()

    def get_last_source_documents(self) -> List[Document]:
        """
        (示例) 获取上一次查询的源文档。
        这需要 RAGPipeline 或其底层的链能够存储这些信息。
        RetrievalQA 的结果中通常包含 source_documents。
        QueryService 需要一种方式来缓存上一次的结果（如果需要这个功能）。
        目前，我们假设调用者会从 ask 的完整结果中获取。
        """
        # self.logger_instance.warning("get_last_source_documents is a conceptual method and needs proper implementation if required.")
        # For now, this is a placeholder. One would typically get this from the output of 'ask'.
        return []