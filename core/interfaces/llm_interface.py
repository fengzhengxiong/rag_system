#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：llm_interface.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:44 
'''

from abc import ABC, abstractmethod
from typing import Any, List, AsyncIterator, Iterator, Union # For streaming
from langchain_core.language_models.llms import LLM as LangchainLLM # 使用别名避免与接口名冲突
from langchain_core.outputs import LLMResult, GenerationChunk, ChatGenerationChunk # ChatGenerationChunk for chat models

class LLMInterface(ABC):
    """
    大语言模型接口。
    所有具体的 LLM 实现都应继承此类并实现其方法。
    """

    @abstractmethod
    def get_langchain_llm(self) -> LangchainLLM:
        """返回底层的 Langchain 兼容的 LLM 实例。"""
        pass

    @abstractmethod
    async def agenerate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
        """
        异步为一组提示词生成回复。

        Args:
            prompts: 提示词列表。
            **kwargs:传递给底层 LLM 的额外参数。

        Returns:
            LLMResult 对象，包含生成的回复。
        """
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
        """
        同步为一组提示词生成回复。

        Args:
            prompts: 提示词列表。
            **kwargs: 传递给底层 LLM 的额外参数。

        Returns:
            LLMResult 对象，包含生成的回复。
        """
        pass

    @abstractmethod
    async def astream(self, input: str, **kwargs: Any) -> AsyncIterator[Union[GenerationChunk, ChatGenerationChunk, str]]: # noqa: E501
        """
        异步流式生成回复。
        返回的类型取决于底层模型是 LLM 还是 ChatModel，或者是否直接返回字符串。
        对于 Ollama 这样的 LLM，通常是 GenerationChunk 或 str。
        """
        pass

    @abstractmethod
    def stream(self, input: str, **kwargs: Any) -> Iterator[Union[GenerationChunk, ChatGenerationChunk, str]]: # noqa: E501
        """
        同步流式生成回复。
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前 LLM 实例所使用的模型名称。"""
        pass

    # 可以考虑添加获取模型元数据的方法，例如最大上下文长度等
    # def get_model_metadata(self) -> Dict[str, Any]:
    #     pass