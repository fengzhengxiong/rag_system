#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：ollama_llm.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:48 
'''

from typing import Any, List, Optional, Iterator, AsyncIterator, Union
from langchain_community.llms import Ollama as LangchainOllama # 从 langchain_community 导入
from langchain_core.outputs import LLMResult, GenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

from core.interfaces.llm_interface import LLMInterface
from utils.config_models import OllamaLLMConfig # 导入具体的配置模型
from utils.logger import RAGLogger # 导入日志记录器

class OllamaLLMImpl(LLMInterface):
    """
    使用 Ollama 的大语言模型实现。
    """
    def __init__(self, config: OllamaLLMConfig, logger: RAGLogger):
        self.config = config
        self.logger_instance = logger.get_logger(f"{__name__}.{self.__class__.__name__}") # 获取带类名的logger

        try:
            self._model = LangchainOllama(
                model=self.config.model_name,
                temperature=self.config.temperature,
                num_ctx=self.config.num_ctx,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                # base_url=self.config.base_url, # 如果在配置中定义了 base_url
                # 其他 Ollama 支持的参数...
            )
            self.logger_instance.info(
                f"Ollama LLM initialized successfully with model: '{self.config.model_name}'."
            )
        except Exception as e:
            self.logger_instance.error(
                f"Failed to initialize Ollama LLM with model '{self.config.model_name}': {e}",
                exc_info=True
            )
            # 可以选择在这里抛出自定义异常，或者让 AppBuilder 处理
            raise ConnectionError(f"Could not connect or initialize Ollama model '{self.config.model_name}'. Ensure Ollama server is running and the model is available. Original error: {e}")


    def get_langchain_llm(self) -> LangchainOllama:
        return self._model

    @property
    def model_name(self) -> str:
        return self.config.model_name

    async def agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        self.logger_instance.debug(f"Async generating for {len(prompts)} prompts. First prompt: '{prompts[0][:50]}...'")
        try:
            return await self._model.agenerate(prompts, stop=stop, **kwargs)
        except Exception as e:
            self.logger_instance.error(f"Error during async generation: {e}", exc_info=True)
            # 根据需要返回一个空的LLMResult或重新抛出异常
            # return LLMResult(generations=[[] for _ in prompts], llm_output={"error": str(e)})
            raise

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        self.logger_instance.debug(f"Sync generating for {len(prompts)} prompts. First prompt: '{prompts[0][:50]}...'")
        try:
            return self._model.generate(prompts, stop=stop, **kwargs)
        except Exception as e:
            self.logger_instance.error(f"Error during sync generation: {e}", exc_info=True)
            # return LLMResult(generations=[[] for _ in prompts], llm_output={"error": str(e)})
            raise

    async def astream(
        self,
        input: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, # Langchain 0.1.0+
        **kwargs: Any,
    ) -> AsyncIterator[Union[GenerationChunk, str]]: # Ollama 通常是 str
        self.logger_instance.debug(f"Async streaming for input: '{input[:50]}...'")
        try:
            # Langchain Ollama 的 astream 可能直接返回 AsyncIterator[str]
            # 或者 AsyncIterator[GenerationChunk]
            # 需要检查你使用的 Langchain 版本中 Ollama 的具体实现
            # Langchain 0.1.0+ 的 stream/astream 签名有所改变
            # async for chunk in self._model.astream(input, stop=stop, **kwargs): # 老版本 langchain
            #    yield chunk
            # 对于较新版本的 Langchain (例如 0.1.x)，astream 的调用方式和返回类型可能略有不同
            # 确保你的 Langchain 版本与这里的用法兼容
            # 通常，Ollama 的流式输出是字符串片段
            async for chunk_text in self._model.astream(
                input,
                stop=stop,
                # run_manager=run_manager, # 如果需要传递 run_manager
                **kwargs
            ):
                yield chunk_text # Ollama 通常直接 yield 字符串
        except Exception as e:
            self.logger_instance.error(f"Error during async streaming: {e}", exc_info=True)
            # 在流中抛出异常可能难以处理，可以考虑 yield 一个错误标记或记录后停止
            # raise # 或者不重新抛出，让调用者处理空的迭代器

    def stream(
        self,
        input: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, # Langchain 0.1.0+
        **kwargs: Any,
    ) -> Iterator[Union[GenerationChunk, str]]: # Ollama 通常是 str
        self.logger_instance.debug(f"Sync streaming for input: '{input[:50]}...'")
        try:
            # for chunk in self._model.stream(input, stop=stop, **kwargs): # 老版本 langchain
            #    yield chunk
            for chunk_text in self._model.stream(
                input,
                stop=stop,
                # run_manager=run_manager,
                **kwargs
            ):
                yield chunk_text
        except Exception as e:
            self.logger_instance.error(f"Error during sync streaming: {e}", exc_info=True)
            # raise