#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：app_builder.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:43 
'''


from typing import Optional

# 从 config_models 导入 Pydantic 模型定义
from utils.config_models import AppConfig, OllamaLLMConfig, LLMConfigType
# 如果将来有其他 LLM 配置模型，如 OpenAILLMConfig，也从这里导入
# from utils.config_models import OpenAILLMConfig, AnotherLLMConfig

from core.interfaces.llm_interface import LLMInterface
# from core.interfaces.embedding_interface import EmbeddingInterface # 暂时注释，下一步用
# from core.interfaces.vector_store_interface import VectorStoreInterface # 暂时注释
# from core.interfaces.retriever_interface import RetrieverInterface # 暂时注释
# from core.interfaces.cache_interface import CacheInterface # 暂时注释

# 导入具体的 LLM 实现
from components.llms.ollama_llm import OllamaLLMImpl
# 如果有其他 LLM 实现，也在这里导入
# from components.llms.another_llm import AnotherLLMImpl

from utils.logger import RAGLogger


# from services.ingestion_service import IngestionService # 暂时注释
# from services.query_service import QueryService # 暂时注释
# from core.rag_pipeline import RAGPipeline # 暂时注释
# from core.prompt_manager import PromptManager # 暂时注释
# import langchain # 用于设置全局缓存 # 暂时注释

class AppBuilder:
    def __init__(self, config: AppConfig, logger: RAGLogger):
        self.config = config
        self.logger_instance = logger  # 直接使用传入的 RAGLogger 实例
        self.log = self.logger_instance.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.log.info("AppBuilder initialized.")
        self._llm_instance: Optional[LLMInterface] = None  # 缓存创建的LLM实例

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

    # --- 其他 create_xxx 方法占位符 ---
    # def create_embedding_model(self) -> EmbeddingInterface: ...
    # def create_vector_store(self, embedding_model: EmbeddingInterface) -> VectorStoreInterface: ...
    # def create_retriever(self, vector_store: VectorStoreInterface) -> RetrieverInterface: ...
    # def create_cache(self) -> Optional[CacheInterface]: ...
    # def build_ingestion_service(self) -> IngestionService: ...
    # def build_query_service(self) -> QueryService: ...