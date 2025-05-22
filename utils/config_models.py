#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：config_models.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 15:54 
'''

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Union

# --- 基础组件配置模型 ---
class BaseComponentModel(BaseModel):
    """所有组件配置模型的基类，方便统一处理或扩展"""
    type: str # 用于动态加载不同类型的组件实现

# --- 日志配置 ---
class LoggingConfig(BaseModel):
    log_dir: str = Field("logs", description="日志文件存放目录")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="日志级别"
    )
    max_bytes: int = Field(10 * 1024 * 1024, description="单个日志文件最大大小 (字节)") # 10MB
    backup_count: int = Field(5, description="保留的备份文件数量")

# --- 路径配置 ---
class PathsConfig(BaseModel):
    data_dir: str = Field("data", description="原始数据目录")
    vector_store_dir: str = Field("vector_store_data", description="向量数据库持久化路径")
    cache_db_path: str = Field(".rag_cache.db", description="SQLite缓存数据库路径")

# --- LLM 具体配置模型 ---
class OllamaLLMConfig(BaseComponentModel):
    type: Literal["ollama"] = "ollama"
    model_name: str = Field("deepseek-coder:6.7b-instruct", description="Ollama 模型的名称 (例如 'llama2', 'mistral')")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="生成文本的随机性")
    num_ctx: Optional[int] = Field(None, description="上下文窗口大小 (tokens)，如果为None则使用模型默认")
    top_k: Optional[int] = Field(None, description="Top-k 采样")
    top_p: Optional[float] = Field(None, description="Top-p (nucleus) 采样")
    repeat_penalty: Optional[float] = Field(None, description="重复惩罚系数")
    # 可以根据 Ollama 支持的参数添加更多字段
    # base_url: str = Field("http://localhost:11434", description="Ollama 服务的基础 URL") # 如果需要非默认

    @validator('model_name')
    def model_name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Ollama model_name cannot be empty')
        return v

class AnotherLLMConfig(BaseComponentModel): # 示例：如果你有另一个 LLM 类型
    type: Literal["another_llm_provider"] = "another_llm_provider"
    model_name: str
    api_key: Optional[str] = Field(None, description="API 密钥")
    # ... 其他特定于此提供商的参数

# LLM 配置的联合类型，允许 AppConfig 接受多种 LLM 配置
LLMConfigType = Union[OllamaLLMConfig, AnotherLLMConfig] # 在这里添加所有支持的LLM配置类型

# --- Embedding 配置占位符 (后续步骤会具体化) ---
class GenericEmbeddingConfig(BaseComponentModel):
    model_name: str

# --- 缓存配置 ---
class CacheConfig(BaseModel):
    enable: bool = Field(True, description="是否启用缓存")
    type: Literal["sqlite", "memory", "disabled"] = Field("sqlite", description="缓存类型")
    max_size: int = Field(1000, description="缓存最大条目数", gt=0)
    ttl_seconds: Optional[int] = Field(3600, description="缓存过期时间 (秒), None表示永不过期, 主要用于内存缓存")

    @validator('type')
    def disable_cache_if_type_disabled(cls, v, values):
        if v == "disabled":
            values['enable'] = False
        return v

    @validator('enable')
    def ensure_type_not_disabled_if_enabled(cls, v, values):
        if v and values.get('type') == "disabled":
            raise ValueError("Cache cannot be enabled if type is 'disabled'. Set type to 'sqlite' or 'memory'.")
        return v

# --- 主应用配置模型 ---
class AppConfig(BaseModel):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfigType # <--- 修改这里，使用联合类型
    embedding: GenericEmbeddingConfig # 暂时保持不变
    cache: CacheConfig = Field(default_factory=CacheConfig)

    class Config:
        validate_assignment = True