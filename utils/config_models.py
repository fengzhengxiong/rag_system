#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：config_models.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 15:54 
'''

from pydantic import BaseModel, Field, validator, DirectoryPath
from typing import Literal, Optional, Union, List
from pathlib import Path

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

# --- 路径配置 (修改，明确vector_store_dir的用途) ---
class PathsConfig(BaseModel):
    data_dir: DirectoryPath = Field("data", description="原始数据文档目录") # 使用DirectoryPath确保是目录
    # vector_store_dir 移到具体的 VectorStoreConfig 中，因为不同 VS 可能有不同存储方式
    cache_db_path: str = Field(".rag_cache.db", description="SQLite缓存数据库路径")

    @validator('data_dir', pre=True, always=True)
    def ensure_data_dir_exists(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

# --- LLM 具体配置模型 ---
class OllamaLLMConfig(BaseComponentModel):
    type: Literal["ollama"] = "ollama"
    model_name: str = Field("deepseek-r1:8b", description="Ollama 模型的名称 (例如 'llama2', 'mistral')")
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

# --- Embedding 具体配置模型 ---
class OllamaEmbeddingConfig(BaseComponentModel):
    type: Literal["ollama_embedding"] = "ollama_embedding"
    model_name: str = Field("nomic-embed-text:latest", description="Ollama 提供的嵌入模型名称 (例如 'nomic-embed-text', 'mxbai-embed-large')")
    # base_url: str = Field("http://localhost:11434", description="Ollama 服务的基础 URL") # 如果需要非默认
    # 可以根据 langchain_community.embeddings.OllamaEmbeddings 支持的参数添加更多字段
    #例如： num_thread, headers 等

    @validator('model_name')
    def model_name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Ollama embedding model_name cannot be empty')
        return v

class AnotherEmbeddingConfig(BaseComponentModel): # 示例：如果你有另一个 Embedding 提供商
    type: Literal["another_embedding_provider"] = "another_embedding_provider"
    model_name: str
    api_key: Optional[str] = Field(None, description="API 密钥")
    # ... 其他特定于此提供商的参数

# Embedding 配置的联合类型
EmbeddingConfigType = Union[OllamaEmbeddingConfig, AnotherEmbeddingConfig]

# --- Vector Store 具体配置模型 ---
class FaissVectorStoreConfig(BaseComponentModel):
    type: Literal["faiss"] = "faiss"
    # 持久化路径现在是 VS 配置的一部分
    persist_directory: DirectoryPath = Field("vector_store_data/faiss_index", description="FAISS索引持久化目录")
    index_name: str = Field("index", description="FAISS索引文件名 (不含扩展名)") # 例如 "index", 会生成 "index.faiss", "index.pkl"
    allow_dangerous_deserialization: bool = Field(True, description="是否允许FAISS进行可能不安全的反序列化 (pickle)")
    # FAISS 特有参数 (可以根据需要添加更多)
    # index_type: Literal["Flat", "IVF"] = "IVF" # 之前在全局 Config 中的
    # index_params: Dict[str, Any] = Field(default_factory=dict)

    # 在初始化或加载时创建目录
    @validator('persist_directory', pre=True, always=True)
    def ensure_persist_directory_exists(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

class AnotherVectorStoreConfig(BaseComponentModel): # 示例
    type: Literal["another_vs_provider"] = "another_vs_provider"
    # ...特定于此VS的配置...

# Vector Store 配置的联合类型
VectorStoreConfigType = Union[FaissVectorStoreConfig, AnotherVectorStoreConfig]

# --- Document Loader 具体配置模型 ---
class DirectoryLoaderConfig(BaseComponentModel):
    type: Literal["directory_loader"] = "directory_loader"
    # data_dir 将从 AppConfig.paths.data_dir 获取，这里定义特定于加载器的参数
    glob_pattern: str = Field("**/*.txt", description="用于匹配文件的 glob 模式 (例如 '**/*.txt', '*.md')")
    # loader_cls_name: Optional[str] = Field(None, description="Langchain Loader 类的名称 (如 'TextLoader', 'UnstructuredFileLoader')，如果loader_kwargs需要特定加载器")
    use_multithreading: bool = Field(False, description="是否使用多线程加载文件")
    # max_concurrency: Optional[int] = Field(None, description="最大并发线程数 (如果 use_multithreading 为 True)") # Langchain DirectoryLoader 可能没有此参数
    silent_errors: bool = Field(False, description="加载单个文件出错时是否静默处理")
    # loader_kwargs: Dict[str, Any] = Field(default_factory=dict, description="传递给具体文件加载器 (如 TextLoader) 的参数")
    # 例如 TextLoader 的 autodetect_encoding
    text_loader_autodetect_encoding: bool = Field(True, description="TextLoader是否自动检测编码")


class WebLoaderConfig(BaseComponentModel): # 示例：未来可以添加网页加载器
    type: Literal["web_loader"] = "web_loader"
    urls: List[str] = Field(..., description="要加载的网页 URL 列表")
    # ... 其他网页加载特定参数 ...

# Document Loader 配置的联合类型
DocumentLoaderConfigType = Union[DirectoryLoaderConfig, WebLoaderConfig]

# --- Text Splitter 具体配置模型 ---
class RecursiveCharacterTextSplitterConfig(BaseComponentModel):
    type: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = Field(500, gt=0, description="每个文本块的最大长度 (字符数)")
    chunk_overlap: int = Field(50, ge=0, description="文本块之间的重叠字符数")
    # length_function_name: str = Field("len", description="用于计算长度的函数名 ('len' 或自定义)")
    # separators: Optional[List[str]] = Field(None, description="自定义分隔符列表")
    keep_separator: bool = Field(True, description="是否保留分隔符")
    # add_start_index: bool = Field(False, description="是否在元数据中添加块的起始索引") # Langchain 参数


class CharacterTextSplitterConfig(BaseComponentModel):
    type: Literal["character"] = "character"
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    separator: str = Field("\n\n", description="用于分割的字符") # CharacterTextSplitter 使用单个 separator
    # ... 其他 CharacterTextSplitter 参数

# Text Splitter 配置的联合类型
TextSplitterConfigType = Union[RecursiveCharacterTextSplitterConfig, CharacterTextSplitterConfig]


# --- Prompt 配置 (可以放在这里，或者由 PromptManager 内部定义) ---
class PromptConfigModel(BaseModel): # 与 PromptManager 中的 PromptConfig 对应
    qa_template_str: str = Field(
        """基于以下上下文已知信息，请简洁并专业地回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。

上下文内容:
{context}

问题:
{question}""",
        description="问答任务的提示词模板字符串"
    )
    # condense_question_template_str: Optional[str] = Field(None, description="用于压缩问题的提示词模板")


# --- 对话记忆配置 ---
class MemoryConfig(BaseModel):
    enable: bool = Field(True, description="是否启用对话记忆")
    memory_type: Literal["conversation_buffer"] = Field("conversation_buffer", description="记忆类型")
    memory_key: str = Field("chat_history", description="记忆在链中的键名")
    input_key: Optional[str] = Field("question", description="用户输入的键名 (ConversationBufferMemory需要)")
    output_key: Optional[str] = Field("result", description="LLM输出的键名 (用于某些记忆类型)") # RetrievalQA 通常是 'result'
    max_token_limit: Optional[int] = Field(None, description="记忆的最大token限制 (如 ConversationTokenBufferMemory)")
    # return_messages: bool = Field(True, description="是否以消息对象列表形式返回记忆") # RetrievalQA的memory通常不用这个


# --- 检索器配置 (细化) ---
class BM25RetrieverConfig(BaseComponentModel): # BM25的独立配置
    type: Literal["bm25_retriever"] = "bm25_retriever"
    # k1, b 等参数可以放在这里，如果它们不是全局RetrieverConfig的一部分
    k1: float = Field(1.5, description="BM25 k1 参数")
    b: float = Field(0.75, description="BM25 b 参数")
    # top_k 也可以是 BM25 特有的，如果与向量检索的 top_k 不同
    top_k: int = Field(3, gt=0, description="BM25检索返回的文档数量")

class VectorRetrieverConfig(BaseComponentModel): # 向量检索的独立配置
    type: Literal["vector_retriever"] = "vector_retriever"
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        "similarity", description="向量搜索类型"
    )
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="相似度分数阈值"
    )
    # top_k 也可以是向量检索特有的
    top_k: int = Field(3, gt=0, description="向量检索返回的文档数量")
    # mmr_fetch_k: Optional[int] = Field(20, description="MMR中初始获取的文档数") # 如果用MMR
    # mmr_lambda_mult: Optional[float] = Field(0.5, description="MMR中多样性参数") # 如果用MMR


class HybridRetrieverConfig(BaseComponentModel):
    type: Literal["hybrid_retriever"] = "hybrid_retriever"
    bm25_weight: float = Field(0.4, ge=0.0, le=1.0, description="混合检索中 BM25 的权重")
    vector_weight: float = Field(0.6, ge=0.0, le=1.0, description="混合检索中向量的权重")
    # Hybrid retriever 自身也需要一个 top_k，或者它依赖于其子retriever的top_k并进行融合后的top_k
    top_k: int = Field(3, gt=0, description="混合检索最终返回的文档数量")

# Retriever 配置的联合类型 (现在包含更具体的类型)
RetrieverComponentConfigType = Union[BM25RetrieverConfig, VectorRetrieverConfig]

class MainRetrieverConfig(BaseModel): # 主检索器配置，决定使用哪种策略和组件
    strategy: Literal["vector", "bm25", "hybrid"] = Field(
        "vector", description="检索策略"
    )
    # 为每种策略组件指定配置，AppBuilder 会根据 strategy 选择使用哪个
    bm25_config: Optional[BM25RetrieverConfig] = Field(default_factory=BM25RetrieverConfig)
    vector_config: Optional[VectorRetrieverConfig] = Field(default_factory=VectorRetrieverConfig)
    hybrid_config: Optional[HybridRetrieverConfig] = Field(default_factory=HybridRetrieverConfig) # 如果 strategy 是 hybrid，则使用此配置


# --- RAG Chain 配置 ---
class RAGChainConfig(BaseModel):
    chain_type: Literal["stuff", "map_reduce", "refine", "map_rerank"] = Field(
        "stuff", description="RetrievalQA 的 chain_type"
    )
    return_source_documents: bool = Field(True, description="是否返回源文档")
    # verbose: bool = Field(False, description="是否启用详细日志 (Langchain链级别)") # 通常由全局日志控制


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
    embedding: EmbeddingConfigType # <--- 修改这里，使用联合类型
    vector_store: VectorStoreConfigType  # <--- 新增
    document_loader: DocumentLoaderConfigType  # <--- 新增
    text_splitter: TextSplitterConfigType
    prompts: PromptConfigModel = Field(default_factory=PromptConfigModel)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retriever: MainRetrieverConfig = Field(default_factory=MainRetrieverConfig)
    rag_chain: RAGChainConfig = Field(default_factory=RAGChainConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    class Config:
        validate_assignment = True