#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：logger.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:53 
'''

# utils/logger.py
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union

# 从新的配置模型导入 LoggingConfig
from .config_models import LoggingConfig

class RAGLogger:
    _instance = None
    _initialized_with_config = False # 新增标志，防止重复使用配置初始化

    # __new__ 方法保持单例逻辑，但初始化推迟
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 不在这里调用 _initialize，或者让 _initialize 能够被重复调用但只生效一次配置
        return cls._instance

    def _initialize(self, name: str, config: LoggingConfig):
        """
        使用 LoggingConfig 对象初始化日志记录器。
        现在 name 可以是一个默认值，或者也从配置中读取（如果需要）。
        """
        if RAGLogger._initialized_with_config and hasattr(self, 'logger'): # 如果已经用配置初始化过，直接返回
            return

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.main_log_path = self.log_dir / f"{name.lower().replace(' ', '_')}_rag.log" # 基于name生成日志文件名

        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(process)d - %(threadName)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
        ) # 更详细的日志格式

        # 清空现有处理器，确保配置更改生效
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close() # 关闭旧的 handler

        # 主日志文件处理器(带轮转)
        file_handler = logging.handlers.RotatingFileHandler(
            self.main_log_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        RAGLogger._initialized_with_config = True
        self.logger.info(f"RAGLogger '{name}' initialized with config: Level={config.log_level}, Dir={config.log_dir}")

    # 提供一个静态方法或一个初始化函数来设置日志记录器
    @classmethod
    def setup_logger(cls, name: str = "RAGSystem", config: Optional[LoggingConfig] = None):
        """
        获取或初始化 RAGLogger 实例。
        如果提供了配置，则使用该配置初始化（或重新初始化）。
        """
        instance = cls() # 获取单例实例
        if config:
            # 强制重新初始化处理器，即使之前已初始化，以应用新配置
            RAGLogger._initialized_with_config = False # 允许用新配置重新初始化
            instance._initialize(name, config)
        elif not hasattr(instance, 'logger'):
            # 如果没有提供配置，且从未初始化过，则使用默认配置（或者报错提示需要配置）
            print("警告: RAGLogger 未配置，将使用默认设置创建。建议提供 LoggingConfig。")
            default_log_config = LoggingConfig() # 使用Pydantic模型的默认值
            instance._initialize(name, default_log_config)
        return instance.get_logger() # 返回底层的 logging.Logger 对象

    # 旧的便捷方法可以保留，它们会使用已经配置好的 self.logger
    def info(self, message: str, **kwargs) -> None:
        if hasattr(self, 'logger'): self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        if hasattr(self, 'logger'): self.logger.warning(message, **kwargs)

    def error(self, message: str, exc_info: Optional[bool] = False, **kwargs) -> None:
        if hasattr(self, 'logger'): self.logger.error(message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: Optional[bool] = False, **kwargs) -> None:
        if hasattr(self, 'logger'): self.logger.critical(message, exc_info=exc_info, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        if hasattr(self, 'logger'): self.logger.debug(message, **kwargs)

    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        获取配置好的 logger 实例。
        如果提供了 module_name，则返回一个子 logger，便于按模块区分日志来源。
        """
        if not hasattr(self, 'logger'):
            # 如果logger未初始化（例如直接调用get_logger而没有先setup），进行一次默认初始化
            RAGLogger.setup_logger() # 使用默认配置初始化

        if module_name:
            return logging.getLogger(f"{self.logger.name}.{module_name}")
        return self.logger

    def change_log_level(self, level: Union[int, str]) -> None:
        if not hasattr(self, 'logger'):
            print("错误: Logger尚未初始化，无法更改日志级别。")
            return

        if isinstance(level, str):
            level_int = getattr(logging, level.upper(), None)
            if level_int is None:
                self.logger.warning(f"无效的日志级别字符串: {level}。级别未更改。")
                return
            level = level_int

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level) # 确保所有 handler 的级别也更新
        self.logger.info(f"日志级别已更改为: {logging.getLevelName(level)}")