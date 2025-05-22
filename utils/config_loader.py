#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：config_loader.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:53 
'''

import yaml
from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel
from .config_models import AppConfig # 从同级目录的 config_models.py 导入

T = TypeVar('T', bound=BaseModel)

def load_config_from_yaml(config_path: Path, config_model: Type[T]) -> T:
    """
    从 YAML 文件加载配置并使用 Pydantic 模型进行校验。

    Args:
        config_path: YAML 配置文件的路径。
        config_model: 用于校验配置的 Pydantic 模型类。

    Returns:
        经过校验的配置对象。

    Raises:
        FileNotFoundError: 如果配置文件不存在。
        yaml.YAMLError: 如果 YAML 文件格式错误。
        pydantic.ValidationError: 如果配置数据不符合模型定义。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # 在实际应用中，这里应该使用日志记录器记录错误
        print(f"解析 YAML 配置文件 '{config_path}' 时出错: {e}")
        raise

    if raw_config is None:
        # YAML 文件为空或只包含注释
        # 根据需求，可以返回一个使用默认值的模型实例或抛出错误
        print(f"警告: 配置文件 '{config_path}' 为空或只包含注释。将使用默认配置。")
        return config_model() # 返回一个包含所有默认值的模型实例

    try:
        return config_model(**raw_config)
    except Exception as e: # Pydantic 通常抛出 ValidationError，但这里捕获通用异常以防万一
        # 在实际应用中，这里应该使用日志记录器记录错误
        print(f"校验配置文件 '{config_path}' 时出错: {e}")
        # 可以考虑打印 Pydantic 的 e.errors() 获取更详细的错误信息
        raise


# 一个便捷函数，用于加载主应用配置
def load_app_config(config_file_name: str = "default_config.yaml") -> AppConfig:
    """
    加载主应用程序配置。

    Args:
        config_file_name: 配置文件的名称 (在 'configs' 目录下)。

    Returns:
        AppConfig 实例。
    """
    # 假设项目根目录是 config_loader.py 向上两级
    # 或者根据你的项目结构调整 base_path
    # 一个更健壮的方法是使用一个环境变量指向项目根目录，或者在main.py中确定根目录并传递
    current_file_path = Path(__file__).resolve()
    # 假设 utils 在项目根目录下，configs 也在项目根目录下
    # 如果 utils 在 core 或其他子目录下，需要调整 parent 的层级
    # 例如，如果 utils 在 project_root/core/utils/ 下，configs 在 project_root/configs/ 下
    # base_path = current_file_path.parent.parent.parent / "configs"
    base_path = current_file_path.parent.parent / "configs" # 假设 utils 是 project_root/utils/

    config_path = base_path / config_file_name
    # print(f"Attempting to load config from: {config_path}") # 调试时使用
    return load_config_from_yaml(config_path, AppConfig)

# --- 如何使用 (可以放在 main.py 或测试文件中) ---
# if __name__ == "__main__":
#     try:
#         app_config = load_app_config()
#         print("配置加载成功!")
#         print("日志配置:", app_config.logging)
#         print("LLM 类型:", app_config.llm.type)
#         print("LLM 模型名:", app_config.llm.model_name)
#         print("缓存是否启用:", app_config.cache.enable)
#
#         # 测试修改并校验 (如果 AppConfig.Config.validate_assignment = True)
#         # app_config.logging.log_level = "INVALID_LEVEL" # 这会引发 ValidationError
#
#         # 测试从不存在的文件加载
#         # load_app_config("non_existent_config.yaml")
#
#         # 测试一个空的YAML文件，或者只有注释的YAML文件
#         # (需要你手动创建一个这样的文件，例如 empty.yaml)
#         # empty_config = load_app_config("empty.yaml")
#         # print("空配置文件加载:", empty_config)
#
#     except FileNotFoundError as e:
#         print(f"错误: {e}")
#     except yaml.YAMLError as e:
#         print(f"YAML 解析错误: {e}")
#     except Exception as e: # 主要捕获 Pydantic 的 ValidationError
#         print(f"配置校验错误: {e}")
#         # import traceback
#         # traceback.print_exc() # 打印详细的堆栈信息