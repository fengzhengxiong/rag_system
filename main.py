#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：main.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:42 
'''


import asyncio # 用于测试异步方法
from typing import Optional
from utils.config_loader import load_app_config, AppConfig
from utils.logger import RAGLogger
from app_builder import AppBuilder # 导入 AppBuilder
from core.interfaces.llm_interface import LLMInterface # 导入 LLM 接口

async def test_llm_async(llm: LLMInterface):
    """测试 LLM 的异步方法"""
    logger = RAGLogger().get_logger(__name__) # 获取 logger

    logger.info(f"Testing LLM: {llm.model_name} (Async Generate)")
    prompts = ["天空为什么是蓝色的？请用中文简洁回答。", "写一个关于太空旅行的短故事。"]
    try:
        results = await llm.agenerate(prompts)
        for i, generation_list in enumerate(results.generations):
            for generation in generation_list:
                logger.info(f"Async Prompt {i+1} Result: {generation.text[:100]}...")
    except Exception as e:
        logger.error(f"Async generate test failed: {e}", exc_info=True)

    logger.info(f"\nTesting LLM: {llm.model_name} (Async Stream)")
    try:
        async for chunk in llm.astream("写一句关于编程的禅语。"):
            print(chunk, end="", flush=True)
        print("\nAsync Stream test finished.")
        logger.info("Async stream test finished successfully.")
    except Exception as e:
        logger.error(f"Async stream test failed: {e}", exc_info=True)


def test_llm_sync(llm: LLMInterface):
    """测试 LLM 的同步方法"""
    logger = RAGLogger().get_logger(__name__)

    logger.info(f"Testing LLM: {llm.model_name} (Sync Generate)")
    prompts = ["What is the capital of France?"]
    try:
        results = llm.generate(prompts)
        for i, generation_list in enumerate(results.generations):
            for generation in generation_list:
                logger.info(f"Sync Prompt {i+1} Result: {generation.text[:100]}...")
    except Exception as e:
        logger.error(f"Sync generate test failed: {e}", exc_info=True)

    logger.info(f"\nTesting LLM: {llm.model_name} (Sync Stream)")
    try:
        for chunk in llm.stream("Tell me a short joke."):
            print(chunk, end="", flush=True)
        print("\nSync Stream test finished.")
        logger.info("Sync stream test finished successfully.")
    except Exception as e:
        logger.error(f"Sync stream test failed: {e}", exc_info=True)


async def main_async(): # 将 main 函数改为异步，以便调用异步测试
    # 1. 加载应用配置 (不变)
    try:
        app_config: AppConfig = load_app_config("default_config.yaml")
    except Exception as e:
        print(f"关键错误: 无法加载应用配置. {e}")
        return

    # 2. 设置日志系统 (不变)
    logger = RAGLogger.setup_logger(name="RAGApplication", config=app_config.logging)

    logger.info("RAG 应用启动 (LLM Test)...")

    # 3. 创建 AppBuilder 实例
    try:
        builder = AppBuilder(config=app_config, logger=RAGLogger()) # RAGLogger()会返回已配置的单例
    except Exception as e:
        logger.critical(f"无法初始化 AppBuilder: {e}", exc_info=True)
        return

    # 4. 创建 LLM 实例
    llm_instance: Optional[LLMInterface] = None
    try:
        logger.info("尝试创建 LLM 实例...")
        llm_instance = builder.create_llm()
        if llm_instance:
            logger.info(f"LLM 实例 '{llm_instance.model_name}' 创建成功!")
        else:
            logger.error("LLM 实例创建失败，返回 None。")
            return # 如果LLM是核心，没有它无法继续
    except Exception as e:
        logger.critical(f"创建 LLM 实例时发生严重错误: {e}", exc_info=True)
        logger.info("请确保 Ollama 服务正在运行并且模型已下载/可用。")
        return # 无法继续

    # 5. 测试 LLM 实例 (如果创建成功)
    if llm_instance:
        test_llm_sync(llm_instance)
        await test_llm_async(llm_instance) # 调用异步测试函数

    logger.info("RAG 应用 (LLM Test) 关闭。")

if __name__ == "__main__":
    # 使用 asyncio.run() 来运行异步的 main 函数
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n操作被用户中断。")
    except Exception as e:
        # 捕获在 main_async 中未被捕获的顶层异常
        # (理想情况下，main_async 应该处理其内部所有预期的异常)
        print(f"应用程序顶层发生未处理的错误: {e}")
        # 这里的日志记录器可能还未初始化，或者我们想用一个独立的简单日志
        RAGLogger().get_logger("MAIN_FALLBACK").critical(f"应用程序顶层错误: {e}", exc_info=True)