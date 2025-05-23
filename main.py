#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：main.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:42 
'''


import asyncio
import argparse
from pathlib import Path # 用于创建测试文档
# from langchain_core.documents import Document # 用于创建测试文档
# from typing import Optional, List
from utils.config_loader import load_app_config, AppConfig
from utils.logger import RAGLogger
from app_builder import AppBuilder
# from core.interfaces.llm_interface import LLMInterface
# from core.interfaces.embedding_interface import EmbeddingInterface
# from core.interfaces.vector_store_interface import VectorStoreInterface
# from core.interfaces.document_loader_interface import DocumentLoaderInterface
# from core.interfaces.text_splitter_interface import TextSplitterInterface

# async def test_llm_async(llm: LLMInterface):
#     """测试 LLM 的异步方法"""
#     logger = RAGLogger().get_logger(__name__) # 获取 logger
#
#     logger.info(f"Testing LLM: {llm.model_name} (Async Generate)")
#     prompts = ["天空为什么是蓝色的？请用中文简洁回答。", "写一个关于太空旅行的短故事。"]
#     try:
#         results = await llm.agenerate(prompts)
#         for i, generation_list in enumerate(results.generations):
#             for generation in generation_list:
#                 logger.info(f"Async Prompt {i+1} Result: {generation.text[:100]}...")
#     except Exception as e:
#         logger.error(f"Async generate test failed: {e}", exc_info=True)
#
#     logger.info(f"\nTesting LLM: {llm.model_name} (Async Stream)")
#     try:
#         async for chunk in llm.astream("写一句关于编程的禅语。"):
#             print(chunk, end="", flush=True)
#         print("\nAsync Stream test finished.")
#         logger.info("Async stream test finished successfully.")
#     except Exception as e:
#         logger.error(f"Async stream test failed: {e}", exc_info=True)
#
#
# def test_llm_sync(llm: LLMInterface):
#     """测试 LLM 的同步方法"""
#     logger = RAGLogger().get_logger(__name__)
#
#     logger.info(f"Testing LLM: {llm.model_name} (Sync Generate)")
#     prompts = ["What is the capital of France?"]
#     try:
#         results = llm.generate(prompts)
#         for i, generation_list in enumerate(results.generations):
#             for generation in generation_list:
#                 logger.info(f"Sync Prompt {i+1} Result: {generation.text[:100]}...")
#     except Exception as e:
#         logger.error(f"Sync generate test failed: {e}", exc_info=True)
#
#     logger.info(f"\nTesting LLM: {llm.model_name} (Sync Stream)")
#     try:
#         for chunk in llm.stream("Tell me a short joke."):
#             print(chunk, end="", flush=True)
#         print("\nSync Stream test finished.")
#         logger.info("Sync stream test finished successfully.")
#     except Exception as e:
#         logger.error(f"Sync stream test failed: {e}", exc_info=True)
#
# def test_embedding_model(embedding_model: EmbeddingInterface): # <--- 新增测试函数
#     logger = RAGLogger().get_logger(__name__)
#     logger.info(f"Testing Embedding Model: {embedding_model.model_name}")
#
#     docs_to_embed = [
#         "天空是蓝色的因为瑞利散射。",
#         "Langchain 是一个强大的AI开发框架。",
#         "Ollama 使得在本地运行大语言模型变得容易。"
#     ]
#     query_to_embed = "什么是最好的编程语言？"
#
#     try:
#         logger.info("Testing document embedding...")
#         doc_embeddings = embedding_model.embed_documents(docs_to_embed)
#         if doc_embeddings and len(doc_embeddings) == len(docs_to_embed):
#             logger.info(f"Successfully embedded {len(doc_embeddings)} documents.")
#             logger.info(f"Dimension of first document embedding: {len(doc_embeddings[0])}")
#         else:
#             logger.error("Document embedding failed or returned unexpected result.")
#
#         logger.info("\nTesting query embedding...")
#         query_embedding = embedding_model.embed_query(query_to_embed)
#         if query_embedding:
#             logger.info(f"Successfully embedded query.")
#             logger.info(f"Dimension of query embedding: {len(query_embedding)}")
#         else:
#             logger.error("Query embedding failed.")
#
#     except Exception as e:
#         logger.error(f"Embedding model test failed: {e}", exc_info=True)
#
#
# def test_vector_store(vs_instance: VectorStoreInterface, app_config: AppConfig): # <--- 新增测试函数
#     logger = RAGLogger().get_logger(__name__)
#     logger.info(f"Testing Vector Store. Type: {app_config.vector_store.type}")
#
#     if vs_instance.is_initialized:
#         logger.info("Vector Store was successfully initialized (likely loaded from disk).")
#     else:
#         logger.info("Vector Store is not initialized. It may need to be built from documents.")
#         # 在这里我们可以模拟一次构建，如果VS是空的
#         # 这部分逻辑更适合放在 IngestionService 中
#         logger.info("Attempting a mock build for testing purposes...")
#         sample_docs_for_build = [
#             Document(page_content="这是一个用于构建向量数据库的示例文档。", metadata={"source": "test_build_1"}),
#             Document(page_content="FAISS 是一个高效的相似性搜索库。", metadata={"source": "test_build_2"}),
#         ]
#         if vs_instance.build_from_documents(sample_docs_for_build):
#             logger.info("Mock build successful.")
#             if vs_instance.is_initialized:
#                 logger.info("Vector Store is now initialized after mock build.")
#                 # 测试保存 (如果构建成功且配置了持久化)
#                 # vs_instance.save_local() # build_from_documents 内部可能已经调用了save
#             else:
#                  logger.warning("Mock build reported success, but VS still not initialized. Check implementation.")
#         else:
#             logger.error("Mock build failed.")
#
#
#     # 简单的搜索测试 (如果已初始化)
#     if vs_instance.is_initialized:
#         logger.info("Attempting a simple similarity search...")
#         try:
#             # 使用一个通用的查询，这个查询可能在mock build的文档中有相关内容
#             search_results = vs_instance.similarity_search("FAISS库", k=1)
#             if search_results:
#                 logger.info(f"Similarity search returned {len(search_results)} result(s).")
#                 for doc in search_results:
#                     logger.info(f"  - Found: {doc.page_content[:50]}... (Source: {doc.metadata.get('source')})")
#             else:
#                 logger.info("Similarity search returned no results (this is OK if DB is empty or query is irrelevant).")
#         except Exception as e:
#             logger.error(f"Error during test similarity search: {e}", exc_info=True)
#     else:
#         logger.info("Skipping similarity search test as Vector Store is not initialized.")
#
#
# def test_document_loader_and_splitter(
#         doc_loader: DocumentLoaderInterface,
#         text_splitter: TextSplitterInterface,
#         app_config: AppConfig
# ):  # <--- 新增测试函数
#     logger = RAGLogger().get_logger(__name__)
#     logger.info("Testing Document Loader and Text Splitter...")
#     logger.info(f"Document Loader Source: {doc_loader.source_description}")
#     logger.info(f"Text Splitter Description: {text_splitter.description}")
#
#     # 准备一些测试文件在 data_dir 中
#     data_path = Path(app_config.paths.data_dir)
#     data_path.mkdir(parents=True, exist_ok=True)
#     test_file_1 = data_path / "test_doc_1.txt"
#     test_file_2 = data_path / "another_doc.txt"  # 如果你的 glob 是 **/*.txt
#
#     content1 = "这是第一个测试文档。\n它包含多行内容。\n目的是测试文档加载和分割功能。"
#     content2 = "第二个文件用于演示。\nRecursiveCharacterTextSplitter 应该能够处理它。\n让我们看看效果如何。"
#
#     try:
#         with open(test_file_1, "w", encoding="utf-8") as f:
#             f.write(content1)
#         with open(test_file_2, "w", encoding="utf-8") as f:
#             f.write(content2)
#         logger.info(f"Created test files in '{data_path}'.")
#     except Exception as e:
#         logger.error(f"Failed to create test files: {e}", exc_info=True)
#         return
#
#     loaded_docs: List[Document] = []
#     try:
#         logger.info("Attempting to load documents...")
#         # loaded_docs = doc_loader.load() # 测试 load
#         # 测试 lazy_load
#         temp_docs = []
#         for doc in doc_loader.lazy_load():
#             logger.info(
#                 f"  Lazy loaded doc from: {doc.metadata.get('source', 'Unknown source')}, content starts: '{doc.page_content[:30]}...'")
#             temp_docs.append(doc)
#         loaded_docs = temp_docs
#
#         if loaded_docs:
#             logger.info(f"Successfully loaded {len(loaded_docs)} documents using loader.")
#             for i, doc in enumerate(loaded_docs):
#                 logger.debug(f"  Loaded Doc {i + 1} metadata: {doc.metadata}, content length: {len(doc.page_content)}")
#         else:
#             logger.warning("Document loader returned no documents.")
#             # 如果这里没有文档，后续的分割测试就没意义了
#             # 清理测试文件
#             # test_file_1.unlink(missing_ok=True)
#             # test_file_2.unlink(missing_ok=True)
#             return  # 提前返回
#
#     except Exception as e:
#         logger.error(f"Error during document loading test: {e}", exc_info=True)
#         # 清理测试文件
#         # test_file_1.unlink(missing_ok=True)
#         # test_file_2.unlink(missing_ok=True)
#         return
#
#     if not loaded_docs:
#         logger.info("Skipping text splitting test as no documents were loaded.")
#         # 清理测试文件
#         # test_file_1.unlink(missing_ok=True)
#         # test_file_2.unlink(missing_ok=True)
#         return
#
#     try:
#         logger.info("Attempting to split loaded documents...")
#         split_chunks = text_splitter.split_documents(loaded_docs)
#         if split_chunks:
#             logger.info(f"Successfully split documents into {len(split_chunks)} chunks.")
#             for i, chunk in enumerate(split_chunks):
#                 logger.info(
#                     f"  Chunk {i + 1} (from {chunk.metadata.get('source')}): '{chunk.page_content[:50]}...' (Length: {len(chunk.page_content)})")
#         else:
#             logger.warning("Text splitter returned no chunks.")
#     except Exception as e:
#         logger.error(f"Error during text splitting test: {e}", exc_info=True)
#     finally:
#         # 清理测试文件
#         logger.info(f"Cleaning up test files from '{data_path}'.")
#         test_file_1.unlink(missing_ok=True)  # missing_ok=True (Python 3.8+)
#         test_file_2.unlink(missing_ok=True)

async def main_async(run_ingestion: bool, force_rebuild_vs: bool): # 将 main 函数改为异步，以便调用异步测试
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

    # # 4. 创建 LLM 实例
    # llm_instance: Optional[LLMInterface] = None
    # try:
    #     logger.info("尝试创建 LLM 实例...")
    #     llm_instance = builder.create_llm()
    #     if llm_instance:
    #         logger.info(f"LLM 实例 '{llm_instance.model_name}' 创建成功!")
    #     else:
    #         logger.error("LLM 实例创建失败，返回 None。")
    #         return # 如果LLM是核心，没有它无法继续
    # except Exception as e:
    #     logger.critical(f"创建 LLM 实例时发生严重错误: {e}", exc_info=True)
    #     logger.info("请确保 Ollama 服务正在运行并且模型已下载/可用。")
    #     return # 无法继续
    #
    # # 5. 创建 Embedding 模型实例
    # embedding_instance: Optional[EmbeddingInterface] = None
    # try:
    #     logger.info("尝试创建 Embedding 模型实例...")
    #     embedding_instance = builder.create_embedding_model()
    #     if embedding_instance:
    #         logger.info(f"Embedding 模型实例 '{embedding_instance.model_name}' 创建成功!")
    #     else:
    #         logger.error("Embedding 模型实例创建失败，返回 None。")
    #         return # 无法继续
    # except Exception as e:
    #     logger.critical(f"创建 Embedding 模型实例时发生严重错误: {e}", exc_info=True)
    #     logger.info("请确保 Ollama 服务正在运行并且 Embedding 模型已下载/可用。")
    #     return # 无法继续
    #
    # # 6. 创建 Vector Store 实例
    # vector_store_instance: Optional[VectorStoreInterface] = None
    # try:
    #     logger.info("尝试创建 Vector Store 实例...")
    #     vector_store_instance = builder.create_vector_store()  # AppBuilder会处理注入Embedding
    #     if vector_store_instance:
    #         logger.info(f"Vector Store 实例 (type: {app_config.vector_store.type}) 创建成功!")
    #     else:
    #         logger.error("Vector Store 实例创建失败，返回 None。")
    #         return  # 无法继续
    # except Exception as e:
    #     logger.critical(f"创建 Vector Store 实例时发生严重错误: {e}", exc_info=True)
    #     return  # 无法继续
    #
    # # 7. 创建 Document Loader 和 Text Splitter 实例
    # doc_loader_instance: Optional[DocumentLoaderInterface] = None
    # text_splitter_instance: Optional[TextSplitterInterface] = None
    # try:
    #     logger.info("尝试创建 Document Loader 实例...")
    #     doc_loader_instance = builder.create_document_loader()
    #     if doc_loader_instance:
    #         logger.info(f"Document Loader 实例 (type: {app_config.document_loader.type}) 创建成功!")
    #     else:
    #         logger.error("Document Loader 实例创建失败。")
    #
    #     logger.info("尝试创建 Text Splitter 实例...")
    #     text_splitter_instance = builder.create_text_splitter()
    #     if text_splitter_instance:
    #         logger.info(f"Text Splitter 实例 (type: {app_config.text_splitter.type}) 创建成功!")
    #     else:
    #         logger.error("Text Splitter 实例创建失败。")
    #
    # except Exception as e:
    #     logger.critical(f"创建文档处理组件时发生严重错误: {e}", exc_info=True)
    #
    # # 测试
    # if llm_instance:
    #     test_llm_sync(llm_instance)
    #     await test_llm_async(llm_instance)
    #
    # if embedding_instance:
    #     test_embedding_model(embedding_instance)
    #
    # if vector_store_instance:
    #     test_vector_store(vector_store_instance, app_config)
    #
    # if doc_loader_instance and text_splitter_instance:
    #     test_document_loader_and_splitter(doc_loader_instance, text_splitter_instance, app_config)

    # --- 构建和运行 IngestionService ---

    if run_ingestion:
        ingestion_service_instance = None
        try:
            logger.info("尝试构建 IngestionService...")
            ingestion_service_instance = builder.build_ingestion_service()
            if ingestion_service_instance:
                logger.info("IngestionService 构建成功! 开始数据注入...")
                # 确保 app_config.paths.data_dir 存在，IngestionService 可能需要它
                Path(app_config.paths.data_dir).mkdir(parents=True, exist_ok=True)

                success = ingestion_service_instance.ingest_data(force_rebuild=force_rebuild_vs)
                if success:
                    logger.info("数据注入流程执行完毕。")
                else:
                    logger.error("数据注入流程中发生错误。")
            else:
                logger.error("IngestionService 构建失败。")
        except Exception as e:
            logger.critical(f"构建或运行 IngestionService 时发生严重错误: {e}", exc_info=True)
    else:
        logger.info("跳过数据注入流程 (根据命令行参数)。")
        # 如果不注入，但后续有查询逻辑，可能需要确保VectorStore已加载
        # 例如，可以尝试获取VS实例，它内部会尝试load_local
        try:
            logger.info("尝试获取 VectorStore 实例 (因跳过注入)...")
            vs_instance = builder.create_vector_store()  # 这会尝试加载或返回已缓存的
            if vs_instance and vs_instance.is_initialized:
                logger.info("VectorStore 已初始化/加载。")
            elif vs_instance:
                logger.warning("VectorStore 实例已创建但未初始化/加载，查询可能失败。")
            else:
                logger.error("无法获取 VectorStore 实例。")
        except Exception as e:
            logger.error(f"获取 VectorStore 实例时出错: {e}", exc_info=True)

    # --- 构建 QueryService ---
    query_service_instance = None
    try:
        logger.info("尝试构建 QueryService...")
        query_service_instance = builder.build_query_service()
        if query_service_instance:
            logger.info("QueryService 构建成功! 系统准备就绪。")
        else:
            logger.error("QueryService 构建失败。无法进行查询。")
            return  # 没有查询服务，应用无法交互
    except Exception as e:
        logger.critical(f"构建 QueryService 时发生严重错误: {e}", exc_info=True)
        return

    # --- 交互式查询循环 ---
    if query_service_instance:
        print("\n欢迎来到RAG问答系统！输入 'q' 退出, 'c' 清空对话历史。")
        while True:
            try:
                user_input = input("\n你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'q':
                    logger.info("用户请求退出。")
                    break
                if user_input.lower() == 'c':
                    logger.info("用户请求清空对话历史。")
                    query_service_instance.clear_conversation_history()
                    print("AI: 对话历史已清空。")
                    continue

                logger.info(f"用户提问: {user_input[:100]}...")
                print("AI 思考中...")
                answer = query_service_instance.ask(user_input)
                print(f"AI: {answer}")

            except KeyboardInterrupt:
                logger.info("用户通过 Ctrl+C 中断。")
                print("\n操作已取消。输入 'q' 退出。")
            except Exception as e:
                logger.error(f"查询循环中发生错误: {e}", exc_info=True)
                print("抱歉，处理您的请求时发生了内部错误。")

    logger.info("RAG 应用关闭。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System Application")
    parser.add_argument(
        "--ingest",
        action="store_true",  # 如果提供了 --ingest 参数，则为 True
        help="运行数据注入流程。"
    )
    parser.add_argument(
        "--rebuild-vs",
        action="store_true",
        help="强制重新构建向量数据库 (仅当 --ingest 也被设置时有效)。"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_async(run_ingestion=args.ingest, force_rebuild_vs=args.rebuild_vs))
    except KeyboardInterrupt:
        # 在这里获取一个logger实例来记录中断，因为main_async中的logger可能还没完全配置好
        # 或者如果 RAGLogger 已经通过 setup_logger 配置过，可以直接使用
        # RAGLogger().get_logger("MAIN").info("操作被用户中断。")
        print("\n操作被用户中断。")  # 对于中断，简单打印可能更合适
    except Exception as e:
        # 这里的日志记录器可能还未初始化，或者我们想用一个独立的简单日志
        # RAGLogger().get_logger("MAIN_FALLBACK").critical(f"应用程序顶层错误: {e}", exc_info=True)
        print(f"应用程序顶层发生未处理的错误: {e}")
        import traceback

        traceback.print_exc()  # 打印详细堆栈信息，帮助调试