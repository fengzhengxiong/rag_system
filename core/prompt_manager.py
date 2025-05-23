#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：rag_system 
@File    ：prompt_manager.py
@Author  ：fengzhengxiong
@Date    ：2025/5/22 11:44 
'''

# core/prompt_manager.py
from typing import Dict, Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 可以在 config_models.py 中定义 PromptConfig
# 或者在这里定义一个简单的结构
class PromptConfig(BaseModel):
    qa_template_str: str = Field(
        """基于以下上下文已知信息，请简洁并专业地回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。

上下文内容:
{context}

问题:
{question}""",
        description="问答任务的提示词模板字符串"
    )
    # 可以添加其他类型的提示词，例如 condense_question_prompt 等

class PromptManager:
    def __init__(self, config: Optional[PromptConfig] = None):
        if config is None:
            config = PromptConfig() # 使用默认模板
        self.config = config
        self._qa_prompt_template: Optional[PromptTemplate] = None

    def get_qa_prompt_template(self) -> PromptTemplate:
        if self._qa_prompt_template is None:
            self._qa_prompt_template = PromptTemplate(
                template=self.config.qa_template_str,
                input_variables=["context", "question"]
            )
        return self._qa_prompt_template

    # 可以添加获取其他提示词模板的方法
    # def get_condense_question_prompt(self) -> PromptTemplate: ...