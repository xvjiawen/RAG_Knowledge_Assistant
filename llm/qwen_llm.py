#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wenxin_llm.py
@Time    :   2023/10/16 18:53:26
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于百度文心大模型自定义 LLM 类
'''

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from llm.self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun


class QwenLLM(Self_LLM):
    """针对 Qwen API 的 LangChain LLM Wrapper"""

    # 1. 覆盖默认的接口地址
    url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    # url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 2. 使用 Qwen 自己的模型名
    model: str = "qwen-1.0"

    # 3. 设置默认超时时间（秒）、温度等
    request_timeout: float = 60.0
    temperature: float = 0.3

    # 4. 从环境变量或初始化参数中读取 API Key
    api_key: str = Field(..., env="DASHSCOPE_API_KEY")

    # 5. Qwen 特有的其他参数
    top_p: int=0.9
    max_tokens: int=1024


    @property
    def _llm_type(self) -> str:
        """LangChain 用来标识第三方模型类型"""
        return "qwen"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:
        """
        同步调用 Qwen 接口。
        - prompt: 用户输入
        - stop: 可选，停止词列表
        """
        # 组装消息格式（以 Chat Completions API 为例）
        # print(f"***********model:{self.model}\nmessages:{prompt}")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # 内置的 temperature、timeout 等
            **self._default_params,
            # "top_p":self.top_p,
            # "max_tokens":self.max_tokens
        }
        print(f"***********payload:\n{payload}")
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # 根据 Qwen 返回结构解析出模型回复
        return data["choices"][0]["message"]["content"]
