import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import ModelScopeEmbeddings
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    if embedding == "modelscope":
        return ModelScopeEmbeddings()
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    if embedding == "openai":
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    elif embedding == "zhipuai":
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"embedding {embedding} not support ")
