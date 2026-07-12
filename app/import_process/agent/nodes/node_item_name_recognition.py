# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.node_decorator import node_log
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def get_chunks(state):
    """
    获取文档切片
    :param state:
    :return:
    """
    file_title = state.get("file_title")
    chunks = state.get("chunks")
    if not chunks:
        raise ValueError(f"get_chunks:文档切片为空: {file_title}")
    if not file_title:
        # md_path中获取文件名
        file_title = os.path.splitext(os.path.basename(state.get("md_path")))[0]
        state["file_title"] = file_title

    return file_title, chunks


@node_log
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    try:
        # 获取文档切片
        file_title,chunks = get_chunks(state)
        # 构建上下文环境
        context = build_context(chunks)
        # 调用大模型识别物品名称
        item_name = call_llm_recognize_item_name(context, file_title)
        # 回写数据state
        update_chunks_and_state(state, chunks, item_name)
        # item_name生成稀疏稠密向量
        sparse_vector, dense_vector = generate_embeddings(item_name)
        # 将向量存储值向量数据库
        store_vectors_to_milvus(item_name, sparse_vector, dense_vector)
    except Exception as e:
        logger.error(f"node_item_name_recognition:获取文档切片失败: {e}")

    return state