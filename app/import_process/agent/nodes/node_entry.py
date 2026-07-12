import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    负责接收外部输入并决定流程走向。
    实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """

    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{func_name}] 开始执行,节点状态: {state}")
    add_running_task(state["task_id"],func_name)

    local_file_path = state["local_file_path"]
    if not "local_file_path" in state:
        logger.error("文件路径不存在")

    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path

    #提取file_title
    state["file_title"] = local_file_path.split("/")[-1].split(".")[0]

    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{func_name}] 结束执行,节点状态: {state}")
    add_done_task(state["task_id"], func_name)

    return state