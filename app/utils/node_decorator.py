
import functools
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

def node_log(func):
    """
    节点 AOP 装饰器：自动注入"开始执行"/"结束执行"日志 + 任务状态追踪。
    用法：
        @node_log
        def node_xxx(state: ImportGraphState) -> ImportGraphState:
            ...
    """
    @functools.wraps(func)
    def wrapper(state):
        func_name = func.__name__
        logger.info(f">>> [{func_name}] 开始执行, 节点状态: {state}")
        add_running_task(state["task_id"], func_name)

        result = func(state)

        logger.info(f">>> [{func_name}] 结束执行, 节点状态: {result}")
        add_done_task(state["task_id"], func_name)
        return result

    return wrapper