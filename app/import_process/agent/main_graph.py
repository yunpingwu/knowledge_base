from dotenv import load_dotenv
from langgraph.graph import StateGraph,END, START

from app.import_process.agent.nodes import node_pdf_to_md, node_md_img, node_document_split, node_item_name_recognition, \
    node_bge_embedding, node_import_milvus
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.state import ImportGraphState

load_dotenv()

#初始化langgraph
workflow = StateGraph(ImportGraphState)

#注册所有节点
workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition",node_item_name_recognition)
workflow.add_node("node_bge_embedding",node_bge_embedding)
workflow.add_node("node_import_milvus",node_import_milvus)

#注册入口
workflow.set_entry_point("node_entry")

#定义条件边路由函数
def condition_edge_router(state: ImportGraphState) -> str:
    """
    根据状态决定走哪条边
    :param state:
    :return:
    """
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "node_md_img"
    else:
        return END

#定义边
workflow.add_conditional_edges(
    "node_entry",
    condition_edge_router,
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END
    }
)
workflow.add_edge("node_pdf_to_md","node_md_img")
workflow.add_edge("node_md_img","node_document_split")
workflow.add_edge("node_document_split","node_item_name_recognition")
workflow.add_edge("node_item_name_recognition","node_bge_embedding")
workflow.add_edge("node_bge_embedding","node_import_milvus")
workflow.add_edge("node_import_milvus",END)

kb_import_app = workflow.compile()

