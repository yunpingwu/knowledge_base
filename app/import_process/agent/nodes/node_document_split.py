import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple, final
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.node_decorator import node_log
# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500


def get_content(state):
    """
    获取内容
    :param state:
    :return:
    """
    md_content = state.get("md_content")
    if not md_content:
        logger.error("get_content:md_content为空")
        raise ValueError("get_content:md_content为空")
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "default_title")
    return md_content, file_title


def split_by_title(md_content, file_title):
    """
    根据标题进行切割
    :param md_content:
    :param file_title:
    :return:
    """
    # 标题匹配模式
    title_pattern = r'\s*#{1,6}\s+.+'
    lines = md_content.split('\n')
    # 临时存储变量
    current_title = ""
    current_lines = []
    title_count = 0
    is_code_block = False

    sections = []

    for line in lines:
        strip_line = line.strip()
        if strip_line.startswith("```"):
            is_code_block = not is_code_block
            current_lines.append(line)
        elif is_code_block:
            current_lines.append(line)
        elif re.match(title_pattern, line):
            title_count += 1
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            current_title = strip_line
            current_lines = [current_title]
        else:
            current_lines.append(line)

    # 处理最后一个部分
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    logger.info(f"split_by_title:文档切割完成，共{title_count}个标题，{len(lines)}行内容")
    return sections, title_count, len(lines)


def split_long_chunk(section, max_length):
    """
    对过长的Chunk进行二次切割
    :param section:
    :param max_length:
    :return:
    """
    content = section.get("content", "")
    if len(content) <= max_length:
        logger.info(f"split_long_chunk:Chunk长度未超过阈值，无需切割")
        return [section]
    logger.info(f"split_long_chunk:Chunk长度超过阈值，进行二次切割")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！"," "]
    )
    chunks = text_splitter.split_text(content)
    sub_sections = []
    for index, chunk in enumerate(chunks):
        sub_sections.append({
            "title": section.get("title", f"Chunk_{index+1}"),
            "content": chunk,
            "file_title": section.get("file_title"),
            "parent_title": section.get("title"),
            "part": index + 1
        })
    return sub_sections


def merge_short_chunks(final_sections, min_length):
    """
    对过短的Chunk进行合并
    :param final_sections:
    :param min_length:
    :return:
    """
    merge_sections = []
    pre_section = None
    for section in final_sections:
        if pre_section is None:
            pre_section = section
        is_current_short = len(pre_section.get("content", "")) <= min_length
        is_same_parent_title = pre_section.get("parent_title") and (section.get("parent_title") == pre_section.get("parent_title"))
        if is_current_short and is_same_parent_title:
            pre_section["content"] += "\n" + section.get("content", "")
            pre_section["part"] = section.get("part", 1)
        else:
            merge_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merge_sections.append(pre_section)
    return merge_sections


def refine_chunks(sections, max_length, min_length):
    """
    细粒度切割，大于DEFAULT_MAX_CONTENT_LENGTH的Chunk会进行二次切割，小于MIN_CONTENT_LENGTH的Chunk会进行合并
    :param sections:
    :param MIN_CONTENT_LENGTH:
    :return:
    """
    final_sections = []
    for section in sections:
        sub_section = split_long_chunk(section, max_length)
        final_sections.extend(sub_section)

    final_sections = merge_short_chunks(final_sections, min_length)
    # 补全属性参数
    for section in sections:
        section['part'] = section.get('part', 1)
        section['parent_title'] = section.get('parent_title') or section.get('title')
    return final_sections


def backup_chunks(state, sections):
    """
    存储Chunks
    :param state:
    :param sections:
    :return:
    """
    local_dir = state.get("local_dir")
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    logger.info(f"backup_chunks:Chunks存储完成，文件路径: {backup_file_path}")


@node_log
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    try:
        # 参数校验
        md_content, file_title = get_content(state)
        # 粗粒度切割（标题切割）
        sections,title_count,lines_count = split_by_title(md_content, file_title)
        # 当文档没有标题，给出默认标题
        if title_count == 0:
            sections.append({
                "title": "默认标题",
                "content": md_content,
                "file_title": file_title
            })
        # 细粒度切割
        sections = refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        # 更新状态
        state["chunks"] = sections
        # 存储Chunks
        backup_chunks(state,sections)
    except Exception as e:
        logger.error(f"node_document_split:文档切分失败: {e}")
        raise

    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")