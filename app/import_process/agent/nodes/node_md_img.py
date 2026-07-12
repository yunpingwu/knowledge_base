import base64
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Tuple

from minio.deleteobjects import DeleteObject

from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.node_decorator import node_log
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.clients.minio_utils import get_minio_client

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(file_name: str) -> bool:
    """
    判断文件是否为支持的图片格式
    :param file_name:
    :return:
    """
    return os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS


def get_content(state) -> tuple[str, Path, Path]:
    """
    获取内容
    :param state:
    :return:
    """
    # 获取 Markdown 文件路径
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError(f"get_content:Markdown 文件不存在: {md_file_path}")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"get_content:Markdown 文件不存在: {md_file_path}")

    if not state["md_content"]:
        with md_path_obj.open("r", encoding="utf-8") as f:
           md_content  = f.read()
    state["md_content"] = md_content

    # 图片文件夹获取
    images_dir = md_path_obj.parent / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"get_content:图片文件夹不存在: {images_dir}")

    return md_content, md_path_obj, images_dir


def find_image_in_content(md_content, image_file,context_length):
    """
    在Markdown内容中查找图片
    :param md_content:
    :param image_file:
    :param context_length:
    :return:
    """
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")
    results = []
    for match in pattern.finditer(md_content):
        start,end = match.span()
        pre_text = md_content[max(start-context_length, 0):start]
        post_text = md_content[end:min(end+context_length, len(md_content))]
        results.append((pre_text, post_text))

    if results:
        logger.info(f"find_image_in_content:图片上下文存在: {image_file}")
        return results[0]
    return None


def scan_images(md_content, images_dir_obj) -> list[Tuple[str,str,Tuple[str,str]]]:
    """
    扫描图片
    :param md_content:
    :param images_dir_obj:
    :return:
    """
    targets = []
    for image_file in os.listdir(images_dir_obj):
        if not is_supported_image(image_file):
            logger.warning(f"scan_images:图片格式不支持: {image_file}")
            continue

        #获取图片上下文
        context_data = find_image_in_content(md_content, image_file, context_length=100)
        if not context_data:
            logger.warning(f"scan_images:图片上下文不存在: {image_file}")
            continue

        targets.append((image_file, str(images_dir_obj / image_file), context_data))
    return targets


def generate_img_summaries(targets, stem) -> dict[str, str]:
    """
    生成图片摘要
    :param targets:
    :param name:
    :return:
    """
    summaries = {}
    request_times = deque()
    for image_file, image_path, context_data in targets:
        apply_api_rate_limit(request_times,max_requests=9)
        vm_model = get_llm_client(model=lm_config.lv_model)
        prompt = load_prompt('image_summary',root_folder=stem,image_content=context_data)
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                     },
                    {"type": "text", "text": f"{prompt}"},

                ]
            }
        ]

        resp = vm_model.invoke(messages)
        summary = resp.content.strip().replace("\n", "")
        summaries[image_file] = summary
        logger.info(f"generate_img_summaries:图片摘要生成成功: {image_file}")
    return summaries


def upload_images_to_minio_and_replace_links(md_content, stem, summaries, targets):
    """
    上传图片到MinIO并替换链接,替换原md中的图片和描述
    :param md_content:
    :param name:
    :param summaries:
    :param targets:
    :return:
    """
    # 删除原有图片
    minio_client = get_minio_client()
    object_list = minio_client.list_objects(bucket_name=minio_config.bucket_name,
                              prefix=f"{minio_config.minio_img_dir}/{stem}",
                              recursive=True)
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    minio_client.remove_objects(minio_config.bucket_name, delete_object_list)

    images_url = {}
    # 上传图片到minio服务器
    for image_file, image_path, context_data in targets:
        try:
            minio_client.fput_object(bucket_name=minio_config.bucket_name,
                                    object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                                    file_path=image_path,
                                    content_type="image/jpeg")
            # 格式：http://<minio_endpoint>/<minio_bucket_name>/<minio_img_dir>/<stem>/<image_file>
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"upload_images_to_minio_and_replace_links:图片上传成功: {image_file}")
        except Exception as e:
            logger.error(f"upload_images_to_minio_and_replace_links:上传图片到MinIO出错: {image_file}, 错误信息: {e}")
    # 替换原md中的图片和描述
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = {"url": url, "summary": summary}
        logger.info(f"upload_images_to_minio_and_replace_links:图片信息获取成功: {image_infos}")
        if image_infos:
            rep = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")
            md_content = rep.sub(f"![{summary}]({url})", md_content)
    return md_content


def replace_md_and_save(new_md_content, md_path_obj):
    """
    替换MD文件并保存,返回老地址
    :param new_md_content:
    :param md_path_obj:
    :return:
    """
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"
    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"replace_md_and_save:新MD文件保存成功: {new_md_path_str}")
    return new_md_path_str


@node_log
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 校验操作的数据合法性
    md_content,md_path_obj,images_dir_obj = get_content(state)
    # 识别md中使用的图片，总结
    # 如果没有图片直接返回
    if not images_dir_obj.exists():
        logger.info("node_md_img:没有图片，直接返回")
        return state
    targets = scan_images(md_content,images_dir_obj)
    # 视觉模型处理图片
    summaries = generate_img_summaries(targets,md_path_obj.name)
    #上传图片到minio，替换md中图片链接
    new_md_content = upload_images_to_minio_and_replace_links(md_content, md_path_obj.name, summaries,targets)
    # 保存新MD文件
    new_md_file_path = replace_md_and_save(new_md_content, md_path_obj)

    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")