import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.node_decorator import node_log
from app.conf.mineru_config import mineru_config

def validate_path(state):
    """
    验证路径
    :param state: 
    :return: 
    """
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]
    if not pdf_path or not local_dir:
        raise ValueError("validate_path: pdf_path or local_dir 缺失")

    pdf_path = Path(pdf_path)
    output_dir = Path(local_dir)

    if not pdf_path.exists():
        raise ValueError("validate_path: pdf_path 不存在")
    if not output_dir.exists():
        raise ValueError("validate_path: local_dir 不存在")

    if not output_dir.exists():
        logger.info("validate_path: output_dir 不存在，正在创建")
        output_dir.mkdir(parents=True, exist_ok=True)

    return pdf_path, output_dir


def upload_poll(pdf_path,output_dir):
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("upload_poll: mineru_config.base_url or mineru_config.api_key 缺失")

    logger.info("upload_poll: 开始上传文件到MinerU")
    # 1. 调用批量接口，获取上传Signed URL和任务batch_id
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key}"
    }

    url_upload = f"{mineru_config.base_url}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path.name}],
        "model_version": "vlm"  # 官方推荐解析模型
    }

    resp = requests.post(url_upload, headers=request_headers, json=req_data,timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"upload_poll: 上传文件到MinerU失败")
    resp_data = resp.json()
    if resp_data["code"] != 0:
        raise ValueError(f"upload_poll: 上传文件到MinerU失败，错误信息: {resp_data['message']}")
    sign_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    # 2. 读取PDF二进制数据，准备上传
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    # 创建Session（复用TCP连接，禁用代理避免签名验证失败）
    upload_session = requests.Session()
    upload_session.trust_env = False

    try:
        resp_upload = upload_session.put(sign_url, data=pdf_data, timeout=30)
        if resp_upload.status_code != 200:
            raise ValueError(f"upload_poll: 上传文件到MinerU失败")

    except Exception as e:
        raise ValueError(f"upload_poll: 上传文件到MinerU失败，错误信息: {e}")
    finally:
        upload_session.close()
    # 3. 根据batch_id轮询任务状态，直至完成/失败/超时
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    poll_interval = 3
    timeout = 600
    while True:
        if time.time() - start_time > timeout:
            raise ValueError("upload_poll: 轮询任务状态超时")
        resp_poll = requests.get(poll_url, headers=request_headers, timeout=30)
        if resp_poll.status_code != 200:
            if  500 <= resp_poll.status_code < 600:
                time.sleep(poll_interval)
                raise ValueError("upload_poll: 轮询任务状态失败")
        resp_data = resp_poll.json()
        if resp_data["code"] != 0:
            raise ValueError(f"upload_poll: 轮询任务状态失败，错误信息: {resp_data['message']}")
        extract_result = resp_data["data"]["extract_result"][0]
        if extract_result["state"] == "done":
            full_zip_url = extract_result["full_zip_url"]
            logger.info("upload_poll: 上传文件到MinerU成功,耗时: {:.2f}秒".format(time.time() - start_time))
            return full_zip_url
        else:
            time.sleep(poll_interval)

def download_extract(zip_url, local_dir, stem):
    """
    下载指定的md.zip文件，解压后返回md文件路径
    :param zip_url:
    :param local_dir:
    :param stem:
    :return:
    """
    resp = requests.get(zip_url, timeout=30)
    if resp.status_code != 200:
        raise ValueError("download_extract: 下载文件失败")
    zip_path = local_dir / f"{stem}.zip"
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    logger.info("download_extract: 下载文件成功, 路径: {}".format(zip_path))

    extract_target_dir = local_dir / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_target_dir)

    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise ValueError("download_extract: 解压文件为空")
    target_md_file = None
    for md_file in md_file_list:
        if md_file.name == stem + ".md":
            target_md_file = md_file
        break

    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0]

    #统一重命名
    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    final_md_file = target_md_file.resolve()
    logger.info("download_extract: 解压文件成功, 路径: {}".format(final_md_file))
    return str(final_md_file)



@node_log
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    try:
        # 验证路径
        pdf_path,output_dir = validate_path(state)
        # 调用 MinerU (magic-pdf) 工具,返回下载地址
        zip_url = upload_poll(pdf_path,output_dir)
        # 下载文件
        md_path = download_extract(zip_url, output_dir, pdf_path.stem)
        
        state["md_path"] = md_path
        state["local_dir"] = str(output_dir)

        with open(md_path, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()
    except Exception as e:
        logger.error(f">>> [node_pdf_to_md]使用minerU解析异常: {e}")

    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")