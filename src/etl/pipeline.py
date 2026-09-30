# 本文件包含部分从 Dify 项目移植的代码。
# 原始来源: https://github.com/langgenius/dify
# 遵循修改后的 Apache License 2.0 许可证。详情请参阅项目根目录下的 DIFY_LICENSE 文件。

import time
from pathlib import Path
from typing import Any

from src.etl.cleaners.base import BaseCleaner
from src.etl.cleaners.basic_cleaner import BasicCleaner
from src.etl.extractors.base import BaseExtractor
from src.etl.extractors.markdown_extractor import MarkdownExtractor
from src.etl.splitters.base import BaseSplitter
from src.etl.splitters.recursive_text_splitter import RecursiveTextSplitter
from src.models.document import Document
from src.utils.config import safe_repr
from src.utils.log_manager import get_module_logger

logger = get_module_logger(__name__)


def _require_chunk_list(chunks: Any, stage: str) -> list[Document]:
    """确认流水线某阶段返回的是「文档列表」。

    与 ``RetrievalService._require_document_list`` 同一判据、同一收场：三个阶段
    都是 ``BaseExtractor``／``BaseCleaner``／``BaseSplitter`` 的实现，抽象基类
    正是为替换实现而存在的，而 ``Pipeline.__init__`` 接受任意实现。实测
    ``split`` 返回 ``None`` 时，下面那条日志里的 ``len(final_chunks)`` 抛裸
    ``TypeError: object of type 'NoneType' has no len()``——报错现场在日志行，
    消息里既没有阶段名也没有「是谁违反了契约」。

    这里抛 ``TypeError`` 而不是 ``ValueError``，理由与
    ``_require_document_list`` 逐字相同：这是**接口契约**问题，不是数据内容
    问题。入参 ``document`` 形态错误（调用方给了非 ``Document``）才是数据问题，
    那一处用 ``ValueError``。

    不静默丢弃不合规条目：丢弃会让「实现错了」表现为「分片变少了」。
    """
    if not isinstance(chunks, list):
        raise TypeError(
            f"{stage}返回的必须是文档列表，但得到 {type(chunks).__name__}。"
            " 请检查 BaseExtractor/BaseCleaner/BaseSplitter 实现是否遵守契约。"
        )
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, Document):
            raise TypeError(
                f"{stage}返回的第 {position} 项不是 Document"
                f"（{type(chunk).__name__}）。"
                " 请检查 BaseExtractor/BaseCleaner/BaseSplitter 实现是否遵守契约。"
            )
    return chunks


class Pipeline:
    """
    文档处理流水线。
    根据配置动态组合抽取器、清洗器和分割器来处理文档。
    已注入 CSE 性能传感器。
    """

    def __init__(self, extractor: BaseExtractor, cleaner: BaseCleaner, splitter: BaseSplitter):
        """
        初始化 Pipeline。
        Args:
            extractor (BaseExtractor): 文档抽取器实例。
            cleaner (BaseCleaner): 文档清洗器实例。
            splitter (BaseSplitter): 文档分割器实例。
        """
        self.extractor = extractor
        self.cleaner = cleaner
        self.splitter = splitter
        logger.info("ETL Pipeline 初始化完成。")

    @classmethod
    def from_file_path(
        cls, file_path: Path, splitter_structure_mode: str = "standard"
    ) -> "Pipeline":
        """
        根据文件路径创建并初始化 Pipeline 实例。
        """
        logger.info(f"正在从文件路径 '{file_path}' 创建 ETL Pipeline。")
        extractor_instance: BaseExtractor = MarkdownExtractor()
        cleaner_instance: BaseCleaner = BasicCleaner()
        splitter_instance: BaseSplitter = RecursiveTextSplitter(
            structure_mode=splitter_structure_mode
        )

        return cls(
            extractor=extractor_instance, cleaner=cleaner_instance, splitter=splitter_instance
        )

    def process(self, document: Document) -> list[Document]:
        """
        处理单个文档，并监控每一步的耗时 (CSE Sensor)。
        """
        # 形态校验先做：``process`` 的第一行就读 ``document.metadata``，此前
        # ``5``／``"abc"``／``[5]``／``None`` 全部抛裸
        # ``AttributeError: ... object has no attribute 'metadata'``——不在
        # ``KnowledgeBuildService.build`` 按 ``ValueError`` 建立的分类内，而
        # 那是建库主路径（``_process_file`` 对知识库里的每个文件调一次）。
        # 判据与 ``RecursiveTextSplitter._require_documents`` 一致：要求
        # ``Document`` 实例而不是鸭子类型，本方法后续每一步都读它的
        # ``content``／``metadata``。
        if not isinstance(document, Document):
            raise ValueError(f"Pipeline 处理的文档必须是 Document，但得到 {safe_repr(document)}。")
        source = document.metadata.get("source", "未知来源")
        logger.info(f"开始处理文档: {source}")

        start_total = time.perf_counter()

        # 抽取
        start_step = time.perf_counter()
        extracted_docs = _require_chunk_list(self.extractor.extract(document), "抽取阶段")
        logger.info(f"抽取完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s")

        # 清洗
        start_step = time.perf_counter()
        cleaned_docs = _require_chunk_list(self.cleaner.clean(extracted_docs), "清洗阶段")
        logger.info(f"清洗完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s")

        # 分割
        start_step = time.perf_counter()
        final_chunks = _require_chunk_list(self.splitter.split(cleaned_docs), "分割阶段")
        logger.info(
            f"分割完成: {source}, 耗时: {time.perf_counter() - start_step:.4f}s, 分片数: {len(final_chunks)}"
        )

        total_duration = time.perf_counter() - start_total
        logger.info(f"文档总处理时长: {source}, 总计: {total_duration:.2f}s")

        return final_chunks
