# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from ..utils.log_manager import get_module_logger

logger = get_module_logger(__name__)

class ExcelLogger:
    """
    一个用于将召回测试结果记录到 Excel 文件的日志记录器。
    """
    def __init__(self, log_dir: str = "data/logs"):
        """
        初始化 ExcelLogger。

        Args:
            log_dir (str): 存储日志文件的目录。
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成带时间戳的唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"recall_test_log_{timestamp}.xlsx")
        
        # 创建一个新的 Excel 工作簿和工作表
        self.workbook = Workbook()
        self.worksheet: Worksheet = self.workbook.active # type: ignore
        self.worksheet.title = "Recall Test Log"
        
        # 写入表头
        self._write_header()
        
        logger.info(f"Excel 日志记录器初始化成功，日志将保存至: {self.filepath}")

    def _write_header(self):
        """写入 Excel 文件的表头。"""
        headers = [
            "时间戳 (Timestamp)",
            "查询 (Query)",
            "排名 (Rank)",
            "综合得分 (Score)",
            "文档来源 (Source)",
            "页码 (Page)",
            "文档内容 (Content)",
            "详细分数 (All Scores)"
        ]
        self.worksheet.append(headers)
        self.workbook.save(self.filepath)

    def log_results(self, query: str, results: List[Dict[str, Any]]):
        """
        将单次查询的结果记录到 Excel 文件中。

        Args:
            query (str):用户的查询。
            results (List[Dict[str, Any]]): 从检索器返回的文档列表。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not results:
            # 如果没有结果，也记录一条信息
            row = [timestamp, query, "N/A", "N/A", "无结果", "N/A", "N/A", "N/A"]
            self.worksheet.append(row)
        else:
            for i, doc in enumerate(results):
                rank = i + 1
                score = doc.get("score", 0)
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "未知")
                page = metadata.get("page", "N/A")
                
                # 提取所有可用的分数
                all_scores = {
                    "score": score,
                    "semantic_score": doc.get("semantic_score"),
                    "keyword_score": doc.get("keyword_score")
                }
                # 过滤掉值为 None 的分数
                all_scores_str = json.dumps({k: v for k, v in all_scores.items() if v is not None})

                row = [
                    timestamp,
                    query,
                    rank,
                    score,
                    source,
                    page,
                    content,
                    all_scores_str
                ]
                self.worksheet.append(row)
        
        try:
            self.workbook.save(self.filepath)
            logger.debug(f"成功将查询 '{query}' 的 {len(results)} 条结果记录到 {self.filepath}")
        except Exception as e:
            logger.error(f"保存 Excel 日志文件失败: {e}", exc_info=True)
