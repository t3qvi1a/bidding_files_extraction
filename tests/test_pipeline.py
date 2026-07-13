"""目录处理入口和固定输出文件离线集成测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bidding_ocr.models import OCRLine, PageText, ProcessingConfig
from bidding_ocr.pipeline import process_pdf_tree


class FakePDFTextEngine:
    """
    【类功能】模拟单页中标公告 PDF，引导完整输出流水线而不加载 OCR 模型。
    :Attributes:
        page_count: int+固定页面总数
        ocr_pages: set[int]+固定 OCR 页码
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def __init__(self, pdf_path: Path, cache_dir: Path, config: ProcessingConfig) -> None:
        """
        【方法功能】保存测试参数并初始化单页状态。
        :param pdf_path: Path+测试 PDF 路径
        :param cache_dir: Path+测试缓存路径
        :param config: ProcessingConfig+测试处理配置
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.config = config
        self.page_count = 1
        self.ocr_pages = {1}

    def native_text(self, page_number: int) -> str:
        """
        【方法功能】返回空文本层以模拟扫描件。
        :param page_number: int+页码
        :return: str+空文本
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return ""

    def get_page(self, page_number: int, dpi: int) -> PageText:
        """
        【方法功能】返回固定项目和中标人 OCR 页面。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :return: PageText+固定页面
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        lines = [
            OCRLine("项目名称：离线测试项目", 0.99, []),
            OCRLine("中标人：甲建设有限公司", 0.99, []),
        ]
        return PageText(page_number, "\n".join(line.text for line in lines), lines, "ocr", dpi)

    def cover_bookmark_pages(self) -> list[int]:
        """
        【方法功能】返回空书签列表。
        :return: list[int]+空书签页码
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return []


class PipelineIntegrationTests(unittest.TestCase):
    """
    【类功能】验证统一入口生成七类 CSV、最终结果、复核清单和运行摘要。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def test_process_tree_writes_all_fixed_outputs(self) -> None:
        """
        【方法功能】验证最小目录运行后固定输出文件齐全且最终记录正确。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            output_dir = root / "results"
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(root / "pdf_files", output_dir)

            expected_files = {
                "tender_cover.csv",
                "bid_evaluation_report.csv",
                "bid_candidates.csv",
                "award_notice.csv",
                "bid_announcement.csv",
                "bid_list.csv",
                "archive_info.csv",
                "final.csv",
                "review_queue.csv",
                "run_summary.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))
            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.failed_files, 0)
            self.assertIn("甲建设有限公司", (output_dir / "final.csv").read_text(encoding="utf-8-sig"))
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run_summary["文件总数"], 1)

    def test_category_filter_emits_file_progress(self) -> None:
        """
        【方法功能】验证类别筛选仅处理目标目录文件，并输出开始和完成进度消息。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 16:25:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            announcement_dir = root / "pdf_files" / "bid_announcement"
            notice_dir = root / "pdf_files" / "award_notice"
            announcement_dir.mkdir(parents=True)
            notice_dir.mkdir(parents=True)
            (announcement_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            (notice_dir / "notice.pdf").write_bytes(b"fake-pdf")
            messages: list[str] = []
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    category_filter="bid_announcement",
                    progress_callback=messages.append,
                )

            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.files[0].category, "bid_announcement")
            self.assertTrue(any("处理中 1/1" in message for message in messages))
            self.assertTrue(any("完成 1/1" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
