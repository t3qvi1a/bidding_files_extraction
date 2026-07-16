"""自动文件发现、中文名称分类和命令行类别筛选测试。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from pypdf import PdfWriter

from bidding_ocr.pipeline import classify_pdf_for_plan, discover_pdf_files
from bidding_ocr.utils import classify_pdf, is_tender_cover_text
from main import build_argument_parser, default_worker_count, parse_category_list


def write_blank_pdf(path: Path, page_count: int) -> None:
    """
    【函数功能】创建指定页数的空白 PDF，用于验证封面页数边界。
    :param path: Path+目标 PDF 路径
    :param page_count: int+待写入的空白页数量
    :return: None
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:12:03
    Example: write_blank_pdf(Path("封面.pdf"), 1)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as stream:
        writer.write(stream)


class ClassificationTests(unittest.TestCase):
    """
    【类功能】验证七类中文文件名规则、封面限制和首页文本确认逻辑。
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:12:03
    """

    def test_chinese_filename_keywords_map_to_expected_categories(self) -> None:
        """
        【方法功能】验证新增中文关键词均映射到需求指定的标准类别。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        cases = {
            "项目备案材料.pdf": "archive_info",
            "项目归档资料.PDF": "archive_info",
            "施工中标通知书.pdf": "award_notice",
            "施工中标人公告.pdf": "bid_announcement",
            "施工中标候选人.pdf": "bid_candidates",
            "施工中标公示.pdf": "bid_candidates",
            "项目评标报告.pdf": "bid_evaluation_report",
            "投标单位名单.pdf": "bid_list",
        }
        root = Path("input")
        for file_name, expected in cases.items():
            with self.subTest(file_name=file_name):
                self.assertEqual(classify_pdf(root / file_name, root, 0), expected)

    def test_tender_cover_page_limit_and_construction_design_exclusion(self) -> None:
        """
        【方法功能】验证封面仅接受一至三页，并排除施工组织设计目录中的 1.pdf。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        root = Path("input")
        self.assertEqual(classify_pdf(root / "封面.pdf", root, 3), "tender_cover")
        self.assertEqual(classify_pdf(root / "封面.pdf", root, 4), "unknown")
        self.assertEqual(
            classify_pdf(root / "第一信封" / "施工组织设计" / "1.pdf", root, 1),
            "unknown",
        )
        self.assertEqual(
            classify_pdf(root / "某公司" / "第一信封" / "1.pdf", root, 1),
            "tender_cover",
        )

    def test_ambiguous_one_pdf_requires_native_cover_text(self) -> None:
        """
        【方法功能】验证路径不明确的 1.pdf 需要首页标题或多个封面字段才能分类。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        root = Path("input")
        pdf_path = root / "普通目录" / "1.pdf"
        self.assertEqual(classify_pdf(pdf_path, root, 1), "unknown")
        self.assertEqual(
            classify_pdf(pdf_path, root, 1, "项目名称：测试项目\n投标人：测试公司"),
            "tender_cover",
        )
        self.assertTrue(is_tender_cover_text("参与文件\n测试项目"))
        self.assertFalse(is_tender_cover_text("施工组织设计第一章"))

    def test_cover_plan_reads_real_page_count(self) -> None:
        """
        【方法功能】验证批处理预分类读取真实页数，不再以零页默认接受封面。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted = root / "第一信封" / "封面.PDF"
            rejected = root / "第一信封" / "1.pdf"
            write_blank_pdf(accepted, 3)
            write_blank_pdf(rejected, 4)
            self.assertEqual(classify_pdf_for_plan(accepted, root), "tender_cover")
            self.assertEqual(classify_pdf_for_plan(rejected, root), "unknown")


class DiscoveryAndCLITests(unittest.TestCase):
    """
    【类功能】验证递归 PDF 发现以及 include、exclude 命令行参数约束。
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:12:03
    """

    def test_discovery_accepts_lowercase_and_uppercase_pdf_extensions(self) -> None:
        """
        【方法功能】验证递归扫描同时发现 .pdf 和 .PDF，且忽略非 PDF 文件。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "子目录").mkdir()
            (root / "a.pdf").write_bytes(b"pdf")
            (root / "子目录" / "b.PDF").write_bytes(b"pdf")
            (root / "子目录" / "c.txt").write_text("text", encoding="utf-8")
            _, paths = discover_pdf_files(root)
            self.assertEqual({path.name for path in paths}, {"a.pdf", "b.PDF"})

    def test_category_list_supports_chinese_comma_and_deduplication(self) -> None:
        """
        【方法功能】验证类别列表兼容中文逗号、空白和重复值。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        self.assertEqual(
            parse_category_list("award_notice， bid_candidates,award_notice"),
            ("award_notice", "bid_candidates"),
        )

    def test_include_and_exclude_are_mutually_exclusive(self) -> None:
        """
        【方法功能】验证 include 与 exclude 同时出现时 argparse 拒绝启动。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        parser = build_argument_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--include", "award_notice", "--exclude", "archive_info"])

    def test_unknown_category_is_rejected(self) -> None:
        """
        【方法功能】验证类别列表出现未知名称时立即返回命令行参数错误。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_category_list("award_notice,not_exists")

    def test_workers_argument_uses_safe_default_and_rejects_zero(self) -> None:
        """
        【方法功能】验证 worker 默认值限制在安全范围且拒绝非正数。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 16:00:00
        """
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).workers, default_worker_count())
        self.assertEqual(parser.parse_args(["--workers", "1"]).workers, 1)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--workers", "0"])


if __name__ == "__main__":
    unittest.main()
