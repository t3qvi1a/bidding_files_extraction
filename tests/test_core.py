"""招投标 PDF 解析核心规则单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bidding_ocr.models import ExtractionRecord, OCRLine, PageText
from bidding_ocr.parsers import (
    ParserContext,
    parse_archive_info,
    parse_award_notice,
    parse_bid_candidates,
    parse_bid_evaluation_report,
    parse_bid_list,
)
from bidding_ocr.pipeline import merge_and_deduplicate, write_records_csv
from bidding_ocr.utils import (
    classify_pdf,
    determine_cover_award_status,
    extract_company_names,
    is_readable_chinese_text,
    normalize_text,
)


def make_page(page_number: int, lines: list[str], confidence: float = 0.98) -> PageText:
    """
    【函数功能】创建用于解析器测试的固定 OCR 页面对象。
    :param page_number: int+页码
    :param lines: list[str]+页面文字行
    :param confidence: float+统一 OCR 置信度（默认0.98）
    :return: PageText+测试页面
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: make_page(1, ["项目名称：测试项目"])
    """
    ocr_lines = [
        OCRLine(
            text,
            confidence,
            [[0, index * 10], [10, index * 10], [10, index * 10 + 5], [0, index * 10 + 5]],
        )
        for index, text in enumerate(lines)
    ]
    return PageText(page_number, "\n".join(lines), ocr_lines, "ocr", 300)


class UtilityTests(unittest.TestCase):
    """
    【类功能】验证文本质量、分类、路径状态与企业名称规范化规则。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def test_unawarded_path_has_precedence(self) -> None:
        """
        【方法功能】验证同时含“中标”和“未”时优先判定为未中标。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.assertEqual(determine_cover_award_status("未中标单位投标文件/封面.pdf"), "否")
        self.assertEqual(determine_cover_award_status("中标资料/封面.pdf"), "是")
        self.assertEqual(determine_cover_award_status("tender_cover/封面.pdf"), "未知")

    def test_directory_alias_classification(self) -> None:
        """
        【方法功能】验证 archived_info 目录映射为标准 archive_info 类别。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        root = Path("pdf_files")
        self.assertEqual(
            classify_pdf(root / "archived_info" / "a.pdf", root, 300),
            "archive_info",
        )

    def test_text_quality_and_normalization(self) -> None:
        """
        【方法功能】验证可读中文判定和全半角空白规范化。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.assertTrue(is_readable_chinese_text("项目名称：测试高标准农田建设项目，投标人名称：测试公司。"))
        self.assertFalse(is_readable_chinese_text("��վ������ȱ��� abc 123"))
        self.assertEqual(normalize_text(" 江苏（测试） 有限公司 "), "江苏(测试)有限公司")

    def test_extract_multiple_companies(self) -> None:
        """
        【方法功能】验证同一文本中多个企业名称按顺序提取。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.assertEqual(
            extract_company_names("甲建设有限公司、乙工程集团有限公司"),
            ["甲建设有限公司", "乙工程集团有限公司"],
        )


class ParserTests(unittest.TestCase):
    """
    【类功能】验证候选人公示和中标通知书的分类解析行为。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def _context(self, category: str) -> ParserContext:
        """
        【方法功能】构造指定类别的解析器测试上下文。
        :param category: str+文件类别
        :return: ParserContext+测试上下文
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return ParserContext(
            pdf_path=Path("test.pdf"),
            relative_path=f"{category}/test.pdf",
            category=category,
            generated_at="2026-07-13T11:08:59+08:00",
            confidence_threshold=0.80,
        )

    def test_bid_candidates_first_company_wins(self) -> None:
        """
        【方法功能】验证候选人公示第一家企业中标且其余企业未中标。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        page = make_page(
            1,
            [
                "项目名称：测试农田建设项目",
                "项目编号：ABC20260001-S01",
                "第一中标候选人",
                "甲建设有限公司",
                "乙工程有限公司",
            ],
        )
        records = parse_bid_candidates([page], self._context("bid_candidates"))
        self.assertEqual(
            [(record.company_name, record.award_status) for record in records],
            [("甲建设有限公司", "是"), ("乙工程有限公司", "否")],
        )

    def test_award_notice_explicit_winner(self) -> None:
        """
        【方法功能】验证中标通知书按明确中标人字段提取唯一中标企业。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        page = make_page(1, ["项目名称：测试工程项目", "中标人：甲建设有限公司"])
        records = parse_award_notice([page], self._context("award_notice"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].company_name, "甲建设有限公司")
        self.assertEqual(records[0].award_status, "是")

    def test_evaluation_report_keeps_cross_page_ranking(self) -> None:
        """
        【方法功能】验证评标报告排序表跨页时仍提取下一页投标人。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        pages = [
            make_page(1, ["项目名称：测试农田建设项目", "项目编号：ABC20260002-S01"]),
            make_page(2, ["投标人排序及推荐的中标候选人名单", "甲建设有限公司"]),
            make_page(3, ["乙工程有限公司"]),
        ]
        records = parse_bid_evaluation_report(pages, self._context("bid_evaluation_report"))
        self.assertEqual(
            [(record.company_name, record.award_status) for record in records],
            [("甲建设有限公司", "是"), ("乙工程有限公司", "否")],
        )

    def test_archive_combines_winner_and_bidder_list(self) -> None:
        """
        【方法功能】验证备案资料合并中标通知书企业和投标名单企业。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        pages = [
            make_page(1, ["工程名称：测试农田建设项目"]),
            make_page(5, ["中标通知书", "中标人：甲建设有限公司"]),
            make_page(8, ["按时送达投标文件的投标人名单", "甲建设有限公司", "乙工程有限公司"]),
        ]
        records = parse_archive_info(pages, self._context("archive_info"))
        self.assertEqual(
            [(record.company_name, record.award_status) for record in records],
            [("甲建设有限公司", "是"), ("乙工程有限公司", "否")],
        )

    def test_missing_project_enters_review_queue(self) -> None:
        """
        【方法功能】验证投标名单缺少项目名称时记录仍保留并标记待复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        records = parse_bid_list(
            [make_page(1, ["投标名单", "甲建设有限公司"])],
            self._context("bid_list"),
        )
        self.assertEqual(records[0].company_name, "甲建设有限公司")
        self.assertEqual(records[0].review_status, "待复核")


class MergeAndOutputTests(unittest.TestCase):
    """
    【类功能】验证最终去重、来源优先级、冲突和 CSV 编码行为。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def test_merge_detects_status_conflict(self) -> None:
        """
        【方法功能】验证同项目同企业状态冲突时保留高优先级结论并标记复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        base = {
            "project_name": "测试项目",
            "project_code": "ABC20260001",
            "company_name": "甲建设有限公司",
            "confidence": 0.95,
            "generated_at": "2026-07-13T11:08:59+08:00",
        }
        records = [
            ExtractionRecord(
                **base,
                category="tender_cover",
                source_path="未中标/1.pdf",
                award_status="否",
                evidence="路径判断",
            ),
            ExtractionRecord(
                **base,
                category="award_notice",
                source_path="通知书.pdf",
                award_status="是",
                evidence="中标人字段",
            ),
        ]
        merged = merge_and_deduplicate(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].award_status, "是")
        self.assertEqual(merged[0].review_status, "冲突待复核")
        self.assertIn("通知书.pdf", merged[0].source_path)
        self.assertIn("未中标/1.pdf", merged[0].source_path)

    def test_csv_has_utf8_bom_and_header(self) -> None:
        """
        【方法功能】验证 CSV 使用 UTF-8 BOM 并包含统一中文表头。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.csv"
            write_records_csv(path, [ExtractionRecord(project_name="测试项目")])
            content = path.read_bytes()
            self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
            self.assertIn("项目名称", path.read_text(encoding="utf-8-sig"))

    def test_missing_project_code_merges_with_named_record(self) -> None:
        """
        【方法功能】验证缺少项目编号的同名项目企业可归并到唯一编号记录。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        records = [
            ExtractionRecord(
                project_name="测试项目",
                project_code="ABC20260003",
                company_name="甲建设有限公司",
                category="bid_announcement",
                award_status="是",
                source_path="公告.pdf",
                generated_at="2026-07-13T11:08:59+08:00",
            ),
            ExtractionRecord(
                project_name="测试项目",
                company_name="甲建设有限公司",
                category="bid_list",
                source_path="名单.pdf",
                generated_at="2026-07-13T11:08:59+08:00",
            ),
        ]
        merged = merge_and_deduplicate(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].project_code, "ABC20260003")
        self.assertIn("名单.pdf", merged[0].source_path)


if __name__ == "__main__":
    unittest.main()
