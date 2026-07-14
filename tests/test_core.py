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
    parse_bid_announcement,
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


def make_positioned_page(
    page_number: int,
    lines: list[tuple[str, float, float, float]],
) -> PageText:
    """
    【函数功能】创建带表格坐标的 OCR 页面对象，用于验证列级解析逻辑。
    :param page_number: int+页码
    :param lines: list[tuple[str, float, float, float]]+文字、置信度、横向中心和纵向中心
    :return: PageText+带真实列位置的测试页面
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: make_positioned_page(1, [("单位名称", 0.99, 300, 100)])
    """
    ocr_lines = [
        OCRLine(
            text,
            confidence,
            [[center_x - 40, center_y - 10], [center_x + 40, center_y - 10],
             [center_x + 40, center_y + 10], [center_x - 40, center_y + 10]],
        )
        for text, confidence, center_x, center_y in lines
    ]
    return PageText(page_number, "\n".join(line.text for line in ocr_lines), ocr_lines, "ocr", 300)


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
                "测试农田建设项目中标候选人公示",
                "项目编号：ABC20260001-S01",
                "五、所有投标人得分汇总表：",
                "序号 投标人名称 排名",
                "1 甲建设有限公司 1",
                "2 乙工程有限公司 2",
                "六、拟定中标人：甲建设有限公司",
            ],
        )
        records = parse_bid_candidates([page], self._context("bid_candidates"))
        self.assertEqual(
            [(record.company_name, record.award_status) for record in records],
            [("甲建设有限公司", "是"), ("乙工程有限公司", "否")],
        )

    def test_bid_candidates_prefers_inline_page_title(self) -> None:
        """
        【函数功能】验证候选人公示优先使用同一行标题且忽略后续表头项目名称。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:30:00
        """
        page = make_page(
            1,
            [
                "首页 > 中标候选人公示",
                "测试农田建设项目施工中标候选人公示",
                "项目编号：ABC20260001-S01",
                "投标人业绩、奖项（包含项目名称、获奖具体情况）",
                "五、所有投标人得分汇总表：",
                "序号 投标人名称 排名",
                "1 甲建设有限公司 1",
                "六、拟定中标人：甲建设有限公司",
            ],
        )

        records = parse_bid_candidates([page], self._context("bid_candidates"))

        self.assertEqual(records[0].project_name, "测试农田建设项目施工")
        self.assertEqual(records[0].project_code, "ABC20260001-S01")
        self.assertEqual(records[0].review_status, "通过")

    def test_bid_candidates_extracts_split_and_spaced_page_titles(self) -> None:
        """
        【函数功能】验证候选人公示兼容分行标题与字符间排版空白。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:30:00
        """
        cases = (
            (
                [
                    "前洲片区生态综合整治项目（农田水利部分）水利基建项目三标段",
                    "中标候选人公示",
                    "五、所有投标人得分汇总表：",
                    "序号 投标人名称 排名",
                    "1 甲建设有限公司 1",
                    "六、拟定中标人：甲建设有限公司",
                ],
                "前洲片区生态综合整治项目（农田水利部分）水利基建项目三标段",
            ),
            (
                [
                    "前 洲 片 区 生 态 综 合 整 治 项 目 （农 田 水 利 部 分）水 利 基 建 项 目 一 标 段",
                    "中 标 候 选 人 公 示",
                    "五、所有投标人得分汇总表：",
                    "序号 投标人名称 排名",
                    "1 甲建设有限公司 1",
                    "六、拟定中标人：甲建设有限公司",
                ],
                "前洲片区生态综合整治项目（农田水利部分）水利基建项目一标段",
            ),
        )
        for lines, expected_project_name in cases:
            with self.subTest(expected_project_name=expected_project_name):
                records = parse_bid_candidates(
                    [make_page(1, lines)],
                    self._context("bid_candidates"),
                )
                self.assertEqual(records[0].project_name, expected_project_name)
        self.assertEqual(records[0].review_status, "通过")

    def test_bid_candidates_reads_only_score_table_companies(self) -> None:
        """
        【函数功能】验证候选人公示仅提取得分表企业并补齐跨行有限责任公司名称。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:20:00
        """
        pages = [
            make_page(
                1,
                [
                    "测试农田建设项目中标候选人公示",
                    "投标人名称：江苏正鹏水利工程有限公司",
                    "五、所有投标人得分汇总表：",
                ],
            ),
            make_page(
                2,
                [
                    "序号 投标人名称 排名",
                    "1 甲建设有限公司 1",
                    "如东县水利电力建筑工程有限责任公      户",
                    "2 18911593.93 2",
                    "司      录",
                    "六、拟定中标人：甲建设有限公司",
                    "征询：无锡六垄田农业发展有限公司",
                ],
            ),
        ]

        records = parse_bid_candidates(pages, self._context("bid_candidates"))

        self.assertEqual(
            [record.company_name for record in records],
            ["甲建设有限公司", "如东县水利电力建筑工程有限责任公司"],
        )
        self.assertEqual([record.award_status for record in records], ["是", "否"])
        self.assertEqual([record.rank for record in records], ["1", "2"])

    def test_bid_candidates_missing_title_enters_review_queue(self) -> None:
        """
        【函数功能】验证未找到首页标题时清空项目名称并进入待复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:30:00
        """
        page = make_page(
            1,
            [
                "项目名称：表头中的错误项目名称",
                "第一中标候选人：甲建设有限公司",
            ],
        )

        records = parse_bid_candidates([page], self._context("bid_candidates"))

        self.assertEqual(records[0].project_name, "")
        self.assertEqual(records[0].review_status, "待复核")
        self.assertTrue(records[0].evidence.startswith("未识别到首页中标候选人公示标题；"))

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
        self.assertEqual(records[0].project_name, "测试工程项目")
        self.assertEqual(records[0].company_name, "甲建设有限公司")
        self.assertEqual(records[0].award_status, "是")

    def test_award_notice_extracts_project_from_wrapped_review_narrative(self) -> None:
        """
        【方法功能】验证中标通知书跨 OCR 行的评审结束叙述会去除招标人和结束语。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:20:55
        """
        page = make_page(
            1,
            [
                "江苏岳泰建设工程有限公司：",
                "无锡市惠山区洛社镇人民政府的花苑村高标准农田建设项目（二期）施工的评审工作已结",
                "束，根据有关法律、法规、规章和本工程交易文件的规定，确定你单位为交易结果确定人。",
            ],
        )

        records = parse_award_notice([page], self._context("award_notice"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].project_name, "花苑村高标准农田建设项目（二期）施工")
        self.assertEqual(records[0].company_name, "江苏岳泰建设工程有限公司")

    def test_bid_announcement_truncates_merged_metadata_fields(self) -> None:
        """
        【函数功能】验证合并行中的项目、标段及中标人字段分别被正确截断。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:00:00
        """
        page = make_page(
            1,
            [
                "项目编号：WXHS20210908001项目名称：前洲片区生态综合整治项目（农田水利部分）"
                "水利基建项目      台标段编号：WXHS20210908001-S04标段名称：前洲片区生态综合整治项目"
                "（农田水利部分）水利基建项目一标段招标人名称：无锡市惠山区人民政府前洲街道办事处"
                "中标人：江阴市水利工程公司"
            ],
        )
        records = parse_bid_announcement([page], self._context("bid_announcement"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].project_name, "前洲片区生态综合整治项目（农田水利部分）水利基建项目")
        self.assertEqual(records[0].project_code, "WXHS20210908001")
        self.assertEqual(records[0].lot_name, "前洲片区生态综合整治项目（农田水利部分）水利基建项目一标段")
        self.assertEqual(records[0].company_name, "江阴市水利工程公司")

    def test_evaluation_report_reads_ranking_table_and_recommended_winner(self) -> None:
        """
        【方法功能】验证评标报告排除目录和候选人区重复企业，并按推荐第一名判定中标。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        pages = [
            make_page(1, ["项目名称：测试农田建设项目", "项目编号：ABC20260002-S01"]),
            make_page(2, ["目录", "投标人排序及推荐的中标候选人名单"]),
            make_page(
                3,
                [
                    "投标人排序及推荐的中标候选人名单",
                    "投标人名称",
                    "1",
                    "甲建设有限公司",
                    "2",
                    "乙工程有限公司",
                    "推荐的中标候",
                    "第1名",
                    "乙工程有限公司",
                    "选人",
                    "第2名",
                    "甲建设有限公司",
                    "评标委员会：丙建设有限公司",
                ],
            ),
        ]
        records = parse_bid_evaluation_report(pages, self._context("bid_evaluation_report"))
        self.assertEqual(
            [(record.company_name, record.award_status, record.rank) for record in records],
            [("甲建设有限公司", "否", "1"), ("乙工程有限公司", "是", "2")],
        )

    def test_evaluation_report_uses_basic_info_engineering_name(self) -> None:
        """
        【方法功能】验证评标报告项目名称取基本情况表工程名称右侧同一行的值。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:30:00
        """
        basic_info_page = make_positioned_page(
            3,
            [
                ("一、基本情况一览表", 0.99, 300, 100),
                ("工程名称", 0.99, 200, 200),
                ("花苑村高标准农田建设项目（二期）施工", 0.99, 620, 200),
                ("招标范围", 0.99, 200, 240),
            ],
        )
        ranking_page = make_positioned_page(
            8,
            [
                ("投标人排序及推荐的中标候选人名单", 0.99, 500, 100),
                ("甲建设有限公司", 0.99, 700, 160),
                ("推荐的中标候选人", 0.99, 300, 220),
                ("第一名", 0.99, 500, 220),
                ("甲建设有限公司", 0.99, 700, 220),
            ],
        )

        records = parse_bid_evaluation_report(
            [basic_info_page, ranking_page],
            self._context("bid_evaluation_report"),
        )

        self.assertEqual(records[0].project_name, "花苑村高标准农田建设项目（二期）施工")
        self.assertEqual(records[0].review_status, "通过")

    def test_evaluation_report_keeps_cross_page_ranking(self) -> None:
        """
        【方法功能】验证评标报告排序表跨页时仍提取下一页投标人。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        pages = [
            make_page(1, ["项目名称：测试农田建设项目"]),
            make_page(2, ["投标人排序及推荐的中标候选人名单", "甲建设有限公司"]),
            make_page(
                3,
                ["乙工程有限公司", "推荐的中标候选人", "第一名", "甲建设有限公司"],
            ),
        ]
        records = parse_bid_evaluation_report(pages, self._context("bid_evaluation_report"))
        self.assertEqual(
            [(record.company_name, record.award_status, record.rank) for record in records],
            [("甲建设有限公司", "是", "1"), ("乙工程有限公司", "否", "2")],
        )

    def test_evaluation_report_marks_records_for_review_without_first_candidate(self) -> None:
        """
        【方法功能】验证未识别推荐第一名时投标企业保持未知并进入复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        pages = [
            make_page(1, ["项目名称：测试农田建设项目"]),
            make_page(2, ["投标人排序及推荐的中标候选人名单", "甲建设有限公司"]),
            make_page(3, ["投标人名称", "乙工程有限公司"]),
        ]
        records = parse_bid_evaluation_report(pages, self._context("bid_evaluation_report"))
        self.assertEqual([record.award_status for record in records], ["未知", "未知"])
        self.assertTrue(all(record.review_status == "待复核" for record in records))

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

    def test_bid_list_reads_only_unit_name_column_and_normalizes_metadata(self) -> None:
        """
        【方法功能】验证投标名单仅提取单位名称列并由标段信息生成项目元数据。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        """
        page = make_positioned_page(
            1,
            [
                ("标段（包）编号：WXHS20210908001-S04", 0.99, 500, 40),
                ("标段（包）名称：前洲片区生态综合整治项目（农田水利部分）水利基建项目一标段", 0.99, 500, 70),
                ("序号", 0.99, 100, 120),
                ("单位名称", 0.99, 320, 120),
                ("投标联系人", 0.99, 620, 120),
                ("联系方式", 0.99, 820, 120),
                ("1", 0.99, 100, 160),
                ("甲建设有限公司", 0.99, 320, 160),
                ("联系人建设有限公司", 0.99, 620, 160),
                ("2", 0.99, 100, 200),
                ("乙水利工程有限公司", 0.99, 320, 200),
                ("另一联系人有限公司", 0.99, 620, 200),
            ],
        )

        records = parse_bid_list([page], self._context("bid_list"))

        self.assertEqual([record.company_name for record in records], ["甲建设有限公司", "乙水利工程有限公司"])
        self.assertEqual(records[0].project_name, "前洲片区生态综合整治项目（农田水利部分）水利基建项目")
        self.assertEqual(records[0].lot_name, "前洲片区生态综合整治项目（农田水利部分）水利基建项目一标段")
        self.assertEqual(records[0].project_code, "WXHS20210908001-S04")
        self.assertEqual([record.award_status for record in records], ["未知", "未知"])
        self.assertEqual([record.review_status for record in records], ["通过", "通过"])

    def test_bid_list_reuses_unit_column_for_headerless_continuation_page(self) -> None:
        """
        【方法功能】验证跨页投标名单在续页缺少表头时仍按上一页单位列提取并去重。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        """
        first_page = make_positioned_page(
            1,
            [
                ("序号", 0.99, 100, 120),
                ("单位名称", 0.99, 320, 120),
                ("投标联系人", 0.99, 620, 120),
                ("甲建设有限公司", 0.99, 320, 160),
            ],
        )
        second_page = make_positioned_page(
            2,
            [
                ("甲建设有限公司", 0.99, 320, 40),
                ("乙水利工程有限公司", 0.99, 320, 80),
                ("联系人建设有限公司", 0.99, 620, 80),
            ],
        )

        records = parse_bid_list([first_page, second_page], self._context("bid_list"))

        self.assertEqual([record.company_name for record in records], ["甲建设有限公司", "乙水利工程有限公司"])
        self.assertEqual([record.source_pages for record in records], ["1", "2"])

    def test_bid_list_missing_header_falls_back_to_review_queue(self) -> None:
        """
        【方法功能】验证缺少单位列表头时回退全文提取并标记人工复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        """
        page = make_page(1, ["项目名称：测试农田建设项目", "甲建设有限公司"])

        records = parse_bid_list([page], self._context("bid_list"))

        self.assertEqual(records[0].company_name, "甲建设有限公司")
        self.assertEqual(records[0].review_status, "待复核")

    def test_bid_list_low_confidence_falls_back_to_review_queue(self) -> None:
        """
        【方法功能】验证单位名称置信度低于阈值时回退全文提取并标记人工复核。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        """
        page = make_positioned_page(
            1,
            [
                ("序号", 0.99, 100, 120),
                ("单位名称", 0.99, 320, 120),
                ("投标联系人", 0.99, 620, 120),
                ("甲建设有限公司", 0.70, 320, 160),
            ],
        )

        records = parse_bid_list([page], self._context("bid_list"))

        self.assertEqual(records[0].company_name, "甲建设有限公司")
        self.assertEqual(records[0].review_status, "待复核")

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
            self.assertIn("标段编号", path.read_text(encoding="utf-8-sig"))

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
