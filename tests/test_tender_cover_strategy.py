"""投标文件封面四种现有版式、去红章和专属页面读取策略测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pypdf import PdfWriter

from bidding_ocr.models import OCRLine, PageText, ProcessingConfig
from bidding_ocr.parsers import ParserContext, parse_tender_cover
from bidding_ocr.pdf_engine import OCRBackend, PDFTextEngine
from bidding_ocr.pipeline import load_pages_for_category
from bidding_ocr.tender_cover_strategy import (
    build_cover_image_variants,
    extract_tender_cover_fields,
    is_valid_project_name_candidate,
    is_fragmented_cover_text,
    remove_red_seal,
    split_tender_cover_project_and_lot_code,
    strip_bid_file_project_noise,
    suppress_red_seal_by_red_channel,
)


COVER_CASES = (
    (
        "tender_cover_01",
        """新点网上招投标系统
花苑村高标准农田建设项目（二期）
（工程名称）
施工（标段名称）
参与文件
交易编号：HSLS2021011-01
参与单位：江苏仪征苏中建设有限公司（盖单位章）""",
        "花苑村高标准农田建设项目(二期)施工",
        "HSLS2021011",
        "HSLS2021011-01",
        "江苏仪征苏中建设有限公司",
        "ocr",
    ),
    (
        "tender_cover_02",
        """花苑村高标准农田建设项目（二期）（工程名称）
施工（标段名称）
参与文件
交易编号：HSLS2021011-01
参与单位：江苏嘉奕建设有限公司（盖单位章）""",
        "花苑村高标准农田建设项目(二期)施工",
        "HSLS2021011",
        "HSLS2021011-01",
        "江苏嘉奕建设有限公司",
        "ocr",
    ),
    (
        "tender_cover_03",
        """花苑村高标准农田建设项目（二期）（工程名称）
施工（标段名称）
参与文件
交易编号：HSLS2021011-01
参与单位：无锡润华市政绿化有限公司（盖单位章）""",
        "花苑村高标准农田建设项目(二期)施工",
        "HSLS2021011",
        "HSLS2021011-01",
        "无锡润华市政绿化有限公司",
        "ocr",
    ),
    (
        "tender_cover_04",
        """2021年洛社镇石塘湾片区高标准农田建设项目
施工招标
投标文件
项目编号：WXHS20221008001
项目名称：2021年洛社镇石塘湾片区高标准农田建设项目施工
投 标 人：无锡市银河建筑安装有限公司
日期：2022年11月3日""",
        "2021年洛社镇石塘湾片区高标准农田建设项目施工",
        "WXHS20221008001",
        "WXHS20221008001",
        "无锡市银河建筑安装有限公司",
        "text",
    ),
)


class VariantOCRBackend(OCRBackend):
    """
    【类功能】根据候选图像是否含红章返回不同质量文本，验证去红章候选选择。
    :Attributes:
        calls: list[str]+已识别候选图像类型
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化候选调用记录。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        self.calls: list[str] = []

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        """
        【方法功能】原图返回低质量文本，去红章图返回完整封面字段。
        :param image: np.ndarray+候选 RGB 图像
        :return: list[OCRLine]+固定 OCR 文字行
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        contains_red_seal = bool(np.any((image[:, :, 0] > 200) & (image[:, :, 1] < 80)))
        self.calls.append("original" if contains_red_seal else "remove_red_seal")
        if not contains_red_seal:
            texts = [
                "项目名称：测试高标准农田建设项目",
                "项目编号：TEST20260001-S01",
                "投标人：测试建设有限公司",
            ]
        else:
            texts = ["投标文件"]
        return [OCRLine(text, 0.98, []) for text in texts]


class CoverRenderPDFTextEngine(PDFTextEngine):
    """
    【类功能】渲染带红色像素的固定封面图片，隔离测试多策略 OCR。
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """

    def _render_page_image(self, page_number: int, dpi: int) -> np.ndarray:
        """
        【方法功能】生成包含红色区域的最小 RGB 封面图像。
        :param page_number: int+页码
        :param dpi: int+渲染分辨率
        :return: np.ndarray+测试封面 RGB 图像
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        edge = max(40, round(40 * dpi / 150))
        image = np.full((edge, edge, 3), 255, dtype=np.uint8)
        image[edge // 4 : edge // 2, edge // 4 : edge // 2] = [220, 30, 30]
        return image


class CompanyCandidateOCRBackend(OCRBackend):
    """
    【类功能】模拟全文公司名受红章干扰、局部红通道候选完整的 OCR 结果。
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    """

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        """
        【方法功能】按候选图像尺寸和颜色返回缺失、残缺或完整企业名称。
        :param image: np.ndarray+候选 RGB 图像
        :return: list[OCRLine]+模拟 OCR 行
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        is_gray = bool(np.array_equal(image[:, :, 0], image[:, :, 1]))
        if is_gray and image.shape[0] <= 12:
            texts = ["参与单位江苏仪征苏中建设有限公司（盖单位章）"]
            confidence = 0.99
        elif image.shape[0] >= 50:
            texts = [
                "项目名称：测试高标准农田建设项目",
                "项目编号：TEST20260001-S01",
                "参与单汪苏仪征臺中建设有限公司",
            ]
            confidence = 0.91
        else:
            texts = ["投标文件"]
            confidence = 0.98
        return [OCRLine(text, confidence, []) for text in texts]


class RecordingPageEngine:
    """
    【类功能】记录普通页面和封面专属页面读取次数，验证类别隔离。
    :Attributes:
        page_count: int+固定页数
        tender_calls: int+封面专属读取次数
        generic_calls: int+普通读取次数
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化固定页数与调用计数。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        self.page_count = 1
        self.tender_calls = 0
        self.generic_calls = 0

    def get_tender_cover_page(self, page_number: int, dpi: int) -> PageText:
        """
        【方法功能】记录封面专属页面读取并返回固定页面。
        :param page_number: int+页码
        :param dpi: int+分辨率
        :return: PageText+固定封面页面
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        self.tender_calls += 1
        return PageText(page_number, "投标文件", [OCRLine("投标文件", 0.9, [])], "ocr", dpi)

    def get_page(self, page_number: int, dpi: int) -> PageText:
        """
        【方法功能】记录普通页面读取并返回固定页面。
        :param page_number: int+页码
        :param dpi: int+分辨率
        :return: PageText+固定普通页面
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        self.generic_calls += 1
        return PageText(page_number, "普通页面", [OCRLine("普通页面", 0.9, [])], "ocr", dpi)

    def cover_bookmark_pages(self) -> list[int]:
        """
        【方法功能】返回空封面书签列表。
        :return: list[int]+空页码列表
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        return []


class EvaluationReportPageEngine:
    """
    【类功能】模拟多页评标报告并记录不同 OCR 分辨率的页面读取。
    :Attributes:
        page_count: int+固定页面总数
        calls: list[tuple[int, int]]+页面与分辨率调用记录
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化固定评标报告页面和调用记录。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        self.page_count = 4
        self.calls: list[tuple[int, int]] = []

    def get_page(self, page_number: int, dpi: int) -> PageText:
        """
        【方法功能】记录读取参数并返回对应的模拟 OCR 页面。
        :param page_number: int+从1开始的页面序号
        :param dpi: int+OCR 分辨率
        :return: PageText+模拟评标报告页面
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        self.calls.append((page_number, dpi))
        texts = {
            1: ["项目名称：测试农田建设项目"],
            2: ["目录", "投标人排序及推荐的中标候选人名单"],
            3: [
                "投标人排序及推荐的中标候选人名单",
                "投标人名称",
                "甲建设有限公司",
                "推荐的中标候选人",
                "第一名",
                "甲建设有限公司",
            ],
            4: ["评标委员会签名"],
        }[page_number]
        lines = [OCRLine(text, 0.99, []) for text in texts]
        return PageText(page_number, "\n".join(texts), lines, "ocr", dpi)


class TenderCoverStrategyTests(unittest.TestCase):
    """
    【类功能】验证四种封面版式、红章预处理、多策略选择和类别隔离。
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """

    def test_extract_four_existing_cover_formats(self) -> None:
        """
        【方法功能】验证当前四种封面版式均能提取项目、编号和企业名称。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        for name, text, project_name, project_code, lot_code, company_name, _ in COVER_CASES:
            with self.subTest(name=name):
                fields = extract_tender_cover_fields(text, prefer_title=name != "tender_cover_04")
                self.assertEqual(fields.project_name, project_name)
                self.assertEqual(fields.project_code, project_code)
                self.assertEqual(fields.lot_code, lot_code)
                self.assertEqual(fields.company_name, company_name)

    def test_split_tender_cover_project_and_lot_code(self) -> None:
        """
        【方法功能】验证封面标段编号可拆出项目编号，无短横线时保留相同编号。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 12:30:00
        """
        self.assertEqual(
            split_tender_cover_project_and_lot_code("WXHS20231116001-S01"),
            ("WXHS20231116001", "WXHS20231116001-S01"),
        )
        self.assertEqual(
            split_tender_cover_project_and_lot_code("WXHS20221008001"),
            ("WXHS20221008001", "WXHS20221008001"),
        )

    def test_strip_project_template_fragments_with_mixed_brackets(self) -> None:
        """
        【方法功能】验证项目名清理可移除 OCR 产生的混合括号模板字段残留。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:00:00
        """
        expected = "花苑村高标准农田建设项目(二期)施工"
        candidates = (
            "花苑村高标准农田建设项目(二期)【工程名称)施工标段名称)",
            "花苑村高标准农田建设项目(二期)【工程名称)施工",
            "花苑村高标准农田建设项目(二期）（工程名称)施工（标段名称)",
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(strip_bid_file_project_noise(candidate), expected)

    def test_extract_project_name_from_actual_template_ocr_text(self) -> None:
        """
        【方法功能】验证实际封面 OCR 模板残留经字段提取后统一保留项目施工名称。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:00:00
        """
        expected = "花苑村高标准农田建设项目(二期)施工"
        texts = (
            "花苑村高标准农田建设项目(二期）【工程名称)\n施工 标段名称)\n参与文件\n交易编号：HSLS2021011-01\n参与单位：江苏嘉奕建设有限公司盖单位章）",
            "花苑村高标准农田建设项目(二期）【工程名称)\n施工（标段名称）\n参与文件\n交易编号：HSLS2021011-01\n参与单位：无锡润华市政绿化有限公司",
        )
        for text in texts:
            with self.subTest(text=text):
                fields = extract_tender_cover_fields(text, prefer_title=True)
                self.assertEqual(fields.project_name, expected)

    def test_project_name_prefers_explicit_label_over_incomplete_title(self) -> None:
        """
        【方法功能】验证明确项目名称字段优先于标题碎片，并正确拼接断行字段值。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 12:00:00
        """
        cases = (
            (
                """前洲街道张皋庄村高标准农田建设工程（工
程项目名称）前洲街道张皋庄村高标准农田
建设工程（标段名称）工程施工招标
投标文件
项目编号：WXHS20231116001-S01
项目名称：前洲街道张皋庄村高标准农田建设工程
投标人：无锡盛佳亿建设工程有限公司（盖公章)""",
                "前洲街道张皋庄村高标准农田建设工程",
            ),
            (
                """投标人：汇苏祥通设有公司
项目名称：2024年度江苏省无锡市惠山区玉祁街道水稻园
区高标准农田建设改造提升项目 （财政补助）
投标文件内：设际件
投标人：汇苏祥通设有卧公司口（盖公章）""",
                "2024年度江苏省无锡市惠山区玉祁街道水稻园区高标准农田建设改造提升项目(财政补助)",
            ),
            (
                """投标人：宜兴市才利工程有限公司
2024年度江苏省无锡市惠山区玉祁街道水稻园区高标
准农田建设改造提升项目（财政补助）施工招标
投标文件
WXHS20240801001-S01
项目编号：_
项目名称：2024年度江苏省无锡市惠山区玉祁街道水稻园区高标准
农田建设改造提升项旦（财政补助）
投标人：_宜兴市力利工程有公司（盖公章)""",
                "2024年度江苏省无锡市惠山区玉祁街道水稻园区高标准农田建设改造提升项目(财政补助)",
            ),
        )
        for text, expected in cases:
            with self.subTest(expected=expected):
                fields = extract_tender_cover_fields(text, prefer_title=True)
                self.assertEqual(fields.project_name, expected)

    def test_generic_project_title_is_rejected(self) -> None:
        """
        【方法功能】验证通用工程词和混入投标人字段的候选不能作为项目名称。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 12:00:00
        """
        self.assertFalse(is_valid_project_name_candidate("建设工程"))
        self.assertFalse(
            is_valid_project_name_candidate("投标人：某公司：某高标准农田建设项目")
        )

    def test_parse_tender_cover_keeps_unified_record_fields(self) -> None:
        """
        【方法功能】验证四种封面经 parse_tender_cover 后仍输出统一记录字段。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        for name, text, project_name, project_code, lot_code, company_name, method in COVER_CASES:
            with self.subTest(name=name):
                lines = [OCRLine(line, 0.98, []) for line in text.splitlines() if line]
                page = PageText(1, text, lines, method, 300 if method == "ocr" else 0)
                context = ParserContext(
                    pdf_path=Path(f"{name}.pdf"),
                    relative_path=f"tender_cover/{name}.pdf",
                    category="tender_cover",
                    generated_at="2026-07-13T14:20:13+08:00",
                    confidence_threshold=0.80,
                )
                record = parse_tender_cover([page], context)[0]
                self.assertEqual(record.project_name, project_name)
                self.assertEqual(record.project_code, project_code)
                self.assertEqual(record.lot_code, lot_code)
                self.assertEqual(record.company_name, company_name)
                self.assertEqual(record.category, "tender_cover")
                self.assertEqual(record.lot_name, "")

    def test_remove_red_seal_preserves_non_red_pixels(self) -> None:
        """
        【方法功能】验证红章像素被置白且黑色正文像素保持不变。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        image[0, 0] = [220, 30, 30]
        image[0, 1] = [20, 20, 20]
        processed = remove_red_seal(image)
        red_channel = suppress_red_seal_by_red_channel(image)
        self.assertEqual(processed[0, 0].tolist(), [255, 255, 255])
        self.assertEqual(processed[0, 1].tolist(), [20, 20, 20])
        self.assertEqual(red_channel[0, 0].tolist(), [220, 220, 220])
        self.assertEqual(red_channel[0, 1].tolist(), [20, 20, 20])
        self.assertEqual(len(build_cover_image_variants(image)), 2)
        retry_names = [
            name
            for name, _ in build_cover_image_variants(
                image,
                include_top_crop=True,
                retry_only=True,
            )
        ]
        self.assertEqual(len(retry_names), 3)
        self.assertIn("company_crop_red_channel", retry_names)

    def test_multi_strategy_prefers_seal_removed_result_and_caches_it(self) -> None:
        """
        【方法功能】验证封面读取选择去红章候选并使用独立 OCR 缓存。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "cover.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            backend = VariantOCRBackend()
            engine = CoverRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(dpi=150),
                backend,
            )
            first = engine.get_tender_cover_page(1, 150)
            second = engine.get_tender_cover_page(1, 150)
            self.assertIn("测试建设有限公司", first.text)
            self.assertEqual(first.text, second.text)
            self.assertEqual(len(backend.calls), 2)
            self.assertIn("remove_red_seal", backend.calls)
            cache_files = list((root / "cache").rglob("*.json"))
            self.assertEqual(len(cache_files), 1)
            self.assertIn("tender-cover-rapidocr-v1", cache_files[0].name)

    def test_multi_strategy_prefers_complete_company_candidate(self) -> None:
        """
        【方法功能】验证局部红通道中的完整企业名称覆盖全文残缺候选。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 15:45:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "cover.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            engine = CoverRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(dpi=150),
                CompanyCandidateOCRBackend(),
            )
            messages: list[str] = []
            engine.progress_callback = messages.append
            page = engine.get_tender_cover_page(1, 150)
            fields = extract_tender_cover_fields(page.text, prefer_title=True)
            self.assertEqual(fields.company_name, "江苏仪征苏中建设有限公司")
            self.assertTrue(any("基础候选 1/2" in message for message in messages))
            self.assertTrue(any("高分辨率定向候选 1/3" in message for message in messages))
            self.assertTrue(any("候选 5 个" in message for message in messages))

    def test_malformed_plain_company_suffix_triggers_retry(self) -> None:
        """
        【方法功能】验证“有限公司”被错识为普通“公司”时仍触发封面高分辨率重试。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 16:02:00
        """
        malformed = "项目名称：测试高标准农田建设项目\n项目编号：TEST001\n参与单江苏嘉建设有阳公司"
        complete = "项目名称：测试高标准农田建设项目\n项目编号：TEST001\n参与单位：江苏嘉奕建设有限公司"
        self.assertTrue(is_fragmented_cover_text(malformed, 0.91))
        self.assertFalse(is_fragmented_cover_text(complete, 0.91))

    def test_only_tender_cover_uses_multi_strategy_page_reader(self) -> None:
        """
        【方法功能】验证封面类别调用专属读取，其他类别仍调用普通读取。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        engine = RecordingPageEngine()
        config = ProcessingConfig(dpi=150)
        load_pages_for_category(engine, "tender_cover", config)
        load_pages_for_category(engine, "bid_list", config)
        self.assertEqual(engine.tender_calls, 1)
        self.assertEqual(engine.generic_calls, 1)

    def test_evaluation_report_scans_then_reloads_only_target_table(self) -> None:
        """
        【方法功能】验证评标报告低清扫描全文后仅高精度重读真实排序表页。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 16:00:00
        """
        engine = EvaluationReportPageEngine()
        pages, warnings = load_pages_for_category(
            engine,
            "bid_evaluation_report",
            ProcessingConfig(dpi=300, archive_scan_dpi=150),
        )

        self.assertEqual(engine.calls, [(1, 150), (2, 150), (3, 150), (4, 150), (3, 300)])
        self.assertEqual([page.page_number for page in pages], [1, 2, 3, 4])
        self.assertEqual(next(page for page in pages if page.page_number == 3).dpi, 300)
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
