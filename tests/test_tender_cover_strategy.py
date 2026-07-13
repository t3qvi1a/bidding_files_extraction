"""投标文件封面四种现有版式、去红章和专属页面读取策略测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfWriter

from bidding_ocr.models import OCRLine, PageText, ProcessingConfig
from bidding_ocr.parsers import ParserContext, parse_tender_cover
from bidding_ocr.pdf_engine import OCRBackend, PDFTextEngine
from bidding_ocr.pipeline import load_pages_for_category
from bidding_ocr.tender_cover_strategy import (
    build_cover_image_variants,
    extract_tender_cover_fields,
    is_fragmented_cover_text,
    remove_red_seal,
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
        "无锡市银河建筑安装有限公司",
        "text",
    ),
)


class VariantOCRBackend(OCRBackend):
    """
    【类功能】根据候选图片文件名返回不同质量文本，验证去红章候选选择。
    :Attributes:
        calls: list[str]+已识别候选图片名称
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

    def recognize(self, image_path: Path) -> list[OCRLine]:
        """
        【方法功能】原图返回低质量文本，去红章图返回完整封面字段。
        :param image_path: Path+候选图片路径
        :return: list[OCRLine]+固定 OCR 文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        self.calls.append(image_path.name)
        if "remove_red_seal" in image_path.name:
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

    def _render_page(self, page_number: int, dpi: int, output_path: Path) -> None:
        """
        【方法功能】生成包含红色区域的最小 RGB PNG 封面。
        :param page_number: int+页码
        :param dpi: int+渲染分辨率
        :param output_path: Path+输出图片路径
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[10:20, 10:20] = [220, 30, 30]
        Image.fromarray(image).save(output_path)


class CompanyCandidateOCRBackend(OCRBackend):
    """
    【类功能】模拟全文公司名受红章干扰、局部红通道候选完整的 OCR 结果。
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    """

    def recognize(self, image_path: Path) -> list[OCRLine]:
        """
        【方法功能】按候选名称返回缺失、残缺或完整企业名称。
        :param image_path: Path+候选图片路径
        :return: list[OCRLine]+模拟 OCR 行
        :Author: gexinyan
        :CreateTime: 2026-07-13 15:45:00
        """
        if "company_crop_red_channel" in image_path.name:
            texts = ["参与单位江苏仪征苏中建设有限公司（盖单位章）"]
            confidence = 0.99
        elif "-200-" in image_path.name:
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
        for name, text, project_name, project_code, company_name, _ in COVER_CASES:
            with self.subTest(name=name):
                fields = extract_tender_cover_fields(text, prefer_title=name != "tender_cover_04")
                self.assertEqual(fields.project_name, project_name)
                self.assertEqual(fields.project_code, project_code)
                self.assertEqual(fields.company_name, company_name)

    def test_parse_tender_cover_keeps_unified_record_fields(self) -> None:
        """
        【方法功能】验证四种封面经 parse_tender_cover 后仍输出统一记录字段。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        for name, text, project_name, project_code, company_name, method in COVER_CASES:
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
            self.assertTrue(any("remove_red_seal" in name for name in backend.calls))

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


if __name__ == "__main__":
    unittest.main()
