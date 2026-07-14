"""PDF 渲染前置、OCR 坐标排序和缓存行为测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pypdf import PdfWriter

from bidding_ocr.models import OCRLine, ProcessingConfig
from bidding_ocr.pdf_engine import OCRBackend, PDFTextEngine, RapidOCRBackend, sort_ocr_lines


class FakeOCRBackend(OCRBackend):
    """
    【类功能】为 PDF 引擎测试提供不依赖模型的固定 OCR 结果。
    :Attributes:
        calls: int+识别调用次数
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化识别调用计数。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.calls = 0

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        """
        【方法功能】验证测试图像数组并返回固定中文 OCR 文字。
        :param image: np.ndarray+测试页面 RGB 图像
        :return: list[OCRLine]+固定识别结果
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        self.calls += 1
        if not isinstance(image, np.ndarray):
            raise AssertionError("测试页面图像未使用内存数组传递")
        return [OCRLine("项目名称：缓存测试项目", 0.99, [[0, 0], [10, 0], [10, 10], [0, 10]])]


class EmptyOCRBackend(OCRBackend):
    """
    【类功能】为 PDF 引擎测试模拟无文字的空白 OCR 页面。
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    """

    def recognize(self, image: np.ndarray) -> list[OCRLine]:
        """
        【函数功能】返回空 OCR 结果以模拟无内容页面。
        :param image: np.ndarray+测试页面 RGB 图像
        :return: list[OCRLine]+空文字行列表
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        Example: recognize(np.zeros((10, 10, 3)))
        """
        return []


class StubRenderPDFTextEngine(PDFTextEngine):
    """
    【类功能】用固定内存图像替代 PDFium 渲染，隔离测试 OCR JSON 缓存。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def _render_page_image(self, page_number: int, dpi: int) -> np.ndarray:
        """
        【方法功能】返回最小测试 RGB 图像以触发假 OCR 后端。
        :param page_number: int+页码
        :param dpi: int+渲染分辨率
        :return: np.ndarray+测试 RGB 图像
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        return np.full((10, 10, 3), 255, dtype=np.uint8)


class PDFEngineTests(unittest.TestCase):
    """
    【类功能】验证无文本 PDF 的 OCR 回退、缓存命中和书签容错。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def test_ocr_cache_avoids_second_recognition(self) -> None:
        """
        【方法功能】验证相同文件、页码和分辨率第二次读取直接命中缓存。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            backend = FakeOCRBackend()
            first_engine = StubRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(),
                backend,
            )
            first = first_engine.get_page(1, 150)
            second_engine = StubRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(),
                backend,
            )
            second = second_engine.get_page(1, 150)
            self.assertEqual(first.text, second.text)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(first_engine.cover_bookmark_pages(), [])
            cache_files = list((root / "cache").rglob("*.json"))
            self.assertEqual(len(cache_files), 1)
            self.assertIn("rapidocr-v1", cache_files[0].name)

    def test_native_text_uses_layout_extraction_mode(self) -> None:
        """
        【函数功能】验证原生文本提取请求 pypdf 保留页面版式。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:00:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            engine = PDFTextEngine(pdf_path, root / "cache", ProcessingConfig())
            page = MagicMock()
            page.extract_text.return_value = "项目名称：版式测试项目"
            engine.reader = MagicMock(pages=[page])
            engine._reader_kind = "pypdf"

            self.assertEqual(engine.native_text(1), "项目名称：版式测试项目")
            page.extract_text.assert_called_once_with(extraction_mode="layout")

    def test_unreadable_native_text_falls_back_to_ocr(self) -> None:
        """
        【函数功能】验证乱码原生文本层会调用 OCR 并返回 OCR 页面。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:00:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            backend = FakeOCRBackend()
            engine = StubRenderPDFTextEngine(pdf_path, root / "cache", ProcessingConfig(), backend)

            with patch.object(engine, "native_text", return_value="\ufffd\ufffd\ufffd\u65e0\u6548\u6587\u672c"):
                page = engine.get_page(1, 150)

            self.assertEqual(page.method, "ocr")
            self.assertEqual(backend.calls, 1)

    def test_empty_ocr_page_returns_empty_page_without_error(self) -> None:
        """
        【方法功能】验证空白页 OCR 无文字时返回空页面而非抛出异常。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 14:30:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            engine = StubRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(),
                EmptyOCRBackend(),
            )

            page = engine.get_page(1, 150)

            self.assertEqual(page.method, "ocr")
            self.assertEqual(page.text, "")
            self.assertEqual(page.lines, [])

    def test_generic_page_ocr_emits_page_timing_progress(self) -> None:
        """
        【方法功能】验证通用页面 OCR、缓存命中与原生文本路径均输出明确进度消息。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 11:30:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with pdf_path.open("wb") as stream:
                writer.write(stream)
            backend = FakeOCRBackend()
            messages: list[str] = []
            engine = StubRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(),
                backend,
                messages.append,
            )
            engine.get_page(1, 150)
            self.assertTrue(any("第1页开始，DPI 150" in message for message in messages))
            self.assertTrue(any("第1页完成，DPI 150，识别 1 行" in message for message in messages))

            cached_messages: list[str] = []
            cached_engine = StubRenderPDFTextEngine(
                pdf_path,
                root / "cache",
                ProcessingConfig(),
                backend,
                cached_messages.append,
            )
            cached_engine.get_page(1, 150)
            self.assertTrue(any("命中缓存，跳过识别" in message for message in cached_messages))

            native_messages: list[str] = []
            with patch.object(
                engine,
                "native_text",
                return_value="项目名称：这是一个用于验证原生文本处理流程的高标准农田建设项目",
            ):
                engine.progress_callback = native_messages.append
                engine.get_page(1, 150)
            self.assertTrue(any("使用原生文本层，跳过 OCR" in message for message in native_messages))

    def test_ocr_lines_are_sorted_by_rows_and_columns(self) -> None:
        """
        【方法功能】验证横向表格 OCR 文本按先行后列顺序恢复。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        lines = [
            OCRLine("右列", 0.9, [[100, 10], [120, 10], [120, 20], [100, 20]]),
            OCRLine("下一行", 0.9, [[0, 30], [20, 30], [20, 40], [0, 40]]),
            OCRLine("左列", 0.9, [[0, 10], [20, 10], [20, 20], [0, 20]]),
        ]
        sorted_lines = sort_ocr_lines(lines)
        self.assertEqual([line.text for line in sorted_lines], ["左列", "右列", "下一行"])

    def test_rapidocr_backend_converts_result_to_ocr_lines(self) -> None:
        """
        【方法功能】验证 RapidOCR 原始结果保留坐标、置信度并按阅读顺序转换。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        backend = RapidOCRBackend()
        backend._engine = lambda image: (
            [
                [[[100, 10], [120, 10], [120, 20], [100, 20]], "右列", 0.9],
                [[[0, 10], [20, 10], [20, 20], [0, 20]], "左列", 0.8],
            ],
            0.01,
        )
        lines = backend.recognize(np.zeros((20, 140, 3), dtype=np.uint8))
        self.assertEqual([line.text for line in lines], ["左列", "右列"])
        self.assertEqual(lines[0].confidence, 0.8)
        self.assertEqual(lines[0].bbox[0], [0, 10])


if __name__ == "__main__":
    unittest.main()
