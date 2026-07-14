"""PDF 渲染前置、OCR 坐标排序和缓存行为测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
