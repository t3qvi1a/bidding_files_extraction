"""PDF 渲染前置、OCR 坐标排序和缓存行为测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from bidding_ocr.models import OCRLine, ProcessingConfig
from bidding_ocr.pdf_engine import OCRBackend, PDFTextEngine, PaddleOCRBackend


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

    def recognize(self, image_path: Path) -> list[OCRLine]:
        """
        【方法功能】验证测试图片存在并返回固定中文 OCR 文字。
        :param image_path: Path+测试页面图片
        :return: list[OCRLine]+固定识别结果
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.calls += 1
        if not image_path.exists():
            raise AssertionError("测试页面图片不存在")
        return [OCRLine("项目名称：缓存测试项目", 0.99, [[0, 0], [10, 0], [10, 10], [0, 10]])]


class StubRenderPDFTextEngine(PDFTextEngine):
    """
    【类功能】用固定图片文件替代 Poppler 渲染，隔离测试 OCR JSON 缓存。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def _render_page(self, page_number: int, dpi: int, output_path: Path) -> None:
        """
        【方法功能】写入最小测试图片占位内容以触发假 OCR 后端。
        :param page_number: int+页码
        :param dpi: int+渲染分辨率
        :param output_path: Path+测试图片路径
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        output_path.write_bytes(f"page={page_number},dpi={dpi}".encode("ascii"))


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
        sorted_lines = PaddleOCRBackend._sort_lines(lines)
        self.assertEqual([line.text for line in sorted_lines], ["左列", "右列", "下一行"])


if __name__ == "__main__":
    unittest.main()
