"""PDF 文本提取、页面渲染、PaddleOCR 与缓存适配器。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from bidding_ocr.models import OCRLine, PageText, ProcessingConfig
from bidding_ocr.utils import is_readable_chinese_text


logging.getLogger("pypdf").setLevel(logging.ERROR)


class OCRBackend(ABC):
    """
    【类功能】定义页面图片文字识别后端的统一接口。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    @abstractmethod
    def recognize(self, image_path: Path) -> list[OCRLine]:
        """
        【方法功能】识别图片并返回带坐标的文字行。
        :param image_path: Path+待识别页面图片
        :return: list[OCRLine]+OCR 文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """

    def prepare(self) -> None:
        """
        【方法功能】在页面渲染前准备 OCR 后端，默认无需额外操作。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """


class PaddleOCRBackend(OCRBackend):
    """
    【类功能】延迟初始化并兼容 PaddleOCR 2.x、3.x 返回格式。
    :Attributes:
        _engine: Any+PaddleOCR 实例，首次识别时创建
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化延迟加载状态，避免普通导入时下载或加载模型。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        """
        【方法功能】创建并缓存本地中文 PaddleOCR 引擎。
        :return: Any+PaddleOCR 引擎实例
        :raises RuntimeError: PaddleOCR 或 PaddlePaddle 未正确安装时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PaddleOCR。请执行：python -m pip install -r requirements.txt"
            ) from exc
        try:
            self._engine = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
        except (TypeError, ValueError):
            self._engine = PaddleOCR(lang="ch", use_angle_cls=True, show_log=False)
        return self._engine

    def recognize(self, image_path: Path) -> list[OCRLine]:
        """
        【方法功能】调用 PaddleOCR 识别页面并统一不同版本的返回格式。
        :param image_path: Path+页面 PNG 图片
        :return: list[OCRLine]+按页面阅读顺序排列的 OCR 文字行
        :raises RuntimeError: OCR 返回格式不可识别或模型执行失败时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        engine = self._get_engine()
        if hasattr(engine, "predict"):
            try:
                result = engine.predict(str(image_path))
                lines = self._parse_modern_result(result)
                if lines:
                    return self._sort_lines(lines)
            except (AttributeError, KeyError, TypeError, ValueError):
                lines = []
        try:
            result = engine.ocr(str(image_path), cls=True)
        except TypeError:
            result = engine.ocr(str(image_path))
        lines = self._parse_legacy_result(result)
        if not lines:
            raise RuntimeError(f"PaddleOCR 未返回可解析文字：{image_path}")
        return self._sort_lines(lines)

    def prepare(self) -> None:
        """
        【方法功能】提前检查并初始化 PaddleOCR，避免缺依赖时先执行 PDF 渲染。
        :return: None
        :raises RuntimeError: PaddleOCR 环境不可用时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self._get_engine()

    @staticmethod
    def _parse_modern_result(result: Any) -> list[OCRLine]:
        """
        【函数功能】解析 PaddleOCR 3.x predict 接口返回数据。
        :param result: Any+PaddleOCR 原始结果
        :return: list[OCRLine]+标准文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        lines: list[OCRLine] = []
        for item in result or []:
            data = item
            if hasattr(item, "json"):
                data = item.json
                if callable(data):
                    data = data()
            if not isinstance(data, dict):
                continue
            data = data.get("res", data)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys") or data.get("dt_polys") or []
            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 0.0
                bbox = boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else (
                    boxes[index] if index < len(boxes) else []
                )
                if str(text).strip():
                    lines.append(OCRLine(str(text).strip(), confidence, bbox))
        return lines

    @staticmethod
    def _parse_legacy_result(result: Any) -> list[OCRLine]:
        """
        【函数功能】解析 PaddleOCR 2.x ocr 接口的嵌套返回数据。
        :param result: Any+PaddleOCR 原始结果
        :return: list[OCRLine]+标准文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        lines: list[OCRLine] = []
        pages = result or []
        if pages and isinstance(pages[0], list) and len(pages[0]) == 2 and isinstance(pages[0][1], tuple):
            pages = [pages]
        for page in pages:
            for item in page or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                bbox, value = item[0], item[1]
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                text = str(value[0]).strip()
                confidence = float(value[1]) if len(value) > 1 else 0.0
                if text:
                    lines.append(OCRLine(text, confidence, bbox or []))
        return lines

    @staticmethod
    def _sort_lines(lines: list[OCRLine]) -> list[OCRLine]:
        """
        【函数功能】依据纵向和横向坐标恢复 OCR 文字阅读顺序。
        :param lines: list[OCRLine]+未排序文字行
        :return: list[OCRLine]+排序后的文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sorted(lines, key=lambda line: (round(line.center_y / 8), line.center_x))


class PDFTextEngine:
    """
    【类功能】统一提供 PDF 页数、书签、文本层、渲染、OCR 和缓存能力。
    :Attributes:
        pdf_path: Path+PDF 文件路径
        cache_dir: Path+OCR JSON 缓存目录
        config: ProcessingConfig+处理配置
        reader: PdfReader+PDF 读取器
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def __init__(
        self,
        pdf_path: Path,
        cache_dir: Path,
        config: ProcessingConfig,
        ocr_backend: OCRBackend | None = None,
    ) -> None:
        """
        【方法功能】打开 PDF 并初始化 OCR 后端和缓存路径。
        :param pdf_path: Path+PDF 文件路径
        :param cache_dir: Path+OCR 缓存根目录
        :param config: ProcessingConfig+处理配置
        :param ocr_backend: OCRBackend|None+可注入的 OCR 后端（默认PaddleOCR）
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.config = config
        self.ocr_backend = ocr_backend or PaddleOCRBackend()
        self.reader = PdfReader(str(pdf_path), strict=False)
        self._file_hash: str | None = None
        self.ocr_pages: set[int] = set()

    @property
    def page_count(self) -> int:
        """
        【方法功能】返回 PDF 页面总数。
        :return: int+页面总数
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return len(self.reader.pages)

    def native_text(self, page_number: int) -> str:
        """
        【方法功能】提取指定页面原生文本层，异常时返回空字符串。
        :param page_number: int+从1开始的页码
        :return: str+原生文本
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        try:
            return self.reader.pages[page_number - 1].extract_text() or ""
        except Exception:
            return ""

    def cover_bookmark_pages(self) -> list[int]:
        """
        【方法功能】从有效书签中查找名称含“封面”的页面。
        :return: list[int]+从1开始的封面页码列表
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        pages: list[int] = []

        def walk(items: Any) -> None:
            """
            【函数功能】递归遍历 PDF 多级书签并收集封面目标页。
            :param items: Any+书签节点或节点列表
            :return: None
            :Author: gexinyan
            :CreateTime: 2026-07-13 11:08:59
            """
            if isinstance(items, list):
                for item in items:
                    walk(item)
                return
            title = str(items.get("/Title", "")) if isinstance(items, dict) else str(getattr(items, "title", ""))
            if "封面" not in title:
                return
            try:
                page = self.reader.get_destination_page_number(items) + 1
            except Exception:
                return
            if 1 <= page <= self.page_count and page not in pages:
                pages.append(page)

        try:
            walk(self.reader.outline)
        except Exception:
            return []
        return sorted(pages)

    def get_page(self, page_number: int, dpi: int | None = None, force_ocr: bool = False) -> PageText:
        """
        【方法功能】优先读取可用文本层，否则渲染并 OCR 指定页面。
        :param page_number: int+从1开始的页码
        :param dpi: int|None+OCR 分辨率（默认使用配置高精度分辨率）
        :param force_ocr: bool+是否强制忽略原生文本层
        :return: PageText+页面文字信息
        :raises ValueError: 页码超出 PDF 范围时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        if page_number < 1 or page_number > self.page_count:
            raise ValueError(f"页码超出范围：{page_number}/{self.page_count}")
        native = self.native_text(page_number)
        if not force_ocr and is_readable_chinese_text(native):
            lines = [OCRLine(line.strip(), 1.0, []) for line in native.splitlines() if line.strip()]
            return PageText(page_number, "\n".join(line.text for line in lines), lines, "text", 0)

        actual_dpi = dpi or self.config.dpi
        cached = self._read_cache(page_number, actual_dpi)
        if cached is not None and not self.config.force_ocr:
            self.ocr_pages.add(page_number)
            return cached

        self.ocr_backend.prepare()
        with tempfile.TemporaryDirectory(prefix="bidding_ocr_") as temp_dir:
            image_path = Path(temp_dir) / f"page-{page_number}.png"
            self._render_page(page_number, actual_dpi, image_path)
            lines = self.ocr_backend.recognize(image_path)
        page = PageText(
            page_number=page_number,
            text="\n".join(line.text for line in lines),
            lines=lines,
            method="ocr",
            dpi=actual_dpi,
        )
        self._write_cache(page)
        self.ocr_pages.add(page_number)
        return page

    def _render_page(self, page_number: int, dpi: int, output_path: Path) -> None:
        """
        【方法功能】使用 Poppler 将指定 PDF 页面渲染为 PNG。
        :param page_number: int+从1开始的页码
        :param dpi: int+渲染分辨率
        :param output_path: Path+目标 PNG 路径
        :return: None
        :raises RuntimeError: Poppler 不可用或渲染失败时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        executable = find_pdftoppm(self.config.poppler_path)
        prefix = output_path.with_suffix("")
        command = [
            str(executable),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(self.pdf_path),
            str(prefix),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output_path.exists():
            message = (completed.stderr or completed.stdout or "未知错误").strip()
            raise RuntimeError(f"PDF 页面渲染失败：{message}")

    def _hash(self) -> str:
        """
        【方法功能】计算并缓存 PDF SHA-256 摘要作为 OCR 缓存键。
        :return: str+十六进制文件摘要
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        if self._file_hash is None:
            digest = hashlib.sha256()
            with self.pdf_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._file_hash = digest.hexdigest()
        return self._file_hash

    def _cache_path(self, page_number: int, dpi: int) -> Path:
        """
        【方法功能】生成指定文件、页码和分辨率对应的 OCR 缓存路径。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :return: Path+JSON 缓存文件路径
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return self.cache_dir / self._hash() / f"page-{page_number}-{dpi}.json"

    def _read_cache(self, page_number: int, dpi: int) -> PageText | None:
        """
        【方法功能】读取已有 OCR JSON 缓存。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :return: PageText|None+缓存页面，不存在或损坏时返回None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        cache_path = self._cache_path(page_number, dpi)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            lines = [OCRLine(**line) for line in data["lines"]]
            return PageText(page_number, data["text"], lines, "ocr", dpi)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, page: PageText) -> None:
        """
        【方法功能】以 UTF-8 JSON 写入 OCR 结果缓存。
        :param page: PageText+待缓存页面
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        cache_path = self._cache_path(page.page_number, page.dpi)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": page.text,
            "lines": [
                {"text": line.text, "confidence": line.confidence, "bbox": line.bbox}
                for line in page.lines
            ],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def find_pdftoppm(configured_path: Path | None = None) -> Path:
    """
    【函数功能】跨平台定位 Poppler 的 pdftoppm 可执行文件。
    :param configured_path: Path|None+用户配置的可执行文件或目录
    :return: Path+可执行文件路径
    :raises RuntimeError: 无法找到可执行文件时触发
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: find_pdftoppm()
    """
    candidates: list[Path] = []
    environment_path = os.getenv("POPPLER_PATH")
    for value in (configured_path, Path(environment_path) if environment_path else None):
        if value is None:
            continue
        candidates.extend([value, value / "pdftoppm.exe", value / "pdftoppm"] if value.is_dir() else [value])

    discovered = shutil.which("pdftoppm")
    if discovered:
        command_path = Path(discovered)
        if command_path.suffix.lower() == ".cmd" and len(command_path.parents) >= 3:
            candidates.append(
                command_path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
            )
        candidates.append(command_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("未找到 Poppler pdftoppm，请安装 Poppler 或设置 POPPLER_PATH。")
