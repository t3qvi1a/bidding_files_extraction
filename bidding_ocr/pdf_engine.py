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

import numpy as np
from PIL import Image

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[assignment,misc]

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

from bidding_ocr.models import OCRLine, PageText, ProcessingConfig
from bidding_ocr.tender_cover_strategy import (
    build_cover_image_variants,
    cover_text_needs_ocr,
    extract_company_name_candidate,
    is_fragmented_cover_text,
    score_company_name_candidate,
    score_cover_ocr_text,
)
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
        project_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(project_root / ".paddlex_cache"))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PaddleOCR。请执行：python -m pip install -r requirements.txt"
            ) from exc
        try:
            self._engine = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (TypeError, ValueError):
            self._engine = PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)
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
        if PdfReader is not None:
            self.reader = PdfReader(str(pdf_path), strict=False)
            self._reader_kind = "pypdf"
        elif pdfium is not None:
            self.reader = pdfium.PdfDocument(str(pdf_path))
            self._reader_kind = "pdfium"
        else:
            raise RuntimeError("缺少 PDF 读取依赖，请安装 pypdf 或 pypdfium2。")
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
        return len(self.reader.pages) if self._reader_kind == "pypdf" else len(self.reader)

    def native_text(self, page_number: int) -> str:
        """
        【方法功能】提取指定页面原生文本层，异常时返回空字符串。
        :param page_number: int+从1开始的页码
        :return: str+原生文本
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        try:
            if self._reader_kind == "pypdf":
                return self.reader.pages[page_number - 1].extract_text() or ""
            page = self.reader[page_number - 1]
            text_page = page.get_textpage()
            return text_page.get_text_range() or ""
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

        if self._reader_kind == "pdfium":
            try:
                for bookmark in self.reader.get_toc():
                    if "封面" not in bookmark.get_title():
                        continue
                    destination = bookmark.get_dest()
                    page = destination.get_index() + 1 if destination else 0
                    if 1 <= page <= self.page_count and page not in pages:
                        pages.append(page)
            except Exception:
                return []
            return sorted(pages)

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

    def get_tender_cover_page(self, page_number: int, dpi: int | None = None) -> PageText:
        """
        【方法功能】为投标封面执行原图、去红章和低质量重试的多策略 OCR。
        :param page_number: int+从1开始的页码
        :param dpi: int|None+基础 OCR 分辨率（默认使用配置高精度分辨率）
        :return: PageText+质量评分最高的封面页面文字
        :raises ValueError: 页码超出范围时触发
        :raises RuntimeError: 所有封面 OCR 候选均失败时触发
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        if page_number < 1 or page_number > self.page_count:
            raise ValueError(f"页码超出范围：{page_number}/{self.page_count}")
        native = self.native_text(page_number)
        if not cover_text_needs_ocr(native):
            lines = [OCRLine(line.strip(), 1.0, []) for line in native.splitlines() if line.strip()]
            return PageText(page_number, "\n".join(line.text for line in lines), lines, "text", 0)

        actual_dpi = dpi or self.config.dpi
        cache_profile = "tender-cover-v4"
        cached = self._read_cache(page_number, actual_dpi, cache_profile)
        if cached is not None and not self.config.force_ocr:
            self.ocr_pages.add(page_number)
            return cached

        self.ocr_backend.prepare()
        candidates: list[tuple[list[OCRLine], float]] = []
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="bidding_cover_ocr_") as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / f"page-{page_number}-{actual_dpi}.png"
            self._render_page(page_number, actual_dpi, image_path)
            image = np.asarray(Image.open(image_path).convert("RGB"))
            candidates.extend(
                self._recognize_cover_variants(
                    image,
                    temp_path,
                    f"page-{page_number}-{actual_dpi}",
                    include_top_crop=False,
                    errors=errors,
                )
            )
            best_lines, best_score = max(candidates, key=lambda item: item[1], default=([], -1.0))
            best_text = "\n".join(line.text for line in best_lines)
            best_average = self._average_line_confidence(best_lines)
            if is_fragmented_cover_text(best_text, best_average):
                retry_dpi = max(actual_dpi + 50, round(actual_dpi * 1.25))
                retry_path = temp_path / f"page-{page_number}-{retry_dpi}.png"
                self._render_page(page_number, retry_dpi, retry_path)
                retry_image = np.asarray(Image.open(retry_path).convert("RGB"))
                candidates.extend(
                    self._recognize_cover_variants(
                        retry_image,
                        temp_path,
                        f"page-{page_number}-{retry_dpi}",
                        include_top_crop=True,
                        errors=errors,
                    )
                )
                best_lines, best_score = max(candidates, key=lambda item: item[1], default=([], -1.0))

        if not best_lines:
            error_message = "；".join(errors) if errors else "OCR 未返回文字"
            raise RuntimeError(f"投标封面多策略 OCR 失败：{error_message}")
        best_lines = self._merge_cover_company_candidate(best_lines, candidates)
        page = PageText(
            page_number=page_number,
            text="\n".join(line.text for line in best_lines),
            lines=best_lines,
            method="ocr",
            dpi=actual_dpi,
        )
        self._write_cache(page, cache_profile)
        self.ocr_pages.add(page_number)
        return page

    def _recognize_cover_variants(
        self,
        image: Any,
        temp_dir: Path,
        prefix: str,
        include_top_crop: bool,
        errors: list[str],
    ) -> list[tuple[list[OCRLine], float]]:
        """
        【方法功能】保存并识别一组封面图像候选，返回文字行和封面质量分。
        :param image: Any+RGB 图像数组
        :param temp_dir: Path+候选图片临时目录
        :param prefix: str+候选图片稳定前缀
        :param include_top_crop: bool+是否加入顶部裁剪候选
        :param errors: list[str]+用于收集候选失败信息的列表
        :return: list[tuple[list[OCRLine], float]]+成功候选及质量分
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        candidates: list[tuple[list[OCRLine], float]] = []
        for variant_name, variant_image in build_cover_image_variants(image, include_top_crop):
            variant_path = temp_dir / f"{prefix}-{variant_name}.png"
            Image.fromarray(np.asarray(variant_image, dtype=np.uint8)).save(variant_path)
            try:
                lines = self.ocr_backend.recognize(variant_path)
            except Exception as exc:
                errors.append(f"{variant_name}:{type(exc).__name__}:{exc}")
                continue
            text = "\n".join(line.text for line in lines)
            average_score = self._average_line_confidence(lines)
            candidates.append((lines, score_cover_ocr_text(text, average_score)))
        return candidates

    @staticmethod
    def _merge_cover_company_candidate(
        best_lines: list[OCRLine],
        candidates: list[tuple[list[OCRLine], float]],
    ) -> list[OCRLine]:
        """
        【函数功能】从局部裁剪候选补充最佳全文候选缺失的投标人企业名称。
        :param best_lines: list[OCRLine]+质量分最高的全文文字行
        :param candidates: list[tuple[list[OCRLine], float]]+全部成功 OCR 候选
        :return: list[OCRLine]+必要时追加企业字段的文字行
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        company_candidates: dict[str, tuple[float, float, int]] = {}
        for lines, _ in candidates:
            for line in lines:
                company_name = extract_company_name_candidate(line.text)
                if not company_name:
                    continue
                current_score, current_confidence, count = company_candidates.get(
                    company_name,
                    (0.0, 0.0, 0),
                )
                candidate_score = score_company_name_candidate(company_name, line.confidence)
                company_candidates[company_name] = (
                    max(current_score, candidate_score),
                    max(current_confidence, line.confidence),
                    count + 1,
                )
        if not company_candidates:
            return best_lines
        company_name, (_, confidence, _) = max(
            company_candidates.items(),
            key=lambda item: (item[1][0], item[1][2], item[1][1]),
        )
        current_company = extract_company_name_candidate("\n".join(line.text for line in best_lines))
        if company_name == current_company:
            return best_lines
        return [OCRLine(f"投标人：{company_name}", confidence, []), *best_lines]

    @staticmethod
    def _average_line_confidence(lines: list[OCRLine]) -> float:
        """
        【函数功能】计算非空 OCR 文字行的平均置信度。
        :param lines: list[OCRLine]+OCR 文字行
        :return: float+平均置信度，无文字时返回0
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        scores = [line.confidence for line in lines if line.text.strip()]
        return sum(scores) / len(scores) if scores else 0.0

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

    def _cache_path(self, page_number: int, dpi: int, profile: str = "default") -> Path:
        """
        【方法功能】生成指定文件、页码和分辨率对应的 OCR 缓存路径。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :param profile: str+OCR 策略缓存标识（默认default）
        :return: Path+JSON 缓存文件路径
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        suffix = "" if profile == "default" else f"-{profile}"
        return self.cache_dir / self._hash() / f"page-{page_number}-{dpi}{suffix}.json"

    def _read_cache(self, page_number: int, dpi: int, profile: str = "default") -> PageText | None:
        """
        【方法功能】读取已有 OCR JSON 缓存。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :param profile: str+OCR 策略缓存标识（默认default）
        :return: PageText|None+缓存页面，不存在或损坏时返回None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        cache_path = self._cache_path(page_number, dpi, profile)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            lines = [OCRLine(**line) for line in data["lines"]]
            return PageText(page_number, data["text"], lines, "ocr", dpi)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, page: PageText, profile: str = "default") -> None:
        """
        【方法功能】以 UTF-8 JSON 写入 OCR 结果缓存。
        :param page: PageText+待缓存页面
        :param profile: str+OCR 策略缓存标识（默认default）
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        cache_path = self._cache_path(page.page_number, page.dpi, profile)
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
