"""PDF 文本提取、PDFium 渲染、RapidOCR 与缓存适配器。"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

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


ProgressCallback = Callable[[str], None]
RAPIDOCR_CACHE_PROFILE = "rapidocr-v1"
TENDER_COVER_RAPIDOCR_CACHE_PROFILE = "tender-cover-rapidocr-v1"


class OCRBackend(ABC):
    """
    【类功能】定义页面图片文字识别后端的统一接口。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    @abstractmethod
    def recognize(self, image: Any) -> list[OCRLine]:
        """
        【方法功能】识别 RGB 图像并返回带坐标的文字行。
        :param image: Any+待识别 RGB 图像数组
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


def sort_ocr_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """
    【函数功能】依据纵向和横向坐标恢复 OCR 文字阅读顺序。
    :param lines: list[OCRLine]+未排序文字行
    :return: list[OCRLine]+排序后的文字行
    :Author: gexinyan
    :CreateTime: 2026-07-14 10:30:00
    """
    return sorted(lines, key=lambda line: (round(line.center_y / 8), line.center_x))


class RapidOCRBackend(OCRBackend):
    """
    【类功能】延迟初始化 RapidOCR，并将识别结果转换为统一 OCR 行对象。
    :Attributes:
        _engine: Any+RapidOCR 实例，首次识别时创建
    :Author: gexinyan
    :CreateTime: 2026-07-14 10:30:00
    """

    def __init__(self) -> None:
        """
        【方法功能】初始化 RapidOCR 的延迟加载状态。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        """
        【方法功能】创建并缓存 RapidOCR ONNX Runtime 引擎。
        :return: Any+RapidOCR 引擎实例
        :raises RuntimeError: rapidocr-onnxruntime 未正确安装时触发
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "缺少 RapidOCR。请执行：python -m pip install -r requirements.txt"
            ) from exc
        self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: Any) -> list[OCRLine]:
        """
        【方法功能】调用 RapidOCR 识别 RGB 图像并转换为统一文字行。
        :param image: Any+页面 RGB 图像数组
        :return: list[OCRLine]+按页面阅读顺序排列的 OCR 文字行
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        result = self._get_engine()(np.asarray(image))
        raw_items = result[0] if isinstance(result, tuple) else result
        lines: list[OCRLine] = []
        for item in raw_items or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box, text, score = item[0], str(item[1]).strip(), item[2]
            if not text:
                continue
            bbox = np.asarray(box).tolist() if box is not None else []
            lines.append(OCRLine(text, float(score), bbox))
        return sort_ocr_lines(lines)

    def prepare(self) -> None:
        """
        【方法功能】提前检查并初始化 RapidOCR，避免缺依赖时先执行 PDF 渲染。
        :return: None
        :raises RuntimeError: RapidOCR 环境不可用时触发
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        self._get_engine()

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
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        【方法功能】打开 PDF 并初始化 OCR 后端和缓存路径。
        :param pdf_path: Path+PDF 文件路径
        :param cache_dir: Path+OCR 缓存根目录
        :param config: ProcessingConfig+处理配置
        :param ocr_backend: OCRBackend|None+可注入的 OCR 后端（默认RapidOCR）
        :param progress_callback: ProgressCallback|None+可选 OCR 进度消息回调（默认不输出）
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.config = config
        self.ocr_backend = ocr_backend or RapidOCRBackend()
        self.progress_callback = progress_callback
        if PdfReader is not None:
            self.reader = PdfReader(str(pdf_path), strict=False)
            self._reader_kind = "pypdf"
        elif pdfium is not None:
            self.reader = pdfium.PdfDocument(str(pdf_path))
            self._reader_kind = "pdfium"
        else:
            raise RuntimeError("缺少 PDF 读取依赖，请安装 pypdf 或 pypdfium2。")
        self._file_hash: str | None = None
        self._render_document: Any | None = None
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
                return self.reader.pages[page_number - 1].extract_text(extraction_mode="layout") or ""
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
            self._emit_progress(f"页面 OCR：第{page_number}页使用原生文本层，跳过 OCR。")
            return PageText(page_number, "\n".join(line.text for line in lines), lines, "text", 0)

        actual_dpi = dpi or self.config.dpi
        cached = self._read_cache(page_number, actual_dpi, RAPIDOCR_CACHE_PROFILE)
        if cached is not None and not self.config.force_ocr:
            self._emit_progress(f"页面 OCR：第{page_number}页命中缓存，跳过识别。")
            self.ocr_pages.add(page_number)
            return cached

        self._emit_progress(f"页面 OCR：第{page_number}页开始，DPI {actual_dpi}。")
        started_at = time.perf_counter()
        try:
            self.ocr_backend.prepare()
            image = self._render_page_image(page_number, actual_dpi)
            lines = self.ocr_backend.recognize(image)
        except Exception:
            self._emit_progress(
                f"页面 OCR：第{page_number}页失败，总耗时 {time.perf_counter() - started_at:.1f} 秒。"
            )
            raise
        page = PageText(
            page_number=page_number,
            text="\n".join(line.text for line in lines),
            lines=lines,
            method="ocr",
            dpi=actual_dpi,
        )
        self._write_cache(page, RAPIDOCR_CACHE_PROFILE)
        self.ocr_pages.add(page_number)
        self._emit_progress(
            f"页面 OCR：第{page_number}页完成，DPI {actual_dpi}，识别 {len(lines)} 行，"
            f"总耗时 {time.perf_counter() - started_at:.1f} 秒。"
        )
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
            self._emit_progress(f"封面 OCR：第{page_number}页使用原生文本层，跳过 OCR。")
            return PageText(page_number, "\n".join(line.text for line in lines), lines, "text", 0)

        actual_dpi = dpi or self.config.dpi
        cache_profile = TENDER_COVER_RAPIDOCR_CACHE_PROFILE
        cached = self._read_cache(page_number, actual_dpi, cache_profile)
        if cached is not None and not self.config.force_ocr:
            self._emit_progress(f"封面 OCR：第{page_number}页命中缓存，跳过候选识别。")
            self.ocr_pages.add(page_number)
            return cached

        started_at = time.perf_counter()
        self.ocr_backend.prepare()
        candidates: list[tuple[list[OCRLine], float]] = []
        errors: list[str] = []
        image = self._render_page_image(page_number, actual_dpi)
        candidates.extend(
            self._recognize_cover_variants(
                image,
                include_top_crop=False,
                retry_only=False,
                stage_name="基础候选",
                errors=errors,
            )
        )
        best_lines, best_score = max(candidates, key=lambda item: item[1], default=([], -1.0))
        best_text = "\n".join(line.text for line in best_lines)
        best_average = self._average_line_confidence(best_lines)
        if is_fragmented_cover_text(best_text, best_average):
            retry_dpi = max(actual_dpi + 50, round(actual_dpi * 1.25))
            self._emit_progress(
                f"封面 OCR：基础候选质量不足，启动 {retry_dpi} DPI 定向重试（3 个候选）。"
            )
            retry_image = self._render_page_image(page_number, retry_dpi)
            candidates.extend(
                self._recognize_cover_variants(
                    retry_image,
                    include_top_crop=True,
                    retry_only=True,
                    stage_name="高分辨率定向候选",
                    errors=errors,
                )
            )
            best_lines, best_score = max(candidates, key=lambda item: item[1], default=([], -1.0))

        if not best_lines:
            error_message = "；".join(errors) if errors else "OCR 未返回文字"
            self._emit_progress(
                f"封面 OCR：第{page_number}页失败，总耗时 {time.perf_counter() - started_at:.1f} 秒。"
            )
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
        self._emit_progress(
            f"封面 OCR：第{page_number}页完成，候选 {len(candidates)} 个，总耗时 "
            f"{time.perf_counter() - started_at:.1f} 秒。"
        )
        return page

    def _recognize_cover_variants(
        self,
        image: Any,
        include_top_crop: bool,
        retry_only: bool,
        stage_name: str,
        errors: list[str],
    ) -> list[tuple[list[OCRLine], float]]:
        """
        【方法功能】在内存中识别一组封面图像候选，返回文字行和封面质量分。
        :param image: Any+RGB 图像数组
        :param include_top_crop: bool+是否加入顶部裁剪候选
        :param retry_only: bool+是否只保留高分辨率定向裁剪候选
        :param stage_name: str+当前候选阶段名称
        :param errors: list[str]+用于收集候选失败信息的列表
        :return: list[tuple[list[OCRLine], float]]+成功候选及质量分
        :Author: gexinyan
        :CreateTime: 2026-07-13 14:20:13
        """
        candidates: list[tuple[list[OCRLine], float]] = []
        variants = build_cover_image_variants(image, include_top_crop, retry_only)
        for variant_index, (variant_name, variant_image) in enumerate(variants, start=1):
            self._emit_progress(
                f"封面 OCR：{stage_name} {variant_index}/{len(variants)}（{variant_name}）开始。"
            )
            started_at = time.perf_counter()
            try:
                lines = self.ocr_backend.recognize(variant_image)
            except Exception as exc:
                errors.append(f"{variant_name}:{type(exc).__name__}:{exc}")
                self._emit_progress(
                    f"封面 OCR：{stage_name} {variant_index}/{len(variants)}（{variant_name}）失败，"
                    f"耗时 {time.perf_counter() - started_at:.1f} 秒。"
                )
                continue
            text = "\n".join(line.text for line in lines)
            average_score = self._average_line_confidence(lines)
            candidates.append((lines, score_cover_ocr_text(text, average_score)))
            self._emit_progress(
                f"封面 OCR：{stage_name} {variant_index}/{len(variants)}（{variant_name}）完成，"
                f"耗时 {time.perf_counter() - started_at:.1f} 秒，识别 {len(lines)} 行。"
            )
        return candidates

    def _emit_progress(self, message: str) -> None:
        """
        【方法功能】向调用方发送一条 OCR 阶段进度消息。
        :param message: str+待发送的中文进度消息
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 16:45:00
        Example: self._emit_progress("封面 OCR：基础候选 1/2 开始。")
        """
        if self.progress_callback is not None:
            self.progress_callback(message)

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

    def _render_page_image(self, page_number: int, dpi: int) -> Any:
        """
        【方法功能】使用 PDFium 将指定 PDF 页面渲染为内存 RGB 图像。
        :param page_number: int+从1开始的页码
        :param dpi: int+渲染分辨率
        :return: Any+RGB uint8 图像数组
        :raises RuntimeError: PDFium 不可用或渲染失败时触发
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        if pdfium is None:
            raise RuntimeError("缺少 pypdfium2，请执行：python -m pip install -r requirements.txt")
        if self._render_document is None:
            self._render_document = pdfium.PdfDocument(str(self.pdf_path))
        page = self._render_document[page_number - 1]
        bitmap = None
        try:
            bitmap = page.render(scale=max(dpi, 1) / 72.0)
            return np.asarray(bitmap.to_pil().convert("RGB")).copy()
        finally:
            if bitmap is not None:
                bitmap.close()
            page.close()

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
