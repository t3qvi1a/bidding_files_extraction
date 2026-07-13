"""招投标 PDF 目录处理、分类输出、去重合并与运行摘要。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from bidding_ocr.models import (
    CATEGORIES,
    CSV_FIELDS,
    ExtractionRecord,
    FileProcessSummary,
    PageText,
    ParsedDocument,
    ProcessingConfig,
    ProcessSummary,
)
from bidding_ocr.parsers import ParserContext, parse_document
from bidding_ocr.pdf_engine import PDFTextEngine
from bidding_ocr.utils import classify_pdf, compact_for_match, normalize_text


ARCHIVE_KEYWORDS = (
    "推荐的中标候选人",
    "成交通知书",
    "中标通知书",
    "招投标情况书面报告",
    "招投标基本情况报告",
    "原件确认签收表",
    "项目负责人答辩评分表",
    "按时送达投标文件的投标人名单",
)

SOURCE_PRIORITY = {
    "award_notice": 6,
    "bid_announcement": 6,
    "bid_candidates": 5,
    "bid_evaluation_report": 5,
    "archive_info": 4,
    "tender_cover": 3,
    "bid_list": 2,
    "unknown": 1,
}

ProgressCallback = Callable[[str], None]


def current_timestamp() -> str:
    """
    【函数功能】生成带本地时区的解析结果日期时间。
    :return: str+ISO 风格日期时间字符串
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: current_timestamp()
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def process_pdf_tree(
    input_dir: Path | str,
    output_dir: Path | str,
    config: ProcessingConfig | None = None,
    category_filter: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ProcessSummary:
    """
    【函数功能】统一处理输入目录中的全部 PDF，输出分类、合并及复核结果。
    :param input_dir: Path|str+PDF 输入目录
    :param output_dir: Path|str+CSV 和运行摘要输出目录
    :param config: ProcessingConfig|None+可选处理配置
    :param category_filter: str|None+可选标准类别，仅处理该类别 PDF（默认处理全部）
    :param progress_callback: ProgressCallback|None+可选运行进度消息回调（默认不输出）
    :return: ProcessSummary+本次运行统计
    :raises FileNotFoundError: 输入目录不存在时触发
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: process_pdf_tree("pdf_files", "results")
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"PDF 输入目录不存在：{input_path}")
    if category_filter is not None and category_filter not in CATEGORIES:
        raise ValueError(f"不支持的 PDF 类别：{category_filter}")
    actual_config = config or ProcessingConfig()
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / ".ocr_cache"
    started_at = current_timestamp()
    all_records: list[ExtractionRecord] = []
    file_summaries: list[FileProcessSummary] = []
    all_pdf_paths = sorted(input_path.rglob("*.pdf"), key=lambda path: str(path).lower())
    planned_categories = {
        pdf_path: classify_pdf(pdf_path, input_path, page_count=0)
        for pdf_path in all_pdf_paths
    }
    pdf_paths = [
        pdf_path
        for pdf_path in all_pdf_paths
        if category_filter is None or planned_categories[pdf_path] == category_filter
    ]
    category_totals: dict[str, int] = defaultdict(int)
    for pdf_path in pdf_paths:
        category_totals[planned_categories[pdf_path]] += 1
    category_positions: dict[str, int] = defaultdict(int)
    total_files = len(pdf_paths)
    _emit_progress(
        progress_callback,
        f"发现 {total_files} 个待处理 PDF"
        + (f"，类别筛选：{category_filter}" if category_filter else "，开始解析。"),
    )

    for file_index, pdf_path in enumerate(pdf_paths, start=1):
        relative_path = str(pdf_path.relative_to(input_path))
        planned_category = planned_categories[pdf_path]
        category_positions[planned_category] += 1
        category_index = category_positions[planned_category]
        category_total = category_totals[planned_category]
        _emit_progress(
            progress_callback,
            (
                f"{_format_progress_bar(file_index - 1, total_files)} "
                f"处理中 {file_index}/{total_files} | 类别 {planned_category} "
                f"{category_index}/{category_total} | 文件：{relative_path}"
            ),
        )
        try:
            document, warnings = process_single_pdf(
                pdf_path,
                input_path,
                cache_dir,
                actual_config,
            )
            all_records.extend(document.records)
            review_count = sum(record.review_status != "通过" for record in document.records)
            status = "待复核" if review_count or warnings else "成功"
            file_summary = FileProcessSummary(
                path=relative_path,
                category=document.category,
                pages=document.page_count,
                ocr_pages=sorted(document.ocr_pages),
                records=len(document.records),
                review_records=review_count,
                status=status,
                error="；".join(warnings),
            )
            file_summaries.append(file_summary)
            _emit_progress(
                progress_callback,
                (
                    f"{_format_progress_bar(file_index, total_files)} "
                    f"完成 {file_index}/{total_files} | 类别 {file_summary.category} "
                    f"| 页数 {file_summary.pages} | OCR 页 {file_summary.ocr_pages or '无'} "
                    f"| 记录 {file_summary.records} | 状态 {file_summary.status}"
                ),
            )
        except Exception as exc:
            failed_record = ExtractionRecord(
                category="unknown",
                source_path=relative_path,
                evidence=f"文件解析失败：{exc}",
                review_status="待复核",
                generated_at=current_timestamp(),
            )
            all_records.append(failed_record)
            file_summary = FileProcessSummary(
                path=relative_path,
                category="unknown",
                pages=0,
                ocr_pages=[],
                records=1,
                review_records=1,
                status="失败",
                error=str(exc),
            )
            file_summaries.append(file_summary)
            _emit_progress(
                progress_callback,
                (
                    f"{_format_progress_bar(file_index, total_files)} "
                    f"完成 {file_index}/{total_files} | 类别 {planned_category} "
                    f"| 文件解析失败：{relative_path} | 原因：{exc}"
                ),
            )

    _emit_progress(progress_callback, "PDF 解析完成，正在生成分类 CSV、最终汇总和复核清单。")
    _write_category_csv_files(all_records, output_path)
    final_records = merge_and_deduplicate(all_records)
    write_records_csv(output_path / "final.csv", final_records)
    review_records = [record for record in final_records if record.review_status != "通过"]
    write_records_csv(output_path / "review_queue.csv", review_records)

    summary = ProcessSummary(
        started_at=started_at,
        finished_at=current_timestamp(),
        input_dir=str(input_path),
        output_dir=str(output_path),
        files=file_summaries,
    )
    (output_path / "run_summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _emit_progress(progress_callback, f"结果已写入：{output_path}")
    return summary


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    """
    【函数功能】在存在回调时发送一条可即时显示的运行进度消息。
    :param callback: ProgressCallback|None+进度消息接收函数
    :param message: str+待发送的中文进度消息
    :return: None
    :Author: gexinyan
    :CreateTime: 2026-07-13 16:25:00
    Example: _emit_progress(print, "完成 1/10")
    """
    if callback is not None:
        callback(message)


def _format_progress_bar(completed: int, total: int, width: int = 20) -> str:
    """
    【函数功能】按已完成文件数生成无第三方依赖的终端文本进度条。
    :param completed: int+已完成文件数
    :param total: int+总文件数
    :param width: int+进度条字符宽度（默认20）
    :return: str+形如 [##########----------] 5/10 (50%) 的进度条
    :Author: gexinyan
    :CreateTime: 2026-07-13 16:25:00
    Example: _format_progress_bar(5, 10)
    """
    if total <= 0:
        return "[--------------------] 0/0 (0%)"
    bounded_completed = min(max(completed, 0), total)
    ratio = bounded_completed / total
    filled = round(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {bounded_completed}/{total} ({ratio:.0%})"


def process_single_pdf(
    pdf_path: Path,
    input_root: Path,
    cache_dir: Path,
    config: ProcessingConfig,
) -> tuple[ParsedDocument, list[str]]:
    """
    【函数功能】分类、提取页面并解析单个 PDF 文件。
    :param pdf_path: Path+PDF 文件路径
    :param input_root: Path+输入根目录
    :param cache_dir: Path+OCR 缓存目录
    :param config: ProcessingConfig+处理配置
    :return: tuple[ParsedDocument, list[str]]+解析文档与页面告警列表
    :raises ValueError: 文件无法分类或未能读取任何页面时触发
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: process_single_pdf(path, root, cache, config)
    """
    engine = PDFTextEngine(pdf_path, cache_dir, config)
    first_native = engine.native_text(1) if engine.page_count else ""
    category = classify_pdf(pdf_path, input_root, engine.page_count, first_native)
    warnings: list[str] = []
    if category == "unknown":
        first_page = engine.get_page(1, config.dpi)
        category = classify_pdf(pdf_path, input_root, engine.page_count, first_page.text)
    if category == "unknown":
        raise ValueError("无法识别 PDF 文件类别")

    pages, page_warnings = load_pages_for_category(engine, category, config)
    warnings.extend(page_warnings)
    if not pages:
        raise ValueError("未能提取任何可解析页面")
    relative_path = str(pdf_path.relative_to(input_root))
    context = ParserContext(
        pdf_path=pdf_path,
        relative_path=relative_path,
        category=category,
        generated_at=current_timestamp(),
        confidence_threshold=config.ocr_confidence_threshold,
    )
    records = parse_document(pages, context)
    return ParsedDocument(pdf_path, category, engine.page_count, records, engine.ocr_pages), warnings


def load_pages_for_category(
    engine: PDFTextEngine,
    category: str,
    config: ProcessingConfig,
) -> tuple[list[PageText], list[str]]:
    """
    【函数功能】按文件类别选择普通、高精度或备案资料两阶段页面提取策略。
    :param engine: PDFTextEngine+PDF 文本与 OCR 引擎
    :param category: str+标准文件类别
    :param config: ProcessingConfig+处理配置
    :return: tuple[list[PageText], list[str]]+页面列表和页级告警
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: load_pages_for_category(engine, "award_notice", config)
    """
    if category == "archive_info":
        return _load_archive_pages(engine, config)
    if category == "tender_cover":
        if engine.page_count > 3:
            page_numbers = engine.cover_bookmark_pages() or list(range(1, min(5, engine.page_count) + 1))
        else:
            page_numbers = [1] if engine.page_count else []
        return _load_tender_cover_pages(engine, page_numbers, config.dpi)
    page_numbers = list(range(1, engine.page_count + 1))
    return _load_selected_pages(engine, page_numbers, config.dpi)


def _load_tender_cover_pages(
    engine: PDFTextEngine,
    page_numbers: Iterable[int],
    dpi: int,
) -> tuple[list[PageText], list[str]]:
    """
    【函数功能】仅为投标封面页启用原图、去红章与低质量重试的多策略 OCR。
    :param engine: PDFTextEngine+PDF 文本与 OCR 引擎
    :param page_numbers: Iterable[int]+封面候选页码
    :param dpi: int+基础 OCR 分辨率
    :return: tuple[list[PageText], list[str]]+成功页面及页级告警
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """
    pages: list[PageText] = []
    warnings: list[str] = []
    for page_number in sorted(set(page_numbers)):
        try:
            pages.append(engine.get_tender_cover_page(page_number, dpi))
        except Exception as exc:
            warnings.append(f"第{page_number}页封面多策略提取失败：{exc}")
    return pages, warnings


def _load_selected_pages(
    engine: PDFTextEngine,
    page_numbers: Iterable[int],
    dpi: int,
) -> tuple[list[PageText], list[str]]:
    """
    【函数功能】逐页提取指定页码并保留页级异常，不因单页损坏终止整个文件。
    :param engine: PDFTextEngine+PDF 文本与 OCR 引擎
    :param page_numbers: Iterable[int]+待提取页码
    :param dpi: int+OCR 分辨率
    :return: tuple[list[PageText], list[str]]+成功页面和告警信息
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    pages: list[PageText] = []
    warnings: list[str] = []
    for page_number in sorted(set(page_numbers)):
        try:
            pages.append(engine.get_page(page_number, dpi))
        except Exception as exc:
            warnings.append(f"第{page_number}页提取失败：{exc}")
    return pages, warnings


def _load_archive_pages(
    engine: PDFTextEngine,
    config: ProcessingConfig,
) -> tuple[list[PageText], list[str]]:
    """
    【函数功能】对备案资料先低清全文检索关键词，再高精度识别命中页及相邻页。
    :param engine: PDFTextEngine+PDF 文本与 OCR 引擎
    :param config: ProcessingConfig+处理配置
    :return: tuple[list[PageText], list[str]]+封面和命中高精度页面及告警
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    matched: set[int] = set()
    warnings: list[str] = []
    compact_keywords = tuple(compact_for_match(keyword) for keyword in ARCHIVE_KEYWORDS)
    for page_number in range(1, engine.page_count + 1):
        try:
            page = engine.get_page(page_number, config.archive_scan_dpi)
        except Exception as exc:
            warnings.append(f"第{page_number}页粗检失败：{exc}")
            continue
        compact = compact_for_match(page.text)
        if any(keyword in compact for keyword in compact_keywords):
            matched.add(page_number)

    selected = set(range(1, min(3, engine.page_count) + 1))
    for page_number in matched:
        selected.update(
            number
            for number in (page_number - 1, page_number, page_number + 1)
            if 1 <= number <= engine.page_count
        )
    pages, high_warnings = _load_selected_pages(engine, selected, config.dpi)
    warnings.extend(high_warnings)
    if not matched:
        warnings.append("备案资料全文未命中目标关键词")
    return pages, warnings


def write_records_csv(path: Path, records: list[ExtractionRecord]) -> None:
    """
    【函数功能】以 UTF-8 BOM 和固定中文表头写入解析记录 CSV。
    :param path: Path+目标 CSV 路径
    :param records: list[ExtractionRecord]+待写入记录
    :return: None
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: write_records_csv(Path("results/final.csv"), records)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sequence, record in enumerate(records, start=1):
            writer.writerow(record.to_csv_row(sequence))


def _write_category_csv_files(records: list[ExtractionRecord], output_dir: Path) -> None:
    """
    【函数功能】为七个标准类别分别生成结构一致的 CSV，即使无记录也写表头。
    :param records: list[ExtractionRecord]+全部未合并记录
    :param output_dir: Path+结果目录
    :return: None
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    for category in CATEGORIES:
        category_records = [record for record in records if record.category == category]
        write_records_csv(output_dir / f"{category}.csv", category_records)


def merge_key(record: ExtractionRecord) -> tuple[str, str, str, str]:
    """
    【函数功能】生成项目、标段与企业组成的稳定去重键。
    :param record: ExtractionRecord+解析记录
    :return: tuple[str, str, str, str]+去重键
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: merge_key(record)
    """
    company = normalize_text(record.company_name)
    lot = normalize_text(record.lot_name)
    if record.project_code:
        return "code", normalize_text(record.project_code).upper(), lot, company
    return "name", normalize_text(record.project_name), lot, company


def merge_and_deduplicate(records: list[ExtractionRecord]) -> list[ExtractionRecord]:
    """
    【函数功能】按项目、标段和企业去重，依据来源优先级合并状态和证据。
    :param records: list[ExtractionRecord]+分类解析记录
    :return: list[ExtractionRecord]+合并后的最终记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: merge_and_deduplicate(records)
    """
    fallback_to_primary: dict[
        tuple[str, str, str, str],
        set[tuple[str, str, str, str]],
    ] = defaultdict(set)
    for record in records:
        if not (record.project_code and record.project_name and record.company_name):
            continue
        fallback_key = (
            "name",
            normalize_text(record.project_name),
            normalize_text(record.lot_name),
            normalize_text(record.company_name),
        )
        fallback_to_primary[fallback_key].add(merge_key(record))

    groups: dict[tuple[str, str, str, str], list[ExtractionRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        key = merge_key(record)
        if not record.company_name or not (record.project_code or record.project_name):
            key = ("review", record.source_path, str(index), normalize_text(record.company_name))
        elif not record.project_code:
            code_keys = fallback_to_primary.get(key, set())
            if len(code_keys) == 1:
                key = next(iter(code_keys))
        groups[key].append(record)

    merged: list[ExtractionRecord] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: SOURCE_PRIORITY.get(item.category, 0), reverse=True)
        explicit = [item for item in ordered if item.award_status in {"是", "否"}]
        chosen = explicit[0] if explicit else ordered[0]
        statuses = {item.award_status for item in explicit}
        conflict = len(statuses) > 1
        source_paths = _unique_join(item.source_path for item in ordered)
        evidence = _unique_join(item.evidence for item in ordered if item.evidence)
        page_evidence = _unique_join(
            f"{item.source_path}#第{item.source_pages}页" for item in ordered if item.source_pages
        )
        merged_record = replace(
            chosen,
            project_name=_first_value(item.project_name for item in ordered),
            project_code=_first_value(item.project_code for item in ordered),
            lot_name=_first_value(item.lot_name for item in ordered),
            company_name=_first_value(item.company_name for item in ordered),
            category=_unique_join(item.category for item in ordered),
            source_path=source_paths,
            source_pages=page_evidence,
            extraction_method=_unique_join(item.extraction_method for item in ordered),
            evidence=("存在中标状态冲突；" if conflict else "") + evidence,
            confidence=max(item.confidence for item in ordered),
            review_status=(
                "冲突待复核"
                if conflict
                else ("待复核" if any(item.review_status != "通过" for item in ordered) else "通过")
            ),
            generated_at=max(item.generated_at for item in ordered),
        )
        merged.append(merged_record)
    return sorted(
        merged,
        key=lambda item: (
            normalize_text(item.project_code or item.project_name),
            normalize_text(item.lot_name),
            normalize_text(item.company_name),
        ),
    )


def _unique_join(values: Iterable[str]) -> str:
    """
    【函数功能】按输入顺序去重并使用分号连接非空文本。
    :param values: Iterable[str]+待连接文本
    :return: str+连接结果
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return "；".join(result)


def _first_value(values: Iterable[str]) -> str:
    """
    【函数功能】获取迭代序列中的第一个非空字符串。
    :param values: Iterable[str]+候选文本
    :return: str+首个非空值，全部为空时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    return next((value for value in values if value), "")
