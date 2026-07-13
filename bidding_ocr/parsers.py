"""七类招投标 PDF 的规则解析器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bidding_ocr.models import ExtractionRecord, PageText
from bidding_ocr.utils import compact_for_match, extract_company_names, normalize_text


PROJECT_LABELS = ("项目名称", "工程名称", "标段(包)名称", "标段（包）名称")
LOT_LABELS = ("标段名称", "标段(包)名称", "标段（包）名称")
PROJECT_CODE_LABELS = ("项目编号", "项目代码", "标段编号", "交易编号")
NON_BIDDER_LABELS = ("招标人", "招标代理", "代理机构", "建设单位", "采购人", "监督部门")


@dataclass(slots=True)
class ParserContext:
    """
    【类功能】向分类解析函数传递文件、类别、时间及复核阈值。
    :Attributes:
        pdf_path: Path+PDF 文件路径
        relative_path: str+相对输入目录的路径
        category: str+标准文件类别
        generated_at: str+解析生成时间
        confidence_threshold: float+人工复核置信度阈值
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    pdf_path: Path
    relative_path: str
    category: str
    generated_at: str
    confidence_threshold: float


@dataclass(slots=True)
class CompanyOccurrence:
    """
    【类功能】保存企业名称在页面中的位置、文本与置信度。
    :Attributes:
        name: str+企业名称
        page_number: int+来源页码
        evidence: str+来源文字行
        confidence: float+OCR 或文本层置信度
        method: str+提取方式
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    name: str
    page_number: int
    evidence: str
    confidence: float
    method: str


def _line_value(lines: list[str], labels: tuple[str, ...]) -> str:
    """
    【函数功能】从当前行冒号后或下一行提取指定字段值。
    :param lines: list[str]+按阅读顺序排列的页面文字行
    :param labels: tuple[str, ...]+候选字段标签
    :return: str+字段值，未找到时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: _line_value(["项目名称：测试项目"], ("项目名称",))
    """
    for index, line in enumerate(lines):
        compact = compact_for_match(line)
        for label in labels:
            normalized_label = compact_for_match(label)
            position = compact.find(normalized_label)
            if position < 0:
                continue
            raw_match = re.search(rf"{re.escape(label)}\s*[:：]?\s*(.+)$", line)
            if raw_match and normalize_text(raw_match.group(1)):
                return raw_match.group(1).strip()
            suffix = compact[position + len(normalized_label) :]
            if suffix:
                return suffix
            if index + 1 < len(lines):
                return lines[index + 1].strip()
    return ""


def extract_project_metadata(pages: list[PageText]) -> tuple[str, str, str]:
    """
    【函数功能】从页面文本中提取项目名称、项目编号和标段名称。
    :param pages: list[PageText]+待解析页面
    :return: tuple[str, str, str]+项目名称、项目编号、标段名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: extract_project_metadata([page])
    """
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    project_name = _line_value(lines, PROJECT_LABELS)
    lot_name = _line_value(lines, LOT_LABELS)
    project_code = _line_value(lines, PROJECT_CODE_LABELS)
    if project_code:
        code_match = re.search(r"[A-Za-z0-9][A-Za-z0-9_\-/]{5,}", project_code)
        project_code = code_match.group(0) if code_match else normalize_text(project_code)

    project_name = _clean_project_value(project_name)
    lot_name = _clean_project_value(lot_name)
    if not project_name:
        candidates: list[str] = []
        for line in lines[:80]:
            compact = normalize_text(line)
            if (
                "项目" in compact
                and 6 <= len(compact) <= 100
                and not any(noise in compact for noise in ("项目编号", "项目负责人", "项目经理", "工程项目管理"))
            ):
                candidates.append(line.strip())
        if candidates:
            project_name = _clean_project_value(candidates[0])
    return project_name, normalize_text(project_code), lot_name


def _clean_project_value(value: str) -> str:
    """
    【函数功能】清理项目字段标签、公告标题后缀和异常标点。
    :param value: str+原始项目文本
    :return: str+清理后的项目或标段名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: _clean_project_value("项目名称：测试项目中标公告")
    """
    value = re.sub(r"^(?:项目名称|工程名称|标段名称|标段[（(]包[）)]名称)\s*[:：]?", "", value or "")
    narrative_match = re.search(
        r"(?:^|的)([^，。；;]{4,80}?项目(?:[（(][^）)]+[）)])?(?:施工|监理|采购)?)的(?:评标|交易|招标)",
        value,
    )
    if narrative_match:
        value = narrative_match.group(1)
    value = re.sub(r"(?:中标候选人公示|中标人公告|中标公告|中标通知书)$", "", value.strip())
    return value.strip(" ：:，,。")


def find_company_occurrences(pages: list[PageText]) -> list[CompanyOccurrence]:
    """
    【函数功能】从页面文字行中提取投标相关企业并按首次出现顺序去重。
    :param pages: list[PageText]+待解析页面
    :return: list[CompanyOccurrence]+企业出现列表
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: find_company_occurrences([page])
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    for page in sorted(pages, key=lambda item: item.page_number):
        for line in page.lines:
            compact = compact_for_match(line.text)
            if any(label in compact for label in NON_BIDDER_LABELS) and not any(
                label in compact for label in ("投标人", "中标人", "中标单位")
            ):
                continue
            for name in extract_company_names(line.text):
                key = normalize_text(name)
                if key in seen:
                    continue
                seen.add(key)
                occurrences.append(
                    CompanyOccurrence(
                        name=name,
                        page_number=page.page_number,
                        evidence=line.text.strip()[:300],
                        confidence=line.confidence if page.method == "ocr" else 1.0,
                        method=page.method,
                    )
                )
    return occurrences


def _pages_with_keywords(pages: list[PageText], keywords: tuple[str, ...]) -> list[PageText]:
    """
    【函数功能】筛选正文包含任一业务关键词的页面。
    :param pages: list[PageText]+全部候选页面
    :param keywords: tuple[str, ...]+业务关键词
    :return: list[PageText]+命中页面
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: _pages_with_keywords(pages, ("中标通知书",))
    """
    return [
        page
        for page in pages
        if any(compact_for_match(keyword) in compact_for_match(page.text) for keyword in keywords)
    ]


def _explicit_company(pages: list[PageText], labels: tuple[str, ...]) -> CompanyOccurrence | None:
    """
    【函数功能】优先从“中标人”等字段行或通知书冒号称呼中提取企业。
    :param pages: list[PageText]+候选页面
    :param labels: tuple[str, ...]+明确企业字段标签
    :return: CompanyOccurrence|None+明确企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: _explicit_company(pages, ("中标人",))
    """
    for page in pages:
        for line in page.lines:
            compact = compact_for_match(line.text)
            names = extract_company_names(line.text)
            has_explicit_label = any(compact_for_match(label) in compact for label in labels)
            if names and (has_explicit_label or line.text.rstrip().endswith((":", "："))):
                return CompanyOccurrence(
                    names[0],
                    page.page_number,
                    line.text.strip()[:300],
                    line.confidence if page.method == "ocr" else 1.0,
                    page.method,
                )
    return None


def _records_from_occurrences(
    occurrences: list[CompanyOccurrence],
    pages: list[PageText],
    context: ParserContext,
    status_by_name: dict[str, str] | None = None,
    rank_by_name: dict[str, str] | None = None,
    unknown_requires_review: bool = False,
) -> list[ExtractionRecord]:
    """
    【函数功能】将企业出现列表转换为统一解析记录并补齐项目信息和复核状态。
    :param occurrences: list[CompanyOccurrence]+企业出现列表
    :param pages: list[PageText]+用于提取项目元数据的页面
    :param context: ParserContext+解析上下文
    :param status_by_name: dict[str, str]|None+按企业名称指定中标状态
    :param rank_by_name: dict[str, str]|None+按企业名称指定排名
    :param unknown_requires_review: bool+未知状态是否进入复核
    :return: list[ExtractionRecord]+统一记录列表
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    project_name, project_code, lot_name = extract_project_metadata(pages)
    records: list[ExtractionRecord] = []
    status_by_name = status_by_name or {}
    rank_by_name = rank_by_name or {}
    for occurrence in occurrences:
        status = status_by_name.get(normalize_text(occurrence.name), "未知")
        needs_review = (
            not project_name
            or not occurrence.name
            or occurrence.confidence < context.confidence_threshold
            or (unknown_requires_review and status == "未知")
        )
        records.append(
            ExtractionRecord(
                project_name=project_name,
                project_code=project_code,
                lot_name=lot_name,
                company_name=occurrence.name,
                award_status=status,
                rank=rank_by_name.get(normalize_text(occurrence.name), ""),
                category=context.category,
                source_path=context.relative_path,
                source_pages=str(occurrence.page_number),
                extraction_method=occurrence.method,
                evidence=occurrence.evidence,
                confidence=occurrence.confidence,
                review_status="待复核" if needs_review else "通过",
                generated_at=context.generated_at,
            )
        )
    if not records:
        records.append(
            ExtractionRecord(
                project_name=project_name,
                project_code=project_code,
                lot_name=lot_name,
                category=context.category,
                source_path=context.relative_path,
                source_pages=",".join(str(page.page_number) for page in pages),
                extraction_method="/".join(sorted({page.method for page in pages})),
                evidence="未提取到企业名称",
                confidence=0.0,
                review_status="待复核",
                generated_at=context.generated_at,
            )
        )
    return records


def parse_tender_cover(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析投标文件封面的项目名称、企业名称及路径中标状态。
    :param pages: list[PageText]+封面候选页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+封面解析记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_tender_cover(pages, context)
    """
    from bidding_ocr.utils import determine_cover_award_status

    occurrences = find_company_occurrences(pages)
    if not occurrences:
        filename_companies = extract_company_names(context.pdf_path.stem)
        occurrences = [
            CompanyOccurrence(name, 1, context.pdf_path.stem, 0.85, "filename")
            for name in filename_companies
        ]
    status = determine_cover_award_status(context.relative_path)
    statuses = {normalize_text(item.name): status for item in occurrences}
    return _records_from_occurrences(occurrences[:1], pages, context, statuses, unknown_requires_review=True)


def parse_bid_evaluation_report(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析评标报告的投标人排序与第一中标候选人。
    :param pages: list[PageText]+评标报告页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+投标企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_bid_evaluation_report(pages, context)
    """
    ranking_pages = _pages_with_keywords(pages, ("投标人排序及推荐的中标候选人", "推荐的中标候选人"))
    fallback = False
    if not ranking_pages:
        ranking_pages = _pages_with_keywords(pages, ("投标文件初步评审", "投标人名单"))
        fallback = True
    if ranking_pages:
        target_numbers = {
            number
            for page in ranking_pages
            for number in (page.page_number, page.page_number + 1)
        }
        ranking_pages = [page for page in pages if page.page_number in target_numbers]
    target_pages = ranking_pages or pages
    occurrences = find_company_occurrences(target_pages)
    statuses: dict[str, str] = {}
    ranks: dict[str, str] = {}
    for index, occurrence in enumerate(occurrences, start=1):
        statuses[normalize_text(occurrence.name)] = (
            "是" if index == 1 and not fallback else ("否" if not fallback else "未知")
        )
        ranks[normalize_text(occurrence.name)] = str(index) if not fallback else ""
    records = _records_from_occurrences(occurrences, pages, context, statuses, ranks, unknown_requires_review=fallback)
    if fallback:
        for record in records:
            record.review_status = "待复核"
            record.evidence = f"未定位推荐候选人排序页；{record.evidence}"[:300]
    return records


def parse_bid_candidates(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析中标候选人公示中的全部投标人、得分排名和第一名。
    :param pages: list[PageText]+公示页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+投标企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_bid_candidates(pages, context)
    """
    explicit = _explicit_company(pages, ("拟定中标人", "第一中标候选人"))
    occurrences = find_company_occurrences(pages)
    if explicit and normalize_text(explicit.name) not in {normalize_text(item.name) for item in occurrences}:
        occurrences.insert(0, explicit)
    winner_name = (
        normalize_text(explicit.name)
        if explicit
        else (normalize_text(occurrences[0].name) if occurrences else "")
    )
    statuses: dict[str, str] = {}
    ranks: dict[str, str] = {}
    for index, occurrence in enumerate(occurrences, start=1):
        key = normalize_text(occurrence.name)
        statuses[key] = "是" if key == winner_name else "否"
        ranks[key] = "1" if key == winner_name else str(index)
    return _records_from_occurrences(occurrences, pages, context, statuses, ranks)


def parse_award_notice(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析中标或交易结果通知书的冒号收件企业。
    :param pages: list[PageText]+通知书页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+中标企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_award_notice(pages, context)
    """
    explicit = _explicit_company(pages, ("中标人", "中标单位"))
    occurrences = [explicit] if explicit else find_company_occurrences(pages)[:1]
    statuses = {normalize_text(item.name): "是" for item in occurrences if item}
    return _records_from_occurrences([item for item in occurrences if item], pages, context, statuses)


def parse_bid_announcement(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析中标公告的“中标人”字段。
    :param pages: list[PageText]+公告页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+中标企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_bid_announcement(pages, context)
    """
    explicit = _explicit_company(pages, ("中标人", "中标单位"))
    occurrences = [explicit] if explicit else find_company_occurrences(pages)[:1]
    statuses = {normalize_text(item.name): "是" for item in occurrences if item}
    return _records_from_occurrences([item for item in occurrences if item], pages, context, statuses)


def parse_bid_list(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析投标单位名单表格并将中标状态保留为未知。
    :param pages: list[PageText]+名单页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+全部投标企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_bid_list(pages, context)
    """
    occurrences = find_company_occurrences(pages)
    return _records_from_occurrences(occurrences, pages, context)


def parse_archive_info(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】解析备案资料命中页中的中标企业和投标企业名单。
    :param pages: list[PageText]+备案资料封面及关键词命中页
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+备案资料企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_archive_info(pages, context)
    """
    winner_pages = _pages_with_keywords(pages, ("中标通知书", "成交通知书", "推荐的中标候选人"))
    participant_pages = _pages_with_keywords(
        pages,
        (
            "招投标情况书面报告",
            "招投标基本情况报告",
            "原件确认签收表",
            "项目负责人答辩评分表",
            "按时送达投标文件的投标人名单",
        ),
    )
    winner = _explicit_company(winner_pages, ("中标人", "中标单位"))
    winner_occurrences = find_company_occurrences(winner_pages)
    if winner is None and winner_occurrences:
        winner = winner_occurrences[0]
    occurrences = find_company_occurrences(participant_pages + winner_pages)
    if winner and normalize_text(winner.name) not in {normalize_text(item.name) for item in occurrences}:
        occurrences.insert(0, winner)
    winner_name = normalize_text(winner.name) if winner else ""
    statuses: dict[str, str] = {}
    for occurrence in occurrences:
        key = normalize_text(occurrence.name)
        statuses[key] = "是" if key == winner_name else ("否" if winner_name else "未知")
    return _records_from_occurrences(
        occurrences,
        pages,
        context,
        statuses,
        unknown_requires_review=not bool(winner_name),
    )


PARSER_REGISTRY: dict[str, Callable[[list[PageText], ParserContext], list[ExtractionRecord]]] = {
    "tender_cover": parse_tender_cover,
    "bid_evaluation_report": parse_bid_evaluation_report,
    "bid_candidates": parse_bid_candidates,
    "award_notice": parse_award_notice,
    "bid_announcement": parse_bid_announcement,
    "bid_list": parse_bid_list,
    "archive_info": parse_archive_info,
}


def parse_document(pages: list[PageText], context: ParserContext) -> list[ExtractionRecord]:
    """
    【函数功能】通过解析器注册表调用对应文件类别的处理函数。
    :param pages: list[PageText]+已提取页面
    :param context: ParserContext+解析上下文
    :return: list[ExtractionRecord]+统一解析记录
    :raises ValueError: 文件类别不存在对应解析器时触发
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: parse_document(pages, context)
    """
    parser = PARSER_REGISTRY.get(context.category)
    if parser is None:
        raise ValueError(f"没有对应解析器：{context.category}")
    return parser(pages, context)
