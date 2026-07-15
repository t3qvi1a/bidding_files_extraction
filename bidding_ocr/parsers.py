"""七类招投标 PDF 的规则解析器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bidding_ocr.models import ExtractionRecord, OCRLine, PageText
from bidding_ocr.tender_cover_strategy import extract_tender_cover_fields
from bidding_ocr.utils import compact_for_match, extract_company_names, normalize_text


PROJECT_LABELS = ("项目名称", "工程名称", "标段(包)名称", "标段（包）名称")
LOT_LABELS = ("标段名称", "标段(包)名称", "标段（包）名称")
PROJECT_CODE_LABELS = (
    "项目编号",
    "项目代码",
    "标段编号",
    "标段(包)编号",
    "标段（包）编号",
    "交易编号",
)
LOT_CODE_LABELS = ("标段编号", "标段(包)编号", "标段（包）编号")
NON_BIDDER_LABELS = ("招标人", "招标代理", "代理机构", "建设单位", "采购人", "监督部门")
METADATA_STOP_LABELS = (
    "项目编号",
    "项目代码",
    "交易编号",
    "标段编号",
    "标段(包)编号",
    "标段（包）编号",
    "项目名称",
    "工程名称",
    "标段名称",
    "标段(包)名称",
    "标段（包）名称",
    "招标人名称",
    "招标人",
    "招标代理",
    "代理机构",
    "建设单位",
    "采购人",
    "监督部门",
    "中标人",
    "中标单位",
    "中标总价",
    "中标工期",
    "项目负责人",
    "日期",
)
LAYOUT_MARGIN_RESIDUE_RE = re.compile(r"(?<=\S)\s{2,}\S{1,3}$")
AWARD_NOTICE_NARRATIVE_PROJECT_RE = re.compile(
    r"(?:^|[：:。；;])"
    r"[^：:，。；;]{2,80}?的"
    r"(?P<project>[^，。；;]{4,100}?(?:项目|工程)[^，。；;]{0,40}?)"
    r"的(?:评标|评审|交易|招标)(?:工作)?(?:已经|已)?(?:结束|完成)"
)
AWARD_NOTICE_HEADER_CODE_RE = re.compile(
    r"(?<![A-Z0-9])(?P<code>[A-Z][A-Z0-9]{4,}(?:-[A-Z]{0,3}\d{1,4})?)(?![A-Z0-9])"
)
AWARD_NOTICE_LOT_CODE_SUFFIX_RE = re.compile(r"-[A-Z]{0,3}\d{1,4}$")
AWARD_NOTICE_HEADER_LINE_LIMIT = 8
BID_LIST_UNIT_HEADER_LABELS = ("单位名称", "投标单位名称", "投标人名称", "投标单位")
BID_LIST_TABLE_HEADER_LABELS = (
    "序号",
    *BID_LIST_UNIT_HEADER_LABELS,
    "投标联系人",
    "联系人",
    "联系方式",
    "投标时间",
)
BID_LIST_LOT_SUFFIX_RE = re.compile(r"(?:第?[一二三四五六七八九十百千万零〇0-9]+标段|标段)$")
BID_CANDIDATE_TITLE_MARKER = "中标候选人公示"
BID_CANDIDATE_TITLE_CHROME = (
    "首页",
    "交易分类",
    "信息发布时间",
    "阅读次数",
    "我要打印",
    "关闭",
    "用户登录",
)
BID_CANDIDATE_TITLE_TRAILING_CHROME = (
    "用户登录",
    "交易平台",
    "曝光台",
    "用",
    "户",
    "登",
    "录",
    "交",
    "易",
    "平",
    "台",
    "曝",
    "光",
)
BID_CANDIDATE_PRINT_TIME_RE = re.compile(
    r"^\s*\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}\s*"
)
BID_CANDIDATE_COMPACT_PRINT_TIME_RE = re.compile(
    r"^\d{4}/\d{1,2}/\d{1,2}\d{1,2}:\d{2}(?=\D|$)"
)
BID_CANDIDATE_TITLE_EDITION_RE = re.compile(r"^[（(]二次[）)]$")
BID_CANDIDATE_SCORE_TABLE_TITLES = (
    "所有投标人得分汇总表",
    "所有投标人报价及得分情况",
    "所有投标人投标报价及得分情况",
)
BID_CANDIDATE_SCORE_TABLE_COLUMN = "投标人名称"
BID_CANDIDATE_SCORE_TABLE_HEADERS = (
    "序号",
    BID_CANDIDATE_SCORE_TABLE_COLUMN,
    "投标报价",
    "得分情况",
    "得分总计",
    "排名",
)
BID_CANDIDATE_SCORE_TABLE_STOP_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+|\d+)[、.]?(?:拟定中标人|中标候选人)"
)
EVALUATION_REPORT_TITLE_MARKER = "投标人排序及推荐的中标候选人"
EVALUATION_REPORT_RECOMMENDED_MARKER = "推荐的中标候选人"
EVALUATION_REPORT_FIRST_RANK_RE = re.compile(r"(?:第?[1一]名)")
EVALUATION_REPORT_BASIC_INFO_MARKER = "基本情况一览表"
ARCHIVE_PARTICIPANT_LIST_MARKERS = (
    "按时送达投标文件的投标人名单",
    "投标人名单",
    "投标单位名单",
)
ARCHIVE_LOT_NAME_FALLBACK_LABELS = ("工程名称", "项目名称")
ARCHIVE_LIST_SEPARATOR_RE = re.compile(r"[、，,；;]")
ARCHIVE_SECTION_HEADING_RE = re.compile(r"^(?:[0-9一二三四五六七八九十百]+)[、.．。]")


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


@dataclass(slots=True)
class BidListUnitColumn:
    """
    【类功能】保存投标名单“单位名称”列的横向范围和表头下边界。
    :Attributes:
        left: float+单位名称列左边界
        right: float+单位名称列右边界
        content_top: float+表头下边界，同行或上方文字不作为单位名称
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    """

    left: float
    right: float
    content_top: float


@dataclass(slots=True)
class NativeScoreTableColumn:
    """
    【类功能】保存原生 layout 得分表名称列的字符范围和表头行位置。
    :Attributes:
        left: int+名称列可接受片段的最小字符位置
        right: int+名称列可接受片段的最大字符位置
        content_top: int+表头所在版式行号
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    """

    left: int
    right: int
    content_top: int


def _metadata_label_pattern(label: str) -> str:
    """
    【函数功能】构造允许标签字符之间存在空白的字段标签正则表达式。
    :param label: str+原始字段标签
    :return: str+可用于正则匹配的标签表达式
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:00:00
    Example: _metadata_label_pattern("项目名称")
    """
    return r"\s*".join(re.escape(character) for character in label)


def _truncate_metadata_value(value: str) -> str:
    """
    【函数功能】截断合并行中当前字段值后的下一个元数据标签。
    :param value: str+字段标签之后的原始文本
    :return: str+不含后续字段标签的字段值
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:00:00
    Example: _truncate_metadata_value("测试项目标段编号：ABC001")
    """
    stop_positions = [
        match.start()
        for label in METADATA_STOP_LABELS
        if (match := re.search(_metadata_label_pattern(label), value)) is not None
    ]
    return value[: min(stop_positions)] if stop_positions else value


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
        for label in labels:
            label_match = re.search(_metadata_label_pattern(label), line)
            if label_match is None:
                continue
            value = re.sub(r"^\s*[:：]?\s*", "", line[label_match.end() :])
            value = _truncate_metadata_value(value).strip()
            value = LAYOUT_MARGIN_RESIDUE_RE.sub("", value).strip()
            if normalize_text(value):
                return value
            if index + 1 < len(lines):
                return lines[index + 1].strip()
    return ""


def extract_project_metadata(
    pages: list[PageText],
    prefer_richest: bool = False,
) -> tuple[str, str, str]:
    """
    【函数功能】从页面文本中提取项目名称、项目编号和标段名称。
    :param pages: list[PageText]+待解析页面
    :param prefer_richest: bool+是否在多处项目字段中优先选择信息更完整的值
    :return: tuple[str, str, str]+项目名称、项目编号、标段名称
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:00:00
    Example: extract_project_metadata([page])
    """
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    project_name = _line_value(lines, PROJECT_LABELS)
    lot_name = _line_value(lines, LOT_LABELS)
    if prefer_richest:
        project_candidates = _line_values(lines, PROJECT_LABELS)
        lot_candidates = _line_values(lines, LOT_LABELS)
        if project_candidates:
            project_name = max(project_candidates, key=len)
        if lot_candidates:
            lot_name = max(lot_candidates, key=len)
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


def _line_values(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    """
    【函数功能】提取页面中所有指定元数据字段值，供备案资料择优使用。
    :param lines: list[str]+按阅读顺序排列的页面文字行
    :param labels: tuple[str, ...]+候选字段标签
    :return: list[str]+去除空值后的字段值
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:00:00
    Example: _line_values(["项目名称：甲项目"], ("项目名称",))
    """
    values: list[str] = []
    for index, line in enumerate(lines):
        for label in labels:
            label_match = re.search(_metadata_label_pattern(label), line)
            if label_match is None:
                continue
            value = re.sub(r"^\s*[:：]?\s*", "", line[label_match.end() :])
            value = _truncate_metadata_value(value).strip()
            value = LAYOUT_MARGIN_RESIDUE_RE.sub("", value).strip()
            if not normalize_text(value) and index + 1 < len(lines):
                value = lines[index + 1].strip()
            if normalize_text(value):
                values.append(value)
            break
    return values


def extract_bid_announcement_lot_code(pages: list[PageText]) -> str:
    """
    【函数功能】从中标公告页面提取标段编号并规范化为连续编号。
    :param pages: list[PageText]+中标公告页面
    :return: str+标段编号，未识别时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 17:00:00
    Example: extract_bid_announcement_lot_code([page])
    """
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    value = _line_value(lines, LOT_CODE_LABELS)
    if not value:
        return ""
    code_match = re.search(r"[A-Za-z0-9][A-Za-z0-9_\-/]{5,}", value)
    return code_match.group(0) if code_match else normalize_text(value)


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


def extract_award_notice_project_name(pages: list[PageText]) -> str:
    """
    【函数功能】优先提取明确项目字段，否则从中标通知书评审结束叙述中截取项目名称。
    :param pages: list[PageText]+中标通知书页面
    :return: str+项目名称，未识别时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:20:55
    Example: extract_award_notice_project_name([page])
    """
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    explicit_project_name = _clean_project_value(_line_value(lines, PROJECT_LABELS))
    if explicit_project_name:
        return explicit_project_name

    for page in sorted(pages, key=lambda item: item.page_number):
        compact_page_text = re.sub(r"\s+", "", page.text)
        match = AWARD_NOTICE_NARRATIVE_PROJECT_RE.search(compact_page_text)
        if match:
            project_name = _clean_project_value(match.group("project"))
            if project_name:
                return project_name
    return ""


def extract_award_notice_codes(pages: list[PageText]) -> tuple[str, str]:
    """
    【函数功能】从中标或交易结果通知书首页抬头提取项目编号和标段编号。
    :param pages: list[PageText]+中标或交易结果通知书页面
    :return: tuple[str, str]+项目编号、标段编号；未识别字段返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:25:50
    Example: extract_award_notice_codes([page])
    """
    for page in sorted(pages, key=lambda item: item.page_number):
        header_lines = [line.text for line in page.lines if line.text.strip()][
            :AWARD_NOTICE_HEADER_LINE_LIMIT
        ]
        for line in header_lines:
            normalized_line = normalize_text(line).upper()
            normalized_line = normalized_line.replace("－", "-").replace("—", "-").replace("–", "-")
            match = AWARD_NOTICE_HEADER_CODE_RE.search(normalized_line)
            if match is None:
                continue
            return split_project_and_lot_code(match.group("code"))
    return "", ""


def split_project_and_lot_code(value: str) -> tuple[str, str]:
    """
    【函数功能】将末尾带标段后缀的编号拆分为项目编号和完整标段编号。
    :param value: str+原始项目、交易或标段编号
    :return: tuple[str, str]+项目编号与完整标段编号；无标段后缀时后者为空
    :Author: gexinyan
    :CreateTime: 2026-07-14 17:30:00
    Example: split_project_and_lot_code("WXHS20241212002-S01")
    """
    code = normalize_text(value).upper()
    code = code.replace("－", "-").replace("—", "-").replace("–", "-")
    suffix_match = AWARD_NOTICE_LOT_CODE_SUFFIX_RE.search(code)
    if suffix_match is None:
        return code, ""
    return code[: suffix_match.start()], code


def _project_name_without_lot_suffix(value: str) -> str:
    """
    【函数功能】移除标段名称末尾的标段标识，生成项目名称。
    :param value: str+完整标段名称
    :return: str+不含末尾标段标识的项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _project_name_without_lot_suffix("测试水利项目一标段")
    """
    return BID_LIST_LOT_SUFFIX_RE.sub("", value or "").strip(" ：:，,。")


def _line_bottom(line: OCRLine) -> float:
    """
    【函数功能】获取带坐标 OCR 行的下边界，缺失坐标时返回纵向中心点。
    :param line: OCRLine+带坐标的 OCR 文字行
    :return: float+文字框下边界纵坐标
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _line_bottom(OCRLine("单位名称", 0.99, [[0, 0], [1, 0], [1, 1], [0, 1]]))
    """
    return max((point[1] for point in line.bbox), default=line.center_y)


def _line_height(line: OCRLine) -> float:
    """
    【函数功能】计算 OCR 行文字框高度，用于识别同一表头行。
    :param line: OCRLine+带坐标的 OCR 文字行
    :return: float+文字框高度，缺失坐标时返回最小容差
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _line_height(OCRLine("单位名称", 0.99, [[0, 0], [1, 0], [1, 1], [0, 1]]))
    """
    if not line.bbox:
        return 3.0
    values = [point[1] for point in line.bbox]
    return max(max(values) - min(values), 3.0)


def _is_credible_evaluation_report_project_name(value: str) -> bool:
    """
    【函数功能】判断基本情况一览表中的工程名称值是否可作为项目名称。
    :param value: str+工程名称字段候选值
    :return: bool+候选值可信时返回True
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:30:00
    Example: _is_credible_evaluation_report_project_name("测试高标准农田建设项目施工")
    """
    compact = normalize_text(value)
    return (
        6 <= len(compact) <= 120
        and any(keyword in compact for keyword in ("项目", "工程", "施工", "监理", "采购"))
        and not any(
            label in compact
            for label in ("工程名称", "招标范围", "开标时间", "开标地点", "评标时间", "评标地点")
        )
    )


def extract_bid_evaluation_report_project_name(pages: list[PageText]) -> str:
    """
    【函数功能】从基本情况一览表中按同一行坐标提取工程名称。
    :param pages: list[PageText]+评标报告全部已读取页面
    :return: str+工程名称对应的项目名称，未识别时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:30:00
    Example: extract_bid_evaluation_report_project_name(pages)
    """
    for page in sorted(pages, key=lambda item: item.page_number):
        if EVALUATION_REPORT_BASIC_INFO_MARKER not in compact_for_match(page.text):
            continue
        for label_line in page.lines:
            if compact_for_match(label_line.text) != compact_for_match("工程名称"):
                continue
            candidates = []
            for value_line in page.lines:
                if value_line is label_line or value_line.center_x <= label_line.center_x:
                    continue
                row_tolerance = max(_line_height(label_line), _line_height(value_line)) * 1.8
                if abs(value_line.center_y - label_line.center_y) > row_tolerance:
                    continue
                value = _clean_project_value(value_line.text)
                if _is_credible_evaluation_report_project_name(value):
                    candidates.append((value_line.center_x, value))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
    return ""


def _bid_list_header_label(line: OCRLine) -> str:
    """
    【函数功能】识别 OCR 行是否为投标名单表头并返回规范表头名称。
    :param line: OCRLine+待识别的 OCR 文字行
    :return: str+命中的表头名称，未命中时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _bid_list_header_label(OCRLine("单位名称", 0.99, []))
    """
    compact = compact_for_match(line.text)
    return next((label for label in BID_LIST_TABLE_HEADER_LABELS if compact == compact_for_match(label)), "")


def _find_bid_list_unit_column(page: PageText) -> BidListUnitColumn | None:
    """
    【函数功能】通过同一表头行的相邻列坐标定位“单位名称”列。
    :param page: PageText+待定位的投标名单 OCR 页面
    :return: BidListUnitColumn|None+单位名称列范围，无法可靠定位时返回空
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _find_bid_list_unit_column(page)
    """
    positioned_lines = [line for line in page.lines if line.bbox]
    for unit_line in positioned_lines:
        if _bid_list_header_label(unit_line) not in BID_LIST_UNIT_HEADER_LABELS:
            continue
        row_tolerance = _line_height(unit_line)
        row_headers = sorted(
            (
                line
                for line in positioned_lines
                if abs(line.center_y - unit_line.center_y) <= row_tolerance and _bid_list_header_label(line)
            ),
            key=lambda line: line.center_x,
        )
        try:
            unit_index = row_headers.index(unit_line)
        except ValueError:
            continue
        if unit_index == 0 or unit_index == len(row_headers) - 1:
            continue
        previous_header = row_headers[unit_index - 1]
        next_header = row_headers[unit_index + 1]
        left = (previous_header.center_x + unit_line.center_x) / 2
        right = (unit_line.center_x + next_header.center_x) / 2
        if left < unit_line.center_x < right:
            return BidListUnitColumn(left, right, _line_bottom(unit_line))
    return None


def _find_bid_list_company_occurrences(
    pages: list[PageText],
) -> tuple[list[CompanyOccurrence], bool]:
    """
    【函数功能】从投标名单“单位名称”列提取企业，并支持跨页复用列范围。
    :param pages: list[PageText]+按页码排列的投标名单页面
    :return: tuple[list[CompanyOccurrence], bool]+企业记录与是否成功定位过单位列
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:30:00
    Example: _find_bid_list_company_occurrences([page])
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    active_column: BidListUnitColumn | None = None
    located_column = False
    for page in sorted(pages, key=lambda item: item.page_number):
        column = _find_bid_list_unit_column(page)
        if column is not None:
            active_column = column
            located_column = True
        if active_column is None:
            continue
        content_top = column.content_top if column is not None else float("-inf")
        for line in sorted(page.lines, key=lambda item: (item.center_y, item.center_x)):
            if not line.bbox or line.center_y <= content_top:
                continue
            if not active_column.left < line.center_x < active_column.right:
                continue
            for name in extract_company_names(line.text):
                key = normalize_text(name)
                if not key or key in seen:
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
    return occurrences, located_column


def _compact_bid_candidate_title(value: str) -> str:
    """
    【函数功能】移除候选人公示标题中的排版空白并保留原始中文标点。
    :param value: str+原始标题文本
    :return: str+去除排版空白后的标题文本
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:30:00
    Example: _compact_bid_candidate_title("测试 项目 中 标 候 选 人 公 示")
    """
    return re.sub(r"\s+", "", value or "").strip(" ：:，,。")


def _is_credible_bid_candidate_title(value: str) -> bool:
    """
    【函数功能】判断文本是否可作为中标候选人公示的项目标题主体。
    :param value: str+已移除公示后缀的标题候选
    :return: bool+候选可信时返回True
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:30:00
    Example: _is_credible_bid_candidate_title("测试农田建设项目施工")
    """
    compact = _compact_bid_candidate_title(value)
    return (
        len(compact) >= 6
        and bool(re.search(r"[\u4e00-\u9fff]", compact))
        and ">" not in compact
        and BID_CANDIDATE_TITLE_MARKER not in compact
        and not any(chrome in compact for chrome in BID_CANDIDATE_TITLE_CHROME)
    )


def _strip_bid_candidate_print_time(value: str) -> str:
    """
    【函数功能】仅移除标题行开头的斜杠日期加时分格式打印时间。
    :param value: str+标题原始文本
    :return: str+去除严格时间前缀后的文本
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _strip_bid_candidate_print_time("2024/9/13 11:00 项目中标候选人公示")
    """
    cleaned = BID_CANDIDATE_PRINT_TIME_RE.sub("", value or "", count=1)
    compact = _compact_bid_candidate_title(cleaned)
    return BID_CANDIDATE_COMPACT_PRINT_TIME_RE.sub("", compact, count=1)


def _bid_candidate_title_body(value: str) -> str:
    """
    【函数功能】从包含完整公示标记的标题文本中提取结构有效的项目名称主体。
    :param value: str+单行或相邻行拼成的标题文本
    :return: str+可信项目名称主体，标题结构无效时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _bid_candidate_title_body("测试项目中标候选人公示（二次）")
    """
    compact = _strip_bid_candidate_print_time(value)
    marker_position = compact.find(BID_CANDIDATE_TITLE_MARKER)
    if marker_position < 0:
        return ""
    marker_end = marker_position + len(BID_CANDIDATE_TITLE_MARKER)
    trailing = compact[marker_end:].strip(" ：:，,。")
    trailing_is_structural = (
        not trailing
        or trailing in BID_CANDIDATE_TITLE_TRAILING_CHROME
        or bool(BID_CANDIDATE_TITLE_EDITION_RE.fullmatch(trailing))
    )
    if not trailing_is_structural:
        return ""
    body = compact[:marker_position]
    return body if _is_credible_bid_candidate_title(body) else ""


def extract_bid_candidate_title_project_name(pages: list[PageText]) -> str:
    """
    【函数功能】从中标候选人公示首页标题区提取项目名称主体。
    :param pages: list[PageText]+候选人公示页面
    :return: str+移除“中标候选人公示”后的项目标题，未识别时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 11:30:00
    Example: extract_bid_candidate_title_project_name([page])
    """
    if not pages:
        return ""
    first_page = min(pages, key=lambda page: page.page_number)
    title_lines = [line.text for line in first_page.lines if line.text.strip()][:40]
    project_code_indexes = [
        index
        for index, line in enumerate(title_lines)
        if any(label in compact_for_match(line) for label in PROJECT_CODE_LABELS)
    ]
    candidates: list[tuple[str, int, int]] = []
    for index, line in enumerate(title_lines):
        compact = _compact_bid_candidate_title(line)
        inline_body = _bid_candidate_title_body(line)
        if inline_body:
            code_distance = min(
                (abs(index - code_index) for code_index in project_code_indexes),
                default=len(title_lines),
            )
            candidates.append((inline_body, 0, code_distance))

        if (
            BID_CANDIDATE_TITLE_MARKER not in compact
            and len(compact) > 1
            and index + 1 < len(title_lines)
        ):
            joined_body = _bid_candidate_title_body(f"{line}\n{title_lines[index + 1]}")
            if joined_body:
                code_distance = min(
                    (abs(index - code_index) for code_index in project_code_indexes),
                    default=len(title_lines),
                )
                candidates.append((joined_body, 1, code_distance))

    if not candidates:
        return ""

    deduplicated: dict[str, tuple[str, int, int]] = {}
    for candidate in candidates:
        current = deduplicated.get(candidate[0])
        if current is None or candidate[1:] < current[1:]:
            deduplicated[candidate[0]] = candidate
    unique_candidates = list(deduplicated.values())
    without_layout_residue = [
        candidate
        for candidate in unique_candidates
        if not any(
            other[0] != candidate[0] and candidate[0].startswith(other[0])
            for other in unique_candidates
        )
    ]
    selectable = without_layout_residue or unique_candidates
    return min(selectable, key=lambda candidate: (candidate[1], candidate[2], len(candidate[0])))[0]


def find_company_occurrences(
    pages: list[PageText],
    strict_company_filter: bool = False,
) -> list[CompanyOccurrence]:
    """
    【函数功能】从页面文字行中提取投标相关企业并按首次出现顺序去重。
    :param pages: list[PageText]+待解析页面
    :param strict_company_filter: bool+是否过滤备案资料叙述句中的伪企业名称
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
            for name in extract_company_names(line.text, strict=strict_company_filter):
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


def _evaluation_report_marker_positions(page: PageText, marker: str) -> list[float]:
    """
    【函数功能】定位评标排序表标记文本的纵向位置，并兼容 OCR 拆行。
    :param page: PageText+待检测的页面
    :param marker: str+待定位的规范标记文本
    :return: list[float]+标记起始行的纵向中心坐标
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    Example: _evaluation_report_marker_positions(page, "推荐的中标候选人")
    """
    positions: list[float] = []
    compact_marker = compact_for_match(marker)
    for line in page.lines:
        compact_line = compact_for_match(line.text)
        if compact_marker in compact_line or (
            len(compact_line) >= 4 and compact_marker.startswith(compact_line)
        ):
            positions.append(line.center_y)
    return positions


def _evaluation_report_recommended_positions(page: PageText) -> list[float]:
    """
    【函数功能】定位排序表下方的推荐候选人区，忽略标题中包含的同名短语。
    :param page: PageText+待检测的页面
    :return: list[float]+推荐候选人区标记的纵向中心坐标
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    Example: _evaluation_report_recommended_positions(page)
    """
    title_positions = _evaluation_report_marker_positions(page, EVALUATION_REPORT_TITLE_MARKER)
    title_bottom = max(title_positions) if title_positions else float("-inf")
    return [
        position
        for position in _evaluation_report_marker_positions(
            page,
            EVALUATION_REPORT_RECOMMENDED_MARKER,
        )
        if position > title_bottom + 1
    ]


def find_bid_evaluation_report_target_pages(pages: list[PageText]) -> list[int]:
    """
    【函数功能】定位包含真实投标人排序表的页码，排除仅含目录标题的页面。
    :param pages: list[PageText]+按页码读取的评标报告页面
    :return: list[int]+排序表及必要续表的页码
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    Example: find_bid_evaluation_report_target_pages(pages)
    """
    ordered_pages = sorted(pages, key=lambda item: item.page_number)
    target_numbers: list[int] = []
    for index, page in enumerate(ordered_pages):
        page_text = compact_for_match(page.text)
        has_title = EVALUATION_REPORT_TITLE_MARKER in page_text
        has_company = bool(extract_company_names(page.text))
        if not has_title or not has_company:
            continue

        has_recommended_area = bool(_evaluation_report_recommended_positions(page))
        has_table_structure = "投标人名称" in page_text or has_recommended_area
        if not has_table_structure:
            has_table_structure = any(
                bool(extract_company_names(following_page.text))
                and (
                    "投标人名称" in compact_for_match(following_page.text)
                    or bool(_evaluation_report_recommended_positions(following_page))
                )
                for following_page in ordered_pages[index + 1 : index + 3]
            )
        if not has_table_structure:
            continue

        target_numbers.append(page.page_number)
        following_index = index + 1
        while not has_recommended_area and following_index < len(ordered_pages):
            following_page = ordered_pages[following_index]
            following_text = compact_for_match(following_page.text)
            has_following_company = bool(extract_company_names(following_page.text))
            has_following_structure = (
                "投标人名称" in following_text
                or EVALUATION_REPORT_RECOMMENDED_MARKER in following_text
            )
            if not has_following_company and not has_following_structure:
                break
            target_numbers.append(following_page.page_number)
            has_recommended_area = bool(_evaluation_report_recommended_positions(following_page))
            following_index += 1
        break
    return sorted(set(target_numbers))


def _evaluation_report_company_occurrences(pages: list[PageText]) -> list[CompanyOccurrence]:
    """
    【函数功能】仅提取评标排序表上半部分的投标企业。
    :param pages: list[PageText]+已定位的排序表及续表页面
    :return: list[CompanyOccurrence]+按排序表顺序去重的投标企业
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    Example: _evaluation_report_company_occurrences(pages)
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    started = False
    completed = False
    for page in sorted(pages, key=lambda item: item.page_number):
        title_positions = _evaluation_report_marker_positions(page, EVALUATION_REPORT_TITLE_MARKER)
        if not started:
            if not title_positions:
                continue
            started = True
        title_bottom = max(title_positions) if title_positions else float("-inf")
        recommended_positions = _evaluation_report_recommended_positions(page)
        recommended_top = min(recommended_positions) if recommended_positions else float("inf")
        for line in page.lines:
            if line.center_y <= title_bottom or line.center_y >= recommended_top - 1:
                continue
            for name in extract_company_names(line.text):
                key = normalize_text(name)
                if not key or key in seen:
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
        if recommended_positions:
            completed = True
        if completed:
            break
    return occurrences


def _evaluation_report_recommended_winner(pages: list[PageText]) -> CompanyOccurrence | None:
    """
    【函数功能】从推荐中标候选人区按第一名定位中标企业。
    :param pages: list[PageText]+已定位的排序表及续表页面
    :return: CompanyOccurrence|None+第一推荐候选人，无法定位时返回空
    :Author: gexinyan
    :CreateTime: 2026-07-14 16:00:00
    Example: _evaluation_report_recommended_winner(pages)
    """
    recommended_started = False
    for page in sorted(pages, key=lambda item: item.page_number):
        marker_positions = _evaluation_report_recommended_positions(page)
        marker_top = min(marker_positions) if marker_positions else float("inf")
        if marker_positions:
            recommended_started = True
        if not recommended_started:
            continue
        candidate_lines = [line for line in page.lines if line.center_y >= marker_top - 1]
        for index, rank_line in enumerate(candidate_lines):
            if EVALUATION_REPORT_FIRST_RANK_RE.search(compact_for_match(rank_line.text)) is None:
                continue
            nearby_lines = candidate_lines[max(0, index - 2) : index + 3]
            for candidate_line in nearby_lines:
                if candidate_line.bbox and rank_line.bbox:
                    row_tolerance = max(_line_height(candidate_line), _line_height(rank_line)) * 2.5
                    if abs(candidate_line.center_y - rank_line.center_y) > row_tolerance:
                        continue
                names = extract_company_names(candidate_line.text)
                if names:
                    return CompanyOccurrence(
                        name=names[0],
                        page_number=page.page_number,
                        evidence=candidate_line.text.strip()[:300],
                        confidence=(
                            candidate_line.confidence if page.method == "ocr" else 1.0
                        ),
                        method=page.method,
                    )
    return None


def _append_bid_candidate_occurrence(
    occurrences: list[CompanyOccurrence],
    seen: set[str],
    name: str,
    page: PageText,
    evidence: str,
    confidence: float | None = None,
) -> None:
    """
    【函数功能】将未重复的候选人公示表格企业加入提取结果。
    :param occurrences: list[CompanyOccurrence]+已提取的企业列表
    :param seen: set[str]+已提取企业的规范化名称集合
    :param name: str+待加入的企业名称
    :param page: PageText+企业所在页面
    :param evidence: str+企业名称对应的表格证据文本
    :param confidence: float|None+企业名称片段的最低置信度，默认使用页面置信度
    :return: None
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:20:00
    Example: _append_bid_candidate_occurrence([], set(), "测试建设有限公司", page, "1 测试建设有限公司")
    """
    validated_names = extract_company_names(name, strict=True)
    if not validated_names:
        return
    name = validated_names[0]
    key = normalize_text(name)
    if not key or key in seen:
        return
    seen.add(key)
    occurrences.append(
        CompanyOccurrence(
            name=name,
            page_number=page.page_number,
            evidence=evidence.strip()[:300],
            confidence=page.confidence if confidence is None else confidence,
            method=page.method,
        )
    )


def _incomplete_score_table_company_fragment(line: str) -> str:
    """
    【函数功能】从得分表换行残片中识别可与常见公司后缀续行拼接的未完成企业名称。
    :param line: str+得分表中的原始文本行
    :return: str+待与后续公司后缀拼接的企业名称残片，未命中时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:20:00
    Example: _incomplete_score_table_company_fragment("如东县水利电力建筑工程有限责任公   户")
    """
    incomplete_suffixes = ("有限责任公", "有限公", "有限", "工程", "有", "公")
    for segment in re.split(r"\s{2,}", line.strip()):
        compact = re.sub(r"^\d+\s*", "", _compact_bid_candidate_title(segment))
        if (
            len(compact) >= 6
            and compact.endswith(incomplete_suffixes)
            and not extract_company_names(compact, strict=True)
        ):
            return compact
    return ""


def _complete_score_table_company_fragment(fragment: str, line: str) -> str:
    """
    【函数功能】用后续文本行补齐得分表中的公司后缀拆行。
    :param fragment: str+前一条未完成企业名称片段
    :param line: str+后续候选文本行
    :return: str+可通过严格企业校验的完整名称，无法补齐时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _complete_score_table_company_fragment("测试建设有限", "公司")
    """
    compact = compact_for_match(line)
    continuations = ("有限责任公司", "有限公司", "限公司", "公司", "司")
    for continuation in continuations:
        if not compact.startswith(continuation):
            continue
        names = extract_company_names(f"{fragment}{continuation}", strict=True)
        if names:
            return names[0]
    return ""


def _is_bid_candidate_score_table_title(value: str) -> bool:
    """
    【函数功能】判断文本是否包含任一受支持的候选人得分表标题。
    :param value: str+待检查的原始标题文本
    :return: bool+是否命中标准或报价得分表标题
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:50:36
    Example: _is_bid_candidate_score_table_title("六、所有投标人投标报价及得分情况：")
    """
    compact = compact_for_match(value)
    return any(title in compact for title in BID_CANDIDATE_SCORE_TABLE_TITLES)


def _bid_candidate_score_header_label(line: OCRLine) -> str:
    """
    【函数功能】识别候选人得分表表头文字并返回规范标签。
    :param line: OCRLine+待识别的 OCR 表头行
    :return: str+命中的规范表头，未命中时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:50:36
    Example: _bid_candidate_score_header_label(OCRLine("投标人名称", 0.99, []))
    """
    compact = compact_for_match(line.text)
    return next(
        (label for label in BID_CANDIDATE_SCORE_TABLE_HEADERS if compact == compact_for_match(label)),
        "",
    )


def _find_bid_candidate_score_company_column(
    page: PageText,
    minimum_y: float,
) -> BidListUnitColumn | None:
    """
    【函数功能】在得分表标题下方依据相邻表头坐标定位投标人名称列。
    :param page: PageText+候选人公示 OCR 页面
    :param minimum_y: float+得分表标题纵坐标，仅在该位置下方查找表头
    :return: BidListUnitColumn|None+投标人名称列范围，无法定位时返回空
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:50:36
    Example: _find_bid_candidate_score_company_column(page, 100.0)
    """
    positioned_lines = [line for line in page.lines if line.bbox and line.center_y > minimum_y]
    for company_line in positioned_lines:
        if compact_for_match(company_line.text) != BID_CANDIDATE_SCORE_TABLE_COLUMN:
            continue
        row_tolerance = _line_height(company_line) * 3.0
        row_headers = [
            line
            for line in positioned_lines
            if abs(line.center_y - company_line.center_y) <= row_tolerance
            and _bid_candidate_score_header_label(line)
        ]
        left_headers = [line for line in row_headers if line.center_x < company_line.center_x]
        right_headers = [line for line in row_headers if line.center_x > company_line.center_x]
        if not left_headers or not right_headers:
            continue
        previous_header = max(left_headers, key=lambda line: line.center_x)
        next_header = min(right_headers, key=lambda line: line.center_x)
        left = (previous_header.center_x + company_line.center_x) / 2
        right = (company_line.center_x + next_header.center_x) / 2
        if left < company_line.center_x < right:
            return BidListUnitColumn(left, right, _line_bottom(company_line))
    return None


def _group_bid_candidate_company_lines(lines: list[OCRLine]) -> list[list[OCRLine]]:
    """
    【函数功能】按纵向间距将投标人名称列 OCR 片段重组为表格单元格。
    :param lines: list[OCRLine]+同一页面投标人名称列内的文字片段
    :return: list[list[OCRLine]]+按表格顺序排列的企业名称单元格片段组
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:50:36
    Example: _group_bid_candidate_company_lines([first_line, suffix_line])
    """
    groups: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda item: item.center_y):
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        join_tolerance = max(_line_height(previous), _line_height(line)) * 1.8
        if line.center_y - previous.center_y <= join_tolerance:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _find_positioned_bid_candidate_score_table_occurrences(
    pages: list[PageText],
) -> tuple[list[CompanyOccurrence], bool]:
    """
    【函数功能】按 OCR 表格坐标重组得分表投标人名称列，兼容单元格内多行企业名称。
    :param pages: list[PageText]+候选人公示页面
    :return: tuple[list[CompanyOccurrence], bool]+企业记录与是否成功定位名称列
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:50:36
    Example: _find_positioned_bid_candidate_score_table_occurrences([page])
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    table_started = False
    active_column: BidListUnitColumn | None = None
    column_located = False

    for page in sorted(pages, key=lambda item: item.page_number):
        positioned_lines = sorted(
            (line for line in page.lines if line.bbox),
            key=lambda item: (item.center_y, item.center_x),
        )
        if not positioned_lines:
            continue
        title_y = float("-inf") if table_started else float("inf")
        stop_y = float("inf")
        for line in positioned_lines:
            compact = compact_for_match(line.text)
            if not table_started and _is_bid_candidate_score_table_title(compact):
                table_started = True
                title_y = line.center_y
                continue
            if table_started and BID_CANDIDATE_SCORE_TABLE_STOP_RE.match(compact):
                stop_y = min(stop_y, line.center_y)

        if not table_started:
            continue
        column = _find_bid_candidate_score_company_column(page, title_y)
        if column is not None:
            active_column = column
            column_located = True
        if active_column is None:
            continue
        content_top = column.content_top if column is not None else float("-inf")
        company_lines = [
            line
            for line in positioned_lines
            if content_top < line.center_y < stop_y
            and active_column.left < line.center_x < active_column.right
        ]
        for group in _group_bid_candidate_company_lines(company_lines):
            joined_name = "".join(re.sub(r"\s+", "", line.text) for line in group)
            names = extract_company_names(joined_name, strict=True)
            if not names:
                continue
            _append_bid_candidate_occurrence(
                occurrences,
                seen,
                names[0],
                page,
                " ".join(line.text.strip() for line in group),
                min(line.confidence for line in group),
            )
        if stop_y != float("inf"):
            break
    return occurrences, column_located


def _find_native_score_table_column(
    layout_lines: list[str],
    minimum_index: int = 0,
) -> NativeScoreTableColumn | None:
    """
    【函数功能】从原生版式表头确定序号、投标人名称和报价列之间的字符范围。
    :param layout_lines: list[str]+保留前导空格的原生版式文本行
    :param minimum_index: int+允许定位表头的最小版式行号（默认0）
    :return: NativeScoreTableColumn|None+名称列字符范围，表头不完整时返回空
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _find_native_score_table_column(["序号  投标人名称  投标报价"])
    """
    for index, line in enumerate(layout_lines):
        if index < minimum_index:
            continue
        if "序号" not in line or BID_CANDIDATE_SCORE_TABLE_COLUMN not in line:
            continue
        sequence_start = line.find("序号")
        company_start = line.find(BID_CANDIDATE_SCORE_TABLE_COLUMN)
        if company_start <= sequence_start:
            continue
        following_positions: list[int] = []
        for nearby_line in layout_lines[max(0, index - 2) : index + 3]:
            for label in BID_CANDIDATE_SCORE_TABLE_HEADERS:
                if label in ("序号", BID_CANDIDATE_SCORE_TABLE_COLUMN):
                    continue
                position = nearby_line.find(label)
                if position > company_start:
                    following_positions.append(position)
        next_column_start = min(following_positions, default=company_start + 24)
        left = max(0, (sequence_start + len("序号") + company_start) // 2 - 1)
        right = max(company_start + len(BID_CANDIDATE_SCORE_TABLE_COLUMN), next_column_start + 14)
        return NativeScoreTableColumn(left, right, index)
    return None


def _native_layout_company_fragments(
    line: str,
    column: NativeScoreTableColumn,
) -> list[str]:
    """
    【函数功能】按 layout 字符列从一行中筛选企业名称或后缀片段。
    :param line: str+保留字符列空格的版式行
    :param column: NativeScoreTableColumn+名称列范围
    :return: list[str]+位于名称列内的中文片段
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _native_layout_company_fragments("  1  测试有限公司  100", column)
    """
    fragments: list[str] = []
    for match in re.finditer(r"\S+", line):
        text = match.group(0).strip()
        if not re.search(r"[\u4e00-\u9fff]", text):
            continue
        if match.start() < column.left or match.start() > column.right:
            continue
        compact = compact_for_match(text)
        if (
            compact in BID_CANDIDATE_SCORE_TABLE_HEADERS
            or _is_bid_candidate_score_table_title(compact)
            or BID_CANDIDATE_SCORE_TABLE_STOP_RE.match(compact)
        ):
            continue
        fragments.append(text)
    return fragments


def _native_fragment_y(page: PageText, value: str) -> float | None:
    """
    【函数功能】查找与 layout 片段匹配的原生 visitor 文字纵坐标，用作片段排序辅助。
    :param page: PageText+包含原生定位片段的页面
    :param value: str+待匹配的 layout 文字片段
    :return: float|None+匹配片段纵坐标，无匹配时返回空
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _native_fragment_y(page, "测试有限公司")
    """
    compact_value = compact_for_match(value)
    matched = [
        fragment.center_y
        for fragment in page.native_fragments
        if compact_value
        and (
            compact_value == compact_for_match(fragment.text)
            or compact_value in compact_for_match(fragment.text)
        )
    ]
    return matched[0] if matched else None


def _find_native_bid_candidate_score_table_occurrences(
    pages: list[PageText],
) -> tuple[list[CompanyOccurrence], bool]:
    """
    【函数功能】用原生 layout 字符列、visitor 纵坐标和序号锚点重组跨行企业名称。
    :param pages: list[PageText]+候选人公示页面
    :return: tuple[list[CompanyOccurrence], bool]+企业记录与是否成功定位原生名称列
    :Author: gexinyan
    :CreateTime: 2026-07-15 11:46:28
    Example: _find_native_bid_candidate_score_table_occurrences([page])
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    table_started = False
    active_column: NativeScoreTableColumn | None = None
    column_located = False

    for page in sorted(pages, key=lambda item: item.page_number):
        if not page.layout_text:
            continue
        layout_lines = page.layout_text.splitlines()
        title_indexes = [
            index
            for index, line in enumerate(layout_lines)
            if _is_bid_candidate_score_table_title(line)
        ]
        if title_indexes:
            table_started = True
        if not table_started:
            continue

        page_column = _find_native_score_table_column(
            layout_lines,
            max(title_indexes) + 1 if title_indexes else 0,
        )
        if page_column is not None:
            active_column = page_column
            column_located = True
        if active_column is None:
            continue

        content_start = (
            page_column.content_top + 1
            if page_column is not None
            else 0
        )
        if title_indexes:
            content_start = max(content_start, max(title_indexes) + 1)
        stop_indexes = [
            index
            for index, line in enumerate(layout_lines[content_start:], content_start)
            if BID_CANDIDATE_SCORE_TABLE_STOP_RE.match(compact_for_match(line))
        ]
        content_end = min(stop_indexes, default=len(layout_lines))
        anchors = [
            index
            for index, line in enumerate(layout_lines[content_start:content_end], content_start)
            if re.match(r"^\s*\d{1,3}(?:\s+|$)", line)
        ]
        grouped_fragments: dict[int, list[tuple[int, float | None, str]]] = {
            anchor: [] for anchor in anchors
        }
        continuation_fragments = {"司", "公司", "限公司", "有限公司", "责任公司"}
        for line_index in range(content_start, content_end):
            fragments = _native_layout_company_fragments(
                layout_lines[line_index],
                active_column,
            )
            if not fragments or not anchors:
                continue
            distances = [abs(line_index - anchor) for anchor in anchors]
            minimum_distance = min(distances)
            nearest_indexes = [
                index for index, distance in enumerate(distances) if distance == minimum_distance
            ]
            if len(nearest_indexes) == 1:
                target_anchor = anchors[nearest_indexes[0]]
            elif all(compact_for_match(fragment) in continuation_fragments for fragment in fragments):
                target_anchor = anchors[nearest_indexes[0]]
            else:
                target_anchor = anchors[nearest_indexes[-1]]
            grouped_fragments[target_anchor].extend(
                (line_index, _native_fragment_y(page, fragment), fragment)
                for fragment in fragments
            )

        for anchor in anchors:
            positioned_fragments = grouped_fragments[anchor]
            positioned_fragments.sort(
                key=lambda item: (
                    item[0],
                    -(item[1] if item[1] is not None else float("-inf")),
                )
            )
            joined = "".join(fragment for _, _, fragment in positioned_fragments)
            names = extract_company_names(joined, strict=True)
            if not names:
                continue
            evidence = " ".join(fragment for _, _, fragment in positioned_fragments)
            _append_bid_candidate_occurrence(
                occurrences,
                seen,
                names[0],
                page,
                evidence,
                1.0,
            )
        if stop_indexes:
            break
    return occurrences, column_located


def find_bid_candidate_score_table_occurrences(pages: list[PageText]) -> list[CompanyOccurrence]:
    """
    【函数功能】仅从候选人公示“所有投标人得分汇总表”中提取投标企业。
    :param pages: list[PageText]+候选人公示页面
    :return: list[CompanyOccurrence]+按得分表排名顺序提取的投标企业
    :Author: gexinyan
    :CreateTime: 2026-07-14 14:20:00
    Example: find_bid_candidate_score_table_occurrences([page])
    """
    native_occurrences, native_column_located = (
        _find_native_bid_candidate_score_table_occurrences(pages)
    )
    positioned_occurrences, column_located = (
        _find_positioned_bid_candidate_score_table_occurrences(pages)
    )
    occurrences = native_occurrences if native_column_located else positioned_occurrences
    seen = {normalize_text(occurrence.name) for occurrence in occurrences}
    table_started = False
    column_found = False
    pending_fragment = ""
    pending_page: PageText | None = None
    pending_evidence = ""
    pending_line_budget = 0

    for page in sorted(pages, key=lambda item: item.page_number):
        for line in page.lines:
            compact = compact_for_match(line.text)
            if not table_started:
                if _is_bid_candidate_score_table_title(compact):
                    table_started = True
                    column_found = BID_CANDIDATE_SCORE_TABLE_COLUMN in compact
                continue
            if BID_CANDIDATE_SCORE_TABLE_STOP_RE.match(compact):
                return occurrences
            if not column_found:
                if BID_CANDIDATE_SCORE_TABLE_COLUMN in compact:
                    column_found = True
                continue

            names = extract_company_names(line.text, strict=True)
            for name in names:
                _append_bid_candidate_occurrence(occurrences, seen, name, page, line.text)

            if pending_fragment:
                pending_line_budget -= 1
                completed_name = _complete_score_table_company_fragment(
                    pending_fragment,
                    line.text,
                )
                if completed_name:
                    if pending_page is not None:
                        _append_bid_candidate_occurrence(
                            occurrences,
                            seen,
                            completed_name,
                            pending_page,
                            f"{pending_evidence} {line.text}",
                        )
                    pending_fragment = ""
                    pending_page = None
                    pending_evidence = ""
                    pending_line_budget = 0
                elif pending_line_budget <= 0:
                    pending_fragment = ""
                    pending_page = None
                    pending_evidence = ""

            fragment = _incomplete_score_table_company_fragment(line.text)
            if fragment:
                pending_fragment = fragment
                pending_page = page
                pending_evidence = line.text
                pending_line_budget = 3
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


def _archive_metadata(pages: list[PageText]) -> tuple[str, str, str, str]:
    """
    【函数功能】提取备案资料的标段名称及标段编号，并拆分项目编号。
    :param pages: list[PageText]+备案资料候选页面
    :return: tuple[str, str, str, str]+项目名称、项目编号、标段编号、标段名称
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:30:00
    Example: _archive_metadata([page])
    """
    lines = [line.text.strip() for page in pages for line in page.lines if line.text.strip()]
    lot_name = _clean_project_value(_line_value(lines, LOT_LABELS))
    if not lot_name:
        lot_name = _clean_project_value(_line_value(lines, ARCHIVE_LOT_NAME_FALLBACK_LABELS))
    raw_lot_code = _line_value(lines, LOT_CODE_LABELS)
    if not raw_lot_code:
        # 备案材料中“项目编号”字段有时实际记录的是完整标段编号，按业务规则兜底使用。
        raw_lot_code = _line_value(lines, PROJECT_CODE_LABELS)
    code_match = re.search(r"[A-Za-z0-9][A-Za-z0-9_\-/－—–]{5,}", raw_lot_code)
    normalized_lot_code = normalize_text(code_match.group(0) if code_match else raw_lot_code)
    project_code, lot_code = split_project_and_lot_code(normalized_lot_code)
    return lot_name, project_code, lot_code, lot_name


def _archive_list_occurrences(pages: list[PageText]) -> list[CompanyOccurrence]:
    """
    【函数功能】重组备案资料投标人名单区段并提取完整企业名称。
    :param pages: list[PageText]+备案资料候选页面
    :return: list[CompanyOccurrence]+名单中的企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:30:00
    Example: _archive_list_occurrences([page])
    """
    occurrences: list[CompanyOccurrence] = []
    seen: set[str] = set()
    for page in sorted(pages, key=lambda item: item.page_number):
        collecting = False
        content_lines: list[str] = []
        evidence_lines: list[str] = []
        for line in page.lines:
            compact = compact_for_match(line.text)
            if not collecting:
                marker = next(
                    (item for item in ARCHIVE_PARTICIPANT_LIST_MARKERS if item in compact),
                    None,
                )
                if marker is None:
                    continue
                marker_position = compact.find(marker)
                compact_line = re.sub(r"\s+", "", line.text)
                content = compact_line[marker_position + len(marker) :]
                collecting = True
                if content:
                    content_lines.append(content)
                    evidence_lines.append(line.text.strip())
                continue

            if ARCHIVE_SECTION_HEADING_RE.match(compact):
                break
            content_lines.append(re.sub(r"\s+", "", line.text))
            evidence_lines.append(line.text.strip())

        if not collecting:
            continue
        joined = "".join(content_lines).strip(" ：:：")
        for fragment in ARCHIVE_LIST_SEPARATOR_RE.split(joined):
            names = extract_company_names(fragment, strict=True)
            for name in names:
                key = normalize_text(name)
                if not key or key in seen:
                    continue
                seen.add(key)
                occurrences.append(
                    CompanyOccurrence(
                        name=name,
                        page_number=page.page_number,
                        evidence="".join(evidence_lines)[:300],
                        confidence=min(
                            (line.confidence for line in page.lines if line.text.strip()),
                            default=0.0,
                        )
                        if page.method == "ocr"
                        else 1.0,
                        method=page.method,
                    )
                )
    return occurrences


def _explicit_company(
    pages: list[PageText],
    labels: tuple[str, ...],
    strict_company_filter: bool = False,
) -> CompanyOccurrence | None:
    """
    【函数功能】优先从“中标人”等字段行或通知书冒号称呼中提取企业。
    :param pages: list[PageText]+候选页面
    :param labels: tuple[str, ...]+明确企业字段标签
    :param strict_company_filter: bool+是否过滤叙述句中的伪企业名称
    :return: CompanyOccurrence|None+明确企业记录
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: _explicit_company(pages, ("中标人",))
    """
    for page in pages:
        for line in page.lines:
            compact = compact_for_match(line.text)
            names = extract_company_names(line.text, strict=strict_company_filter)
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
    force_review: bool = False,
    prefer_richest_metadata: bool = False,
) -> list[ExtractionRecord]:
    """
    【函数功能】将企业出现列表转换为统一解析记录并补齐项目信息和复核状态。
    :param occurrences: list[CompanyOccurrence]+企业出现列表
    :param pages: list[PageText]+用于提取项目元数据的页面
    :param context: ParserContext+解析上下文
    :param status_by_name: dict[str, str]|None+按企业名称指定中标状态
    :param rank_by_name: dict[str, str]|None+按企业名称指定排名
    :param unknown_requires_review: bool+未知状态是否进入复核
    :param force_review: bool+是否强制将全部记录送入人工复核
    :param prefer_richest_metadata: bool+是否优先使用信息更完整的项目字段
    :return: list[ExtractionRecord]+统一记录列表
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """
    project_name, project_code, lot_name = extract_project_metadata(
        pages,
        prefer_richest=prefer_richest_metadata,
    )
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
            or force_review
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

    ordered_pages = sorted(pages, key=lambda page: page.page_number)
    cover_text = "\n".join(page.text for page in ordered_pages if page.text)
    cover_fields = extract_tender_cover_fields(
        cover_text,
        prefer_title=any(page.method == "ocr" for page in ordered_pages),
    )
    occurrences: list[CompanyOccurrence] = []
    cover_companies = extract_company_names(cover_fields.company_name)
    if cover_companies:
        company_name = cover_companies[0]
        matched_page = ordered_pages[0] if ordered_pages else None
        matched_line = None
        company_key = normalize_text(company_name)
        for page in ordered_pages:
            for line in page.lines:
                if company_key and company_key in normalize_text(line.text):
                    matched_page = page
                    matched_line = line
                    break
            if matched_line is not None:
                break
        if matched_page is not None:
            occurrences.append(
                CompanyOccurrence(
                    name=company_name,
                    page_number=matched_page.page_number,
                    evidence=(matched_line.text if matched_line else cover_fields.company_name)[:300],
                    confidence=(
                        matched_line.confidence
                        if matched_line is not None and matched_page.method == "ocr"
                        else matched_page.confidence
                    ),
                    method=matched_page.method,
                )
            )
    if not occurrences:
        occurrences = find_company_occurrences(ordered_pages)
    if not occurrences:
        filename_companies = extract_company_names(context.pdf_path.stem)
        occurrences = [
            CompanyOccurrence(name, 1, context.pdf_path.stem, 0.85, "filename")
            for name in filename_companies
        ]
    status = determine_cover_award_status(context.relative_path)
    statuses = {normalize_text(item.name): status for item in occurrences}
    records = _records_from_occurrences(
        occurrences[:1],
        ordered_pages,
        context,
        statuses,
        unknown_requires_review=True,
    )
    for record in records:
        record.project_name = cover_fields.project_name or record.project_name
        record.project_code = cover_fields.project_code or record.project_code
        record.lot_code = cover_fields.lot_code or record.lot_code
        if cover_fields.project_code or cover_fields.lot_name:
            record.lot_name = cover_fields.lot_name
        needs_review = (
            not record.project_name
            or not record.company_name
            or record.confidence < context.confidence_threshold
            or record.award_status == "未知"
        )
        record.review_status = "待复核" if needs_review else "通过"
    return records


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
    target_numbers = set(find_bid_evaluation_report_target_pages(pages))
    target_pages = [page for page in pages if page.page_number in target_numbers]
    occurrences = _evaluation_report_company_occurrences(target_pages)
    winner = _evaluation_report_recommended_winner(target_pages)
    winner_name = normalize_text(winner.name) if winner else ""
    statuses = (
        {
            normalize_text(occurrence.name): (
                "是" if normalize_text(occurrence.name) == winner_name else "否"
            )
            for occurrence in occurrences
        }
        if winner_name
        else {}
    )
    ranks = {
        normalize_text(occurrence.name): str(index)
        for index, occurrence in enumerate(occurrences, start=1)
    }
    if winner and winner_name not in statuses:
        occurrences.append(winner)
        statuses[winner_name] = "是"
        ranks[winner_name] = "1"
    basic_info_project_name = extract_bid_evaluation_report_project_name(pages)
    records = _records_from_occurrences(
        occurrences,
        pages,
        context,
        statuses,
        ranks,
        unknown_requires_review=not bool(winner_name),
    )
    if basic_info_project_name:
        for record in records:
            record.project_name = basic_info_project_name
    else:
        for record in records:
            record.review_status = "待复核"
            record.evidence = f"未从基本情况一览表工程名称提取项目名称；{record.evidence}"[:300]
    if not target_pages:
        for record in records:
            record.review_status = "待复核"
            record.evidence = f"未定位投标人排序及推荐的中标候选人表；{record.evidence}"[:300]
    elif not winner:
        for record in records:
            record.review_status = "待复核"
            record.evidence = f"未识别推荐的中标候选人第一名；{record.evidence}"[:300]
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
    occurrences = find_bid_candidate_score_table_occurrences(pages)
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
    title_project_name = extract_bid_candidate_title_project_name(pages)
    records = _records_from_occurrences(occurrences, pages, context, statuses, ranks)
    for record in records:
        record.project_name = title_project_name
        record.project_code, record.lot_code = split_project_and_lot_code(record.project_code)
        needs_review = (
            not title_project_name
            or not record.company_name
            or record.confidence < context.confidence_threshold
        )
        record.review_status = "待复核" if needs_review else "通过"
        if not title_project_name:
            record.evidence = f"未识别到首页中标候选人公示标题；{record.evidence}"[:300]
    return records


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
    records = _records_from_occurrences([item for item in occurrences if item], pages, context, statuses)
    project_name = extract_award_notice_project_name(pages)
    project_code, lot_code = extract_award_notice_codes(pages)
    if project_name:
        for record in records:
            record.project_name = project_name
    for record in records:
        record.lot_name = record.project_name
    if project_code or lot_code:
        for record in records:
            if project_code:
                record.project_code = project_code
            if lot_code:
                record.lot_code = lot_code
    return records


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
    records = _records_from_occurrences([item for item in occurrences if item], pages, context, statuses)
    lot_code = extract_bid_announcement_lot_code(pages)
    for record in records:
        record.lot_code = lot_code
    return records


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
    occurrences, located_column = _find_bid_list_company_occurrences(pages)
    has_low_confidence = any(item.confidence < context.confidence_threshold for item in occurrences)
    requires_fallback = not located_column or not occurrences or has_low_confidence
    if requires_fallback:
        occurrences = find_company_occurrences(pages)
    records = _records_from_occurrences(
        occurrences,
        pages,
        context,
        force_review=requires_fallback,
    )
    lot_code = extract_bid_announcement_lot_code(pages)
    project_code, normalized_lot_code = split_project_and_lot_code(lot_code)
    for record in records:
        record.project_name = _project_name_without_lot_suffix(record.project_name)
        if project_code:
            record.project_code = project_code
        if normalized_lot_code:
            record.lot_code = normalized_lot_code
    return records


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
    winner = _explicit_company(
        winner_pages,
        ("中标人", "中标单位"),
        strict_company_filter=True,
    )
    winner_occurrences = find_company_occurrences(winner_pages, strict_company_filter=True)
    if winner is None and winner_occurrences:
        winner = winner_occurrences[0]
    participant_occurrences = _archive_list_occurrences(pages)
    if not participant_occurrences:
        participant_occurrences = find_company_occurrences(
            participant_pages,
            strict_company_filter=True,
        )
    occurrences = participant_occurrences + ([winner] if winner else [])
    deduplicated_occurrences: list[CompanyOccurrence] = []
    occurrence_names: set[str] = set()
    for occurrence in occurrences:
        key = normalize_text(occurrence.name)
        if key and key not in occurrence_names:
            occurrence_names.add(key)
            deduplicated_occurrences.append(occurrence)
    occurrences = deduplicated_occurrences
    if winner and normalize_text(winner.name) not in {normalize_text(item.name) for item in occurrences}:
        occurrences.insert(0, winner)
    winner_name = normalize_text(winner.name) if winner else ""
    statuses: dict[str, str] = {}
    for occurrence in occurrences:
        key = normalize_text(occurrence.name)
        statuses[key] = "是" if key == winner_name else ("否" if winner_name else "未知")
    records = _records_from_occurrences(
        occurrences,
        pages,
        context,
        statuses,
        unknown_requires_review=not bool(winner_name),
    )
    project_name, project_code, lot_code, lot_name = _archive_metadata(pages)
    for record in records:
        record.project_name = project_name
        record.project_code = project_code
        record.lot_code = lot_code
        record.lot_name = lot_name
        if not project_name or (not project_code and not lot_code):
            record.review_status = "待复核"
    return records


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
