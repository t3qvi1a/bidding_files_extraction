"""招投标解析通用文本、分类和规范化工具。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from bidding_ocr.models import CATEGORIES


DIRECTORY_CATEGORY_ALIASES = {
    "tender_cover": "tender_cover",
    "bid_evaluation_report": "bid_evaluation_report",
    "bid_candidates": "bid_candidates",
    "award_notice": "award_notice",
    "bid_announcement": "bid_announcement",
    "bid_list": "bid_list",
    "archive_info": "archive_info",
    "archived_info": "archive_info",
}

FILENAME_CATEGORY_KEYWORDS = (
    ("award_notice", ("中标通知书", "awardnotice")),
    ("bid_announcement", ("中标人公告", "bidannouncement")),
    ("bid_candidates", ("中标候选人", "中标公示", "bidcandidates")),
    ("bid_evaluation_report", ("评标报告", "bidevaluationreport")),
    ("bid_list", ("投标单位名单", "bidlist")),
    (
        "archive_info",
        (
            "备案资料",
            "归档资料",
            "备案材料",
            "归档材料",
            "备案",
            "归档",
            "archiveinfo",
            "archivedinfo",
        ),
    ),
)

COVER_FILE_NAMES = {"封面", "1"}
COVER_PATH_KEYWORDS = ("第一信封", "第一封信", "投标文件", "中标单位", "未中标")
COVER_TITLE_KEYWORDS = ("投标文件", "参与文件")
COVER_FIELD_KEYWORDS = ("项目名称", "项目编号", "投标人", "参与单位")

COMPANY_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·&\-]{2,80}?"
    r"(?:有限责任公司|股份有限公司|集团有限公司|有限公司|工程公司|集团公司|研究院|研究所|集团|公司)"
)


def normalize_text(value: str) -> str:
    """
    【函数功能】统一全半角、空白和中文括号，便于文本匹配与去重。
    :param value: str+待规范化文本
    :return: str+规范化后的文本
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: normalize_text(" 江苏（测试） 公司 ")
    """
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", normalized).strip("，,。；;:：")


def compact_for_match(value: str) -> str:
    """
    【函数功能】生成忽略空白和常见标点的关键词匹配文本。
    :param value: str+原始文本
    :return: str+用于关键词检索的紧凑文本
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: compact_for_match("中 标 通知书")
    """
    return re.sub(r"[\s:：,，。；;()（）\-_]", "", unicodedata.normalize("NFKC", value or ""))


def is_readable_chinese_text(value: str, minimum_length: int = 20) -> bool:
    """
    【函数功能】判断 PDF 文本层是否包含足够可读中文，识别乱码并触发 OCR。
    :param value: str+PDF 文本层内容
    :param minimum_length: int+最少非空字符数量（默认20）
    :return: bool+是否可直接使用文本层
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: is_readable_chinese_text("项目名称：测试项目")
    """
    compact = re.sub(r"\s+", "", value or "")
    if len(compact) < minimum_length or "�" in compact:
        return False
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    return chinese_count >= 6 and chinese_count / len(compact) >= 0.12


def classify_pdf(pdf_path: Path, input_root: Path, page_count: int, first_page_text: str = "") -> str:
    """
    【函数功能】依据中文文件名、封面页数、路径语义和兼容目录判定 PDF 类别。
    :param pdf_path: Path+PDF 文件路径
    :param input_root: Path+输入根目录
    :param page_count: int+PDF 页数
    :param first_page_text: str+可选首页文本
    :return: str+标准文件类别，无法识别时返回 unknown
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: classify_pdf(Path("pdf_files/award_notice/a.pdf"), Path("pdf_files"), 1)
    """
    normalized_stem = normalize_text(pdf_path.stem).lower()
    filename_target = compact_for_match(normalized_stem)
    for category, keywords in FILENAME_CATEGORY_KEYWORDS:
        if any(keyword in filename_target for keyword in keywords):
            return category

    relative_parts = _relative_directory_parts(pdf_path, input_root)
    for part in reversed(relative_parts):
        alias = DIRECTORY_CATEGORY_ALIASES.get(part.lower())
        if alias:
            return alias

    if normalized_stem not in COVER_FILE_NAMES or not 1 <= page_count <= 3:
        return "unknown"
    if normalized_stem == "封面":
        return "tender_cover"
    if relative_parts and normalize_text(relative_parts[-1]) == "施工组织设计":
        return "unknown"
    normalized_path = compact_for_match("/".join(relative_parts))
    if any(keyword in normalized_path for keyword in COVER_PATH_KEYWORDS):
        return "tender_cover"
    return "tender_cover" if is_tender_cover_text(first_page_text) else "unknown"


def _relative_directory_parts(pdf_path: Path, input_root: Path) -> tuple[str, ...]:
    """
    【函数功能】获取 PDF 相对于输入根目录的父目录片段，兼容不同路径表示形式。
    :param pdf_path: Path+待分类 PDF 路径
    :param input_root: Path+输入根目录
    :return: tuple[str, ...]+从外到内排列的相对父目录名称
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:12:03
    Example: _relative_directory_parts(Path("input/a/b.pdf"), Path("input"))
    """
    try:
        return pdf_path.resolve().relative_to(input_root.resolve()).parts[:-1]
    except (OSError, ValueError):
        return pdf_path.parts[:-1]


def is_tender_cover_text(value: str) -> bool:
    """
    【函数功能】判断首页原生文本是否包含足以确认投标封面的标题或字段组合。
    :param value: str+PDF 首页原生文本
    :return: bool+是否具有投标封面特征
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:12:03
    Example: is_tender_cover_text("投标文件\n项目名称：测试项目")
    """
    target = compact_for_match(value)
    if any(keyword in target for keyword in COVER_TITLE_KEYWORDS):
        return True
    matched_fields = sum(keyword in target for keyword in COVER_FIELD_KEYWORDS)
    return matched_fields >= 2


def determine_cover_award_status(relative_path: str) -> str:
    """
    【函数功能】按路径关键词判断投标封面对应企业是否中标。
    :param relative_path: str+PDF 相对路径
    :return: str+是、否或未知
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: determine_cover_award_status("未中标单位投标文件/封面.pdf")
    """
    normalized = compact_for_match(relative_path)
    if "未中标单位投标文件" in normalized or "未中标" in normalized:
        return "否"
    if "中标单位投标文件" in normalized or "中标资料" in normalized:
        return "是"
    return "未知"


def clean_company_name(value: str) -> str:
    """
    【函数功能】清理公司名称前的序号、字段标签和 OCR 标点。
    :param value: str+OCR 或文本层识别到的公司文本
    :return: str+清理后的公司名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: clean_company_name("1. 中标人：江苏测试有限公司")
    """
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"^.*?(?:中标人|中标单位|投标人名称|投标人|参与单位|单位名称)\s*[:：]?", "", value)
    value = re.sub(r"^[\s\d一二三四五六七八九十、.．()（）-]+", "", value)
    match = COMPANY_PATTERN.search(value.replace(" ", ""))
    return normalize_text(match.group(0)) if match else ""


def extract_company_names(value: str, strict: bool = False) -> list[str]:
    """
    【函数功能】从一段文字中提取并按出现顺序去重企业名称，可启用严格过滤。
    :param value: str+待解析文本
    :param strict: bool+是否过滤叙述性文本中的泛化企业名称
    :return: list[str]+企业名称列表
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:00:00
    Example: extract_company_names("甲有限公司、乙集团有限公司")
    """
    compact = unicodedata.normalize("NFKC", value or "").replace(" ", "")
    names: list[str] = []
    for match in COMPANY_PATTERN.finditer(compact):
        name = clean_company_name(match.group(0)) or normalize_text(match.group(0))
        if strict:
            name = _clean_strict_company_candidate(name)
        if name and name not in names:
            names.append(name)
    return names


def _clean_strict_company_candidate(value: str) -> str:
    """
    【函数功能】清理备案资料叙述句中的伪企业名称候选。
    :param value: str+普通企业正则提取结果
    :return: str+可信企业名称，无法确认时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-15 10:00:00
    Example: _clean_strict_company_candidate("合同经江苏测试有限公司")
    """
    candidate = normalize_text(value)
    if not candidate:
        return ""

    # 这些词通常说明企业后缀出现在合同叙述中，而不是企业字段或名单单元格中。
    narrative_markers = ("委托", "代理的", "须在", "合同经", "依法", "协调", "见证")
    marker_positions = [candidate.rfind(marker) for marker in narrative_markers]
    marker_position = max(marker_positions, default=-1)
    if marker_position >= 0:
        candidate = candidate[marker_position + max(
            len(marker) for marker in narrative_markers
            if candidate.rfind(marker) == marker_position
        ):]

    generic_names = {"公司", "有限公司", "有限责任公司", "我公司", "本公司", "该公司"}
    if candidate in generic_names:
        return ""

    company_suffixes = ("有限责任公司", "股份有限公司", "集团有限公司", "有限公司", "工程公司", "集团公司", "研究院", "研究所", "集团", "公司")
    suffix = next((item for item in company_suffixes if candidate.endswith(item)), "")
    prefix = candidate[: -len(suffix)] if suffix else ""
    if len(prefix) < 2 or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", prefix):
        return ""
    return candidate


def validate_category(category: str) -> str:
    """
    【函数功能】校验分类名称是否属于标准类别集合。
    :param category: str+分类名称
    :return: str+合法分类名称
    :raises ValueError: 分类名称不受支持时触发
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: validate_category("award_notice")
    """
    if category not in CATEGORIES:
        raise ValueError(f"不支持的 PDF 类别：{category}")
    return category
