"""投标文件封面的图像预处理、OCR 质量评估与字段解析策略。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


COVER_PROJECT_TITLE_KEYWORDS = (
    "高标准农田建设改造提升项目",
    "高标准农田建设工程",
    "高标准农田建设项目",
    "建设改造提升项目",
    "生态综合整治项目",
    "水利基建项目",
    "农田建设项目",
    "改造提升项目",
    "建设项目",
    "建设工程",
    "水利工程",
    "整治工程",
)
COVER_CODE_LABELS = ("标段编号", "项目编号", "招标编号", "交易编号", "编号")
COVER_PROJECT_LABELS = ("项目名称", "工程名称", "招标项目名称")
COVER_BIDDER_LABELS = ("投标人名称", "投标单位名称", "投标人", "参与单位")
COVER_PROJECT_STOP_PREFIXES = (
    "投标文件",
    "参与文件",
    *COVER_BIDDER_LABELS,
    "法定代表人",
    "其委托代理人",
    "日期",
)
COVER_GENERIC_PROJECT_TITLES = ("建设工程", "建设项目", "水利工程", "整治工程", "施工")
COVER_CHINESE_SECTION_RE = re.compile(r"第?[一二三四五六七八九十0-9]{1,3}(?:标段|合同段|施工标)")
COVER_PROJECT_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/]{5,}")
COVER_TEMPLATE_FIELD_RE = re.compile(
    r"[（(【\[]?\s*(?:工程项目名称|工程名称|项目名称|标段名称)\s*[）)】\]]?"
)
COVER_COMPANY_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·&\-]{2,80}?"
    r"(?:有限责任公司|股份有限公司|集团有限公司|有限公司|研究院|研究所|集团|公司)"
)


@dataclass(slots=True)
class TenderCoverFields:
    """
    【类功能】保存投标文件封面的专属字段解析结果。
    :Attributes:
        project_name: str+项目名称
        project_code: str+项目或交易编号
        lot_name: str+中文标段名称
        company_name: str+投标或参与单位名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    """

    project_name: str = ""
    project_code: str = ""
    lot_name: str = ""
    company_name: str = ""


def clean_cell(value: object) -> str:
    """
    【函数功能】移除空字符及全部空白，生成适合封面标签匹配的紧凑文本。
    :param value: object+待清理值
    :return: str+紧凑文本
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: clean_cell("投 标 人")
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\x00", "").strip()
    return re.sub(r"\s+", "", text).strip()


def clean_line(value: object) -> str:
    """
    【函数功能】清理文本行但保留单词和字段值之间的单个空格。
    :param value: object+待清理值
    :return: str+规范化文本行
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: clean_line("项目名称：  测试项目")
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\x00", "").strip()
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def cover_text_needs_ocr(text: str) -> bool:
    """
    【函数功能】判断封面原生文本是否过少或缺少封面业务标记，需要 OCR 兜底。
    :param text: str+PDF 首页原生文本
    :return: bool+是否需要 OCR
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: cover_text_needs_ocr("第2页/共728页")
    """
    compact_text = clean_cell(text)
    markers = (*COVER_PROJECT_LABELS, *COVER_CODE_LABELS, "投标文件", *COVER_BIDDER_LABELS)
    return len(compact_text) < 30 or not any(marker in compact_text for marker in markers)


def remove_red_seal(image: Any) -> Any:
    """
    【函数功能】将封面中满足红章颜色特征的像素置白，降低印章对 OCR 的干扰。
    :param image: Any+RGB uint8 图像数组
    :return: Any+去除红章后的 RGB 图像数组
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: remove_red_seal(image)
    """
    processed = image.copy()
    red_channel = processed[:, :, 0]
    green_channel = processed[:, :, 1]
    blue_channel = processed[:, :, 2]
    red_mask = (
        (red_channel > 110)
        & (green_channel < 170)
        & (blue_channel < 170)
        & ((red_channel.astype(int) - green_channel.astype(int)) > 25)
        & ((red_channel.astype(int) - blue_channel.astype(int)) > 25)
    )
    processed[red_mask] = [255, 255, 255]
    return processed


def suppress_red_seal_by_red_channel(image: Any) -> Any:
    """
    【函数功能】使用红色通道生成灰度图，在保留黑字的同时弱化红章。
    :param image: Any+RGB uint8 图像数组
    :return: Any+三通道灰度图像数组
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    Example: suppress_red_seal_by_red_channel(image)
    """
    red_channel = np.asarray(image)[:, :, 0]
    return np.repeat(red_channel[:, :, np.newaxis], 3, axis=2)


def crop_top_image(image: Any, ratio: float = 0.72) -> Any:
    """
    【函数功能】裁剪封面上部区域，优先保留项目标题、编号和封面标题。
    :param image: Any+RGB 图像数组
    :param ratio: float+顶部保留比例（默认0.72）
    :return: Any+顶部裁剪图像
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: crop_top_image(image)
    """
    crop_height = max(1, int(image.shape[0] * ratio))
    return image[:crop_height, :, :]


def crop_bottom_image(image: Any, start_ratio: float = 0.55) -> Any:
    """
    【函数功能】裁剪封面下部投标人区域，提高印章覆盖企业名称的 OCR 相对分辨率。
    :param image: Any+RGB 图像数组
    :param start_ratio: float+底部区域起始高度比例（默认0.55）
    :return: Any+底部裁剪图像
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: crop_bottom_image(image)
    """
    start = min(image.shape[0] - 1, max(0, int(image.shape[0] * start_ratio)))
    return image[start:, :, :]


def crop_company_name_image(
    image: Any,
    top_ratio: float = 0.66,
    bottom_ratio: float = 0.86,
    left_ratio: float = 0.12,
    right_ratio: float = 0.92,
) -> Any:
    """
    【函数功能】裁剪扫描封面中常见的底部投标单位字段区域。
    :param image: Any+RGB 图像数组
    :param top_ratio: float+裁剪上边界比例（默认0.66）
    :param bottom_ratio: float+裁剪下边界比例（默认0.86）
    :param left_ratio: float+裁剪左边界比例（默认0.12）
    :param right_ratio: float+裁剪右边界比例（默认0.92）
    :return: Any+投标单位字段区域图像
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    Example: crop_company_name_image(image)
    """
    height, width = image.shape[:2]
    top = min(height - 1, max(0, int(height * top_ratio)))
    bottom = min(height, max(top + 1, int(height * bottom_ratio)))
    left = min(width - 1, max(0, int(width * left_ratio)))
    right = min(width, max(left + 1, int(width * right_ratio)))
    return image[top:bottom, left:right, :]


def build_cover_image_variants(
    image: Any,
    include_top_crop: bool = False,
    retry_only: bool = False,
) -> list[tuple[str, Any]]:
    """
    【函数功能】生成封面 OCR 候选图，重试时仅保留定向裁剪候选以减少重复识别。
    :param image: Any+RGB 图像数组
    :param include_top_crop: bool+是否加入标题、底部和企业区域裁剪候选（默认False）
    :param retry_only: bool+是否只生成高分辨率定向重试候选（默认False）
    :return: list[tuple[str, Any]]+候选名称与图像数组
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: build_cover_image_variants(image, True)
    """
    candidates: list[tuple[str, Any]] = []
    if not retry_only:
        candidates.append(("original", image))
        seal_removed = remove_red_seal(image)
        if not np.array_equal(image, seal_removed):
            candidates.append(("remove_red_seal", seal_removed))
    if include_top_crop:
        top_image = crop_top_image(image)
        top_without_seal = remove_red_seal(top_image)
        candidates.append(
            (
                "top_crop_remove_red_seal" if not np.array_equal(top_image, top_without_seal) else "top_crop",
                top_without_seal if not np.array_equal(top_image, top_without_seal) else top_image,
            )
        )
        bottom_image = crop_bottom_image(image)
        bottom_without_seal = remove_red_seal(bottom_image)
        candidates.append(
            (
                (
                    "bottom_crop_remove_red_seal"
                    if not np.array_equal(bottom_image, bottom_without_seal)
                    else "bottom_crop"
                ),
                bottom_without_seal if not np.array_equal(bottom_image, bottom_without_seal) else bottom_image,
            )
        )
        company_image = crop_company_name_image(image)
        candidates.append(
            (
                "company_crop_red_channel",
                suppress_red_seal_by_red_channel(company_image),
            )
        )
    return candidates


def is_fragmented_cover_text(text: str, average_score: float = 1.0) -> bool:
    """
    【函数功能】判断 OCR 封面文本是否过短、缺少业务标记或置信度过低。
    :param text: str+OCR 封面文本
    :param average_score: float+OCR 平均置信度（默认1.0）
    :return: bool+是否属于低质量结果
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: is_fragmented_cover_text("程公司", 0.3)
    """
    compact_text = clean_cell(text)
    markers = (*COVER_PROJECT_LABELS, *COVER_CODE_LABELS, "投标文件", *COVER_BIDDER_LABELS)
    bidder_name = extract_company_name_candidate(text)
    has_company = is_complete_company_name_candidate(bidder_name)
    return (
        len(compact_text) < 30
        or not any(marker in compact_text for marker in markers)
        or not has_company
        or average_score < 0.45
    )


def is_complete_company_name_candidate(company_name: str) -> bool:
    """
    【函数功能】判断公司候选是否具有足以停止高分辨率重试的完整企业后缀。
    :param company_name: str+企业名称候选
    :return: bool+是否具有完整法定或机构后缀
    :Author: gexinyan
    :CreateTime: 2026-07-13 16:02:00
    Example: is_complete_company_name_candidate("某建设有限公司")
    """
    cleaned = strip_company_label(strip_stamp(company_name))
    complete_suffixes = (
        "有限责任公司",
        "股份有限公司",
        "集团有限公司",
        "有限公司",
        "股份公司",
        "集团公司",
        "研究院",
        "研究所",
        "集团",
    )
    return any(cleaned.endswith(suffix) for suffix in complete_suffixes)


def score_cover_ocr_text(text: str, average_score: float) -> float:
    """
    【函数功能】按封面关键词、文本长度和 OCR 置信度计算候选质量分。
    :param text: str+OCR 封面文本
    :param average_score: float+OCR 平均置信度
    :return: float+候选质量分，越大越可靠
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: score_cover_ocr_text("项目编号：WXHS20231114002-S01", 0.9)
    """
    compact_text = clean_cell(text)
    markers = (*COVER_PROJECT_LABELS, *COVER_CODE_LABELS, "投标文件", *COVER_BIDDER_LABELS)
    marker_score = sum(1 for marker in markers if marker in compact_text)
    company_score = 15 if extract_company_name_candidate(text) else 0
    return marker_score * 20 + company_score + min(len(compact_text), 240) * 0.2 + average_score * 10


def normalize_cover_text_for_parse(text: str) -> str:
    """
    【函数功能】规范化封面文本并修正少量高确定性 OCR 错字。
    :param text: str+PDF 或 OCR 封面文本
    :return: str+适合字段解析的文本
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: normalize_cover_text_for_parse("项且二标段")
    """
    normalized = text.replace("\r", "\n")
    replacements = {
        "项且": "项目",
        "项日": "项目",
        "项口": "项目",
        "项旦": "项目",
        "建没": "建设",
        "健设": "建设",
        "高标准农日": "高标准农田",
        "高标淮农田": "高标准农田",
        "注苏": "江苏",
    }
    for wrong, right in replacements.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def cover_lines(text: str) -> list[str]:
    """
    【函数功能】按行清理封面文本并保留非空行顺序。
    :param text: str+封面文本
    :return: list[str]+清理后的文本行
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: cover_lines("项目编号：ABC\n投标文件")
    """
    normalized = normalize_cover_text_for_parse(text)
    return [clean_line(line) for line in normalized.splitlines() if clean_line(line)]


def is_template_project_line(line: str) -> bool:
    """
    【函数功能】判断文本行是否为封面模板占位提示。
    :param line: str+待判断文本行
    :return: bool+是否属于模板提示
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: is_template_project_line("（工程项目名称）")
    """
    return clean_cell(line) in {
        "（工程项目名称）",
        "(工程项目名称)",
        "（工程名称）",
        "(工程名称)",
        "（标段名称）",
        "(标段名称)",
        "工程项目名称",
        "标段名称",
    }


def is_cover_title_noise_line(line: str) -> bool:
    """
    【函数功能】判断标题候选行是否为封面类型、页码或日期噪声。
    :param line: str+待判断文本行
    :return: bool+是否不应参与项目标题拼接
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: is_cover_title_noise_line("投标文件")
    """
    compact_line = clean_cell(line)
    if (
        "新点网" in compact_line
        or "网上招投标系统" in compact_line
        or ("新点" in compact_line and "招投标系统" in compact_line)
    ):
        return True
    if not compact_line or compact_line in {"投标文件", "参与文件", "商务标", "技术标", "资格审查资料"}:
        return True
    if any(compact_line.startswith(prefix) for prefix in COVER_PROJECT_STOP_PREFIXES):
        return True
    if is_template_project_line(compact_line):
        return True
    if re.fullmatch(r"第?\d+页[/／共\d+页]*", compact_line):
        return True
    return bool(re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", compact_line))


def has_project_title_keyword(value: str) -> bool:
    """
    【函数功能】判断候选文本是否包含常见工程或项目标题关键词。
    :param value: str+项目名称候选
    :return: bool+是否像项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: has_project_title_keyword("某高标准农田建设工程")
    """
    compact_value = clean_cell(value)
    if compact_value in COVER_GENERIC_PROJECT_TITLES:
        return False
    if any(keyword in compact_value for keyword in COVER_PROJECT_TITLE_KEYWORDS):
        return True
    return bool(re.search(r"高标准农田.*?(?:项目|工程)", compact_value))


def is_valid_project_name_candidate(value: str) -> bool:
    """
    【函数功能】判断项目名称候选是否完整且未混入投标主体、文档类型或项目编号。
    :param value: str+已清理项目名称候选
    :return: bool+是否可作为最终项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-14 12:00:00
    Example: is_valid_project_name_candidate("某高标准农田建设项目")
    """
    compact_value = clean_cell(value)
    if not compact_value or compact_value in COVER_GENERIC_PROJECT_TITLES:
        return False
    if any(marker in compact_value for marker in COVER_PROJECT_STOP_PREFIXES):
        return False
    if COVER_PROJECT_CODE_RE.search(compact_value):
        return False
    return has_project_title_keyword(compact_value)


def strip_bid_file_project_noise(value: str) -> str:
    """
    【函数功能】清除项目名称候选中的封面占位标签、字段和文件类型尾缀。
    :param value: str+项目名称候选
    :return: str+清理后的项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: strip_bid_file_project_noise("2024年项目（工程项目名称）")
    """
    text = clean_cell(value).strip("_:：")
    stop_labels = (
        "投标文件内容",
        "投标文件",
        "参与文件",
        "投标人名称",
        "投标人",
        "参与单位",
        "法定代表人",
        "其委托代理人",
        "日期",
    )
    for label in stop_labels:
        index = text.find(clean_cell(label))
        if index > 0:
            text = text[:index]
    text = COVER_TEMPLATE_FIELD_RE.sub("", text)
    text = re.sub(r"^(?:项目名称|工程名称|招标项目名称)[:：]*", "", text)
    text = re.sub(r"(?:工程)?施工招标$", "", text)
    text = re.sub(r"招标文件$", "", text)
    text = re.sub(r"(?:投标|参与)文件$", "", text)
    text = re.sub(r"(?:商务标|技术标)$", "", text)
    return text.strip("_:：")


def value_after_compact_label_joined(
    text: str,
    labels: Iterable[str],
    stop_labels: Iterable[str] = (),
    max_next_lines: int = 3,
) -> str:
    """
    【函数功能】按紧凑标签读取当前行值，并拼接有限数量的后续断行文本。
    :param text: str+封面文本
    :param labels: Iterable[str]+候选字段标签
    :param stop_labels: Iterable[str]+遇到后停止读取的字段标签
    :param max_next_lines: int+最多拼接后续行数（默认3）
    :return: str+字段值
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: value_after_compact_label_joined("项目名称：某\n项目", ["项目名称"])
    """
    lines = [clean_cell(line) for line in normalize_cover_text_for_parse(text).splitlines()]
    compact_labels = [clean_cell(label) for label in labels]
    compact_stops = [clean_cell(label) for label in stop_labels]
    for line_index, line in enumerate(lines):
        if not line:
            continue
        for label in compact_labels:
            label_index = line.find(label)
            if label_index < 0:
                continue
            next_index = label_index + len(label)
            previous_text = line[max(0, label_index - 2) : label_index]
            next_char = line[next_index : next_index + 1]
            if previous_text in {"工程", "标段"} or next_char in {"）", ")"}:
                continue
            values: list[str] = []
            tail = line[next_index:].strip("_:：")
            if tail:
                values.append(tail)
            for offset in range(1, max_next_lines + 1):
                following_index = line_index + offset
                if following_index >= len(lines):
                    break
                value = lines[following_index].strip("_:：")
                if not value:
                    continue
                if any(value.startswith(stop) for stop in compact_stops):
                    break
                values.append(value)
            return "".join(values)
    return ""


def extract_project_name_before_code(text: str) -> str:
    """
    【函数功能】从“标题区域+项目编号”版式中向前读取项目名称。
    :param text: str+封面文本
    :return: str+项目名称候选
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_project_name_before_code("某建设工程\n项目编号：ABC123\n投标文件")
    """
    lines = cover_lines(text)
    compact_labels = [clean_cell(label) for label in COVER_CODE_LABELS]
    for line_index, line in enumerate(lines):
        if not any(label in clean_cell(line) for label in compact_labels):
            continue
        pieces: list[str] = []
        for previous_line in reversed(lines[max(0, line_index - 40) : line_index]):
            compact_previous = clean_cell(previous_line)
            if any(marker in compact_previous for marker in (*COVER_BIDDER_LABELS, "法定代表人", "日期")):
                break
            if is_template_project_line(compact_previous):
                continue
            if is_cover_title_noise_line(compact_previous):
                continue
            if len(compact_previous) <= 3:
                continue
            if COVER_PROJECT_CODE_RE.fullmatch(compact_previous):
                continue
            pieces.insert(0, compact_previous)
            if has_project_title_keyword(compact_previous):
                break
        candidate = strip_bid_file_project_noise("".join(pieces))
        if is_valid_project_name_candidate(candidate):
            return candidate
    return ""


def extract_chinese_bid_section(text: str) -> str:
    """
    【函数功能】从项目名称、标段名称或标题中提取中文标段描述。
    :param text: str+封面文本
    :return: str+中文标段描述
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_chinese_bid_section("某项目二标段工程施工招标")
    """
    normalized = normalize_cover_text_for_parse(text)
    label_value = value_after_compact_label_joined(
        normalized,
        labels=("标段名称", "项目名称", "工程名称"),
        stop_labels=("投标文件内容", "投标文件", "参与文件", "投标人", "参与单位", "日期"),
    )
    for candidate in (label_value, extract_bid_file_title_project_name(normalized), clean_cell(normalized)):
        match = COVER_CHINESE_SECTION_RE.search(clean_cell(candidate))
        if match:
            return match.group(0)
    return ""


def strip_section_from_project_name(project_name: str, bid_section: str) -> str:
    """
    【函数功能】从项目名称尾部保守移除已经单独识别的中文标段。
    :param project_name: str+项目名称候选
    :param bid_section: str+中文标段名称
    :return: str+去除重复标段后的项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: strip_section_from_project_name("某项目二标段", "二标段")
    """
    cleaned_project = strip_bid_file_project_noise(project_name)
    cleaned_section = clean_cell(bid_section)
    if not cleaned_project or not COVER_CHINESE_SECTION_RE.fullmatch(cleaned_section):
        return cleaned_project
    pattern = re.escape(cleaned_section) + r"(?:工程施工招标|施工招标|工程)?$"
    stripped = re.sub(pattern, "", cleaned_project)
    return stripped or cleaned_project


def value_after_compact_label(
    text: str,
    labels: Iterable[str],
    stop_labels: Iterable[str] = (),
    max_next_lines: int = 2,
) -> str:
    """
    【函数功能】按紧凑标签读取同一行或下一非空行的字段值。
    :param text: str+封面文本
    :param labels: Iterable[str]+候选字段标签
    :param stop_labels: Iterable[str]+停止字段标签
    :param max_next_lines: int+最多向后读取行数（默认2）
    :return: str+字段值
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: value_after_compact_label("项目编号：ABC", ["项目编号"])
    """
    lines = [clean_cell(line) for line in normalize_cover_text_for_parse(text).splitlines()]
    compact_labels = [clean_cell(label) for label in labels]
    compact_stops = [clean_cell(label) for label in stop_labels]
    for line_index, line in enumerate(lines):
        if not line:
            continue
        for label in compact_labels:
            search_start = 0
            while True:
                label_index = line.find(label, search_start)
                if label_index < 0:
                    break
                next_index = label_index + len(label)
                previous_text = line[max(0, label_index - 2) : label_index]
                next_char = line[next_index : next_index + 1]
                if previous_text in {"工程", "标段"} or next_char in {"）", ")"}:
                    search_start = label_index + 1
                    continue
                tail = line[next_index:].strip("_:：")
                if tail:
                    return tail
                for offset in range(1, max_next_lines + 1):
                    following_index = line_index + offset
                    if following_index >= len(lines):
                        break
                    value = lines[following_index].strip("_:：")
                    if not value:
                        continue
                    if any(value.startswith(stop) for stop in compact_stops):
                        return ""
                    return value
                break
    return ""


def extract_bid_file_section(text: str) -> str:
    """
    【函数功能】从封面提取项目编号、交易编号或中文标段描述。
    :param text: str+封面文本
    :return: str+编号或中文标段
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_bid_file_section("项目编号：WXHS20231129001-S01")
    """
    normalized = normalize_cover_text_for_parse(text)
    value = value_after_compact_label(
        normalized,
        labels=COVER_CODE_LABELS,
        stop_labels=("项目名称", "工程名称", "投标文件", "参与文件", "投标人", "参与单位", "日期"),
    )
    match = COVER_PROJECT_CODE_RE.search(clean_cell(value))
    return match.group(0) if match else extract_chinese_bid_section(normalized)


def extract_bid_file_title_project_name(text: str) -> str:
    """
    【函数功能】从封面标题区域提取项目名称候选。
    :param text: str+封面文本
    :return: str+标题项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_bid_file_title_project_name("2024年某建设项目\n投标文件")
    """
    prefix_lines: list[str] = []
    compact_labels = [clean_cell(value) for value in COVER_CODE_LABELS]
    for line in cover_lines(text):
        compact_line = clean_cell(line)
        if (
            any(compact_line.startswith(prefix) for prefix in ("投标文件", "参与文件"))
            or any(label in compact_line for label in compact_labels)
            or COVER_PROJECT_CODE_RE.fullmatch(compact_line)
        ):
            break
        if is_template_project_line(compact_line):
            continue
        if is_cover_title_noise_line(compact_line):
            if prefix_lines:
                break
            continue
        prefix_lines.append(compact_line)
    cleaned = strip_bid_file_project_noise("".join(prefix_lines))
    return cleaned if is_valid_project_name_candidate(cleaned) else ""


def extract_bid_file_project_name(text: str, prefer_title: bool = False) -> str:
    """
    【函数功能】按明确字段、顶部标题和编号前标题顺序提取项目名称。
    :param text: str+封面文本
    :param prefer_title: bool+保留的调用兼容参数，明确字段始终优先（默认False）
    :return: str+项目名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_bid_file_project_name("项目名称：2024年某建设项目")
    """
    normalized = normalize_cover_text_for_parse(text)
    label_candidate = strip_bid_file_project_noise(
        value_after_compact_label_joined(
            normalized,
            labels=COVER_PROJECT_LABELS,
            stop_labels=(
                *COVER_CODE_LABELS,
                "投标文件内容",
                "投标文件",
                "参与文件",
                *COVER_BIDDER_LABELS,
                "法定代表人",
                "日期",
            ),
        )
    )
    before_code_candidate = extract_project_name_before_code(normalized)
    title_candidate = extract_bid_file_title_project_name(normalized)
    candidates = [label_candidate, title_candidate, before_code_candidate]
    for candidate in candidates:
        cleaned = strip_bid_file_project_noise(candidate)
        if is_valid_project_name_candidate(cleaned):
            return cleaned
    return ""


def strip_stamp(value: str) -> str:
    """
    【函数功能】清理企业名称后的公章、单位章和盖章提示。
    :param value: str+企业名称文本
    :return: str+清理后的企业名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: strip_stamp("某公司（盖公章）")
    """
    cleaned = re.sub(r"\s*\(公章\)\s*", "", value)
    cleaned = re.sub(r"\s*\(盖单位章\)\s*", "", cleaned)
    cleaned = re.sub(r"\s*[（(]\s*盖?公章\s*[）)]\s*", "", cleaned)
    cleaned = re.sub(r"\s*[（(]\s*盖?单位章\s*[）)]\s*", "", cleaned)
    cleaned = re.sub(r"\s*盖?公章[）)]?\s*", "", cleaned)
    cleaned = re.sub(r"\s*盖?单位章[）)]?\s*", "", cleaned)
    return clean_cell(cleaned)


def strip_company_label(value: str) -> str:
    """
    【函数功能】清除企业候选前粘连的投标人标签及高频 OCR 近似标签。
    :param value: str+企业名称候选行
    :return: str+移除标签后的企业候选
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    Example: strip_company_label("参与单江苏某建设有限公司")
    """
    cleaned = clean_cell(value).strip("_:：")
    label_pattern = (
        r"^(?:投标单位名称|投标人名称|参与单位|投标单位|投标人|参与单[位立]?|投标单[位立]?)[_:：]*"
    )
    return re.sub(label_pattern, "", cleaned).strip("_:：")


def extract_bid_file_bidder_name(text: str) -> str:
    """
    【函数功能】从投标人、投标单位或参与单位标签提取企业名称。
    :param text: str+封面文本
    :return: str+企业名称
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_bid_file_bidder_name("投标人：某公司（盖公章）")
    """
    value = value_after_compact_label(
        text,
        labels=COVER_BIDDER_LABELS,
        stop_labels=("法定代表人", "其委托代理人", "日期"),
        max_next_lines=1,
    )
    return strip_stamp(value)


def extract_company_name_candidate(text: str) -> str:
    """
    【函数功能】优先按投标人标签提取公司，失败时从独立 OCR 行保守匹配企业后缀。
    :param text: str+封面或局部裁剪 OCR 文本
    :return: str+企业名称，未找到时返回空字符串
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_company_name_candidate("参与单位：某建设有限公司")
    """
    labeled = strip_company_label(strip_stamp(extract_bid_file_bidder_name(text)))
    labeled_match = COVER_COMPANY_RE.search(labeled)
    if labeled_match:
        return labeled_match.group(0)
    for line in cover_lines(text):
        cleaned_line = strip_company_label(strip_stamp(line))
        match = COVER_COMPANY_RE.search(cleaned_line)
        if match:
            return strip_stamp(match.group(0))
    return ""


def score_company_name_candidate(company_name: str, confidence: float) -> float:
    """
    【函数功能】按企业法定后缀完整度、长度和 OCR 行置信度评价公司候选。
    :param company_name: str+已清理的企业名称候选
    :param confidence: float+候选所在 OCR 行置信度
    :return: float+公司候选质量分，越大越可靠
    :Author: gexinyan
    :CreateTime: 2026-07-13 15:45:00
    Example: score_company_name_candidate("某建设有限公司", 0.95)
    """
    cleaned = strip_company_label(strip_stamp(company_name))
    suffix_scores = (
        ("有限责任公司", 50.0),
        ("股份有限公司", 50.0),
        ("集团有限公司", 48.0),
        ("有限公司", 45.0),
        ("股份公司", 42.0),
        ("集团公司", 42.0),
        ("研究院", 38.0),
        ("研究所", 38.0),
        ("集团", 32.0),
        ("公司", 25.0),
    )
    suffix_score = next((score for suffix, score in suffix_scores if cleaned.endswith(suffix)), 0.0)
    length_score = min(len(cleaned), 30) * 0.2
    return suffix_score + length_score + max(0.0, min(confidence, 1.0)) * 20


def extract_tender_cover_fields(text: str, prefer_title: bool = False) -> TenderCoverFields:
    """
    【函数功能】统一提取投标封面的项目名称、编号、中文标段和企业名称。
    :param text: str+封面全文
    :param prefer_title: bool+是否优先标题项目名称（默认False）
    :return: TenderCoverFields+封面字段结果
    :Author: gexinyan
    :CreateTime: 2026-07-13 14:20:13
    Example: extract_tender_cover_fields("项目名称：某项目\n投标人：某公司")
    """
    section = extract_bid_file_section(text)
    project_name = extract_bid_file_project_name(text, prefer_title=prefer_title)
    project_name = strip_section_from_project_name(project_name, section)
    if COVER_PROJECT_CODE_RE.fullmatch(clean_cell(section)):
        project_code = clean_cell(section)
        lot_name = ""
    else:
        project_code = ""
        lot_name = clean_cell(section)
    return TenderCoverFields(
        project_name=project_name,
        project_code=project_code,
        lot_name=lot_name,
        company_name=extract_company_name_candidate(text),
    )
