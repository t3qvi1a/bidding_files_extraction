"""招投标 PDF 解析数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CATEGORIES = (
    "tender_cover",
    "bid_evaluation_report",
    "bid_candidates",
    "award_notice",
    "bid_announcement",
    "bid_list",
    "archive_info",
)

CSV_FIELDS = (
    "序号",
    "项目名称",
    "项目编号",
    "标段编号",
    "标段名称",
    "公司名称",
    "中标与否",
    "投标排名",
    "文件类别",
    "依据文件路径",
    "来源页码",
    "提取方式",
    "证据文本",
    "置信度",
    "复核状态",
    "解析结果生成日期时间",
)


@dataclass(slots=True)
class ProcessingConfig:
    """
    【类功能】保存 PDF 渲染、OCR、缓存及复核阈值配置。
    :Attributes:
        dpi: int+普通页面高精度 OCR 分辨率
        archive_scan_dpi: int+备案资料全文粗检分辨率
        ocr_confidence_threshold: float+低于该值时进入人工复核
        force_ocr: bool+是否忽略已有 OCR 缓存
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    dpi: int = 300
    archive_scan_dpi: int = 150
    ocr_confidence_threshold: float = 0.80
    force_ocr: bool = False


@dataclass(slots=True)
class OCRLine:
    """
    【类功能】表示带坐标与置信度的一条 OCR 文本。
    :Attributes:
        text: str+识别文本
        confidence: float+识别置信度
        bbox: list[list[float]]+四点坐标框
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    text: str
    confidence: float
    bbox: list[list[float]] = field(default_factory=list)

    @property
    def center_x(self) -> float:
        """
        【方法功能】计算文字框横向中心坐标。
        :return: float+横向中心坐标，无坐标时返回0
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sum(point[0] for point in self.bbox) / len(self.bbox) if self.bbox else 0.0

    @property
    def center_y(self) -> float:
        """
        【方法功能】计算文字框纵向中心坐标。
        :return: float+纵向中心坐标，无坐标时返回0
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sum(point[1] for point in self.bbox) / len(self.bbox) if self.bbox else 0.0


@dataclass(slots=True)
class PageText:
    """
    【类功能】保存一个 PDF 页面的文本及来源信息。
    :Attributes:
        page_number: int+从1开始的 PDF 页码
        text: str+页面合并文本
        lines: list[OCRLine]+按阅读顺序排列的文字行
        method: str+文本提取方式，取值 text 或 ocr
        dpi: int+OCR 分辨率，文本层提取时为0
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    page_number: int
    text: str
    lines: list[OCRLine]
    method: str
    dpi: int = 0

    @property
    def confidence(self) -> float:
        """
        【方法功能】计算页面所有识别行的平均置信度。
        :return: float+平均置信度，文本层页面返回1
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        if self.method == "text":
            return 1.0
        values = [line.confidence for line in self.lines if line.text.strip()]
        return sum(values) / len(values) if values else 0.0


@dataclass(slots=True)
class ExtractionRecord:
    """
    【类功能】表示最终 CSV 中的一条企业投标解析记录。
    :Attributes:
        project_name: str+项目名称
        project_code: str+项目编号
        lot_code: str+标段编号
        lot_name: str+标段名称
        company_name: str+公司名称
        award_status: str+中标状态
        rank: str+投标排名
        category: str+文件类别
        source_path: str+依据文件相对路径
        source_pages: str+来源页码
        extraction_method: str+提取方式
        evidence: str+关键证据文本
        confidence: float+综合置信度
        review_status: str+复核状态
        generated_at: str+解析生成时间
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    project_name: str = ""
    project_code: str = ""
    lot_code: str = ""
    lot_name: str = ""
    company_name: str = ""
    award_status: str = "未知"
    rank: str = ""
    category: str = ""
    source_path: str = ""
    source_pages: str = ""
    extraction_method: str = ""
    evidence: str = ""
    confidence: float = 0.0
    review_status: str = "通过"
    generated_at: str = ""

    def to_csv_row(self, sequence: int) -> dict[str, Any]:
        """
        【方法功能】将解析记录转换为固定中文表头的 CSV 行。
        :param sequence: int+当前 CSV 中从1开始的序号
        :return: dict[str, Any]+符合统一字段规范的字典
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return {
            "序号": sequence,
            "项目名称": self.project_name,
            "项目编号": self.project_code,
            "标段编号": self.lot_code,
            "标段名称": self.lot_name,
            "公司名称": self.company_name,
            "中标与否": self.award_status,
            "投标排名": self.rank,
            "文件类别": self.category,
            "依据文件路径": self.source_path,
            "来源页码": self.source_pages,
            "提取方式": self.extraction_method,
            "证据文本": self.evidence,
            "置信度": f"{self.confidence:.4f}",
            "复核状态": self.review_status,
            "解析结果生成日期时间": self.generated_at,
        }


@dataclass(slots=True)
class FileProcessSummary:
    """
    【类功能】记录单个 PDF 的解析统计与错误信息。
    :Attributes:
        path: str+PDF 相对路径
        category: str+文件类别
        pages: int+总页数
        ocr_pages: list[int]+执行过 OCR 的页码
        records: int+生成记录数量
        review_records: int+待复核记录数量
        status: str+文件处理状态
        error: str+异常摘要
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    path: str
    category: str
    pages: int
    ocr_pages: list[int]
    records: int
    review_records: int
    status: str
    error: str = ""


@dataclass(slots=True)
class ProcessSummary:
    """
    【类功能】汇总一次目录解析运行的整体统计。
    :Attributes:
        started_at: str+开始时间
        finished_at: str+结束时间
        input_dir: str+输入目录
        output_dir: str+输出目录
        files: list[FileProcessSummary]+逐文件统计
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    started_at: str
    finished_at: str
    input_dir: str
    output_dir: str
    files: list[FileProcessSummary] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        """
        【方法功能】统计本次运行发现的 PDF 数量。
        :return: int+PDF 总数
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return len(self.files)

    @property
    def total_records(self) -> int:
        """
        【方法功能】统计所有文件生成的记录数量。
        :return: int+记录总数
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sum(item.records for item in self.files)

    @property
    def review_records(self) -> int:
        """
        【方法功能】统计所有待人工复核记录数量。
        :return: int+待复核记录数
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sum(item.review_records for item in self.files)

    @property
    def failed_files(self) -> int:
        """
        【方法功能】统计解析失败的 PDF 数量。
        :return: int+失败文件数
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return sum(item.status == "失败" for item in self.files)

    def to_dict(self) -> dict[str, Any]:
        """
        【方法功能】转换为可写入 JSON 的运行摘要。
        :return: dict[str, Any]+包含总体和逐文件统计的字典
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return {
            "开始时间": self.started_at,
            "结束时间": self.finished_at,
            "输入目录": self.input_dir,
            "输出目录": self.output_dir,
            "文件总数": self.total_files,
            "记录总数": self.total_records,
            "待复核记录数": self.review_records,
            "失败文件数": self.failed_files,
            "文件": [asdict(item) for item in self.files],
        }


@dataclass(slots=True)
class ParsedDocument:
    """
    【类功能】封装单个 PDF 的分类解析结果与 OCR 使用信息。
    :Attributes:
        pdf_path: Path+PDF 绝对路径
        category: str+文件类别
        page_count: int+页面总数
        records: list[ExtractionRecord]+解析记录
        ocr_pages: set[int]+OCR 页码集合
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    pdf_path: Path
    category: str
    page_count: int
    records: list[ExtractionRecord]
    ocr_pages: set[int] = field(default_factory=set)
