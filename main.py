"""招投标 PDF 统一解析命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from bidding_ocr import ProcessingConfig, process_pdf_tree


def build_argument_parser() -> argparse.ArgumentParser:
    """
    【函数功能】构建招投标 PDF 解析工具的命令行参数解析器。
    :return: argparse.ArgumentParser+配置完成的参数解析器
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: build_argument_parser()
    """
    parser = argparse.ArgumentParser(description="批量解析招投标 PDF 并生成分类及汇总 CSV。")
    parser.add_argument("--input", default="pdf_files", help="PDF 输入目录，默认：pdf_files")
    parser.add_argument("--output", default="results", help="结果输出目录，默认：results")
    parser.add_argument("--dpi", type=int, default=300, help="普通页面 OCR 分辨率，默认：300")
    parser.add_argument(
        "--archive-scan-dpi",
        type=int,
        default=150,
        help="备案资料关键词粗检分辨率，默认：150",
    )
    parser.add_argument("--ocr-threshold", type=float, default=0.80, help="OCR 复核阈值，默认：0.80")
    parser.add_argument("--force", action="store_true", help="忽略 OCR 缓存并重新识别")
    return parser


def main() -> int:
    """
    【函数功能】执行命令行解析任务并输出中文运行摘要。
    :return: int+进程退出码，0 表示全部文件已处理，1 表示存在失败文件
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    Example: main()
    """
    args = build_argument_parser().parse_args()
    config = ProcessingConfig(
        dpi=args.dpi,
        archive_scan_dpi=args.archive_scan_dpi,
        ocr_confidence_threshold=args.ocr_threshold,
        force_ocr=args.force,
    )
    summary = process_pdf_tree(Path(args.input), Path(args.output), config)
    print(
        f"处理完成：文件 {summary.total_files} 个，记录 {summary.total_records} 条，"
        f"待复核 {summary.review_records} 条，失败 {summary.failed_files} 个。"
    )
    print(f"结果目录：{Path(args.output).resolve()}")
    return 1 if summary.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
