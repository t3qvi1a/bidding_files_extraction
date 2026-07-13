"""按指定类别运行真实 PDF 解析的命令行测试脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bidding_ocr import ProcessingConfig, process_pdf_tree
from bidding_ocr.models import CATEGORIES


def build_argument_parser() -> argparse.ArgumentParser:
    """
    【函数功能】构建单类别真实 PDF 解析测试脚本的命令行参数。
    :return: argparse.ArgumentParser+配置完成的参数解析器
    :Author: gexinyan
    :CreateTime: 2026-07-13 16:25:00
    Example: build_argument_parser()
    """
    parser = argparse.ArgumentParser(description="仅处理一种招投标 PDF 类别，用于真实样本调试。")
    parser.add_argument("--category", required=True, choices=CATEGORIES, help="待测试的 PDF 类别")
    parser.add_argument("--input", default="pdf_files", help="PDF 输入目录，默认：pdf_files")
    parser.add_argument(
        "--output",
        default="results/category_test",
        help="测试结果目录，默认：results/category_test",
    )
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
    【函数功能】仅运行一个 PDF 类别并以进度日志输出真实样本解析结果。
    :return: int+0 表示目标类别处理成功，1 表示无文件或存在失败
    :Author: gexinyan
    :CreateTime: 2026-07-13 16:25:00
    Example: main()
    """
    args = build_argument_parser().parse_args()
    config = ProcessingConfig(
        dpi=args.dpi,
        archive_scan_dpi=args.archive_scan_dpi,
        ocr_confidence_threshold=args.ocr_threshold,
        force_ocr=args.force,
    )
    output_dir = Path(args.output) / args.category
    summary = process_pdf_tree(
        Path(args.input),
        output_dir,
        config,
        category_filter=args.category,
        progress_callback=lambda message: print(message, flush=True),
    )
    if summary.total_files == 0:
        print(f"未找到类别 {args.category} 的 PDF，请检查输入目录。")
        return 1
    print(
        f"类别测试完成：{args.category}，文件 {summary.total_files} 个，"
        f"记录 {summary.total_records} 条，失败 {summary.failed_files} 个。"
    )
    print(f"测试结果目录：{output_dir.resolve()}")
    return 1 if summary.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
