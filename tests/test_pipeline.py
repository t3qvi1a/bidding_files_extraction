"""目录处理入口和固定输出文件离线集成测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from bidding_ocr.models import ExtractionRecord, OCRLine, PageText, ParsedDocument, ProcessingConfig
from bidding_ocr.pdf_engine import RapidOCRBackend
from bidding_ocr.pipeline import process_pdf_tree


class FakePDFTextEngine:
    """
    【类功能】模拟单页中标公告 PDF，引导完整输出流水线而不加载 OCR 模型。
    :Attributes:
        page_count: int+固定页面总数
        ocr_pages: set[int]+固定 OCR 页码
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    received_ocr_backends: list[object | None] = []

    def __init__(
        self,
        pdf_path: Path,
        cache_dir: Path,
        config: ProcessingConfig,
        ocr_backend: object | None = None,
        progress_callback: object | None = None,
    ) -> None:
        """
        【方法功能】保存测试参数并初始化单页状态。
        :param pdf_path: Path+测试 PDF 路径
        :param cache_dir: Path+测试缓存路径
        :param config: ProcessingConfig+测试处理配置
        :param ocr_backend: object|None+共享 OCR 后端占位参数
        :param progress_callback: object|None+OCR 进度回调占位参数
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self.config = config
        self.ocr_backend = ocr_backend
        self.progress_callback = progress_callback
        type(self).received_ocr_backends.append(ocr_backend)
        self.page_count = 1
        self.ocr_pages = {1}

    def native_text(self, page_number: int) -> str:
        """
        【方法功能】返回空文本层以模拟扫描件。
        :param page_number: int+页码
        :return: str+空文本
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return ""

    def get_page(self, page_number: int, dpi: int) -> PageText:
        """
        【方法功能】返回固定项目和中标人 OCR 页面。
        :param page_number: int+页码
        :param dpi: int+OCR 分辨率
        :return: PageText+固定页面
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        lines = [
            OCRLine("项目名称：离线测试项目", 0.99, []),
            OCRLine("中标人：甲建设有限公司", 0.99, []),
        ]
        return PageText(page_number, "\n".join(line.text for line in lines), lines, "ocr", dpi)

    def cover_bookmark_pages(self) -> list[int]:
        """
        【方法功能】返回空书签列表。
        :return: list[int]+空书签页码
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        return []


class PipelineIntegrationTests(unittest.TestCase):
    """
    【类功能】验证统一入口生成七类 CSV、最终结果、复核清单和运行摘要。
    :Author: gexinyan
    :CreateTime: 2026-07-13 11:08:59
    """

    def test_process_tree_writes_all_fixed_outputs(self) -> None:
        """
        【方法功能】验证最小目录运行后固定输出文件齐全且最终记录正确。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 11:08:59
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            output_dir = root / "results"
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(root / "pdf_files", output_dir)

            expected_files = {
                "tender_cover.csv",
                "bid_evaluation_report.csv",
                "bid_candidates.csv",
                "award_notice.csv",
                "bid_announcement.csv",
                "bid_list.csv",
                "archive_info.csv",
                "final.csv",
                "review_queue.csv",
                "run_summary.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))
            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.failed_files, 0)
            self.assertIn("甲建设有限公司", (output_dir / "final.csv").read_text(encoding="utf-8-sig"))
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run_summary["文件总数"], 1)

    def test_category_filter_emits_file_progress(self) -> None:
        """
        【方法功能】验证类别筛选仅处理目标目录文件，并输出开始和完成进度消息。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 16:25:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            announcement_dir = root / "pdf_files" / "bid_announcement"
            notice_dir = root / "pdf_files" / "award_notice"
            announcement_dir.mkdir(parents=True)
            notice_dir.mkdir(parents=True)
            (announcement_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            (notice_dir / "notice.pdf").write_bytes(b"fake-pdf")
            messages: list[str] = []
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    category_filter="bid_announcement",
                    progress_callback=messages.append,
                )

            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.files[0].category, "bid_announcement")
            self.assertTrue(any("处理中 1/1" in message for message in messages))
            self.assertTrue(any("完成 1/1" in message for message in messages))

    def test_process_tree_reuses_one_ocr_backend_for_all_files(self) -> None:
        """
        【方法功能】验证同一批 PDF 处理时向每个文件引擎传入同一个 OCR 后端实例。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-13 16:45:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement_01.pdf").write_bytes(b"fake-pdf")
            (input_dir / "announcement_02.pdf").write_bytes(b"fake-pdf")
            shared_backend = object()
            FakePDFTextEngine.received_ocr_backends = []
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    ocr_backend=shared_backend,
                )

            self.assertEqual(summary.total_files, 2)
            self.assertEqual(FakePDFTextEngine.received_ocr_backends, [shared_backend, shared_backend])

    def test_process_tree_uses_rapidocr_backend_by_default(self) -> None:
        """
        【方法功能】验证未注入 OCR 后端时统一入口创建并传递 RapidOCR 后端。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-14 10:30:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            FakePDFTextEngine.received_ocr_backends = []
            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                process_pdf_tree(root / "pdf_files", root / "results")

            self.assertEqual(len(FakePDFTextEngine.received_ocr_backends), 1)
            self.assertIsInstance(FakePDFTextEngine.received_ocr_backends[0], RapidOCRBackend)

    def test_parallel_workers_reject_injected_backend(self) -> None:
        """
        【方法功能】验证多进程模式拒绝不可跨进程序列化的外部 OCR backend。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 16:00:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            with self.assertRaisesRegex(ValueError, "ocr_backend"):
                process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    ocr_backend=object(),
                    workers=2,
                )

    def test_include_filter_skips_unknown_files_and_expands_summary(self) -> None:
        """
        【方法功能】验证包含筛选仅处理目标类别，未知文件跳过且摘要记录扫描分类统计。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            announcement_dir = root / "pdf_files" / "bid_announcement"
            notice_dir = root / "pdf_files" / "award_notice"
            announcement_dir.mkdir(parents=True)
            notice_dir.mkdir(parents=True)
            (announcement_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            (notice_dir / "notice.pdf").write_bytes(b"fake-pdf")
            (root / "pdf_files" / "unrelated.pdf").write_bytes(b"fake-pdf")
            output_dir = root / "results"

            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    output_dir,
                    include_categories=("bid_announcement",),
                )

            self.assertEqual(summary.scanned_files, 3)
            self.assertEqual(summary.recognized_files, 2)
            self.assertEqual(summary.unrecognized_files, 1)
            self.assertEqual(summary.filtered_files, 1)
            self.assertEqual(summary.total_files, 1)
            run_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run_summary["扫描PDF总数"], 3)
            self.assertEqual(run_summary["未识别跳过文件数"], 1)
            self.assertEqual(run_summary["本次处理类别"], ["bid_announcement"])

    def test_exclude_filter_processes_remaining_recognized_categories(self) -> None:
        """
        【方法功能】验证排除筛选跳过指定类别并处理其余已识别类别。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-15 10:12:03
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            announcement_dir = root / "pdf_files" / "bid_announcement"
            notice_dir = root / "pdf_files" / "award_notice"
            announcement_dir.mkdir(parents=True)
            notice_dir.mkdir(parents=True)
            (announcement_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            (notice_dir / "notice.pdf").write_bytes(b"fake-pdf")

            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    exclude_categories=("award_notice",),
                )

            self.assertEqual(summary.total_files, 1)
            self.assertEqual(summary.files[0].category, "bid_announcement")
            self.assertEqual(summary.filtered_files, 1)

    def test_completed_callback_runs_once_for_serial_pdf(self) -> None:
        """
        【方法功能】验证串行模式下单个 PDF 成功解析后仅触发一次主进程回调。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-16 10:00:00
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            (input_dir / "announcement.pdf").write_bytes(b"fake-pdf")
            callbacks: list[tuple[str, int]] = []

            def on_completed(document: ParsedDocument, warnings: list[str]) -> None:
                """
                【函数功能】记录测试回调收到的文档信息。
                :param document: ParsedDocument，完成解析的文档
                :param warnings: list[str]，文档告警列表
                :return: None
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """
                callbacks.append((document.pdf_path.name, len(warnings)))

            with patch("bidding_ocr.pipeline.PDFTextEngine", FakePDFTextEngine):
                process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    pdf_completed_callback=on_completed,
                )

            self.assertEqual(callbacks, [("announcement.pdf", 0)])

    def test_completed_callback_runs_in_parallel_result_collection(self) -> None:
        """
        【方法功能】验证并行分支在主进程接收结果时触发回调且保持最终汇总顺序。
        :return: None
        :Author: gexinyan
        :CreateTime: 2026-07-16 10:00:00
        """
        class ImmediateExecutor:
            """
            【类功能】以同步 Future 模拟多进程执行器，用于测试主进程结果收集逻辑。
            :Author: gexinyan
            :CreateTime: 2026-07-16 10:00:00
            """

            def __init__(self, **_: object) -> None:
                """
                【方法功能】初始化测试执行器。
                :return: None
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """

            def __enter__(self) -> "ImmediateExecutor":
                """
                【方法功能】进入测试执行器上下文。
                :return: ImmediateExecutor，当前执行器
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """
                return self

            def __exit__(self, *_: object) -> None:
                """
                【方法功能】退出测试执行器上下文。
                :return: None
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """

            def submit(self, function: object, task: object) -> Future[object]:
                """
                【方法功能】同步执行任务并返回已完成 Future。
                :param function: object，待调用函数
                :param task: object，任务参数
                :return: Future[object]，已完成任务
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """
                future: Future[object] = Future()
                try:
                    future.set_result(function(task))  # type: ignore[operator]
                except Exception as exc:  # noqa: BLE001
                    future.set_exception(exc)
                return future

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdf_files" / "bid_announcement"
            input_dir.mkdir(parents=True)
            first_path = input_dir / "first.pdf"
            second_path = input_dir / "second.pdf"
            first_path.write_bytes(b"fake-pdf")
            second_path.write_bytes(b"fake-pdf")
            callbacks: list[str] = []

            def build_worker_result(task: tuple[Path, Path, Path, ProcessingConfig, str]) -> tuple[ParsedDocument, list[str]]:
                """
                【函数功能】构造不依赖 OCR 模型的并行任务结果。
                :param task: tuple[Path, Path, Path, ProcessingConfig, str]，模拟 worker 任务
                :return: tuple[ParsedDocument, list[str]]，模拟解析结果
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """
                path = task[0]
                return (
                    ParsedDocument(
                        pdf_path=path,
                        category="bid_announcement",
                        page_count=1,
                        records=[ExtractionRecord(company_name=path.stem, source_path=path.name)],
                    ),
                    [],
                )

            def on_completed(document: ParsedDocument, _: list[str]) -> None:
                """
                【函数功能】记录主进程回调收到的文档名。
                :param document: ParsedDocument，完成解析的文档
                :param _: list[str]，文档告警列表
                :return: None
                :Author: gexinyan
                :CreateTime: 2026-07-16 10:00:00
                """
                callbacks.append(document.pdf_path.name)

            with (
                patch("bidding_ocr.pipeline.classify_pdf_for_plan", return_value="bid_announcement"),
                patch("bidding_ocr.pipeline.ProcessPoolExecutor", ImmediateExecutor),
                patch("bidding_ocr.pipeline._process_pdf_worker", side_effect=build_worker_result),
            ):
                summary = process_pdf_tree(
                    root / "pdf_files",
                    root / "results",
                    workers=2,
                    pdf_completed_callback=on_completed,
                )

            self.assertCountEqual(callbacks, ["first.pdf", "second.pdf"])
            self.assertEqual(
                [item.path for item in summary.files],
                ["bid_announcement\\first.pdf", "bid_announcement\\second.pdf"],
            )


if __name__ == "__main__":
    unittest.main()
