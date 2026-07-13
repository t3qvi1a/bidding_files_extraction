# 招投标 PDF 统一解析工具

该工具递归读取 `pdf_files` 下的招投标 PDF，按七种业务类型解析项目与企业信息，并在 `results` 输出分类 CSV、`final.csv`、待复核清单和运行摘要。文件仅在本地处理。

## 环境与安装

- Python 3.10 及以上版本。
- Poppler，需要能够执行 `pdftoppm`。Windows 可通过环境变量 `POPPLER_PATH` 指向 Poppler 的 `bin` 目录。
- PaddleOCR 中文模型。首次识别时 PaddleOCR 会下载模型，需要网络连接；之后可离线使用缓存模型。

```powershell
python -m pip install -r requirements.txt
```

## 运行

```powershell
python main.py --input pdf_files --output results
```

常用参数：

```text
--dpi 300                 普通页面和命中页面的 OCR 分辨率
--archive-scan-dpi 150    备案资料全文关键词粗检分辨率
--ocr-threshold 0.80      低于该置信度的记录进入复核清单
--force                   忽略 OCR JSON 缓存并重新识别
```

OCR 缓存位于 `results/.ocr_cache`，缓存键包含 PDF SHA-256、页码和分辨率。删除该目录或使用 `--force` 可重新识别。

## 输出

分类结果包括 `tender_cover.csv`、`bid_evaluation_report.csv`、`bid_candidates.csv`、`award_notice.csv`、`bid_announcement.csv`、`bid_list.csv` 和 `archive_info.csv`。

汇总结果：

- `final.csv`：按项目、标段和企业去重后的最终数据。
- `review_queue.csv`：字段缺失、低置信度、解析失败或来源冲突的记录。
- `run_summary.json`：逐文件页数、OCR 页码、记录数量、状态和错误摘要。

所有 CSV 使用 UTF-8 BOM 编码，可直接用 Excel 打开。

## 测试

```powershell
python -m unittest discover -s tests -v
```
