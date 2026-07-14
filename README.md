# 招投标 PDF 统一解析工具

该工具递归读取 `pdf_files` 下的招投标 PDF，按七种业务类型解析项目与企业信息，并在 `results` 输出分类 CSV、`final.csv`、待复核清单和运行摘要。文件仅在本地处理。

## 环境与安装

- Python 3.10 及以上版本。
- `pypdfium2` 用于内存渲染 PDF 页面，无需安装 Poppler。
- RapidOCR（ONNX Runtime）用于中文 OCR。首次运行请确保依赖已按下方命令安装完成。

```powershell
python -m pip install -r requirements.txt
```

推荐使用 Python 3.10 的独立 Conda 环境：

```powershell
conda env create -f environment.yml
conda run -n bidding-ocr python main.py --input pdf_files --output results
```

代码使用 `pypdf` 读取原生文本和书签，使用 `pypdfium2` 将扫描页直接渲染为内存 RGB 图像后交给 RapidOCR。

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

OCR 缓存位于 `results/.ocr_cache`，缓存键包含 PDF SHA-256、页码、分辨率和 OCR 策略 profile。RapidOCR 使用独立 profile，不会复用旧 PaddleOCR 缓存；删除该目录或使用 `--force` 可重新识别。

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
