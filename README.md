# 招投标 PDF 统一解析工具

该工具递归读取输入目录及全部子目录中的 PDF，按中文文件名自动识别七种业务类型，解析项目与企业信息，并在输出目录生成分类 CSV、`final.csv`、待复核清单和运行摘要。文件仅在本地处理；Windows 扫描兼容中文超长路径以及 `.pdf`、`.PDF` 等大小写扩展名。

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

处理惠山高标田原始数据：

```powershell
python main.py --input "D:\projects\Internship\Jwmath\biding_files_utils\biding_files\huishan\高标田原始数据" --output "output\huishan_all"
```

仅处理指定类别，或排除耗时较长的类别：

```powershell
python main.py --input pdf_files --output results --include "award_notice,bid_candidates,bid_announcement"
python main.py --input pdf_files --output results --exclude "archive_info,tender_cover"
```

`--include` 和 `--exclude` 不能同时使用。类别名称必须来自：`tender_cover`、`bid_evaluation_report`、`bid_candidates`、`award_notice`、`bid_announcement`、`bid_list`、`archive_info`。旧参数 `--category` 继续保留，用于兼容只处理一个类别的调用，并且同样不能与前两个参数同时使用。

常用参数：

```text
--dpi 300                 普通页面和命中页面的 OCR 分辨率
--archive-scan-dpi 150    备案资料全文关键词粗检分辨率
--ocr-threshold 0.80      低于该置信度的记录进入复核清单
--force                   忽略 OCR JSON 缓存并重新识别
--include 类别1,类别2     仅处理列出的类别
--exclude 类别1,类别2     不处理列出的类别
```

## 自动分类规则

- `archive_info`：文件名包含“备案资料”“归档资料”“备案材料”“归档材料”“备案”或“归档”。
- `award_notice`：文件名包含“中标通知书”。
- `bid_announcement`：文件名包含“中标人公告”。
- `bid_candidates`：文件名包含“中标候选人”或“中标公示”。
- `bid_evaluation_report`：文件名包含“评标报告”。
- `bid_list`：文件名包含“投标单位名单”。
- `tender_cover`：文件名严格为 `封面.pdf` 或 `1.pdf`，且页数为 1 至 3 页；排除 `施工组织设计/1.pdf`。路径语义不明确的 `1.pdf` 还需由首页原生文本中的“投标文件”“参与文件”或项目、投标人字段组合确认。

无法识别的其他 PDF 不执行 OCR，也不会作为解析失败写入 CSV；其数量记录在 `run_summary.json` 中。

OCR 缓存位于 `results/.ocr_cache`，缓存键包含 PDF SHA-256、页码、分辨率和 OCR 策略 profile。RapidOCR 使用独立 profile，不会复用旧 PaddleOCR 缓存；删除该目录或使用 `--force` 可重新识别。

## 输出

分类结果包括 `tender_cover.csv`、`bid_evaluation_report.csv`、`bid_candidates.csv`、`award_notice.csv`、`bid_announcement.csv`、`bid_list.csv` 和 `archive_info.csv`。

汇总结果：

- `final.csv`：按项目、标段和企业去重后的最终数据。
- `review_queue.csv`：字段缺失、低置信度、解析失败或来源冲突的记录。
- `run_summary.json`：扫描、识别、跳过、筛选排除和分类预检统计，以及逐文件页数、OCR 页码、记录数量、状态和错误摘要。

所有 CSV 使用 UTF-8 BOM 编码，可直接用 Excel 打开。

## 测试

```powershell
python -m unittest discover -s tests -v
```
