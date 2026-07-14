"""批量 OCR 提取参与文件封面信息，并输出 CSV。

默认读取 ``pdf_files/cover`` 下的 PDF。图片型 PDF 会调用上级项目中已经
验证过的高清渲染、RapidOCR 和红章去除实现。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
_OCR_ENGINE: Any | None = None


OUTPUT_COLUMNS = ("序号", "项目名称", "交易编号", "参与单位", "封面日期")
PROJECT_TEMPLATE_RE = re.compile(r"[（(【\[]\s*工程名称\s*[）)】\]]\s*$")
COMPANY_SUFFIX_RE = re.compile(r"\s*[（(【\[].*?(?:盖|章).*?[）)】\]]\s*$")
DATE_RE = re.compile(r"(?:日期\s*[：:]?\s*)?(20\d{2})\s*[年\-./]\s*(\d{1,2})\s*[月\-./]\s*(\d{1,2})")
CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\d[A-Z0-9-]*\b")


def clean_text(value: str) -> str:
    """合并 OCR 空白，并修正已由样本验证的高确定性误识别。"""
    value = re.sub(r"\s+", "", value or "")
    # 红章邻近的“江苏”在 RapidOCR 中偶发识别成“注苏”。
    return value.replace("注苏", "江苏")


def render_pdf_page_for_ocr(pdf_path: Path, scale: float = 5.0) -> Any:
    """将 PDF 首页渲染为适合中文 OCR 的 RGB 图像数组。"""
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "缺少 OCR 渲染依赖，请安装：pip install pypdfium2 numpy"
        ) from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if not len(pdf):
            raise ValueError("PDF 没有页面")
        return np.array(pdf[0].render(scale=scale).to_pil().convert("RGB"))
    finally:
        pdf.close()


def get_ocr_engine() -> Any:
    """懒加载并复用 RapidOCR 引擎，避免每个 PDF 重复加载模型。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "缺少 OCR 依赖，请安装：pip install rapidocr-onnxruntime"
            ) from exc
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def remove_red_seal(image: Any) -> Any:
    """将红章像素置白，减少其与黑色正文重叠时的 OCR 干扰。"""
    processed = image.copy()
    red, green, blue = processed[:, :, 0], processed[:, :, 1], processed[:, :, 2]
    red_mask = (
        (red > 110)
        & (green < 170)
        & (blue < 170)
        & ((red.astype(int) - green.astype(int)) > 25)
        & ((red.astype(int) - blue.astype(int)) > 25)
    )
    processed[red_mask] = [255, 255, 255]
    return processed


def ocr_image_to_text(image: Any) -> tuple[str, float]:
    """OCR 图像，按页面阅读顺序合并文本框并返回平均置信度。"""
    result, _ = get_ocr_engine()(image)
    if not result:
        return "", 0.0

    items = []
    for box, text, score in result:
        text = re.sub(r"\s+", "", text or "")
        if not text:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        items.append(
            {
                "text": text,
                "score": float(score),
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "height": max(ys) - min(ys),
            }
        )

    items.sort(key=lambda item: (item["y"], item["x"]))
    lines: list[list[dict[str, Any]]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        line = lines[-1]
        center_y = sum(value["y"] for value in line) / len(line)
        if abs(item["y"] - center_y) <= max(12.0, item["height"] * 0.75):
            line.append(item)
        else:
            lines.append([item])

    text = "\n".join(
        "".join(value["text"] for value in sorted(line, key=lambda value: value["x"]))
        for line in lines
    )
    average_score = sum(item["score"] for item in items) / len(items) if items else 0.0
    return text, average_score


def score_ocr_text(text: str, average_score: float) -> float:
    """用封面关键词、识别文本量和置信度，为 OCR 候选结果评分。"""
    compact_text = clean_text(text)
    markers = ("工程名称", "交易编号", "参与单位", "日期", "参与文件")
    return sum(marker in compact_text for marker in markers) * 20 + min(len(compact_text), 240) * 0.2 + average_score * 10


def choose_ocr_text(pdf_path: Path) -> str:
    """比较原图与去红章图的 OCR 结果，选取质量更好的全文。"""
    image = render_pdf_page_for_ocr(pdf_path, scale=5.0)
    candidates = [
        ocr_image_to_text(image),
        ocr_image_to_text(remove_red_seal(image)),
    ]
    return max(
        candidates,
        key=lambda item: score_ocr_text(item[0], item[1]),
    )[0]


def extract_project_name(text: str) -> str:
    """提取顶部项目标题，并移除封面模板中的“工程名称”提示。"""
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if "工程名称" not in line:
            continue
        value = PROJECT_TEMPLATE_RE.sub("", line)
        if value:
            return value.replace("(", "（").replace(")", "）")
    raise ValueError("未识别到项目名称")


def extract_transaction_no(text: str) -> str:
    """优先从“交易编号”标签行读取编号。"""
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if "交易编号" not in line:
            continue
        match = CODE_RE.search(line)
        if match:
            return match.group(0)
    raise ValueError("未识别到交易编号")


def extract_company(text: str) -> str:
    """提取参与单位，删除随后的盖章说明与 OCR 残留。"""
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if "参与单位" not in line:
            continue
        value = re.split(r"参与单位\s*[：:]?", line, maxsplit=1)[-1]
        value = COMPANY_SUFFIX_RE.sub("", value)
        # OCR 有时将左括号识别为其他符号；“盖单位章”之后均非公司名。
        value = re.split(r"[（(【\[]?盖(?:单位|公)?章", value, maxsplit=1)[0]
        value = value.rstrip("（(【[")
        if value:
            return value
    raise ValueError("未识别到参与单位")


def extract_cover_date(text: str) -> str:
    """提取封面日期并统一为 YYYY-MM-DD。"""
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if "日期" not in line:
            continue
        match = DATE_RE.search(line)
        if match:
            return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    raise ValueError("未识别到封面日期")


def extract_one(pdf_path: Path, sequence: int) -> dict[str, str]:
    text = choose_ocr_text(pdf_path)
    return {
        "序号": str(sequence),
        "项目名称": extract_project_name(text),
        "交易编号": extract_transaction_no(text),
        "参与单位": extract_company(text),
        "封面日期": extract_cover_date(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 提取参与文件封面信息并写入 CSV")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "pdf_files" / "cover")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "cover_extract_results.csv")
    args = parser.parse_args()

    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"未在目录中找到 PDF：{args.input_dir}")

    rows = []
    for sequence, pdf_path in enumerate(pdf_paths, start=1):
        try:
            rows.append(extract_one(pdf_path, sequence))
            print(f"[完成] {pdf_path.name}")
        except Exception as exc:
            raise SystemExit(f"[失败] {pdf_path.name}：{exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已输出 {len(rows)} 条记录：{args.output}")


if __name__ == "__main__":
    main()
