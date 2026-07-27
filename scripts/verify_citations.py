#!/usr/bin/env python3
"""引用验证脚本。

从 Markdown 文件中提取所有 DOI，验证其可解析性，
并从 CrossRef 获取元数据。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests


def extract_dois(text: str) -> list[str]:
    """从文本中提取所有 DOI。

    Args:
        text: Markdown 文本内容。

    Returns:
        DOI 列表（去重）。
    """
    # DOI 正则模式
    pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    dois = re.findall(pattern, text, re.IGNORECASE)
    return list(dict.fromkeys(dois))  # 去重保持顺序


def verify_doi(doi: str, timeout: int = 10) -> dict:
    """验证单个 DOI。

    Args:
        doi: 待验证的 DOI。
        timeout: 请求超时时间（秒）。

    Returns:
        验证结果字典。
    """
    result = {
        "doi": doi,
        "valid": False,
        "resolvable": False,
        "metadata": None,
        "error": None,
    }

    # 检查格式
    if not doi.startswith("10."):
        result["error"] = "DOI 格式无效（必须以 10. 开头）"
        return result

    # 尝试解析
    url = f"https://doi.org/{doi}"
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        result["resolvable"] = response.status_code in (200, 302)

        if result["resolvable"]:
            # 从 CrossRef 获取元数据
            cr_url = f"https://api.crossref.org/works/{doi}"
            cr_response = requests.get(cr_url, timeout=timeout)
            if cr_response.status_code == 200:
                data = cr_response.json()
                result["valid"] = True
                result["metadata"] = {
                    "title": data.get("message", {}).get("title", [""])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}"
                        for a in data.get("message", {}).get("author", [])
                    ],
                    "journal": data.get("message", {}).get("container-title", [""])[0],
                    "year": data.get("message", {}).get("published-print", {}).get("date-parts", [[None]])[0][0],
                    "publisher": data.get("message", {}).get("publisher", ""),
                }
            else:
                result["error"] = f"CrossRef API 返回 {cr_response.status_code}"
        else:
            result["error"] = f"DOI 无法解析（HTTP {response.status_code}）"

    except requests.RequestException as e:
        result["error"] = f"请求失败: {e}"

    return result


def generate_report(results: list[dict], output_path: Path | None = None) -> str:
    """生成验证报告。

    Args:
        results: 验证结果列表。
        output_path: 输出文件路径（可选）。

    Returns:
        报告文本。
    """
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    failed = total - valid

    lines = [
        "# 引用验证报告",
        "",
        f"- 总引用数: {total}",
        f"- 验证成功: {valid}",
        f"- 验证失败: {failed}",
        f"- 成功率: {valid/total*100:.1f}%" if total > 0 else "- 成功率: N/A",
        "",
    ]

    if failed > 0:
        lines.append("## 失败详情")
        lines.append("")
        for r in results:
            if not r["valid"]:
                lines.append(f"- **{r['doi']}**")
                lines.append(f"  - 错误: {r['error']}")
                lines.append("")

    if valid > 0:
        lines.append("## 成功详情")
        lines.append("")
        for r in results:
            if r["valid"] and r["metadata"]:
                m = r["metadata"]
                lines.append(f"- **{r['doi']}**")
                lines.append(f"  - 标题: {m['title']}")
                lines.append(f"  - 作者: {', '.join(m['authors'][:3])}{' et al.' if len(m['authors']) > 3 else ''}")
                lines.append(f"  - 期刊: {m['journal']} ({m['year']})")
                lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report, encoding="utf-8")
        print(f"[OK] 报告已保存: {output_path}")

    return report


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="验证 Markdown 文件中的 DOI")
    parser.add_argument("input_file", type=Path, help="输入 Markdown 文件")
    parser.add_argument("-o", "--output", type=Path, help="输出报告文件")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="请求超时时间（秒）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"[ERROR] 文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # 读取文件
    text = args.input_file.read_text(encoding="utf-8")

    # 提取 DOI
    dois = extract_dois(text)
    print(f"[INFO] 找到 {len(dois)} 个 DOI")

    if not dois:
        print("[INFO] 未找到 DOI，退出")
        sys.exit(0)

    # 验证
    results = []
    for i, doi in enumerate(dois, 1):
        print(f"[INFO] 验证 {i}/{len(dois)}: {doi}")
        result = verify_doi(doi, timeout=args.timeout)
        results.append(result)

    # 输出
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        report = generate_report(results, args.output)
        if not args.output:
            print("\n" + report)


if __name__ == "__main__":
    main()
