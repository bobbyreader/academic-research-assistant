#!/usr/bin/env python3
"""PDF 生成脚本。

将 Markdown 文档转换为 PDF，支持多种引用格式。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies() -> dict[str, bool]:
    """检查系统依赖。

    Returns:
        依赖可用性字典。
    """
    deps = {}

    # 检查 pandoc
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        deps["pandoc"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        deps["pandoc"] = False

    # 检查 xelatex
    try:
        subprocess.run(["xelatex", "--version"], capture_output=True, check=True)
        deps["xelatex"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        deps["xelatex"] = False

    return deps


def generate_pdf(
    input_file: Path,
    output_file: Path,
    citation_style: str = "nature",
    no_toc: bool = False,
    no_numbers: bool = False,
) -> bool:
    """生成 PDF。

    Args:
        input_file: 输入 Markdown 文件。
        output_file: 输出 PDF 文件。
        citation_style: 引用格式。
        no_toc: 不生成目录。
        no_numbers: 不编页码。

    Returns:
        是否成功。
    """
    # 构建 pandoc 命令
    cmd = [
        "pandoc",
        str(input_file),
        "-o", str(output_file),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=11pt",
        "-V", "documentclass=article",
    ]

    # 添加目录
    if not no_toc:
        cmd.append("--toc")

    # 添加页码
    if not no_numbers:
        cmd.extend(["-V", "pagestyle=plain"])

    # 引用格式（通过 CSL）
    csl_map = {
        "nature": "nature.csl",
        "apa": "apa.csl",
        "vancouver": "vancouver.csl",
        "chicago": "chicago.csl",
        "ieee": "ieee.csl",
    }
    if citation_style in csl_map:
        # 注意：需要预先下载对应的 CSL 文件
        cmd.extend(["--csl", csl_map[citation_style]])

    print(f"[INFO] 执行: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[OK] PDF 生成成功: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PDF 生成失败: {e.stderr}", file=sys.stderr)
        return False


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="将 Markdown 转换为 PDF")
    parser.add_argument("input_file", type=Path, help="输入 Markdown 文件")
    parser.add_argument("-o", "--output", type=Path, help="输出 PDF 文件")
    parser.add_argument(
        "--citation-style",
        choices=["nature", "apa", "vancouver", "chicago", "ieee"],
        default="nature",
        help="引用格式（默认: nature）",
    )
    parser.add_argument("--no-toc", action="store_true", help="不生成目录")
    parser.add_argument("--no-numbers", action="store_true", help="不编页码")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖")

    args = parser.parse_args()

    # 检查依赖
    if args.check_deps:
        deps = check_dependencies()
        print("\n依赖检查:")
        for dep, available in deps.items():
            status = "[OK]" if available else "[MISSING]"
            print(f"  {status} {dep}")
        sys.exit(0 if all(deps.values()) else 1)

    # 检查输入文件
    if not args.input_file.exists():
        print(f"[ERROR] 文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # 确定输出文件
    if args.output is None:
        args.output = args.input_file.with_suffix(".pdf")

    # 生成 PDF
    success = generate_pdf(
        args.input_file,
        args.output,
        citation_style=args.citation_style,
        no_toc=args.no_toc,
        no_numbers=args.no_numbers,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
