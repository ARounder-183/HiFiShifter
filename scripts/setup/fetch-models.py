#!/usr/bin/env python3
"""取模型权重：release 资产 / 本地目录 / 本地 zip，三条路都做逐文件校验。

【为什么权重不在仓库里】151.6 MB / 7 个文件让一次 checkout 就是 150 MB，而每个 CI
job 都要 checkout 一次（一次 PR 约 11 次）。仓库里只保留清单
§backend/src-tauri/resources/models/models.json§（1.5 KB）。

【为什么是 .py 而不是 .ps1】同一份实现要在三个平台上跑（build.ps1 / vst3/unix/build.sh
/ CI），而本仓已有 Python 门禁脚本（scripts/checks/check-*.py），三个平台的 runner 与
开发机上都装了它。两份实现（ps1 + sh）只会分叉，且分叉的后果是"某一个平台的构建悄悄
用着旧权重"。

【完整性判据与运行时一致】逐文件比对**字节数 + sha256**，数据来源就是 models.json
（与 model_store::looks_complete 同一份清单；那份只比字节数是因为它跑在每次启动的
路径上，这里可以付出哈希的代价）。版本号是内容派生值，取错版本必然在这里暴露。

用法：
    python scripts/setup/fetch-models.py                # 缺就取（默认从 release 资产）
    python scripts/setup/fetch-models.py --check        # 只检查，缺就退出码 1
    python scripts/setup/fetch-models.py --source DIR   # 从本地目录拷贝（已有权重）
    python scripts/setup/fetch-models.py --archive ZIP  # 从本地 zip 解压
    python scripts/setup/fetch-models.py --base-url URL # 换一个下载源（镜像）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "backend" / "src-tauri" / "resources" / "models"
MANIFEST = MODELS_DIR / "models.json"
# 【为什么是 zip 而不是 tar.zst】三个平台都能解（Expand-Archive / zipfile / unzip），
# 不需要额外的 zstd 二进制。150 MB 的一次性下载里，压缩率差别可以忽略。
DEFAULT_BASE_URL = "https://github.com/ARounder-183/HiFiShifter/releases/download"


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"缺少模型清单：{MANIFEST}（它必须在仓库里）")
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_files(models_dir: Path, manifest: dict) -> list[str]:
    """返回不完整（缺文件 / 字节数不符）的条目。

    【为什么先只比字节数】哈希 150 MB 要几百毫秒到几秒，而这个脚本在每次构建前都跑。
    字节数不符已经能挡住绝大多数故障（没解压完、拷了一半、版本混装）；真正的哈希
    校验放在"解压之后"那一步（见 verify_hashes）。
    """
    problems: list[str] = []
    for entry in manifest["files"]:
        path = models_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size {entry['path']}: {path.stat().st_size} != {entry['bytes']}")
    return problems


def verify_hashes(models_dir: Path, manifest: dict) -> list[str]:
    problems: list[str] = []
    for entry in manifest["files"]:
        path = models_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(f"sha256 {entry['path']}: {actual} != {entry['sha256']}")
    return problems


def install_from_directory(source: Path, manifest: dict) -> None:
    """从一份已有的模型目录拷贝（离线 / 本地镜像）。"""
    for entry in manifest["files"]:
        src = source / entry["path"]
        if not src.is_file():
            raise SystemExit(f"源目录缺少 {entry['path']}：{source}")
        dst = MODELS_DIR / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def install_from_zip(archive: Path, manifest: dict) -> None:
    # 【为什么要先解压到临时目录】直接解到目标目录会让"解压中途失败"留下一个看起来
    # 存在但缺文件的模型目录 —— 那正是运行时最难诊断的状态。
    with tempfile.TemporaryDirectory(prefix="hfs-models-") as tmp:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        root = Path(tmp)
        # zip 可能带一层顶层目录（models-<version>/…），这里自适应。
        candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
        first = manifest["files"][0]["path"]
        for candidate in candidates:
            if (candidate / first).is_file():
                install_from_directory(candidate, manifest)
                return
        raise SystemExit(f"{archive} 里没有 models.json 列出的文件")


def download(url: str, destination: Path) -> None:
    print(f"下载 {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "hifishifter-fetch-models"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="取 HiFiShifter 的模型权重")
    parser.add_argument("--check", action="store_true", help="只检查是否完整（缺则退出码 1）")
    parser.add_argument("--source", type=Path, help="从本地模型目录拷贝")
    parser.add_argument("--archive", type=Path, help="从本地 zip 解压")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HIFISHIFTER_MODELS_BASE_URL", DEFAULT_BASE_URL),
        help="release 资产的基地址（镜像用；也可设 HIFISHIFTER_MODELS_BASE_URL）",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    version = manifest["version"]

    problems = missing_files(MODELS_DIR, manifest)
    if not problems:
        if args.check:
            print(f"模型权重完整（版本 {version}）")
        return 0
    if args.check:
        print(
            f"模型权重不完整（版本 {version}）：{'; '.join(problems[:5])}",
            file=sys.stderr,
        )
        return 1

    print(f"模型权重不完整（版本 {version}），开始获取……")
    if args.source:
        install_from_directory(args.source.expanduser().resolve(), manifest)
    elif args.archive:
        install_from_zip(args.archive.expanduser().resolve(), manifest)
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hfs-models-dl-") as tmp:
            archive = Path(tmp) / f"models-{version}.zip"
            url = f"{args.base_url.rstrip('/')}/models-{version}/models-{version}.zip"
            try:
                download(url, archive)
            except Exception as error:  # noqa: BLE001 - 网络错误的形态太多，统一给人话
                raise SystemExit(
                    f"下载失败：{error}\n"
                    f"  资产地址：{url}\n"
                    "  离线环境请用 --source <已有模型目录> 或 --archive <本地 zip>。"
                ) from error
            install_from_zip(archive, manifest)

    # 【为什么下载完还要哈希校验】字节数对得上但内容不对（换了个同名文件、被截断后
    # 又补齐）是真实发生过的：症状是推理输出悄悄变差，而不是报错。
    hash_problems = verify_hashes(MODELS_DIR, manifest)
    if hash_problems:
        print("模型权重校验失败：", file=sys.stderr)
        for problem in hash_problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"模型权重就绪（版本 {version}，{len(manifest['files'])} 个文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
