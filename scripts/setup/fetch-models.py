#!/usr/bin/env python3
"""取 / 打包模型权重：三条获取路径 + 一条发布路径，四条路都做逐文件校验。

【为什么权重不在仓库里】151.6 MB / 7 个文件让一次 checkout 就是 150 MB，而每个 CI
job 都要 checkout 一次（一次 PR 约 11 次）。仓库里只保留清单
`backend/src-tauri/resources/models/models.json`（1.5 KB）。

【为什么是 .py 而不是 .ps1】同一份实现要在三个平台上跑（build.ps1 / vst3/unix/build.sh
/ CI）。本仓的门禁脚本（scripts/checks/check-*.py）已经依赖 Python，三个平台的 runner
也都预装，没有一个平台需要额外装东西。两份实现（ps1 + sh）只会分叉，分叉的后果是"某一个平台的构建悄悄
用着旧权重"。

【完整性判据与运行时一致】逐文件比对**字节数 + sha256**，数据来源就是 models.json
（与 model_store::looks_complete 同一份清单；那份只比字节数是因为它跑在每次启动的
路径上，这里可以付出哈希的代价）。版本号是内容派生值，取错版本必然在这里暴露。

用法（取权重）：
    python scripts/setup/fetch-models.py                # 缺就取（默认从 release 资产）
    python scripts/setup/fetch-models.py --check        # 只检查，缺就退出码 1
    python scripts/setup/fetch-models.py --source DIR   # 从本地目录拷贝（已有权重）
    python scripts/setup/fetch-models.py --archive ZIP  # 从本地 zip 解压
    python scripts/setup/fetch-models.py --base-url URL # 换一个下载源（镜像）

用法（发布新权重，一年几次；流程见 docs/models.md）：
    python scripts/setup/fetch-models.py --package             # 打出可复算的 release 资产
    python scripts/setup/fetch-models.py --verify-archive ZIP  # 校验某个 zip 是否等于这份清单
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
# 待发布的资产放 dist/（与其它产物同一处，见 scripts/README.md）。构建只会重置
# dist/_stage，因此这里的东西不会被下一次构建抹掉。
ASSET_DIR = REPO_ROOT / "dist" / "models"
# 【为什么是 zip 而不是 tar.zst】三个平台都能解（Expand-Archive / zipfile / unzip），
# 不需要额外的 zstd 二进制。150 MB 的一次性下载里，压缩率差别可以忽略。
DEFAULT_BASE_URL = "https://github.com/ARounder-183/HiFiShifter/releases/download"

# 目录里允许存在、但既不进清单也不进资产的构建产物：macOS 的 build.rs 会把
# pc_nsf_hifigan.onnx 改写成 CoreML 变体（backend/src-tauri/build.rs::
# generate_coreml_model_variant），产物落在同一棵资源树里。它由构建自动生成 ——
# 进了清单就意味着 Windows/Linux 白装 56 MB 用不上的权重，而且版本号会取决于
# "构建环境里有没有编过 macOS 目标"。打包时按清单取文件，它天然被排除；这里只是
# 别把它报成"多余文件"。
GENERATED_ARTIFACTS = frozenset({"nsf_hifigan/pc_nsf_hifigan_coreml.onnx"})

# 资产内部每个条目的时间戳。固定成 zip 能表示的最小值：让"同一份权重打两次包"
# 逐字节相同，sha256 才有资格当发布物的身份。
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


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


def derive_version(manifest: dict) -> str:
    """重算清单应当有的版本号（算法与 scripts/build/write-models-manifest.ps1 相同）。

    【为什么打包前要重算一遍】version 是清单里**唯一不能手写**的字段，而它同时决定
    release 的 tag 与 CI 的缓存键。重算一次，手改过的清单会在打包时当场失败，而不是
    变成一份"版本号与内容无关"的资产被发出去。两份实现万一哪天分叉，表现是拒绝打包
    （fail closed），不是发错东西。
    """
    # 排序键用反斜杠形式：PowerShell 那边按「完整路径」排序，而完整路径里只有分隔符
    # 是反斜杠。这里对齐它，免得两种排序在个别文件名上给出不同顺序。
    entries = sorted(manifest["files"], key=lambda item: item["path"].replace("/", "\\"))
    lines = "\n".join(f"{entry['path']} {entry['sha256']}" for entry in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:12]


def missing_files(models_dir: Path, manifest: dict) -> list[str]:
    """返回不完整（缺文件 / 字节数不符）的条目。

    【为什么先只比字节数】哈希 150 MB 要几百毫秒到几秒，而这个脚本在每次构建前都跑。
    字节数不符已经能挡住绝大多数故障（没解压完、拷了一半、版本混装）；真正的哈希
    校验放在"下载 / 解压之后"那一步（见 verify_hashes）。
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
        # 【为什么按小写比】清单由 write-models-manifest.ps1 生成，Get-FileHash 给的是
        # 大写十六进制；hexdigest() 给的是小写。两边都是同一个哈希，直接比字符会让
        # 每一份**正确**的权重都判成"校验失败"。
        if actual != entry["sha256"].lower():
            problems.append(f"sha256 {entry['path']}: {actual} != {entry['sha256']}")
    return problems


def stray_files(models_dir: Path, manifest: dict) -> list[str]:
    """清单没登记、也不是已知构建产物的文件 —— 通常是"新模型文件忘了进清单"。"""
    listed = {entry["path"] for entry in manifest["files"]}
    found: list[str] = []
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(models_dir).as_posix()
        if relative in listed or relative in GENERATED_ARTIFACTS or relative == "models.json":
            continue
        found.append(relative)
    return found


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


def format_mib(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def build_asset(destination: Path, manifest: dict) -> None:
    """把清单列出的文件写成一个可复算的 zip（发布资产）。

    【为什么不用右键"压缩为 zip"】资产的身份就是它的 sha256：上传之后任何人重打一遍
    包、比一下哈希，就知道自己手上是不是同一份权重。而资源管理器会写进打包那一刻的时间戳、
    用不定的条目顺序 —— 同一份权重每次得到不同字节，那个哈希就没有意义了。

    【为什么存而不压】deflate 的字节流随 zlib 版本变化，跨机器就不再可复算；而 ONNX
    权重是浮点张量，压得动的只有个位数百分点。用几十 MB 的下载体积换"处处可复算"是
    划算的：这个包是故障排查时唯一能把"你装的那份"和"我发的那份"对上的东西。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    # 【为什么先写 .tmp 再改名】150 MB 写到一半失败会留下一个"看起来存在"的资产，
    # 而它是发布物 —— 下一次有人把它传上去就是灾难。
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as bundle:
            for entry in sorted(manifest["files"], key=lambda item: item["path"].replace("/", "\\")):
                info = zipfile.ZipInfo(entry["path"], date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                # 不加这两行，Windows 打的包与 Unix 打的包头部不同，sha256 也就不同。
                info.create_system = 3  # 0 = MS-DOS, 3 = Unix
                info.external_attr = 0o644 << 16
                with (MODELS_DIR / entry["path"]).open("rb") as reader, bundle.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_digest(archive: Path) -> str:
    """写 sha256 边车（与 sha256sum -c 同一格式），下载方不装 Python 也能核对。"""
    digest = sha256_of(archive)
    sidecar = archive.with_name(archive.name + ".sha256")
    with sidecar.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return digest


def verify_asset(archive: Path, manifest: dict) -> list[str]:
    """把一个 zip 当成发布物检查：条目必须**恰好**是清单列的那几项。

    【为什么多一个文件也报错】资产多带东西只有两种来路，都是事故：把上一版权重一起
    打了进去（安装器按清单装，多出来的既进不了共享库、又白占用户的下载），或者把
    CoreML 变体当成了模型文件。少一个文件则更直接：装完就是"模型不可用"。
    """
    problems: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = {name for name in bundle.namelist() if not name.endswith("/")}
        expected = {entry["path"] for entry in manifest["files"]}
        problems += [f"zip 里缺少 {path}" for path in sorted(expected - names)]
        problems += [f"zip 里多出 {path}" for path in sorted(names - expected)]
        if problems:
            return problems
        with tempfile.TemporaryDirectory(prefix="hfs-models-verify-") as tmp:
            bundle.extractall(tmp)
            problems += verify_hashes(Path(tmp), manifest)
    sidecar = archive.with_name(archive.name + ".sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="utf-8").split()[0].lower()
        actual = sha256_of(archive)
        if recorded != actual:
            problems.append(f"{sidecar.name} 与 zip 不符：{actual} != {recorded}")
    return problems


def package_asset(destination: Path, manifest: dict) -> int:
    version = manifest["version"]
    problems = missing_files(MODELS_DIR, manifest)
    if not problems:
        problems = verify_hashes(MODELS_DIR, manifest)
    if not problems:
        derived = derive_version(manifest)
        if derived != version:
            problems.append(f"清单的 version {version} 与内容重算值 {derived} 不符（清单被手改过）")
    if problems:
        print(f"拒绝打包（清单版本 {version}）：", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "  权重改过就重生成清单：pwsh -File scripts/build/write-models-manifest.ps1\n"
            "  没改过就先补齐权重：python scripts/setup/fetch-models.py",
            file=sys.stderr,
        )
        return 1
    for stray in stray_files(MODELS_DIR, manifest):
        print(f"警告：{stray} 不在清单里，不会进资产（新增了模型文件就先重生成清单）", file=sys.stderr)

    build_asset(destination, manifest)
    digest = write_digest(destination)
    sidecar = destination.with_name(destination.name + ".sha256")
    print(f"资产 {destination}（{format_mib(destination.stat().st_size)}）")
    print(f"  sha256 {digest}")
    print(f"  边车   {sidecar}")
    print("下一步：")
    print(f"  1) python scripts/setup/fetch-models.py --verify-archive {destination}")
    print(f"  2) 传到 tag models-{version}：命令与顺序见 docs/models.md")
    print("  3) 线上资产确认能下载之后，再合并清单")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="取 / 打包 HiFiShifter 的模型权重")
    parser.add_argument("--check", action="store_true", help="只检查是否完整（缺则退出码 1）")
    parser.add_argument("--source", type=Path, help="从本地模型目录拷贝")
    parser.add_argument("--archive", type=Path, help="从本地 zip 解压")
    parser.add_argument(
        "--package",
        nargs="?",
        const="",
        metavar="OUT",
        help="打包成 release 资产（默认 dist/models/models-<version>.zip，不联网）",
    )
    parser.add_argument(
        "--verify-archive",
        type=Path,
        metavar="ZIP",
        help="校验一个 zip 资产是否恰好等于这份清单",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HIFISHIFTER_MODELS_BASE_URL", DEFAULT_BASE_URL),
        help="release 资产的基地址（镜像用；也可设 HIFISHIFTER_MODELS_BASE_URL）",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    version = manifest["version"]

    if args.verify_archive:
        archive = args.verify_archive.expanduser().resolve()
        if not archive.is_file():
            raise SystemExit(f"找不到 {archive}")
        problems = verify_asset(archive, manifest)
        if problems:
            print(f"{archive.name} 不是版本 {version} 的资产：", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{archive.name} 与清单一致（版本 {version}，{len(manifest['files'])} 个文件）")
        return 0

    if args.package is not None:
        destination = (
            Path(args.package).expanduser().resolve()
            if args.package
            else ASSET_DIR / f"models-{version}.zip"
        )
        return package_asset(destination, manifest)

    problems = missing_files(MODELS_DIR, manifest)
    if not problems:
        if args.check or args.source or args.archive:
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
