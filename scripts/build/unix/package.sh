#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unix 插件分发入口：一次编译 → 每个 OS 一个用户可安装的产物。
#
#     scripts/build/unix/package.sh                    # VST3 + CLAP → tar.gz / pkg
#     scripts/build/unix/package.sh --formats=vst3     # 只出 VST3
#     scripts/build/unix/package.sh --reuse-frontend   # 复用 frontend/dist
#
# 【为什么构建分开、分发统一】两个格式的**构建**是两件不同的事：不同的 crate、不同的
# bundle 布局、不同的入口符号、不同的黑盒探针。把它们塞进一个脚本，得到的是
# build.ps1 那样一棵 if 分支树 —— 而 assets 嵌套那个 bug 就长在其中一个分支里。
# 但**分发**必须统一：用户不关心格式，而且插件与 App 共用文档格式与模型清单版本，
# 分开下载迟早出现"新插件 + 旧 App"。
#
# 所以：格式脚本各自构建（scripts/<fmt>/unix/build.sh），这里只做三件事 ——
# 编译一次前端、按格式调用、把结果装进同一个产物。
#
# 产物：
#   Linux   dist/HiFiShifter_v<版本>_<os>-<arch>-plugins.tar.gz（含 install.sh）
#   macOS   dist/HiFiShifter_v<版本>_<os>-<arch>-plugins.pkg（装到 /Library/Audio/Plug-Ins）
#
# 签名与公证：设置 HIFISHIFTER_CODESIGN_IDENTITY 后，macOS 用 Developer ID 签名，
# 并可用 HIFISHIFTER_NOTARY_PROFILE 走 notarytool 公证 + stapler 装订。默认 ad-hoc
# 签名 —— 仅够 CI 自检，不够分发给陌生用户（Gatekeeper 会拦）。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

FORMATS="vst3,clap"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
REUSE_FRONTEND=0
SKIP_VERIFY=0
MERGE_ARCH=""

log()  { printf '[package] %s\n' "$*"; }
step() { printf '\n[package] ── %s\n' "$*"; }
die()  { printf '[package] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
用法：scripts/build/unix/package.sh [选项]

  --formats=vst3,clap   要打包的格式（默认两个都打）
  --profile=release|dist
  --out=DIR             产物目录（默认 dist）
  --reuse-frontend      复用已有 frontend/dist（要求它已存在）
  --no-verify           跳过格式自检（只在探针跑不了的机器上用）
  --merge-arch=DIR      把另一个架构的 dist 合并成 universal（需要 lipo，仅 macOS）
  -h, --help
USAGE
}

# 【为什么要同时认 --x=v 与 --x v】工作流里写的是空格形式（可读性更好），
# 而命令行上 `--profile=dist` 更常见。只认一种的代价是 CI 上一句
# "未知参数：--profile" —— 那正是第一次把 package.sh 接进 build.yml 时踩到的。
while [ $# -gt 0 ]; do
    case "$1" in
        --formats=*)      FORMATS="${1#*=}"; shift ;;
        --formats)        [ $# -ge 2 ] || die "--formats 需要值"; FORMATS="$2"; shift 2 ;;
        --profile=*)      PROFILE="${1#*=}"; shift ;;
        --profile)        [ $# -ge 2 ] || die "--profile 需要值"; PROFILE="$2"; shift 2 ;;
        --out=*)          OUT_DIR="${1#*=}"; shift ;;
        --out)            [ $# -ge 2 ] || die "--out 需要值"; OUT_DIR="$2"; shift 2 ;;
        --reuse-frontend) REUSE_FRONTEND=1; shift ;;
        --no-verify)      SKIP_VERIFY=1; shift ;;
        --merge-arch=*)   MERGE_ARCH="${1#*=}"; shift ;;
        --merge-arch)     [ $# -ge 2 ] || die "--merge-arch 需要值"; MERGE_ARCH="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "未知参数：$1（--help 看用法）" ;;
    esac
done

hfs_detect_platform
PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")"
readonly PRODUCT_VERSION
ARTIFACT_BASE="$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-plugins"

log "平台 ${HFS_TAG}｜版本 ${PRODUCT_VERSION}｜格式 ${FORMATS}｜profile $PROFILE"

wants() {
    case ",$FORMATS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# ── 阶段 1：前端（只构建一次，两个格式共用同一份产物）───────────────────────
stage_frontend() {
    if [ "$REUSE_FRONTEND" = "1" ]; then
        [ -s "$FRONTEND_DIST/plugin.html" ] || die "--reuse-frontend 但 $FRONTEND_DIST/plugin.html 不存在"
        log "复用已有前端产物：$FRONTEND_DIST"
        return
    fi
    step "构建前端（vite，含 plugin.html 入口）"
    command -v node >/dev/null 2>&1 || die "缺少 node：需要 ^20.19.0 或 >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci 失败（检查网络与 frontend/package-lock.json）"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "前端构建失败"
    [ -s "$FRONTEND_DIST/plugin.html" ] || die "前端构建完成但 plugin.html 不存在"
}

# ── 阶段 2：按格式构建（格式脚本各管各的布局）──────────────────────────────
build_format() {
    local fmt="$1" flag
    case "$fmt" in
        vst3) flag="--skip-frontend" ;;
        clap) flag="--reuse-frontend" ;;
        *) die "未知格式：$fmt" ;;
    esac
    step "构建 $fmt"
    if [ -s "$FRONTEND_DIST/plugin.html" ]; then
        bash "$REPO_ROOT/scripts/$fmt/unix/build.sh" --stage=bundle \
            --profile="$PROFILE" --out="$OUT_DIR" "$flag"
    else
        bash "$REPO_ROOT/scripts/$fmt/unix/build.sh" --stage=bundle \
            --profile="$PROFILE" --out="$OUT_DIR"
    fi
}

verify_format() {
    local fmt="$1"
    [ "$SKIP_VERIFY" = "1" ] && { log "按 --no-verify 跳过 $fmt 自检"; return 0; }
    step "自检 $fmt"
    case "$fmt" in
        vst3) bash "$REPO_ROOT/scripts/vst3/unix/verify.sh" "$OUT_DIR/HiFiShifter.vst3" ;;
        clap) bash "$REPO_ROOT/scripts/clap/verify.sh" "$OUT_DIR/clap/HiFiShifter.clap" ;;
    esac
}

# ── 阶段 3：合并 universal（可选，仅 macOS）─────────────────────────────────
#
# 【为什么是可选项】App 侧目前是按架构分别发 DMG 的，插件单独做 universal 会得到
# "universal 插件 + 分架构 App"的错配。真正要 universal 时，两边的 CI 腿都产出
# dist/，这里用 --merge-arch 指过去即可。
merge_arch() {
    local other="$1" rel
    [ "$HFS_OS" = "macos" ] || die "--merge-arch 只在 macOS 上有意义"
    command -v lipo >/dev/null 2>&1 || die "缺少 lipo（需要 Xcode 命令行工具）"
    [ -d "$other" ] || die "--merge-arch 指向的目录不存在：$other"
    step "合并 universal（${other}）"
    while IFS= read -r rel; do
        [ -f "$other/$rel" ] || continue
        file -b "$OUT_DIR/$rel" | grep -q 'Mach-O' || continue
        file -b "$other/$rel" | grep -q 'Mach-O' || continue
        lipo -create "$OUT_DIR/$rel" "$other/$rel" -output "$OUT_DIR/$rel.universal" \
            || die "lipo 合并失败：$rel"
        mv -f "$OUT_DIR/$rel.universal" "$OUT_DIR/$rel"
        log "universal：$rel"
    done <<< "$( cd "$OUT_DIR" && find HiFiShifter.vst3 clap -type f 2>/dev/null )"
    hfs_sign_bundle "$OUT_DIR/HiFiShifter.vst3" "$OUT_DIR/HiFiShifter.vst3/Contents/MacOS/HiFiShifter"
    hfs_sign_bundle "$OUT_DIR/clap/HiFiShifter.clap" "$OUT_DIR/clap/HiFiShifter.clap/Contents/MacOS/HiFiShifter"
}

# ── 阶段 4：组装分发产物 ────────────────────────────────────────────────────
package_linux() {
    local stage="$OUT_DIR/stage-plugins" entries=()
    step "打包 Linux 插件包"
    rm -rf "$stage"
    mkdir -p "$stage"
    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "缺少 $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$stage/"
        entries+=("HiFiShifter.vst3")
    fi
    if wants clap; then
        [ -f "$OUT_DIR/clap/HiFiShifter.clap" ] || die "缺少 $OUT_DIR/clap/HiFiShifter.clap"
        cp "$OUT_DIR/clap/HiFiShifter.clap" "$stage/"
        cp -R "$OUT_DIR/clap/HiFiShifter-resources" "$stage/"
        entries+=("HiFiShifter.clap" "HiFiShifter-resources")
    fi
    # 两个格式带的是同一份 152 MB 权重。暂存目录里用硬链接让它们共享 inode，
    # tar 就会把第二份存成硬链接条目 —— 压缩包从 314 MB 回到 160 MB 左右。
    # 解包后仍是两份（cp -R 会断开硬链接），磁盘占用与"各自自包含"的语义都不变。
    if wants vst3 && wants clap; then
        rm -rf "$stage/HiFiShifter-resources/models"
        cp -al "$stage/HiFiShifter.vst3/Contents/Resources/models" "$stage/HiFiShifter-resources/models"
    fi
    # 【为什么要判空】macOS 自带的是 bash 3.2，${arr[@]} 在数组为空时会以
    # "unbound variable" 退出（set -u）。空数组只可能来自 --formats 写错，
    # 那种情况给一句人话，而不是让用户去猜数组展开。
    [ ${#entries[@]} -gt 0 ] || die "没有选中任何格式：--formats=$FORMATS"
    cp "$REPO_ROOT/scripts/plugin/unix/install.sh" "$stage/install.sh"
    chmod +x "$stage/install.sh"
    hfs_stage_licenses "$stage" "$REPO_ROOT"
    hfs_tarball "$ARTIFACT_BASE.tar.gz" "$stage" "${entries[@]}" install.sh LICENSE licenses
    rm -rf "$stage"
}

package_macos() {
    local root="$OUT_DIR/pkgroot" numeric identity
    step "打包 macOS 插件 pkg"
    command -v pkgbuild >/dev/null 2>&1 || die "缺少 pkgbuild（需要 Xcode 命令行工具）"
    rm -rf "$root"
    mkdir -p "$root/Library/Audio/Plug-Ins/VST3" "$root/Library/Audio/Plug-Ins/CLAP"
    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "缺少 $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$root/Library/Audio/Plug-Ins/VST3/"
    fi
    if wants clap; then
        [ -d "$OUT_DIR/clap/HiFiShifter.clap" ] || die "缺少 $OUT_DIR/clap/HiFiShifter.clap"
        cp -R "$OUT_DIR/clap/HiFiShifter.clap" "$root/Library/Audio/Plug-Ins/CLAP/"
    fi
    numeric="$(hfs_numeric_version "$PRODUCT_VERSION")"
    rm -f "$ARTIFACT_BASE.pkg"
    pkgbuild --root "$root" --identifier com.hifishifter.plugins --version "$numeric" \
        --install-location / --ownership recommended "$ARTIFACT_BASE.pkg" >/dev/null \
        || die "pkgbuild 失败"
    rm -rf "$root"

    identity="${HIFISHIFTER_CODESIGN_IDENTITY:--}"
    if [ "$identity" != "-" ]; then
        productsign --sign "$identity" "$ARTIFACT_BASE.pkg" "$ARTIFACT_BASE.signed.pkg" >/dev/null \
            || die "productsign 失败"
        mv -f "$ARTIFACT_BASE.signed.pkg" "$ARTIFACT_BASE.pkg"
        log "pkg 已用 $identity 签名"
        if [ -n "${HIFISHIFTER_NOTARY_PROFILE:-}" ]; then
            step "公证（notarytool）"
            xcrun notarytool submit "$ARTIFACT_BASE.pkg" \
                --keychain-profile "$HIFISHIFTER_NOTARY_PROFILE" --wait || die "公证失败"
            xcrun stapler staple "$ARTIFACT_BASE.pkg" || die "stapler 装订失败"
            log "已公证并装订"
        else
            log "未设置 HIFISHIFTER_NOTARY_PROFILE：跳过公证（用户首次加载会被 Gatekeeper 拦）"
        fi
    else
        log "ad-hoc 签名：仅够 CI 自检，分发前必须换成 Developer ID"
    fi
    local digest
    digest="$(hfs_sha256 "$ARTIFACT_BASE.pkg")"
    printf '%s  %s\n' "$digest" "$(basename "$ARTIFACT_BASE.pkg")" >"$ARTIFACT_BASE.pkg.sha256"
    log "产物 $(du -h "$ARTIFACT_BASE.pkg" | cut -f1)：$ARTIFACT_BASE.pkg"
}

# ── 入口 ─────────────────────────────────────────────────────────────────────
stage_frontend
if wants vst3; then build_format vst3; verify_format vst3; fi
if wants clap; then build_format clap; verify_format clap; fi
if [ -n "$MERGE_ARCH" ]; then merge_arch "$MERGE_ARCH"; fi

if [ "$HFS_OS" = "linux" ]; then
    package_linux
else
    package_macos
fi

log "完成"
