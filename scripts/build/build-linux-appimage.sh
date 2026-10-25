#!/usr/bin/env bash
set -euo pipefail

# Build the HiFiShifter AppImage on Linux.
#
# 用法: ./scripts/build/build-linux-appimage.sh [--release|--debug]
#
# 前置条件：
#   1. Rust 工具链（stable）
#   2. 系统依赖（见 scripts/setup/install_deps_linux.sh）
#   3. appimagetool（没有的话 linux-appimage.sh 会自己下载到 ~/.cache）
#   4. 前端已构建（cd frontend && npm ci && npm run build）
#   5. SoundTouch 源码已就位（见 build.rs 的自动 clone）
#
# 这个脚本只负责"编译 + 让 Tauri 组装 AppDir"。**包里到底放什么、放得对不对**由
# scripts/build/linux-appimage.sh 负责，CI 调用的是同一个脚本、同一条 package
# 子命令：prune（把桌面栈交还宿主）→ verify（ldd 断言）→ repack → smoke。
#
# 【为什么必须共用一份】历史上本地脚本与 CI 用的是两套库清单：CI 会覆盖
# tauri.linux.conf.json 注入一份手工映射，本地脚本则什么都不注入。于是"我在 WSL
# 里构建出来是好的"从来不能证明 CI 产物是好的 —— 0.1.0-beta.13 的 EGL 崩溃正是
# 这样漏出去的。改成单一真源后，本地能跑通就意味着同一套逻辑在 CI 上跑通。

PROFILE="${1:---release}"
PROFILE="${PROFILE#--}"  # strip leading --
CARGO_PROFILE="$PROFILE"
TARGET_PROFILE="$PROFILE"
if [ "$PROFILE" = "debug" ] || [ "$PROFILE" = "dev" ]; then
    # Cargo 把默认 debug profile 叫 dev，但产物目录仍是 target/debug/。
    CARGO_PROFILE="dev"
    TARGET_PROFILE="debug"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BACKEND_DIR="$PROJECT_DIR/backend"

echo "=== HiFiShifter Linux AppImage Build ==="
echo "Project: $PROJECT_DIR"
echo "Profile: $PROFILE"

# Step 1: 编译 Rust 后端并让 Tauri 组装 AppDir。
# 注意：打包那一步在 WSL2 上可能失败（linuxdeploy/FUSE），但 AppDir 在此之前
# 就已经组装完毕，所以这里容忍失败，后面统一由 linux-appimage.sh 重打包。
echo ""
echo "--- Step 1: Building Rust binary + AppDir ---"
cd "$BACKEND_DIR"
set +e
HIFISHIFTER_SKIP_FRONTEND_BUILD=1 APPIMAGE_EXTRACT_AND_RUN=1 cargo tauri build \
    --bundles appimage \
    -- --locked --no-default-features --features onnx \
    ${CARGO_PROFILE:+--profile "$CARGO_PROFILE"} 2>&1 | tail -5
build_rc=${PIPESTATUS[0]}
set -e
if [ "$build_rc" -ne 0 ]; then
    echo "(Tauri bundling exited $build_rc — 只要 AppDir 齐了就可以继续)"
fi

# Step 2: 定位 AppDir。cargo workspace 合并后产物统一落在 backend/target/，
# 部分布局下还会多一层 target triple。
echo ""
echo "--- Step 2: Locating AppDir ---"
RELEASE_DIR="$BACKEND_DIR/target/$TARGET_PROFILE"
if [ ! -d "$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
    TRIPLE_DIR="$BACKEND_DIR/target/x86_64-unknown-linux-gnu/$TARGET_PROFILE"
    if [ -d "$TRIPLE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
        RELEASE_DIR="$TRIPLE_DIR"
    fi
fi
if [ ! -d "$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
    echo "ERROR: 找不到 AppDir：$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" >&2
    echo "Tauri 构建大概率失败了，请检查上面的输出。" >&2
    exit 1
fi

# Step 3: 剪裁 → 校验 → 重打包 → 冒烟（与 CI 完全同一份逻辑）。
echo ""
echo "--- Step 3: Packaging ---"
bash "$SCRIPT_DIR/linux-appimage.sh" package "$RELEASE_DIR"

echo ""
echo "=== SUCCESS ==="
ls -lh "$RELEASE_DIR"/bundle/appimage/*.AppImage
