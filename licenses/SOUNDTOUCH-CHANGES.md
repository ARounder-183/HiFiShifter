# SoundTouch 构建期补丁说明（LGPL-2.1 第 2.a 条披露）

本应用的 Windows / macOS / Linux 发行物以**动态库**形式分发
[SoundTouch](https://codeberg.org/soundtouch/soundtouch)（LGPL-2.1）的
`SoundTouchDLL`。构建期对上游源码（pin 见 `scripts/env/plugin-sdks.json` 的
`soundtouch` 段，tag `2.3.3`）做了以下两处修改，由
`backend/hifishifter-kernel/build.rs` 在编译时自动打上：

1. **`source/SoundTouchDLL/SoundTouchDLL.rc`**
   `#include "afxres.h"` → `#include <windows.h>`，并在缺失时补充
   `IDC_STATIC` 的 `#define`。原因：`afxres.h` 属于 MFC，构建机不总是安装；
   `windows.h` 提供了 DLL 资源编译所需的全部声明。

2. **`CMakeLists.txt`**
   `cmake_minimum_required(VERSION 3.1)` → `3.5`。原因：CMake ≥3.27 弃用、
   ≥4.0 硬性拒绝 `3.1`，不改则现代构建机无法配置。

两处补丁都不触碰音频算法本身（时基、采样率、拉伸逻辑与上游 2.3.3 一致）。

## 源码要约（LGPL-2.1 第 6 条）

- 上游主页：<https://www.surina.net/soundtouch/>，仓库：
  <https://codeberg.org/soundtouch/soundtouch>
- 本仓库锁定的修订：`2.3.3`（commit `e83424d5928ab8513d2d082779c275765dee31b9`）
- 完整许可文本：`licenses/COPYING.LGPL-2.1`（随每个发行物一同分发）
- 按上述修订重建 `SoundTouchDLL` 的最短路径：安装 Rust 工具链与 CMake 后，
  在仓库根目录执行 `cargo build`（内核 build.rs 会按 pin 自动拉取、打补丁并
  构建 DLL）；或手工克隆上游后应用上文两处修改。
