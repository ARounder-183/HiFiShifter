# 波形系统 V1 旧版代码清理

> 日期：2026-03-24  
> 分支：develop

## 背景

HiFiShifter 的波形渲染系统已从 V1（单级峰值缓存）全面升级到 V2（三级 Mipmap 峰值系统 HFSPeaks v2）。
但 V1 代码仍残留在项目中，增加了维护负担和代码复杂度。本次变更删除所有 V1 旧版代码。

---

## 后端变更

### 删除的文件

| 文件 | 说明 |
|------|------|
| `backend/src-tauri/src/audio/waveform.rs` | V1 `CachedPeaks` 结构体 + `segment_from_cached()` |
| `backend/src-tauri/src/audio/waveform_disk_cache.rs` | V1 磁盘缓存（HFSPEAKS v1 格式） |

### 修改的文件

| 文件 | 变更内容 |
|------|---------|
| `commands/waveform.rs` | 将 `WaveformPeaksSegmentPayload` 从 `waveform.rs` 迁移到此处；移除 `use crate::waveform` |
| `commands/common.rs` | `guard_waveform_command` 改用 `super::waveform::WaveformPeaksSegmentPayload` |
| `commands.rs` | 返回类型引用改为 `self::waveform::WaveformPeaksSegmentPayload` |
| `lib.rs` | 删除 `mod waveform` 和 `mod waveform_disk_cache` 声明；缓存目录初始化改用 `hfspeaks_v2::default_cache_dir()` / `ensure_cache_dir()` |
| `state.rs` | 删除 V1 `waveform_cache` 字段和 `get_or_compute_waveform_peaks()` 方法；`clear_waveform_cache()` 改用 `hfspeaks_v2::clear_cache_dir()` |
| `audio/audio_utils.rs` | 删除 `compute_minmax_peaks()`、`compute_minmax_peaks_hound()`、`compute_minmax_peaks_symphonia()` 三个仅被 V1 调用的函数 |
| `audio/hfspeaks_v2.rs` | 删除 V1→V2 迁移代码（`LegacyCachedPeaks`、`try_load_v1_format`、`migrate_v1_to_v2`、`downsample_mipmap`、`try_migrate_from_v1`）；新增从 `waveform_disk_cache.rs` 迁移的 `ClearStats`、`default_cache_dir()`、`ensure_cache_dir()`、`clear_cache_dir()` |

---

## 前端变更

### 删除的文件

| 文件 | 说明 |
|------|------|
| `frontend/src/workers/waveformProcessor.worker.ts` | V1 时代的前端 downsample Web Worker，整个项目中无任何代码导入或引用 |
| `frontend/src/workers/` 目录 | Worker 删除后目录为空，一并删除 |

### 修改的文件

| 文件 | 变更内容 |
|------|---------|
| `features/session/sessionSlice.ts` | 删除 `dragDirection` → `selectDragDirection` 旧版持久化字段迁移 shim（第1497-1503行） |
| `services/api/settings.ts` | 从 `UiSettings` 接口中删除旧版 `dragDirection` 字段定义 |

### 保留的 Fallback（非旧版遗留）

| 文件 | Fallback 机制 | 保留原因 |
|------|-------------|---------|
| `utils/offscreenCanvasCache.ts` | OffscreenCanvas → HTMLCanvasElement | 浏览器兼容性 fallback |
| `utils/waveformMipmapStore.ts` `getNearestLoadedLevel` | 优先级别未加载时使用最近可用级别 | V2 mipmap 运行时 fallback |
| `clearWaveformCache` 全链路 | 清除波形缓存菜单功能 | 当前正在使用的功能 |
| `lockParamLines` → `lockParamLinesEnabled` | 持久化字段名到 state 字段名的映射 | 持久化格式兼容（非版本迁移） |

---

## 注意事项

- 后端代码未经编译验证（本地无 MSVC 环境），请在有编译环境的机器上执行 `cargo check` 确认
- `dragDirection` 旧字段迁移被移除后，从**非常旧版本**（使用 `dragDirection` 字段名保存设置的版本）升级的用户可能丢失此项设置（会回退到默认值 `y-only`）

---

## Phase 2: 波形系统优化

### 陈旧代码修正

| 文件 | 问题 | 修改 |
|------|------|------|
| `types/api.ts` | `MipmapLevel = 0 \| 1 \| 2 \| 3`，实际只有 3 级 | 改为 `0 \| 1 \| 2`，更新注释 |
| `pianoRoll/useClipsPeaksForPianoRoll.ts` | 注释引用不存在的 Level 3 (div ~8192) | 更新为实际三级描述 |

### P0-1: IPC 二进制传输优化（消除 JSON number[] 膨胀）

**问题：** Tauri v2 将 `Vec<u8>` 序列化为 JSON `number[]`，造成 3~5 倍传输膨胀。

**方案：** 后端使用 Base64 编码传输，前端用 `atob()` 高效解码。

| 文件 | 变更内容 |
|------|---------|
| `commands/waveform.rs` | `get_waveform_mipmap_binary` 返回类型从 `Vec<u8>` 改为 `String`（Base64 编码）；添加 `use base64::Engine` |
| `commands.rs` | 门面层返回类型同步改为 `String` |
| `services/api/waveform.ts` | `getWaveformMipmapBinary` 返回类型从 `number[]` 改为 `string` |
| `utils/waveformBinaryCodec.ts` | 移除 `numberArrayToArrayBuffer`（逐字节拷贝）；新增 `base64ToArrayBuffer`（高效解码）；`decodeWaveformFromBase64` 替代 `decodeWaveformFromNumberArray` |
| `utils/waveformMipmapStore.ts` | 导入和调用更新为 `decodeWaveformFromBase64` |

**预估收益：** 传输体积减少 60-80%（Base64 膨胀率 33% vs JSON number[] 300-500%），解码速度提升 5-10x。

### P1-1: Piano Roll 波形请求 Debounce

**问题：** 快速缩放/滚动时，每次 React 渲染周期都可能触发新的异步波形加载请求。

**方案：** 在 `useClipsPeaksForPianoRoll` 的异步 fetch 路径添加 50ms debounce。

| 文件 | 变更内容 |
|------|---------|
| `pianoRoll/useClipsPeaksForPianoRoll.ts` | 缓存命中的同步快速路径保持立即执行；异步 preload+fetch 路径添加 50ms debounce + cleanup |

**设计要点：**
- 缓存命中时立即返回（零延迟），用户无感知
- 仅对需要 IPC 的异步路径延迟 50ms
- useEffect cleanup 自动取消过时的 debounce 定时器
- 配合现有 `requestIdRef` staleness check 双重保护

---

## Phase 3: 批量预加载优化（消除重复 IPC 往返）

### 问题

打开工程时，对每个音频文件执行 `preload` → 1 次 preloadWaveformMipmap + 3 次 getWaveformMipmapBinary = **4 次 IPC/文件**。
10 个音频文件 = 40 次 IPC 往返，每次 IPC 有 ~10-30ms 序列化/调度开销，即使磁盘缓存命中也需 ~800ms-1.2s。

### 方案

新增 `batch_get_waveform_mipmap` 后端命令，接收 `Vec<String>` 源文件路径列表，一次性返回所有文件 3 级 mipmap 的 Base64 数据。
前端 `waveformMipmapStore` 新增 `batchPreload()` 方法，`TimelinePanel.tsx` 改为收集所有新 sourcePath 后批量调用。

### 数据协议

请求：`{ sourcePaths: string[] }`

响应：`Record<string, [string, string, string]>` — sourcePath → [L0_base64, L1_base64, L2_base64]

### 变更文件

| 层级 | 文件 | 变更内容 |
|------|------|---------|
| 后端实现 | `commands/waveform.rs` | 新增 `batch_get_waveform_mipmap()`：遍历所有 sourcePath，调用 `get_or_compute_waveform_peaks_v2` + `to_binary_level` + Base64 编码 |
| 后端门面 | `commands.rs` | 新增 `#[tauri::command] batch_get_waveform_mipmap` |
| 后端注册 | `lib.rs` | `invoke_handler` 注册 `batch_get_waveform_mipmap` |
| 前端 API | `services/api/waveform.ts` | 新增 `batchGetWaveformMipmap()` |
| 前端 invoke | `services/invoke.ts` | 新增 `batch_get_waveform_mipmap` 参数映射 |
| 前端缓存 | `utils/waveformMipmapStore.ts` | 新增 `batchPreload()` 方法，内含 fallback 到逐个 `preload()` |
| 前端调用 | `components/layout/TimelinePanel.tsx` | preload useEffect 改为收集新路径后调用 `batchPreload()` |

### 容错设计

- `batchPreload()` 先过滤已完全缓存（3 级都有）的文件，仅请求缺失的
- 批量 IPC 失败时自动 fallback 到逐个 `preload()`（保持向后兼容）
- 后端计算失败的文件返回 3 个空字符串，前端跳过解码

### 性能预估

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 10 文件项目重新打开（全部有缓存） | 40 次 IPC，~800ms-1.2s | 1 次 IPC，~100-200ms |
| 10 文件项目首次打开（全部需计算） | 40 次 IPC + N 秒计算（串行） | 1 次 IPC + N 秒计算（串行） |
| 单文件新增（运行时导入） | 4 次 IPC | 1 次 IPC（batchPreload 单元素） |
