# F-4：倒放 take 的 ARA region 时间坐标系 —— **已出结果：已镜像坐标**

日期：2026-10-10
主机：REAPER 7.82 / Windows x64；插件 v0.1.0-beta.15（`C:\Program Files\Common Files\VST3\HiFiShifter.vst3`）
对应：`docs/plans/2026-10-09-ara-plugin-fade-rewrite-reverse-auth-and-stability.md` 的 Task 3.0 / 3.2
性质：**只读探针**，不改产品行为。本文件记录方法、原始读数与判决，供 Task 3.4 落地时引用。

## 结论（一句话）

倒放 take 的 ARA region 报的 `start_in_modification_time` / `duration_in_modification_time`
用的是**已镜像坐标（播放顺序坐标）**，**不是**正向源坐标。即：region 报的区间是"正向源窗口
在源时长内的镜像"，而该 take **实际播放的内容仍是正向源窗口本身**（方向翻转，内容不变）。

## 原始读数

夹具 `fixtures/phase3a-asymmetric.wav`（不对称：0–1s 173Hz、1–2s 311Hz，另有 7000–7100
样本的脉冲；44100Hz mono float32，源长 2.0s）。两个 item 做**同一个非对称裁切**：源内
`[0.25, 1.25]`（`D_STARTOFFS=0.25`、`D_LENGTH=1.0`），一个正放、一个用官方 action
**41051**（"Item properties: Toggle take reverse"）倒放，并用
`PCM_Source_GetSectionInfo` 的 `revOut` 核实（`B_REVERSED` setter 是无效证据，见
`EXECUTION-LEDGER.md:639`）。

| item | take `D_STARTOFFS` | ARA region `startMod` | `durationMod` | `startPlay` |
|------|--------------------|------------------------|----------------|-------------|
| 正放 | 0.25 | **0.25** | 1.0 | 0.0 |
| 倒放 | 0.75 | **0.75** | 1.0 | 3.0 |

- 倒放 take 的 `D_STARTOFFS` 被 REAPER 改成了 `0.75 = 源长 − 0.25 − 1.0`，region 与它一致。
- 两个 region 共享同一个 audio source，几何全等，唯一差别是方向。

### 决定性证据：倒放 take 实际播的是哪一段

只放那一个倒放 item、**不挂任何 FX**，把它的播放直接渲染成 WAV，再与源逐样本比对
（`captures/reverse-window.wav`）：

| 候选 | 最大绝对差（掐掉首尾各 256 样本的防爆音淡变） |
|------|---------------------------------------------|
| A：`reverse(源[0.25, 1.25])` | **1.38e-05** |
| B：`reverse(源[0.75, 1.75])` | 0.282 |

⇒ 命中 **A**：倒放 take 播的仍是正向源 `[0.25, 1.25]`（只是反向）。于是 region 报的
`[0.75, 1.75]` 就是它的**镜像**区间 —— 坐标系确认为**已镜像**。

（首尾各几样本的偏差 ~2e-3 来自 REAPER 即便 `D_FADE*LEN=0` 也会加的极短防爆音淡变，
掐掉两端后为 1.38e-05。）

## 对 Task 3.4 的含义

内核的倒放消费模型（`hifishifter-kernel/src/state/model.rs:111-144`）**只**用
`source_end_sec`：`win = [source_end − length×rate, source_end)`，`source_start_sec`
不参与消费数学。也就是说内核要的是**正向**窗口的终点。

而 ARA 映射（`ara/mapping.rs`）写的是 `source_start_sec = startMod`、
`source_end_sec = startMod + durationMod` —— 对倒放 clip 拿到的是镜像区间
（本例 `source_end = 1.75`）。若直接把 `reversed=true` 与这个窗口送进内核，内核会消费
`[0.75, 1.75]` —— **错**（正确是 `[0.25, 1.25]`）。

所以 Task 3.4 落地时，**必须先把镜像窗口翻回正向**再交给内核：对倒放 clip，用源时长 `D`
把 region 区间 `[a, a+d]` 换成 `[D − a − d, D − a]`，于是 `source_end_sec` 应取 `D − a`
（本例 `2.0 − 0.75 = 1.25`）。这正是 plan Task 3.1 注释里说的"必须同时解开'源窗口'这一侧"。

⚠️ 不要只翻 PCM、又把镜像窗口当正向窗口用 —— 那就是 plan 说的"二次镜像 = 又是正放"。

## 怎么复现

无头跑（隔离 profile，不碰 `%APPDATA%\REAPER`；两个探针各一条命令）：

```powershell
# 1) region 坐标 + take 事实（两个非对称裁切 item，一个正放一个倒放）
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -Probe "probe/ara/build_reverse_probe.lua" `
  -Out   "probe/ara/captures/reverse-probe.json" `
  -Env   "HIFISHIFTER_ARA_LOG=probe/ara/captures/reverse-plugin.log" `
  -SettleSeconds 10
powershell -ExecutionPolicy Bypass -File probe/ara/verify_reverse_capture.ps1
# → F-4 region coordinates = mirrored

# 2) 决定性证据：单独渲染倒放 item（无 FX）并逐样本比对
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -Probe "probe/ara/build_reverse_window_probe.lua" `
  -Out   "probe/ara/captures/reverse-window-probe.json" `
  -SettleSeconds 12
python probe/ara/verify_reverse_window.py
# → covered=forward-trim coordinate=mirrored
```

分类器的正确性由本地变异回归保证（真实采集与验证器正确性是两件事，同
`test_task11_capture.ps1` 的纪律）：

```powershell
powershell -ExecutionPolicy Bypass -File probe/ara/test_reverse_capture.ps1
# → 8 条通过：正向/镜像分别判对；缺证据 / 几何不符 / 第三种坐标被拒
```

## 原始产物

- `captures/reverse-script.log` / `captures/reverse-plugin.log`：region 坐标、take 事实、
  宿主 section 核实。
- `captures/reverse-coordinates.json`：region 坐标分类（`coordinateSystem = mirrored`）。
- `captures/reverse-window.wav` / `captures/reverse-window.json`：渲染窗口与候选比对
  （`coveredWindow = forward-trim`）。

## 边界（不得外推）

- 只测了**一个主机版本**（REAPER 7.82 / Windows x64）。其它宿主或其它 REAPER 版本可能
  不同；把"已镜像"当成 ARA 规范行为是错的 —— 这是该宿主该版本的行为。
- 只回答**时间坐标**这一件事。不回答"插件渲染的倒放听感是否与 REAPER 逐样本一致" ——
  那要 Task 3.4 落地后另行验收（Mel Stretch 的 f0 时间基准另有一件独立工作）。
- "ARA 交给插件的 PCM 是正向的"是既有结论（`captures/phase3a-FINDINGS.md`），本探针
  **未复测**，只聚焦时间坐标。
- 倒放 + 内容淡化（content-based fade）等组合未覆盖。
