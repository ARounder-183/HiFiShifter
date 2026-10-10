# Task 3.4：倒放渲染 —— **已落地并端到端验收**

日期：2026-10-10
主机：REAPER 7.82 / Windows x64；插件 `hifishifter-plugin` v0.1.0-beta.15（本机 debug 构建，
换入已安装 bundle 的 `Contents/x86_64-win/HiFiShifterEngine.dll` 做验收，验毕已还原）
对应：`docs/plans/2026-10-09-ara-plugin-fade-rewrite-reverse-auth-and-stability.md` 的 Task 3.4（原 3.2）
性质：**改产品行为**（插件侧），并附**端到端音频证据**。F-4 取证见 `REVERSE-FINDINGS.md`。

## 结论（一句话）

倒放片段现在**真的会渲染**：插件输出的正是 `reverse(源[0.25, 1.25])`（正向窗口的反转），
而不是镜像窗口 `[0.75, 1.75]` 的正放，也不再是"由 REAPER 处理"的占位。

## 落地了什么（都在插件侧，内核只加了一个策略变体）

1. **内核 `host_pcm.rs` 新增 `ReversePolicy::Materialize`**：倒放 clip **照常物化**
   （不再清 `source_path`）。`Reject`（独立 App）与 `Isolate`（旧插件行为）都保留。
2. **`project_host_take_facts_locked` 补上镜像窗口复原**：ARA 报的是**已镜像坐标**
   （F-4），内核的倒放消费窗口要的是**正向**窗口终点。按宿主几何推导
   `forward = [D − host_start − host_span, D − host_start]`，由宿主事实直接得出、
   **幂等**（重建/重认领不累积镜像）。抽成纯函数 `reversed_forward_window` 并单测。
3. **`capture_render_input` 的 `kernel_render` 纳入倒放**：未编辑、无拉伸、无淡化委托的
   片段本来走 `mix_plain_regions` 直通（`RenderInput.timeline = None`）——那是"原样放 ARA
   授权 PCM"。**倒放片段不能直通**：方向翻转只发生在内核 `render_mixdown_internal`
   （`reverse_interleaved_frames`）。不推进内核，它就被按正向原样放出来（这正是中间态
   实测到的现象）。
4. **分析纳入倒放 clip**（`analysis_timeline` 去掉 `retain(|c| !c.reversed)`）：分析读的是
   正向完整源，装配期（`pitch_analysis/schedule.rs`、`pitch_editing.rs`）再整体翻转。
   排除它会让 f0 全零 ⇒ 只剩气声。
5. **参数图鉴逐 clip 跳过倒放**（`capture_inner` / `project_in_domain` / `rebind` 候选 /
   `capture_gaps` 覆盖）：此前 `geometry(clip)` 对倒放返回 `Err`，一条倒放片段会让
   **整份**渲染输入准备失败。现在跳过而不是整份失败（镜像帧映射属独立一期）。

## 端到端验收（决定性证据）

方法：挂**真** HiFiShifter FX，把一个非对称裁切（源内 `[0.25, 1.25]`）**倒放**的 item
的播放渲染成 WAV（`build_reverse_render_probe.lua`），再与源夹具的四个候选逐样本比对
（`verify_reverse_render.py`，±512 样本对齐搜索、掐掉两端各 512 样本）：

| 候选 | 最大绝对差 | 对齐偏移 |
|------|-----------|---------|
| **`reverse(源[0.25, 1.25])`（正确）** | **1.100e-04** | **0** |
| `reverse(源[0.75, 1.75])`（窗口没翻正） | 2.700e-01 | −170 |
| `源[0.25, 1.25]`（压根没倒放） | 2.678e-01 | 304 |
| `源[0.75, 1.75]`（没倒放且窗口镜像） | 2.700e-01 | 433 |

⇒ 命中 `reverse_forward_trim`，其余候选差两个数量级。`captures/reverse-render.json`。

### 中间态（同一探针，修 `kernel_render` 之前）

未把倒放推进内核时，输出 = `源[0.75, 1.75]` **正放**（`forward_mirrored` 4.578e-05，
其余 ~0.27）——即"ARA 授权 PCM 原样直通"。这条读数定位了第 3 项修复，也说明"播种层翻正
窗口"**单独不够**：还得让倒放片段真的走进内核。

## 怎么复现

```powershell
# 1) 构建插件并把引擎 DLL 换进已安装 bundle（先备份！）
cargo build --manifest-path backend/Cargo.toml -p hifishifter-plugin
copy /Y backend\target\debug\hifishifter_plugin.dll `
  "C:\Program Files\Common Files\VST3\HiFiShifter.vst3\Contents\x86_64-win\HiFiShifterEngine.dll"

# 2) 无头渲染倒放 item（挂真 FX）
powershell -ExecutionPolicy Bypass -File probe/ara/run_probe_headless.ps1 `
  -Probe "D:/Projects/HiFiShifter/probe/ara/build_reverse_render_probe.lua" `
  -Out   "D:/Projects/HiFiShifter/probe/ara/captures/reverse-render-probe.json" `
  -Env   "HIFISHIFTER_ARA_LOG=D:/Projects/HiFiShifter/probe/ara/captures/reverse-render-plugin.log" `
  -SettleSeconds 30

# 3) 判定
python probe/ara/verify_reverse_render.py
# → matched=reverse_forward_trim

# 4) 还原原始引擎 DLL（务必）
```

## 边界（不得外推）

- 只测了**一个主机版本**（REAPER 7.82 / Windows x64）与**一个夹具**（非对称、44100、
  mono）。其它宿主/版本可能不同。
- 验收用的是**未编辑**片段（无音高编辑、无拉伸、无淡化委托）——这恰好是最难的一档
  （会走直通），已覆盖。**未**验收"倒放 + 用户音高编辑"（参数图鉴对倒放是**跳过**的，
  见上第 5 项：镜像帧映射属独立一期）。
- **不**宣称倒放听感与 REAPER 逐样本一致：Mel Stretch 的 f0 时间基准是正向轴，倒放 +
  Mel Stretch 组合下 f0 查询与已镜像 PCM 的对齐是**独立一期**（plan 3.7 已记）。
- 插件输出经声码器链，即便"未编辑"也不是逐比特直通；1.1e-04 是 24bit 量化 + 链内
  极短处理下的量级，足以判定"是哪一段"，不等于"逐样本相同"。

## 原始产物

- `captures/reverse-render.wav` / `captures/reverse-render.json`：插件渲染输出与四候选比对。
- `captures/reverse-render-script.log` / `captures/reverse-render-plugin.log`：探针与插件日志。
- 探针与判定器：`build_reverse_render_probe.lua`、`verify_reverse_render.py`。
