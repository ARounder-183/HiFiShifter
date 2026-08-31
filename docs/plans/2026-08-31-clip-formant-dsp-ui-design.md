# Clip 级共振峰：算法重写 + 操作界面优化（设计）

> 日期：2026-08-31
> 范围：`backend/src-tauri/src/audio/formant_morph.rs`（DSP 核心重写）、新增 clip 源共振峰分析命令、clip 共振峰工具窗口 UI。
> 目标症状：**设了目标 F1/F2 但元音基本没变（搬不动）**。

---

## 1. 现状与问题诊断

### 1.1 现有实现

`backend/src-tauri/src/audio/formant_morph.rs`（2026-06-30 由 LPC 极点迁移重写为「包络变形」路线）：

```
STFT(2048 / 1024, hop = N/4) → log|X| → IFFT → 倒谱 lifter(12%) → FFT → 包络 E(k)
→ 固定区间内找 F1/F2 局部峰 → 构造 (0,F1,F2,3.8k) → (0,F1',F2',3.8k) 分段线性频率扭曲
→ E'(k) = E(w⁻¹(f_k)) → Y(k) = X(k) · (E'/E)^(strength·confidence) → iSTFT / OLA
```

链路位置：formant 在**解码重采样之后、变速与声码器之前**执行
（`audio_engine/snapshot.rs:456`、`audio/mixdown.rs:643`），分析对象是原始音高素材，位置本身合理。

### 1.2 「搬不动」的三个根因

| # | 问题 | 位置 | 后果 |
| - | ---- | ---- | ---- |
| **R1** | **包络过度平滑**：`LIFTER_CUTOFF_RATIO = 0.12` → 2048 点 FFT 下保留 245 个 quefrency，等效频率分辨率约 200–400 Hz；真实共振峰 3 dB 带宽只有 50–150 Hz | `formant_morph.rs:66,166` | 校正滤波器 `E'/E` 是**缓坡**而非**峰/谷**，重新加权谐波后只产生"音色有点变化"，不构成元音迁移 |
| **R2** | **锚点常常是假的**：`find_local_peak` 取区间内最高局部峰，找不到就退回**区间中点常量**（F1≈650、F2≈1800）。近讲低频强 / 高 F0 素材会把 F0 区（200–400 Hz）识别成 F1 | `formant_morph.rs:357-408` | 一旦"检测到的 F1"≈目标 F1，整段映射几乎不动 → 正是"设了目标没变化"；识别错时还会把真 F1 搬到错误位置，听感只是"变怪" |
| **R3** | **置信度公式失效**：`confidence` 由 `envelope[f1_lo..f1_bin]` 的最小值算得；切片为空时 → 0（整帧无效果），包络单调下降时 → 1 | `formant_morph.rs:373-388` | 逐帧乱跳的强度 → 忽强忽弱 / 门限感 |

### 1.3 次要问题

- **帧间零平滑**：映射每 ~10 ms 独立重算，滤波器突变 → 调制感。
- **整轴扭曲**（0–3.8 kHz 全部重采样）：把频谱倾斜、低频能量、谷点一起搬走，音色身份受损。
- **无逐帧能量对齐 + 末端硬 clamp 0.99** → 响段削波、响度漂移。
- **前后端范围不一致**：前端 F1∈[250,1000] / F2∈[540,2600]，后端 clamp 到 F1∈[180,1200] / F2∈[F1+250,3200]。
- 元音图上标了元音字母但**不可点击**，且看不到素材当前的共振峰位置，用户无法判断是"没搬动"还是"搬错地方"。

---

## 2. 目标 / 非目标

### 目标

- **G1 搬得动**：输出音频的实测共振峰位置显著向目标移动，并由自动化测试守住。
- **G2 保音色**：只搬 F1/F2 两个极点，其余频段与原始相位完全不动。
- **G3 时域稳定**：无逐帧抖动、无门限感、无相位感。
- **G4 电平安全**：无削波，处理后响度与输入基本持平。
- **G5 可诊断可对比**：UI 上能看到素材当前共振峰，并能一键 A/B。

### 非目标

- 不修改 `ClipFormantMorph` 数据模型（仍为 `enabled / targetF1Hz / targetF2Hz / strength`）。
- 不改动 formant 在渲染链路中的位置，不改 `formant_cache.rs` / `snapshot.rs` / `mixdown.rs` 的调用契约。
- 不做轨道级（参数线）共振峰。
- 不引入新的第三方依赖（不用 WORLD CheapTrick / 不引 nalgebra：避免把分析耦合到 F0 与外部库）。

---

## 3. 算法设计（方案 A′：LPC 极点迁移 + 频域比值滤波）

核心思路：**在低采样率下用低阶 LPC 拿到"真共振峰"级别的锐利包络，只把 F1/F2 两个极点搬到目标位置，然后在 STFT 域用"极点滤波器幅频比值"做校正，保留原始相位。**

### 3.1 处理流程

```
输入 mono PCM (sr)
 │
 ├─ 1. 预加重 (0.97)
 ├─ 2. 抗混叠 FIR 降采样 → 11 025 Hz        （分析域）
 ├─ 3. 逐分析帧（25 ms / hop 10 ms）Hann → 自相关 → Levinson–Durbin → LPC(order 12)
 ├─ 4. Durand–Kerner 复数求根 → 极点 → (中心频率 f, 带宽 bw)
 ├─ 5. 候选筛选 → F1 / F2 原始轨迹
 ├─ 6. 轨迹平滑：5 帧中值 + 限速 + 帧间线性插值
 ├─ 7. 目标极点：f' = f + (target − f) · strength；带宽保持（见 3.6）
 ├─ 8. STFT 域校正：H(k) = |A_orig(e^{-jw})| / |A_target(e^{-jw})|
 │       Y(k) = X(k) · clamp(H(k), ±24 dB)^gate      （相位完全沿用 X）
 ├─ 9. 逐帧 RMS 对齐（clamp 0.5–2.0）+ 去加重
 └─ 10. 末端软限幅（tanh 型）+ 峰值保护
```

立体声策略沿用现状：取通道平均做 mono 分析，把 `wet − dry` 的 delta 加回各通道（保留声像）。

### 3.2 为什么这样能解决 R1/R2/R3

- **R1**：LPC 全极点包络在 11 kHz / order 12 下的分辨率由极点带宽决定（典型 50–150 Hz），校正滤波器 `H` 在原 F1 处有**真实陷波**、在目标 F1 处有**真实尖峰** → 明确可闻的元音迁移。
- **R2**：候选筛选要求同时满足频率范围、带宽范围、F1/F2 最小间距与峰值显著性；**筛选失败则该帧恒等直通**，不再用假常量锚点。
- **R3**：`gate`（浊音权重）由「帧能量 + 检测成功 + 残差能量比」合成，并做 **attack 30 ms / release 80 ms** 的时间平滑，替代逐帧乱跳的 `confidence`。

### 3.3 关键参数（集中为模块常量，便于调参）

| 常量 | 取值 | 说明 |
| ---- | ---- | ---- |
| `ANALYSIS_RATE` | 11 025 Hz | Praat 同款配方：2 × 最高共振峰(5 kHz) + 余量 |
| `LPC_ORDER` | 12 | 5 kHz 内约 2 极点/共振峰 + 2；阶数过高会使极点跟随谐波 |
| 分析帧长 / hop | 25 ms / 10 ms | 与 STFT 帧解耦；不足一帧则直通 |
| STFT `fft_size` / hop | 2048(sr≥24k) / 1024；hop = N/4 | 沿用现状，仅用于施加校正滤波器 |
| F1 候选 | 200–1 000 Hz | |
| F2 候选 | 800–3 000 Hz | |
| 极点带宽 | 30–500 Hz | 超出视为非共振峰（谐波极点 / 数值噪声） |
| F2 − F1 最小间距 | 200 Hz | |
| 峰值显著性 | 相对局部基线 ≥ 3 dB | 用 \|1/A(f)\| 在峰值与两侧谷点间的 dB 差 |
| 轨迹中值窗 / 限速 | 5 帧 / 每帧 ±15% | 抑制倍频程跳变 |
| `H` 限幅 | ±24 dB | 单帧单点最大增益/衰减 |
| `gate` 平滑 | attack 30 ms / release 80 ms | |
| 帧 RMS 对齐 clamp | 0.5–2.0 | |
| 输出峰值上限 | 输入峰值 × 1.6 | 沿用现状 |

### 3.4 降采样

`mixdown::linear_resample_interleaved` 是线性插值、无抗混叠，直接用于 48 k→11 k 会产生混叠并污染 LPC。因此新增一个**专用整数/有理抽取器**：先窗函数法生成截止 0.45×目标奈奎斯特的 FIR（64~96 阶 Hann 窗低通），再按 `round(in_rate / ANALYSIS_RATE)` 做多相抽取；输入采样率低于 `ANALYSIS_RATE × 2` 时不降采样、直接以原速率分析（同时下调 `LPC_ORDER` 对应的最高共振峰假设）。

### 3.5 求根

采用 **Durand–Kerner（Weierstrass）复根迭代**：12 阶多项式，初始点取模递增的复数圆，迭代上限 100 次、收敛阈值 1e-8。该算法对实系数多项式稳定、无需导数、代码约 40 行（复数四则运算手写，不引第三方库）。

每次迭代后做安全处理：非有限值 / `|z| ≥ 1` 的根直接丢弃（不稳定极点不参与重建）；全部失败 → 该帧恒等直通。

### 3.6 目标极点与带宽策略

- 频率：`f' = f + (target − f) · strength`（`strength = 0` 时 `H ≡ 1`，配合入口早退保证严格 bypass）。
- 带宽：**第一版保持原带宽不变**，只搬频率 —— 音色身份保真最好。
- `vowel_synth.py` 中的"收紧到 50/80 Hz"策略作为**待定增强**：若实测咬字不够锐利，再以「锐度」参数形式暴露（YAGNI，先不做）。

### 3.7 频域比值的计算

`A(e^{-jw}) = 1 + Σ a_k e^{-jwk}`。做法：把 LPC 系数补零到 `fft_size` 做一次 FFT，一次性得到全部 bin 的 `A_orig`、`A_target` 复响应，取模相比即可（避免逐 bin 求多项式的 O(bins × order) 开销）。

```
H(k) = |A_orig(k)| / max(|A_target(k)|, floor)
H(k) = clamp(H(k), 10^(-24/20), 10^(+24/20))
Y(k) = X(k) · H(k)^gate
```

---

## 4. 模块拆分与文件清单

现状 `formant_morph` 是单文件（`#[path = "audio/formant_morph.rs"] mod formant_morph;`）。新实现约 600–700 行（含测试），按工程化要求拆为目录模块，`lib.rs` 改为 `#[path = "audio/formant_morph/mod.rs"] mod formant_morph;`：

| 文件 | 职责 |
| ---- | ---- |
| `audio/formant_morph/mod.rs` | 公开入口 `apply_formant_morph_mono` / `apply_formant_morph_interleaved`、`vowel_formant_preset`、整体编排、常量表、测试 |
| `audio/formant_morph/decimator.rs` | 抗混叠 FIR 抽取到分析速率 |
| `audio/formant_morph/lpc.rs` | 自相关、Levinson–Durbin、Durand–Kerner 求根、极点 ↔ (频率, 带宽) 换算、由极点重建系数 |
| `audio/formant_morph/track.rs` | 共振峰候选筛选、F1/F2 轨迹提取、中值滤波 + 限速 + 帧间插值 |
| `audio/formant_morph/correction.rs` | 目标极点、`H(k)` 构造与限幅、`gate` 时间平滑、逐帧能量对齐、软限幅 |
| `audio/formant_morph/analysis.rs` | 供 IPC 复用的源共振峰统计（统计 F1/F2 + 稀疏散点 + 浊音覆盖率） |

所有新文件遵循项目约定：文件头注释说明「主要内容 / 与其他模块的关系 / 维护说明」，关键函数头注释说明「流程 / 作用 / 特殊规则 / 参数」。

**公开 API 与调用方零改动**：`formant_cache.rs`、`audio_engine/snapshot.rs`、`audio/mixdown.rs` 不需要任何修改。

---

## 5. 新增后端接口：源共振峰分析

### 5.1 命令

```
#[tauri::command(rename_all = "camelCase")]
pub async fn analyze_clip_formants(app: tauri::AppHandle, clip_id: String)
    -> Result<ClipFormantAnalysisPayload, String>
```

- 实现放 `backend/src-tauri/src/commands/formant.rs`（`pub(super) fn analyze_clip_formants`），在 `commands.rs` 用 `#[path = "commands/formant.rs"] mod formant;` 登记，并加入 `lib.rs` 的 `invoke_handler`。
- 走 `tauri::async_runtime::spawn_blocking`（与 `commands/waveform.rs` 一致，避免阻塞 UI）。

### 5.2 返回载荷

```rust
pub struct ClipFormantAnalysisPayload {
    pub ok: bool,
    /// 统计意义上的源共振峰（浊音帧的带宽加权中位数）
    pub source_f1_hz: f32,
    pub source_f2_hz: f32,
    /// 用于画轨迹的稀疏采样（归一化时间 0..1）
    pub track: Vec<FormantTrackPointPayload>, // { t, f1_hz, f2_hz }
    /// 浊音帧占比：过低说明素材不适合做共振峰调整（UI 给提示）
    pub voiced_ratio: f32,
    pub message: Option<String>, // "source_too_short" / "no_voiced_frames" 等
}
```

### 5.3 输入与缓存

- 复用 `compute_formant_cache_entry_for_clip` 的取数路径：解码 → 消费窗口切片 → 重采样到 `out_rate` → 转 mono。
- 分析结果放独立小缓存（非 `FormantCache`），键 = `clip_id + source_path + mtime + 窗口量化(1 ms)`；`invalidate_formant_cache_for_clip` 时一并失效。
- **分析核心与 DSP 共用同一套 `lpc.rs` / `track.rs`**，保证 UI 显示的位置与算法实际认定的位置严格一致（否则诊断会误导）。

---

## 6. 前端 UI 设计

### 6.1 `VowelChart.tsx`

- 新增可选 props：`sourceF1Hz / sourceF2Hz / track? / showSource?`。
- 渲染：**源点（空心圆 + 虚线轨迹）→ 目标点（实心圆）** 之间的连线与箭头，让"从哪搬到哪"一目了然。
- 元音字母**可点击** → 一键设为目标（`onPickVowel`），hover 高亮，禁用态下不可点。
- 保持现有 pointer capture 拖拽、`user-select: none`、Space 抑制等行为不变。

### 6.2 `ClipFormantToolWindow.tsx`

- 顶部：状态点 + 标题 + clip 名 + 关闭（不变）。
- 启用改为 **Switch**（比 checkbox 醒目），右侧增加 **Bypass 按钮**（临时旁通试听，不改变数据、不产生撤销步）。
- 元音图区下方：`源 F1/F2 → 目标 F1/F2` 数值对照。
- 强度：滑杆 + 数值输入框（可键盘精确输入）。
- 状态区：
  - `已就绪` / `已关闭` / `重建中` / `失败`
  - 新增：`未检测到稳定元音素材`（`voiced_ratio` 过低时），提示换素材或降低预期。
- 窗口打开时请求一次 `analyze_clip_formants`；clip 源路径 / 消费窗口变化时重新请求。

### 6.3 数据层

- `frontend/src/services/api/timeline.ts` 新增 `analyzeClipFormants(clipId)`。
- 分析结果的请求/状态放在 `sessionSlice`（`clipFormantAnalysis: Record<clipId, ...>`），与现有 `clipFormantStatus` 并列。
- 新增 i18n 键（`zh-CN / zh-TW / en-US / ja-JP / ko-KR` 五份同步）：
  `clip_formant_source`、`clip_formant_target`、`clip_formant_bypass`、`clip_formant_pick_vowel`、`clip_formant_analysis_none`、`clip_formant_no_voiced`。

### 6.4 范围统一

后端 clamp 与前端区间对齐到同一组常量：F1 ∈ [250, 1000]、F2 ∈ [540, 2600]（与元音图坐标域一致，避免"图上能点的位置后端够不着"）。

---

## 7. 测试策略

### 后端（Rust 单测，随模块落地）

保留现有：
- `disabled_is_strict_bypass` / `zero_strength_is_strict_bypass`（严格 bypass）
- `empty_input_returns_empty` / `low_sample_rate_is_bypass` / `silent_input_stays_silent`
- `output_is_finite_and_length_preserving` / `interleaved_stereo_matches_length_and_finite`
- `vowel_preset_table_returns_known_values`

新增（**守住 G1「搬得动」这条核心回归**）：
- `formant_shift_moves_measured_f1_toward_target`：合成 /a/（F1=800, F2=1200），目标 /i/（300, 2300），对输出重新估计共振峰，断言 **输出 F1 比输入 F1 更接近目标**（且方向正确、幅度达到预期比例）。
- `stronger_strength_moves_further`：strength 0.3 / 0.6 / 1.0 单调推进。
- `unvoiced_noise_input_is_not_amplified`：白噪声输入，输出峰值 ≤ 输入峰值 × 1.6，且无 NaN。
- `pathological_input_stays_bounded`：`f32::MAX` 级 / 直流 / 方波 / 极短(1 帧) 输入不炸。
- `stereo_preserves_channel_delta`：立体声输出长度、有限性、左右相关系保持。
- LPC 子模块：对合成共振峰信号，估计的极点频率落在真值 ±8% 内。
- 求根子模块：已知根的多项式（含重根、共轭对）能复原到 1e-5。

### 前端（Vitest）

- 保留 `sessionSlice.formantMorph.test.ts`、`sessionSlice.formantToolWindow.test.ts`、`clipFormantButtonStyle.test.ts`、`clipFormantInteractionGuards.test.ts`。
- 新增：源/目标连线与散点的坐标映射（放到 `vowelChartLayout` 相关纯函数测试）、Bypass 不产生撤销步（`useClipFormantEditor` 的 `dirtyRef` 语义）。

### 手动验收清单

1. 同一段素材，源点与目标点连线可见，拖动目标点 → 元音明显迁移。
2. Bypass 按下立即回到原声，松开恢复。
3. 辅音/气声段不出现爆音；整体响度无明显变化。
4. 长 clip（>30 s）处理耗时可接受；拖动滑杆时防抖预览不卡顿。
5. 实时播放与导出结果一致（走同一缓存与 DSP）。

---

## 8. 风险与缓解

| 风险 | 缓解 |
| ---- | ---- |
| LPC 求根数值不稳 | Durand–Kerner + 迭代上限 + 丢弃非有限 / `|z|≥1` 的根；失败即恒等直通 |
| 高 F0 素材极点跟随谐波 | 分析速率降到 11 kHz 且 order 固定 12（无法分辨单个谐波）；再叠加带宽 30–500 Hz 与显著性筛选 |
| 轨迹倍频程跳变 | 5 帧中值 + 每帧 ±15% 限速 + 帧间插值 |
| 性能回退 | 分析在 11 kHz 域进行（数据量降约 4×）；每帧一次 12 阶求根 + 2 次 FFT，相对现状 STFT 路径开销可控；必要时分析每 2 帧做一次并插值 |
| 过度保护导致"又搬不动" | 候选筛选失败才直通，不做全局降权；`H` 限幅放宽到 ±24 dB；以 G1 自动化测试为闸门 |
| 事件隔离回归（历史 bug） | 现有 `clipFormantInteractionGuards`、body 属性与拖拽中止机制全部保留，UI 改动不触碰这些逻辑 |

**回退面**：所有改动集中在 `audio/formant_morph/` 与新增的 `commands/formant.rs`；公开 API 与三个调用方零改动，出问题可整体回退该目录而不影响其它链路。

---

## 9. 实施顺序

1. 建 `audio/formant_morph/` 模块骨架 + `decimator.rs`（含单测）。
2. `lpc.rs`：自相关 / Levinson–Durbin / Durand–Kerner / 极点参数化（含单测）。
3. `track.rs`：候选筛选 + 轨迹平滑（含单测）。
4. `correction.rs` + `mod.rs` 编排：频域比值校正、gate、能量对齐、软限幅。
5. 端到端测试，重点跑通 `formant_shift_moves_measured_f1_toward_target`；删除旧 STFT/倒谱实现与常量。
6. `analysis.rs` + `commands/formant.rs` + `lib.rs` 注册 `analyze_clip_formants`。
7. 前端：API → slice → `VowelChart` → `ClipFormantToolWindow` → i18n 五语同步。
8. 手动验收 + 更新 `docs/i18n/USERMANUAL.md` 与 README 中 clip formant 相关描述。
