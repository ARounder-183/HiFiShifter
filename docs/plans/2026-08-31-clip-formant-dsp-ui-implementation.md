# Clip Formant DSP 重写 + 分析命令 + UI 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or execute directly with TDD discipline (test-driven-development). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 clip 级共振峰 DSP（LPC 极点迁移 + 频域比值滤波），修复"设了目标却搬不动"；新增源共振峰分析命令与工具窗口 UI 优化。

**Architecture:** 公开 API（`apply_formant_morph_mono/interleaved`）与调用方（`formant_cache.rs` / `snapshot.rs` / `mixdown.rs`）零改动；DSP 核心拆为 `audio/formant_morph/{mod,decimator,lpc,track,correction,analysis}.rs`；新增 Tauri 命令 `analyze_clip_formants`；前端 VowelChart 增加源点可视化、Bypass、元音点击。

**Tech Stack:** Rust（rustfft 既有依赖）、Tauri 2 命令、React + Radix Themes + Vitest。

**设计文档:** `docs/plans/2026-08-31-clip-formant-dsp-ui-design.md`

---

## 关键常量与公式（实现时以此为准）

```
ANALYSIS_RATE      = 11_025 Hz（输入 sr < 2×ANALYSIS_RATE 时不降采样，LPC_ORDER 按最高共振峰 sr/4 折算）
LPC_ORDER          = 12（= 2×5kHz/1k + 2）
分析帧 25ms / hop 10ms（在分析速率域）
STFT fft_size      = 2048 (sr≥24k) / 1024，hop = N/4（沿用现状）
F1 候选 200–1000 Hz；F2 候选 800–3000 Hz；带宽 30–500 Hz；F2−F1 ≥ 200 Hz
峰值显著性：|1/A(f_peak)| 相对两侧谷点 ≥ 3 dB（≈1.41 倍）
轨迹：5 帧中值 → 每帧限速 ±15% → 帧间线性插值到 STFT hop 网格
目标极点：f' = f + (target − f)·strength；带宽不变
H(k) = |A_orig(k)| / max(|A_target(k)|, floor)，clamp ±24 dB
gate = 平滑(能量门限 ∧ 检测成功 ∧ 残差能量比)，attack 30ms / release 80ms
帧 RMS 对齐 clamp 0.5–2.0；输出峰值 ≤ 输入峰值×1.6；软限幅 tanh 型
预加重 0.97（分析前），去加重（输出前）
```

降采样：窗函数法 FIR 低通（截止 0.45×目标奈奎斯特， Hann 窗，64 阶），多相抽取 `decim = round(in_rate / ANALYSIS_RATE)`。

求根：Durand–Kerner，初始点 `z_k = (0.4 + 0.9i)^k`，迭代 ≤100，收敛 1e-8；丢弃非有限 / |z|≥1 的根；共轭对按正虚部枚举。

极点换算：`f = angle(z)·sr/(2π)`（取正虚部根），`bw = −ln|z|·sr/π`。
重建系数：共轭对 → 二阶节 `1 − 2r·cosθ·z⁻¹ + r²·z⁻²`，实根 → `1 − z⁻¹`，连乘。

## 公开 API（mod.rs，签名不变）

```rust
pub fn apply_formant_morph_mono(input: &[f32], sample_rate: u32, params: &ClipFormantMorph)
    -> Result<Vec<f32>, String>;
pub fn apply_formant_morph_interleaved(input: &[f32], sample_rate: u32, channels: usize,
    params: &ClipFormantMorph) -> Result<Vec<f32>, String>;
pub fn vowel_formant_preset(vowel: &str) -> Option<(f64, f64)>;  // 原样保留
```

mono 编排流程：
1. 入口校验（disabled / 空 / sr<8000 / len<512 / strength≤1e-5）→ 严格 bypass（返回 `input.to_vec()`）。
2. 预加重 → decimator → 分析帧序列 → `lpc::analyze_frame` → `track::extract_tracks`（全帧）→ `track::smooth_tracks`。
3. STFT 帧循环（复用现有 pad / OLA / window² 归一化骨架）：对每个 STFT hop，用插值后的 (F1,F2) 与对应 `A_orig/A_target` 计算 `H(k)`，`Y = X·H^gate`。
   - `A_orig(k)`：由该 STFT 帧**邻近分析帧**的 LPC 系数补零 FFT 得到；`A_target` 由迁移后极点重建。
   - gate 逐帧由 `correction::VoicingGate` 维护（内部 attack/release 状态）。
4. iSTFT/OLA → 去加重 → 逐帧 RMS 已在帧内对齐 → 全局峰值保护 + 软限幅。

---

### Task 1: 模块骨架 + decimator（TDD）

**Files:**
- Create: `backend/src-tauri/src/audio/formant_morph/mod.rs`（先放公开 API 的 bypass 分支 + 常量表 + 文件头注释）
- Create: `backend/src-tauri/src/audio/formant_morph/decimator.rs`
- Modify: `backend/src-tauri/src/lib.rs:21-22` → `#[path = "audio/formant_morph/mod.rs"] mod formant_morph;`
- Delete: 旧 `backend/src-tauri/src/audio/formant_morph.rs`（在 Task 5 收尾时删除，Task 1 起由新目录接管）

**Steps:**
- [ ] 1.1 写失败测试：`decimator.rs` 内 `#[cfg(test)]` — `decimates_48k_to_analysis_rate_without_aliasing`（输入 48k 合成 /a/，降采样后与"理想降采样参考"（scipy 不可用 → 用低通后直接抽取的参考实现做互相关）能量差 < 0.5 dB；且 12 kHz 以上成分衰减 ≥ 40 dB：输入加一个 13 kHz 正弦，断言输出中该频率分量 RMS < 原始的 1%）。
- [ ] 1.2 实现 `Decimator::new(in_rate, out_rate) -> Option<Decimator>` 与 `process(&mut self, mono: &[f32]) -> Vec<f32>`（FIR 低通 + 整数抽取；in_rate==out_rate 时恒等）。
- [ ] 1.3 `mod.rs` 迁入公开 API（先保留 bypass 分支 + 旧 `synth_vowel` 测试工具与全部既有测试），`cargo build -p` 编译通过。
- [ ] 1.4 运行 `cargo test formant_morph --quiet`（backend/src-tauri 下）→ PASS。
- [ ] 1.5 Commit: `refactor(formant): split formant_morph into module dir with decimator`

### Task 2: lpc.rs（TDD）

**Files:**
- Create: `backend/src-tauri/src/audio/formant_morph/lpc.rs`

**函数：**
```rust
pub fn autocorrelation(frame: &[f32]) -> Vec<f32>;                  // order+1 阶
pub fn levinson_durbin(ac: &[f32], order: usize) -> Option<LpcCoeffs>; // a[0]=1, 返回激励增益 g
pub fn poly_roots(a: &[f32]) -> Option<Vec<Complex32>>;              // Durand–Kerner
pub fn roots_to_poles(roots: &[Complex32], sr: f32) -> Vec<Pole>;    // 正虚部共轭对 → (freq, bw)
pub fn poles_to_coeffs(pairs: &[(f32, f32)], real_roots: &[f32]) -> Vec<f32>; // 频率+带宽 → 系数
pub fn analyze_frame(frame: &[f32], sr: f32, order: usize) -> Option<Vec<f32>>; // 加窗→LPC，含能量校验
```
`Pole { freq_hz, bandwidth_hz, radius }`。

**Steps:**
- [ ] 2.1 失败测试：
  - `levinson_durbin_recovers_synth_vowel_formants`：合成信号（F1=800, F2=1200, F0=150, sr=11025, 0.3s）分帧求 LPC → 求根 → 主极点频率落在真值 ±8%。
  - `poly_roots_recovers_known_polynomial`：`(z−0.5)(z−(0.3+0.4i))(z−(0.3−0.4i))` 展开系数 → 求根复原 1e-4。
  - `poles_to_coeffs_roundtrip`：任取两组 (f,bw) → 重建系数 → 再求根 → 频率/带宽复原 1e-3。
  - 不稳定/非有限输入 → `None`。
- [ ] 2.2 实现。注意：Levinson-Durbin 需对自相关加 1e-12 正则；激励能量比（残差/信号）供 gate 使用，返回结构里带上。
- [ ] 2.3 `cargo test formant::lpc --quiet` → PASS。
- [ ] 2.4 Commit: `feat(formant): lpc analysis, root finding and pole conversion`

### Task 3: track.rs（TDD）

**Files:**
- Create: `backend/src-tauri/src/audio/formant_morph/track.rs`

**函数：**
```rust
pub struct FormantCandidate { pub f1: Pole, pub f2: Pole, pub voiced: bool, pub residual_ratio: f32 }
pub fn select_f1_f2(poles: &[Pole], envelope_prominence: &dyn Fn(f32) -> f32) -> Option<FormantCandidate>;
pub struct TrackPoint { pub f1_hz: f32, pub f2_hz: f32, pub voiced: f32 }
pub fn extract_tracks(cands: &[Option<FormantCandidate>]) -> Vec<TrackPoint>;  // 未检出帧用最近邻桥接
pub fn smooth_tracks(tracks: &mut Vec<TrackPoint>);  // 5 帧中值 → ±15%/帧限速
pub fn interpolate_at(tracks: &[TrackPoint], t: f32) -> TrackPoint;
```

**Steps:**
- [ ] 3.1 失败测试：
  - `select_rejects_harmonic_like_poles`：极点带宽 800 Hz / 频率 260 Hz → None。
  - `select_requires_f2_above_f1_plus_200`。
  - `smooth_kills_octave_jump`：序列中 1 帧跳到 2 倍频 → 中值后不超 ±15% 限速。
  - `interpolate_is_linear_between_frames`。
- [ ] 3.2 实现。显著性评估：调用方传入包络幅频（|1/A| 经 FFT）后计算；候选筛选同时要求频率范围与带宽。
- [ ] 3.3 `cargo test formant::track --quiet` → PASS。
- [ ] 3.4 Commit: `feat(formant): formant candidate selection and track smoothing`

### Task 4: correction.rs + mod.rs 编排（TDD）

**Files:**
- Create: `backend/src-tauri/src/audio/formant_morph/correction.rs`
- Modify: `backend/src-tauri/src/audio/formant_morph/mod.rs`（完整编排）

**correction.rs 函数：**
```rust
pub fn target_pole(f: f32, target_hz: f32, strength: f32) -> f32;         // f + (t−f)·s
pub struct SpectralRatioFilter { pub coeffs_orig: Vec<f32>, pub coeffs_target: Vec<f32> }
pub fn h_response_db(filter: &SpectralRatioFilter, fft_size: usize, planner: &mut FftPlanner<f32>) -> Vec<f32>; // half bins, clamp ±24 dB
pub struct VoicingGate { state: f32 }  // attack 30ms / release 80ms
pub fn gate_advance(&mut self, raw: f32, dt_sec: f32) -> f32;
pub fn match_frame_energy(dry: &[f32], wet: &mut [f32]);                 // RMS clamp 0.5–2.0
pub fn soft_limit(out: &mut [f32], input: &[f32]);                       // tanh 软限幅 + 峰值×1.6 保护
```

**Steps:**
- [ ] 4.1 失败测试：
  - `h_response_peaks_at_moved_formant`：A_orig 极点 800 Hz → A_target 300 Hz；H 在 300 Hz 为正 dB 峰、800 Hz 为负 dB 谷。
  - `gate_attack_release_smoothing`：raw 0→1 时 gate 不突跳；raw 1→0 时释放更慢。
  - `match_frame_energy_clamps_gain`。
  - `soft_limit_bounds_output`。
- [ ] 4.2 实现并接入 `mod.rs` 编排（替换旧 STFT/倒谱主体）。
- [ ] 4.3 失败测试（端到端，mod.rs tests）：
  - **`formant_shift_moves_measured_f1_toward_target`**（核心闸门）：合成 /a/(F1=800,F2=1200,F0=150, 48k, 0.5s) → 目标 /i/(300,2300)，strength 0.9 → 对输出用 `lpc::analyze_frame+roots_to_poles` 重估 F1/F2，断言 `|out_f1 − 300| < |in_f1 − 300|` 且 `|out_f2 − 2300| < |in_f2 − 2300|`。
  - `stronger_strength_moves_further`（0.3/0.6/1.0 单调）。
  - `unvoiced_noise_input_is_not_amplified`（白噪声，输出峰值 ≤ 输入×1.6，无 NaN）。
  - `pathological_input_stays_bounded`（直流/方波/单帧长度）。
  - 保留全部旧测试（bypass/长度/有限性/静音/立体声/预设表）。
- [ ] 4.4 `cargo test formant_morph --quiet` 全绿。
- [ ] 4.5 Commit: `feat(formant): lpc pole-migration spectral-ratio dsp core`

### Task 5: 清理与文件头注释

- [ ] 5.1 删除旧 `audio/formant_morph.rs`，确认 `lib.rs` 指向新目录。
- [ ] 5.2 每个新文件补全文件头注释（主要内容/关系/维护说明），关键函数头注释（流程/规则/参数）。
- [ ] 5.3 `cargo clippy -p hifishifter -- -D warnings`（或项目现行 lint 命令）通过；`cargo test formant --quiet` 全绿。
- [ ] 5.4 Commit: `chore(formant): remove legacy cepstral implementation`

### Task 6: analysis.rs + analyze_clip_formants 命令

**Files:**
- Create: `backend/src-tauri/src/audio/formant_morph/analysis.rs`
- Create: `backend/src-tauri/src/commands/formant.rs`
- Modify: `backend/src-tauri/src/commands.rs`（`#[path = "commands/formant.rs"] mod formant;` + 转发命令）
- Modify: `backend/src-tauri/src/lib.rs`（invoke_handler 注册）

**接口：**
```rust
// analysis.rs（与 DSP 共用 lpc/track）
pub struct FormantAnalysisSummary {
    pub source_f1_hz: f32, pub source_f2_hz: f32,
    pub track: Vec<(f32, f32, f32)>,   // (t_norm, f1, f2)，≤64 点稀疏
    pub voiced_ratio: f32,
    pub message: Option<&'static str>, // "source_too_short" / "no_voiced_frames"
}
pub fn analyze_clip_formants(mono: &[f32], sr: u32) -> FormantAnalysisSummary;

// commands/formant.rs
pub(super) fn analyze_clip_formants(state: State<'_, AppState>, clip_id: String)
    -> Result<ClipFormantAnalysisPayload, String>;
```
- 取数复用 `compute_formant_cache_entry_for_clip` 的解码/切片/重采样路径（抽公共函数避免复制）。
- 结果小缓存：`Mutex<HashMap<key, payload>>`，键 = clip_id + source_path + mtime + 窗口量化 1 ms；`invalidate_formant_cache_for_clip` 同步失效。
- 前端 payload 命名 `snake_case`，含 `ok / sourceF1Hz…`（用 `#[tauri::command(rename_all = "camelCase")]` 对齐前端习惯）。

**Steps:**
- [ ] 6.1 失败测试（analysis.rs）：合成 /a/ → `source_f1_hz` ∈ 800±10%、`voiced_ratio` > 0.8；静音 → `no_voiced_frames`；过短 → `source_too_short`。
- [ ] 6.2 实现 analysis.rs 与命令、注册；`cargo build` 通过。
- [ ] 6.3 Commit: `feat(formant): clip source formant analysis command`

### Task 7: 前端 UI

**Files:**
- Modify: `frontend/src/services/api/timeline.ts`（`analyzeClipFormants(clipId)`）
- Modify: `frontend/src/features/session/sessionSlice.ts` + `sessionTypes.ts`（`clipFormantAnalysis` 状态与 action）
- Modify: `frontend/src/components/layout/timeline/clip/VowelChart.tsx`（source 点/轨迹/连线箭头 + `onPickVowel`）
- Modify: `frontend/src/components/layout/timeline/clip/ClipFormantToolWindow.tsx`（Switch、Bypass、数值对照、滑杆+数值输入、无元音提示）
- Modify: `frontend/src/features/session/sessionSlice.formantToolWindow.test.ts` 等（补测试）
- Modify: i18n ×5：`clip_formant_source/target/bypass/pick_vowel/analysis_none/no_voiced`

**规则：**
- Bypass：仅本地试听旁通（走既有 commit 通道，`enabled` 临时取反 + 结束恢复），不产生新撤销步语义变化（沿用 `dirtyRef` 逻辑需在测试中确认）。
- 事件隔离 / 拖拽中止 / Space 抑制逻辑（`clipFormantInteractionGuards`、body 属性）全部保留不动。
- F1/F2 范围前后端统一为 F1∈[250,1000]、F2∈[540,2600]（后端 clamp 改到与前端一致）。

**Steps:**
- [ ] 7.1 API + slice + 类型。
- [ ] 7.2 VowelChart 源点/连线/可点击元音。
- [ ] 7.3 ToolWindow 重排（Switch/Bypass/数值/提示）。
- [ ] 7.4 i18n 五语同步。
- [ ] 7.5 `npm --prefix frontend run test` 全绿（含新增用例）；`tsc --noEmit` 通过。
- [ ] 7.6 Commit: `feat(ui): clip formant window source visualization and bypass`

### Task 8: 验收与文档

- [ ] 8.1 `cargo tauri dev` 手动验收（设计文档 §7 清单）。
- [ ] 8.2 更新 `docs/i18n/USERMANUAL.md` 与 README 中 clip formant 描述。
- [ ] 8.3 Commit: `docs: update clip formant manual`
