# ClipProcessor 重构方案

**日期**: 2026-03-08  
**分支**: `feat/clip-processor-refactor`  
**状态**: 草案

---

## 背景与动机

### 现有架构的局限

当前的 `Renderer` trait（位于 `src/renderer/traits.rs`）仅负责"从 PCM + 音高编辑序列 → 合成 PCM"这一步，其 `RenderContext` 只携带音高相关参数：

```rust
pub struct RenderContext<'a> {
    pub mono_pcm: &'a [f32],
    pub sample_rate: u32,
    pub frame_period_ms: f64,
    pub pitch_edit: &'a [f32],   // MIDI cent，每帧
    pub clip_midi: &'a [f32],
    // ...
}
```

**问题一：时间拉伸在声码器体外**  
时间拉伸（`playback_rate`）当前由 `time_stretch.rs::time_stretch_interleaved()` 在调用 `Renderer` 前外部完成，使用 RubberBand 静态链接库。这导致两次 PCM 处理开销，且无法让具备原生时间拉伸能力的声码器（如 vslib）直接处理。

**问题二：无共振峰 / 气声曲线**  
`Clip`、`TrackParamsState` 均不含共振峰曲线、气声曲线字段，前端也没有入口传递这些参数。

**问题三：前端无法动态查询声码器能力**  
前端目前无法知道当前所选声码器支持哪些额外参数曲线，导致 UI 面板只能硬编码（或根本没有动态面板）。

---

## 目标

1. **统一全链路接口**：单个 `ClipProcessor::process()` 调用涵盖音高合成 + 时间拉伸 + 所有声码器参数
2. **向后兼容**：World / HiFiGAN 两个现有声码器通过 `ClipProcessorCompat` 包装，行为不变
3. **vslib 原生支持**：`VslibProcessor` 直接使用 vslib 逐帧控制点 API，不经过外部 RubberBand
4. **前端可查询能力**：通过 `param_descriptors()` 接口，前端可动态渲染对应的自动化曲线面板

---

## 核心数据结构

### ClipProcessContext

```rust
pub struct ClipProcessContext<'a> {
    /// 输入原始 mono PCM
    pub mono_pcm: &'a [f32],
    pub sample_rate: u32,

    /// 当前 clip 在 project 时间轴上的起点（秒），用于定位
    pub clip_start_sec: f64,
    /// 本次渲染段在 clip 内的起点/终点（秒，相对 clip 起点）
    pub seg_start_sec:  f64,
    pub seg_end_sec:    f64,

    /// 声码器帧周期（ms），决定 pitch_edit / extra_curves 长度
    pub frame_period_ms: f64,

    /// 每帧绝对 MIDI 音高（cent），由音高编辑层输出
    pub pitch_edit: &'a [f32],
    /// 每帧 clip 原始 MIDI 音高（cent），用于计算相对偏移
    pub clip_midi:  &'a [f32],

    /// 回放速率（时间拉伸比例）；1.0 = 不拉伸
    pub playback_rate: f64,

    /// 输出帧数（应用 playback_rate 后）
    pub out_frames: usize,

    /// 用于缓存 key 的 clip 唯一 ID
    pub clip_id: &'a str,

    /// 声码器专属自动化曲线（逐帧 Vec<f32>，AutomationCurve 类型参数）
    /// key = ParamDescriptor::id，仅包含已由用户编辑的曲线；缺失 key = 使用该参数默认值
    pub extra_curves: &'a HashMap<String, Vec<f32>>,
    /// 声码器专属静态参数（StaticEnum 类型参数，枚举值以 i32 转 f64 存储）
    /// key = ParamDescriptor::id
    pub extra_params: &'a HashMap<String, f64>,
}
```

### ProcessorCapabilities

```rust
pub struct ProcessorCapabilities {
    /// 是否原生处理 playback_rate（= true 时 compat 层不再调 RubberBand）
    pub handles_time_stretch: bool,
    /// 是否支持逐帧共振峰偏移曲线（"formant_shift_cents"）
    pub supports_formant: bool,
    /// 是否支持逐帧气声强度曲线（"breathiness"）
    pub supports_breathiness: bool,
}
```

### ParamDescriptor

```rust
pub enum ParamKind {
    /// 逐帧自动化曲线，显示在时间轴上，存入 extra_curves
    AutomationCurve {
        unit: &'static str,   // 例："cents", ""（0-1 无单位）
        default_value: f32,
        min_value: f32,
        max_value: f32,
    },
    /// 静态枚举，前端渲染为按钮切换组，存入 extra_params（值为 i32 转 f64）
    StaticEnum {
        options: &'static [(&'static str, i32)],  // (显示名, 整数值)
        default_value: i32,
    },
}

pub struct ParamDescriptor {
    /// AutomationCurve → extra_curves 的 key
    /// StaticEnum      → extra_params 的 key
    /// 同时用于 Tauri command 序列化
    pub id: &'static str,
    pub display_name: &'static str,
    /// 前端面板分组标题
    pub group: &'static str,
    pub kind: ParamKind,
}
```

> **排除参数**：`eq1` / `eq2` / `heq` 不暴露给用户。  
> **固定参数**：`nnOffset` / `nnRange` 在分析阶段由后端固定，不作为可编辑参数。

### ClipProcessor trait

```rust
pub trait ClipProcessor: Send + Sync {
    /// 唯一标识符（与 SynthPipelineKind 对应）
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    /// 运行时可用性检查（例：vslib 仅在 Windows 且 DLL 已加载时为 true）
    fn is_available(&self) -> bool;
    fn capabilities(&self) -> ProcessorCapabilities;
    /// 静态声明支持的额外自动化曲线
    fn param_descriptors(&self) -> &'static [ParamDescriptor];
    /// 全链路处理：PCM → 合成 PCM（含音高 + 拉伸 + 声码器参数）
    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String>;
}
```

---

## ProcessorChain：组件化 Stage 链

用 **Stage 链**取代泛型 `ClipProcessorCompat<T>`。每个 `ProcessingStage` 接收上一步输出的 PCM，返回新 PCM；`ProcessorChain` 串联多个 Stage 并实现 `ClipProcessor`。新增功能（降噪、后处理混响等）只需 push 一个新 Stage，无需修改现有代码。

### ProcessingStage trait

```rust
/// 传递给每个 Stage 的上下文（含完整 ClipProcessContext 引用）
pub struct StageContext<'a> {
    pub clip_ctx: &'a ClipProcessContext<'a>,
}

pub trait ProcessingStage: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    /// Stage 自身贡献的参数描述符（可选）
    fn param_descriptors(&self) -> &'static [ParamDescriptor] { &[] }
    /// 接收上一步 PCM，输出处理后 PCM
    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String>;
}
```

### ProcessorChain

```rust
pub struct ProcessorChain {
    pub id:           String,
    pub display_name: String,
    pub stages:       Vec<Box<dyn ProcessingStage>>,
}

impl ClipProcessor for ProcessorChain {
    fn id(&self) -> &str { &self.id }
    fn display_name(&self) -> &str { &self.display_name }
    fn is_available(&self) -> bool { true }

    fn capabilities(&self) -> ProcessorCapabilities {
        // 聚合所有 Stage 的能力（当前 World/HiFiGAN 链均不原生处理拉伸）
        ProcessorCapabilities {
            handles_time_stretch: false,
            supports_formant:     false,
            supports_breathiness: false,
        }
    }

    /// 聚合所有 Stage 的参数描述符
    fn param_descriptors(&self) -> Vec<&'static ParamDescriptor> {
        self.stages.iter().flat_map(|s| s.param_descriptors()).collect()
    }

    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String> {
        let stage_ctx = StageContext { clip_ctx: ctx };
        let mut pcm = ctx.mono_pcm.to_vec();
        for stage in &self.stages {
            pcm = stage.process(pcm, &stage_ctx)?;
        }
        Ok(pcm)
    }
}
```

### 内置 Stage 实现

```rust
/// Stage 1：RubberBand 时间拉伸（playback_rate == 1.0 时直接透传）
pub struct RubberBandTimeStretchStage;
impl ProcessingStage for RubberBandTimeStretchStage {
    fn id(&self) -> &str { "rubberband_stretch" }
    fn display_name(&self) -> &str { "时间拉伸 (RubberBand)" }
    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let rate = ctx.clip_ctx.playback_rate;
        if (rate - 1.0).abs() > 1e-6 {
            time_stretch_interleaved(&input_pcm, ctx.clip_ctx.sample_rate, rate)
        } else {
            Ok(input_pcm)
        }
    }
}

/// Stage 2a：WORLD 声码器合成
pub struct WorldVocoderStage;
impl ProcessingStage for WorldVocoderStage {
    fn id(&self) -> &str { "world_vocoder" }
    fn display_name(&self) -> &str { "WORLD 声码器" }
    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        let render_ctx = RenderContext {
            mono_pcm: &input_pcm,
            sample_rate: cc.sample_rate,
            frame_period_ms: cc.frame_period_ms,
            pitch_edit: cc.pitch_edit,
            clip_midi: cc.clip_midi,
        };
        WorldRenderer.render(&render_ctx)
    }
}

/// Stage 2b：NSF-HiFiGAN 合成
pub struct HiFiGanStage;
impl ProcessingStage for HiFiGanStage { /* 同上，调用 HiFiGanRenderer */ }
```

### 预设链构造

```rust
pub fn world_chain() -> ProcessorChain {
    ProcessorChain {
        id:           "world".into(),
        display_name: "WORLD Vocoder".into(),
        stages: vec![
            Box::new(RubberBandTimeStretchStage),
            Box::new(WorldVocoderStage),
        ],
    }
}

pub fn hifigan_chain() -> ProcessorChain {
    ProcessorChain {
        id:           "nsf_hifigan".into(),
        display_name: "NSF-HiFiGAN".into(),
        stages: vec![
            Box::new(RubberBandTimeStretchStage),
            Box::new(HiFiGanStage),
        ],
    }
}
```

### 未来扩展示例

```rust
// 在任意链末尾插入后处理 Stage，无需改动现有代码
world_chain().stages.push(Box::new(NoiseReductionStage));
world_chain().stages.push(Box::new(ReverbStage));
```

---

## vslib VslibProcessor 原生实现

vslib 自带时间拉伸、共振峰、气声、Timing 控制点，无需外部 RubberBand。

```rust
/// 仅在 feature = "vslib" 下编译
#[cfg(feature = "vslib")]
pub struct VslibProcessor;

#[cfg(feature = "vslib")]
impl ClipProcessor for VslibProcessor {
    fn id(&self) -> &str { "vslib" }
    fn display_name(&self) -> &str { "VocalShifter (vslib)" }
    fn is_available(&self) -> bool { true /* DLL 已加载 */ }

    fn capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            handles_time_stretch: true,   // 使用 vslib Timing 控制点
            supports_formant:     true,
            supports_breathiness: true,
        }
    }

    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &[
            // ── 合成模式（按钮切换）──────────────────────────────
            ParamDescriptor {
                id: "synth_mode",
                display_name: "合成模式",
                group: "合成",
                kind: ParamKind::StaticEnum {
                    options: &[
                        ("单音",           0),  // SYNTHMODE_M
                        ("单音+共振峰补正", 1),  // SYNTHMODE_MF（启用 formant 曲线）
                        ("和音",           2),  // SYNTHMODE_P
                    ],
                    default_value: 1,  // 默认 SYNTHMODE_MF
                },
            },
            // ── 音量 / 声像（AutomationCurve）────────────────────
            ParamDescriptor {
                id: "volume",
                display_name: "音量",
                group: "动态",
                kind: ParamKind::AutomationCurve {
                    unit: "×", default_value: 1.0, min_value: 0.0, max_value: 4.0,
                },
            },
            ParamDescriptor {
                id: "dyn_edit",
                display_name: "强弱",
                group: "动态",
                kind: ParamKind::AutomationCurve {
                    unit: "×", default_value: 1.0, min_value: 0.0, max_value: 4.0,
                },
            },
            ParamDescriptor {
                id: "pan",
                display_name: "声像",
                group: "动态",
                kind: ParamKind::AutomationCurve {
                    unit: "", default_value: 0.0, min_value: -1.0, max_value: 1.0,
                },
            },
            // ── 声色（AutomationCurve）───────────────────────────
            ParamDescriptor {
                id: "formant_shift_cents",
                display_name: "共振峰偏移",
                group: "声色",
                kind: ParamKind::AutomationCurve {
                    unit: "cents", default_value: 0.0, min_value: -2400.0, max_value: 2400.0,
                },
            },
            ParamDescriptor {
                id: "breathiness",
                display_name: "气声",
                group: "声色",
                kind: ParamKind::AutomationCurve {
                    unit: "", default_value: 0.0, min_value: -10000.0, max_value: 10000.0,
                },
            },
        ]
    }
    // 注：eq1 / eq2 / heq 不暴露；nnOffset / nnRange 固定，不在参数列表中

    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String> {
        // 1. VslibCreateProject
        // 2. 读取 synth_mode（extra_params["synth_mode"]，默认 SYNTHMODE_MF=1）
        // 3. VslibAddItemEx(wav_path, nnOffset_fixed, nnRange_fixed, option)
        //    nnOffset / nnRange 由后端固定，不从 extra_params 读取
        // 4. VslibSetItemInfo → 写入 synthMode
        // 5. VslibSetPitchArray (pitch_edit 每帧)
        // 6. 逐控制点 VslibSetCtrlPntInfoEx2:
        //      volume        ← extra_curves["volume"]（缺失则保持默认 1.0）
        //      dynEdit       ← extra_curves["dyn_edit"]（缺失则保持默认 1.0）
        //      pan           ← extra_curves["pan"]（缺失则保持默认 0.0）
        //      formant       ← extra_curves["formant_shift_cents"]（仅 synthMode=MF/P 时有效）
        //      breathiness   ← extra_curves["breathiness"]
        //      eq1/eq2/heq   ← 不设置，保持 vslib 默认值
        // 7. 若 playback_rate != 1.0 → VslibAddTimeCtrlPnt (Timing 控制点链)
        // 8. VslibGetMixData → 读取输出 PCM
        // 9. VslibDeleteProject
        todo!("VslibProcessor::process")
    }
}
```

### 声码器能力矩阵

| 声码器                     | 时间拉伸           | 合成模式按钮       | 音量/强弱/声像曲线 | 共振峰曲线               | 气声曲线 |
| -------------------------- | ------------------ | ------------------ | ------------------ | ------------------------ | -------- |
| WorldVocoder（链）         | RubberBand Stage   | ✗                  | ✗                  | ✗                        | ✗        |
| NsfHiFiGAN（链）           | RubberBand Stage   | ✗                  | ✗                  | ✗                        | ✗        |
| VocalShifter（vslib 原生） | 原生 Timing 控制点 | ✓ SYNTHMODE_M/MF/P | ✓ 逐帧             | ✓ 逐帧 cent（MF/P 模式） | ✓ 逐帧   |

> **不暴露**：`eq1` / `eq2` / `heq`（vslib 内部字段，不传给用户）  
> **固定值**：`nnOffset` / `nnRange` 由后端在分析阶段写死，不在参数描述符中

---

## 数据模型扩展

### TrackParamsState

```rust
pub struct TrackParamsState {
    // 已有字段...
    pub pitch_orig: Vec<f32>,
    pub pitch_edit: Vec<f32>,

    // 新增：AutomationCurve 类型参数（key = ParamDescriptor::id）
    pub extra_curves: HashMap<String, Vec<f32>>,
    // 新增：StaticEnum 类型参数（key = ParamDescriptor::id，值为枚举整数转 f64）
    pub extra_params: HashMap<String, f64>,
}
```

### Clip

```rust
pub struct Clip {
    // 已有字段...
    pub playback_rate: f32,

    // 新增：clip 级别 AutomationCurve 覆盖（None = 使用 track 级别）
    pub extra_curves: Option<HashMap<String, Vec<f32>>>,
    // 新增：clip 级别 StaticEnum 覆盖（None = 使用 track 级别）
    pub extra_params: Option<HashMap<String, f64>>,
}
```

### synth_clip_cache.rs 哈希更新

`SynthClipCache` 的 `param_hash` 需纳入 `extra_curves` 和 `extra_params`，保证任何参数变化时缓存失效：

```rust
// AutomationCurve：遍历排序后的 key-value
for (k, v) in extra_curves.iter().sorted_by_key(|(k, _)| *k) {
    hash.update(k.as_bytes());
    for &val in v.iter() {
        hash.update(&val.to_le_bytes());
    }
}
// StaticEnum：遍历排序后的 key-value
for (k, v) in extra_params.iter().sorted_by_key(|(k, _)| *k) {
    hash.update(k.as_bytes());
    hash.update(&v.to_le_bytes());
}
```

---

## 迁移分阶段计划

### Phase 1 — 定义新 trait（`renderer/traits.rs`）
- 新增 `ClipProcessContext`、`ProcessorCapabilities`、`ParamDescriptor`、`ParamKind`
- 新增 `ClipProcessor` trait
- 保留原有 `Renderer` trait 不变（向后兼容）

**影响文件**：`src/renderer/traits.rs`

---

### Phase 2 — ProcessorChain 层（`renderer/chain.rs`）
- 新建 `src/renderer/chain.rs`
- 实现 `ProcessingStage` trait、`StageContext`、`ProcessorChain`
- 实现内置 Stage：`RubberBandTimeStretchStage`、`WorldVocoderStage`、`HiFiGanStage`
- 构造 `world_chain()` 和 `hifigan_chain()` 预设链，注册到 `get_processor(SynthPipelineKind)`

**影响文件**：新建 `src/renderer/chain.rs`，修改 `src/renderer/mod.rs`

---

### Phase 3 — vslib 处理器（`renderer/vslib_processor.rs`）
- 新建 `src/renderer/vslib_processor.rs`（`#[cfg(feature = "vslib")]`）
- 实现 `VslibProcessor`（步骤见上方伪代码）
- 注册到 `get_processor(SynthPipelineKind::VocalShifterVslib)`

**影响文件**：新建 `src/renderer/vslib_processor.rs`，修改 `src/renderer/mod.rs`、`src/state.rs`

---

### Phase 4 — 调用方迁移
- `src/pitch/pitch_editing.rs`：将 `get_renderer(...).render(...)` 替换为 `get_processor(...).process(...)`
- `src/commands/playback.rs`：同上，移除外部 `time_stretch_interleaved` 调用（compat 内部处理）

**影响文件**：`src/pitch/pitch_editing.rs`、`src/commands/playback.rs`

---

### Phase 5 — 状态模型扩展
- `src/state.rs`：`TrackParamsState` 新增 `extra_curves` 字段，`Clip` 新增 `extra_curves` 字段
- `src/models.rs`：更新对应的序列化/反序列化结构体
- `SynthPipelineKind`：新增 `VocalShifterVslib` 变体

**影响文件**：`src/state.rs`、`src/models.rs`

---

### Phase 6 — 缓存哈希更新
- `src/synth_clip_cache.rs`：`param_hash` 纳入 `extra_curves` 和 `extra_params`

**影响文件**：`src/synth_clip_cache.rs`

---

### Phase 7 — 缓存目录管理 + 前端清除入口

#### 缓存目录规范

vslib 处理过程中需要对输入 PCM 写临时 WAV，合成缓存片段也落盘。统一存放路径：

```
<exe_dir>/cache/
  vslib_tmp/        ← VslibProcessor 写入的临时 WAV（处理完后立即删除）
  synth/            ← SynthClipCache 落盘的合成片段（哈希命名，长期保留直到手动清除）
```

后端获取路径方式：

```rust
// 在 AppState 初始化时确定并创建目录
fn resolve_cache_dir() -> PathBuf {
    // tauri::api::path::resource_dir() 或 std::env::current_exe() 旁边
    let exe = std::env::current_exe().expect("cannot resolve exe path");
    exe.parent().unwrap().join("cache")
}
```

`VslibProcessor::process()` 使用 `<cache_dir>/vslib_tmp/<clip_id>_<uuid>.wav`，处理完毕后 `std::fs::remove_file` 删除；若 process 中途 panic，由外层 `Drop` guard 清理。

#### 新增 Tauri Command：`clear_cache`

```rust
#[tauri::command]
pub fn clear_cache(state: tauri::State<AppState>) -> Result<u64, String> {
    // 1. 清空 SynthClipCache 内存条目
    // 2. 删除 <cache_dir>/synth/ 下所有文件，统计释放字节数返回给前端
    // 3. 不删除 vslib_tmp/（运行中可能有文件；tmp 文件由 processor 自行负责）
    todo!()
}
```

返回值 `u64` 为释放的字节数，前端可显示 "已释放 xx MB"。

#### 前端清除缓存入口

- 位置：**设置面板 → 存储** 分组，或顶部菜单 **编辑 → 清除合成缓存**
- 交互：点击 → 调用 `clear_cache` command → 显示 Toast "已清除合成缓存（释放 xx MB）"
- 无需确认弹窗（清除后可重新合成，非破坏性操作）

**影响文件**：新建 `src/commands/cache.rs`，修改 `src/commands/mod.rs`（注册命令），`src/synth_clip_cache.rs`（暴露 clear 方法），前端新增设置面板按钮

---

## 需要修改的文件清单

| 文件                              | 操作                                                            | 阶段      |
| --------------------------------- | --------------------------------------------------------------- | --------- |
| `src/renderer/traits.rs`          | 新增 ClipProcessor trait + 相关结构体                           | Phase 1   |
| `src/renderer/chain.rs`           | 新建 ProcessingStage trait、ProcessorChain、内置 Stage、预设链  | Phase 2   |
| `src/renderer/mod.rs`             | 注册 chain + vslib processor                                    | Phase 2-3 |
| `src/renderer/vslib_processor.rs` | 新建 VslibProcessor                                             | Phase 3   |
| `src/state.rs`                    | 扩展 TrackParamsState, Clip, SynthPipelineKind                  | Phase 5   |
| `src/models.rs`                   | 同步序列化结构体                                                | Phase 5   |
| `src/pitch/pitch_editing.rs`      | 迁移到 ClipProcessor 调用                                       | Phase 4   |
| `src/commands/playback.rs`        | 迁移到 ClipProcessor 调用                                       | Phase 4   |
| `src/synth_clip_cache.rs`         | 更新 param_hash 含 extra_curves + extra_params；暴露 clear 方法 | Phase 6   |
| `src/commands/cache.rs`           | 新建 `clear_cache` Tauri command                                | Phase 7   |
| `src/commands/mod.rs`             | 注册 clear_cache                                                | Phase 7   |
| 前端设置面板                      | 新增「清除合成缓存」按钮                                        | Phase 7   |

---

## 风险与注意事项

1. **vslib Windows-only**：`VslibProcessor` 在 feature gate `vslib` 下编译，macOS/Linux 构建不受影响
2. **vslib 输入文件 I/O**：`VslibAddItemEx` 需要一个 WAV 文件路径，临时 WAV 写入 `<exe_dir>/cache/vslib_tmp/`，处理完立即删除；`VslibGetMixData` 直接把混音 PCM 读入调用方缓冲区，不产生输出临时文件。不适合实时预览，仅用于离线渲染（mixdown）或预合成缓存
3. **缓存目录跨版本兼容**：`<exe_dir>/cache/synth/` 中的文件名基于参数哈希，升级版本后哈希算法若有变化需在启动时清空旧缓存目录
3. **compat 层 RubberBand 双重处理**：compat 层对 World/HiFiGAN 仍需先 RubberBand 拉伸再合成，相比直接原生时间拉伸有额外开销；可在未来实现 World 原生时间拉伸后去掉 compat
4. **extra_curves 内存**：逐帧 Vec<f32> 在长 clip 下可能占用可观内存，需在缓存层控制粒度

---

## 参考文件

- `backend/src-tauri/src/renderer/traits.rs` — 现有 Renderer trait
- `backend/src-tauri/src/vocoder/vslib.rs` — vslib FFI 绑定（本 session 创建）
- `backend/src-tauri/third_party/vslib/vslib.h` — vslib C 头文件
- `backend/src-tauri/third_party/vslib/readme.txt` — vslib 用法说明
- `backend/src-tauri/src/time_stretch.rs` — 现有 RubberBand 时间拉伸入口
- `backend/src-tauri/src/synth_clip_cache.rs` — 现有合成缓存
