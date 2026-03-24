# HiFiShifter 后端详细分析

> 文档生成时间：2026-03-16  
> 项目版本：v0.1.0-beta.6

---

## 一、技术栈详解

### 1.1 核心技术

| 技术 | 版本 | 用途 | 选型理由 |
|------|------|------|----------|
| **Rust** | 1.80+ | 系统编程语言 | 内存安全、零成本抽象、高性能 |
| **Tauri** | 2.0 | 桌面应用框架 | 轻量、跨平台、安全 |
| **cpal** | 0.16 | 音频 I/O | 跨平台低延迟音频回调 |
| **rodio** | 0.20 | 音频播放 | 解码与混音 |

### 1.2 音频处理库

| 库名 | 用途 |
|------|------|
| symphonia | 多格式音频解码（WAV/FLAC/MP3等） |
| hound | WAV 编码输出 |
| rubato | 高质量采样率转换 |
| Signalsmith Stretch | 时间伸缩算法 |
| rubato | 重采样

### 1.3 声码器与模型

| 组件 | 用途 |
|------|------|
| WORLD | 传统声码器（Harvest/DIO音高分析） |
| NSF-HiFiGAN ONNX | 深度学习声码器（高音质合成） |
| vslib (VocalShifter) | 兼容 VocalShifter 项目 |

---

## 二、架构分层

```mermaid
flowchart TB
    subgraph Layer1["IPC 命令层 (commands/)"]
        direction LR
        L1A[timeline_commands.rs]
        L1B[project_commands.rs]
        L1C[audio_commands.rs]
        L1D[params_commands.rs]
    end

    subgraph Layer2["状态管理层"]
        direction LR
        L2A[AppState<br/>全局状态]
        L2B[TimelineState<br/>时间线状态]
        L2C[ProjectState<br/>工程状态]
    end

    subgraph Layer3["引擎层"]
        direction LR
        L3A[AudioEngine<br/>实时播放引擎]
        L3B[Renderer<br/>合成渲染器]
    end

    subgraph Layer4["处理层"]
        direction LR
        L4A[ProcessorChain<br/>处理链]
        L4B[Vocoder<br/>声码器]
        L4C[AudioUtils<br/>音频工具]
    end

    subgraph Layer5["存储层"]
        direction LR
        L5A[WaveformCache<br/>波形缓存]
        L5B[ClipPitchCache<br/>音高缓存]
        L5C[FileIO<br/>文件读写]
    end

    Layer1 -->|"读写"| Layer2
    Layer1 -->|"控制"| Layer3
    Layer3 -->|"调用"| Layer4
    Layer4 -->|"缓存"| Layer5
```

---

## 三、目录结构详解

```
backend/src-tauri/src/
├── lib.rs                        # 库入口，注册所有命令
├── main.rs                       # 应用入口，Tauri 配置
├── state.rs                      # AppState 全局状态定义 (~82KB)
│
├── commands/                     # Tauri IPC 命令层
│   ├── mod.rs                    # 模块导出
│   ├── timeline_commands.rs      # 时间线操作命令
│   ├── project_commands.rs       # 工程管理命令
│   ├── audio_commands.rs         # 音频播放命令
│   ├── params_commands.rs        # 参数曲线命令
│   ├── transport_commands.rs     # 播放控制命令
│   ├── waveform_commands.rs      # 波形数据命令
│   └── import_commands.rs        # 导入相关命令
│
├── audio/                        # 音频处理工具
│   ├── mod.rs                    # 模块导出
│   ├── audio_io.rs               # 音频读写（WAV/FLAC等）
│   ├── audio_resample.rs         # 重采样处理
│   ├── audio_stretch.rs          # 时间伸缩
│   ├── audio_stretch_optimized.rs # 优化版时间伸缩
│   ├── audio_utils.rs            # 通用音频工具
│   ├── waveform_cache.rs         # 波形缓存
│   ├── mixdown.rs                # 混音合成
│   ├── player_mixer.rs           # 播放混音器
│   └── rubberband_stretcher.rs   # Rubberband 封装
│
├── audio_engine/                 # 实时音频引擎
│   ├── mod.rs                    # 模块导出
│   ├── audio_engine.rs           # AudioEngine 主类
│   ├── audio_output.rs           # cpal 音频输出
│   ├── event_emitter.rs          # 事件发射器
│   ├── player_track.rs           # 播放轨道
│   ├── mixer.rs                  # 混音器
│   └── clip_player.rs            # 剪辑播放器
│
├── renderer/                     # 合成渲染器
│   ├── mod.rs                    # 模块导出
│   ├── renderer.rs               # Renderer 主接口
│   ├── renderer_types.rs         # 类型定义
│   ├── processor_chain.rs        # 处理链（Stage 组合）
│   ├── time_stretch_stage.rs     # 时间伸缩 Stage
│   ├── pitch_shift_stage.rs      # 音高偏移 Stage
│   ├── param_curve_stage.rs      # 参数曲线 Stage
│   ├── world_processor.rs        # WORLD 声码器处理
│   ├── hifigan_processor.rs      # HiFiGAN 处理
│   └── vslib_processor.rs        # vslib 处理
│
├── vocoder/                      # 声码器实现
│   ├── mod.rs                    # 模块导出
│   ├── world_vocoder.rs          # WORLD 声码器
│   ├── hifigan_vocoder.rs        # HiFiGAN 声码器
│   ├── vslib_vocoder.rs          # vslib 声码器
│   └── pitch_analysis.rs         # 音高分析入口
│
├── pitch/                        # 音高处理
│   ├── mod.rs                    # 模块导出
│   ├── pitch_analysis.rs         # 音高分析核心
│   ├── pitch_utils.rs            # 音高工具
│   └── pitch_shift.rs            # 音高偏移
│
├── pitch_analysis/               # 音高分析调度
│   ├── mod.rs                    # 模块导出
│   ├── dispatcher.rs             # 分析调度器
│   ├── harvest_analyzer.rs       # Harvest 分析器
│   └── dio_analyzer.rs           # DIO 分析器
│
├── import/                       # 项目导入
│   ├── mod.rs                    # 模块导出
│   ├── vocalshifter_import.rs    # VocalShifter 导入
│   └── reaper_import.rs          # REAPER 导入
│
├── models/                       # 数据模型
│   ├── mod.rs                    # 模块导出
│   ├── timeline.rs               # 时间线模型
│   ├── clip.rs                   # 剪辑模型
│   ├── track.rs                  # 轨道模型
│   ├── params.rs                 # 参数模型
│   └── project.rs                # 工程模型
│
├── project/                      # 工程管理
│   ├── mod.rs                    # 模块导出
│   ├── project_loader.rs         # 工程加载
│   ├── project_saver.rs          # 工程保存
│   └── custom_scale.rs           # 自定义音阶
│
├── clip_pitch_cache.rs           # Clip 音高缓存
├── undo.rs                       # 撤销/重做系统
├── ui_bridge.rs                  # UI 桥接层
└── waveform/                     # 波形处理
    ├── mod.rs                    # 模块导出
    └── peaks.rs                  # 峰值计算
```

---

## 四、全局状态管理

### 4.1 AppState 结构

```mermaid
flowchart TB
    subgraph AppState["AppState (全局状态)"]
        direction TB
        
        subgraph Timeline["时间线状态"]
            T1[timeline: Mutex&lt;TimelineState&gt;]
            T2[timeline_history: Mutex&lt;TimelineHistory&gt;]
        end
        
        subgraph Project["工程状态"]
            P1[project: Mutex&lt;ProjectState&gt;]
        end
        
        subgraph Runtime["运行时状态"]
            R1[runtime: Mutex&lt;RuntimeState&gt;]
            R2[ui_locale: RwLock&lt;String&gt;]
        end
        
        subgraph Engine["音频引擎"]
            E1[audio_engine: AudioEngine]
        end
        
        subgraph Cache["缓存"]
            C1[waveform_cache: Mutex&lt;HashMap&gt;]
            C2[clip_pitch_cache: Arc&lt;Mutex&gt;]
            C3[pitch_timeline_snapshot: Mutex]
        end
        
        subgraph Analysis["分析状态"]
            A1[pitch_inflight: Mutex&lt;HashSet&gt;]
            A2[pitch_analysis_progress: RwLock]
        end
        
        subgraph Tauri["Tauri 集成"]
            TA1[app_handle: OnceLock&lt;AppHandle&gt;]
            TA2[config_dir: OnceLock&lt;PathBuf&gt;]
        end
    end
```

### 4.2 核心状态字段

```rust
pub struct AppState {
    // 时间线状态
    pub timeline: std::sync::Mutex<TimelineState>,
    pub timeline_history: std::sync::Mutex<TimelineHistory>,
    
    // 工程状态
    pub project: std::sync::Mutex<ProjectState>,
    
    // 运行时状态
    pub runtime: std::sync::Mutex<RuntimeState>,
    pub ui_locale: RwLock<String>,
    
    // 检查点控制
    pub suppress_checkpoints: std::sync::atomic::AtomicBool,
    
    // 波形缓存
    pub waveform_cache_dir: std::sync::Mutex<PathBuf>,
    pub waveform_cache: std::sync::Mutex<HashMap<String, Arc<CachedPeaks>>>,
    
    // 音高分析缓存
    pub pitch_inflight: std::sync::Mutex<HashSet<String>>,
    pub pitch_analysis_progress: std::sync::RwLock<Option<PitchOrigAnalysisProgressEvent>>,
    pub clip_pitch_cache: Arc<Mutex<ClipPitchCache>>,
    pub pitch_timeline_snapshot: Mutex<HashMap<String, TimelineSnapshot>>,
    
    // 音频引擎
    pub audio_engine: AudioEngine,
    
    // Tauri 句柄
    pub app_handle: OnceLock<tauri::AppHandle>,
    pub config_dir: OnceLock<std::path::PathBuf>,
}
```

---

## 五、音频引擎详解

### 5.1 AudioEngine 架构

```mermaid
flowchart TB
    subgraph AudioEngine["AudioEngine"]
        direction TB
        
        subgraph Output["音频输出层"]
            AO[cpal::Stream<br/>低延迟回调]
        end
        
        subgraph Mixer["混音层"]
            MX[Mixer<br/>多轨混音]
        end
        
        subgraph Tracks["播放轨道层"]
            PT1[PlayerTrack 1]
            PT2[PlayerTrack 2]
            PT3[PlayerTrack N]
        end
        
        subgraph Players["剪辑播放器"]
            CP1[ClipPlayer]
            CP2[ClipPlayer]
            CP3[ClipPlayer]
        end
        
        subgraph Sources["音频源"]
            S1[解码音频数据]
            S2[合成音频数据]
        end
        
        AO -->|"回调"| MX
        MX --> PT1
        MX --> PT2
        MX --> PT3
        PT1 --> CP1
        PT2 --> CP2
        PT3 --> CP3
        CP1 --> S1
        CP2 --> S1
        CP3 --> S2
    end
```

### 5.2 播放流程

```mermaid
sequenceDiagram
    participant Frontend as 前端
    participant Command as play_original
    participant Engine as AudioEngine
    participant Mixer as Mixer
    participant CP as ClipPlayer
    participant Output as cpal Output

    Frontend->>Command: invoke('play_original', startSec)
    Command->>Engine: play(range, target)
    Engine->>Mixer: prepare tracks
    Mixer->>CP: create ClipPlayers
    Engine->>Output: start stream
    
    loop 音频回调 (~44100 Hz)
        Output->>Mixer: callback(samples)
        Mixer->>CP: read_samples()
        CP-->>Mixer: audio samples
        Mixer->>Mixer: mix tracks
        Mixer-->>Output: mixed samples
    end
    
    Engine->>Frontend: emit('playback_ended')
```

### 5.3 播放状态机

```mermaid
stateDiagram-v2
    [*] --> Stopped: 初始化
    
    Stopped --> Playing: play()
    Stopped --> Preparing: preload()
    
    Preparing --> Playing: ready
    Preparing --> Stopped: error
    
    Playing --> Paused: pause()
    Playing --> Stopped: stop()
    Playing --> Seeking: seek()
    Playing --> Stopped: playback_end
    
    Paused --> Playing: resume()
    Paused --> Stopped: stop()
    
    Seeking --> Playing: seek_complete
    Seeking --> Stopped: error
```

---

## 六、声码器系统

### 6.1 声码器架构

```mermaid
flowchart TB
    subgraph VocoderSystem["声码器系统"]
        direction TB
        
        subgraph Analyzers["音高分析器"]
            A1[Harvest<br/>高精度]
            A2[DIO<br/>快速]
        end
        
        subgraph Vocoders["声码器引擎"]
            V1[WORLD Vocoder<br/>传统声码器]
            V2[HiFiGAN ONNX<br/>深度学习声码器]
            V3[vslib<br/>VocalShifter兼容]
        end
        
        subgraph Output["输出"]
            O1[合成音频]
            O2[参数数据]
        end
        
        A1 --> V1
        A2 --> V1
        A1 --> V2
        V1 --> O1
        V2 --> O1
        V3 --> O1
        V1 --> O2
    end
```

### 6.2 声码器选择策略

```mermaid
flowchart TB
    Start[开始合成]
    CheckAlgo{分析算法}
    
    subgraph WORLD["WORLD 流程"]
        W1[Harvest/DIO 分析]
        W2[CheapTrick 谱估计]
        W3[D4C 非周期性]
        W4[合成]
    end
    
    subgraph HiFiGAN["HiFiGAN 流程"]
        H1[ONNX 推理]
        H2[后处理]
    end
    
    subgraph Vslib["vslib 流程"]
        V1[调用 vslib.dll]
    end
    
    Start --> CheckAlgo
    CheckAlgo -->|"WorldDll/None"| WORLD
    CheckAlgo -->|"NsfHifiganOnnx"| HiFiGAN
    CheckAlgo -->|"VocalShifterVslib"| Vslib
    
    W1 --> W2 --> W3 --> W4
    H1 --> H2
    
    W4 --> Output[合成音频]
    H2 --> Output
    V1 --> Output
```

### 6.3 声码器对比

| 特性 | WORLD | NSF-HiFiGAN | vslib |
|------|-------|-------------|-------|
| **音质** | 良好 | 优秀 | 良好 |
| **速度** | 中等 | 快（GPU） | 快 |
| **依赖** | 纯 Rust | ONNX Runtime | DLL |
| **适用场景** | 通用 | 高质量合成 | VS 兼容 |

---

## 七、渲染器系统

### 7.1 ProcessorChain 架构

```mermaid
flowchart LR
    subgraph Input["输入"]
        I[原始音频]
    end
    
    subgraph Chain["ProcessorChain"]
        direction LR
        S1[TimeStretchStage<br/>时间伸缩]
        S2[PitchShiftStage<br/>音高偏移]
        S3[ParamCurveStage<br/>参数曲线]
        S4[VocoderStage<br/>声码器合成]
    end
    
    subgraph Output["输出"]
        O[合成音频]
    end
    
    I --> S1 --> S2 --> S3 --> S4 --> O
```

### 7.2 Stage 接口

```rust
pub trait Stage: Send + Sync {
    /// 处理音频数据
    fn process(&self, input: &AudioBuffer) -> Result<AudioBuffer>;
    
    /// 是否需要重新处理
    fn is_dirty(&self) -> bool;
    
    /// 标记为需要重新处理
    fn mark_dirty(&mut self);
    
    /// 获取处理参数
    fn get_params(&self) -> StageParams;
    
    /// 设置处理参数
    fn set_params(&mut self, params: StageParams);
}
```

### 7.3 渲染流程

```mermaid
sequenceDiagram
    participant Frontend as 前端
    participant Renderer as Renderer
    participant Chain as ProcessorChain
    participant Stage as Stages
    participant Vocoder as Vocoder

    Frontend->>Renderer: synthesize(clip_id, params)
    Renderer->>Chain: build_chain(params)
    
    loop 每个 Stage
        Chain->>Stage: process(audio)
        Stage-->>Chain: processed_audio
    end
    
    Chain->>Vocoder: final_synthesis()
    Vocoder-->>Chain: synthesized_audio
    Chain-->>Renderer: output
    Renderer-->>Frontend: SynthesizeResult
```

---

## 八、命令层详解

### 8.1 命令注册

```mermaid
flowchart TB
    subgraph Commands["Tauri Commands"]
        direction TB
        
        subgraph Timeline["时间线命令"]
            TL1[get_timeline_state]
            TL2[add_track / remove_track]
            TL3[add_clip / remove_clip]
            TL4[move_clip / split_clip]
            TL5[glue_clips]
        end
        
        subgraph Project["工程命令"]
            PR1[new_project / open_project]
            PR2[save_project / save_project_as]
            PR3[get_project_meta]
            PR4[set_project_base_scale]
        end
        
        subgraph Audio["音频命令"]
            AU1[play_original / stop_audio]
            AU2[process_audio]
            AU3[synthesize / save_synthesized]
            AU4[set_transport]
        end
        
        subgraph Params["参数命令"]
            PA1[get_param_frames]
            PA2[set_param_frames]
            PA3[restore_param_frames]
        end
        
        subgraph Waveform["波形命令"]
            WF1[get_waveform_peaks_segment]
            WF2[clear_waveform_cache]
        end
    end
```

### 8.2 主要命令列表

| 命令 | 文件 | 用途 |
|------|------|------|
| `get_timeline_state` | timeline_commands.rs | 获取完整时间线状态 |
| `add_track` | timeline_commands.rs | 添加新轨道 |
| `remove_track` | timeline_commands.rs | 删除轨道 |
| `add_clip` | timeline_commands.rs | 添加剪辑 |
| `move_clip` | timeline_commands.rs | 移动剪辑 |
| `split_clip` | timeline_commands.rs | 分割剪辑 |
| `glue_clips` | timeline_commands.rs | 合并剪辑 |
| `open_project` | project_commands.rs | 打开工程 |
| `save_project` | project_commands.rs | 保存工程 |
| `play_original` | audio_commands.rs | 播放原始音频 |
| `synthesize` | audio_commands.rs | 合成处理后的音频 |
| `get_param_frames` | params_commands.rs | 获取参数曲线帧数据 |
| `set_param_frames` | params_commands.rs | 设置参数曲线 |

---

## 九、后端链路详解

### 9.1 音频加载链路

```mermaid
sequenceDiagram
    participant Frontend as 前端
    participant Import as import_audio_item
    participant AudioIO as audio_io
    participant State as AppState
    participant Cache as WaveformCache

    Frontend->>Import: 拖拽音频文件
    Import->>AudioIO: decode_audio(path)
    AudioIO->>AudioIO: 检测格式
    AudioIO->>AudioIO: 解码为 f32
    AudioIO->>AudioIO: 重采样到目标采样率
    AudioIO-->>Import: AudioData
    
    Import->>State: create Track & Clip
    Import->>Cache: generate_peaks(audio)
    Cache-->>Import: waveform_peaks
    
    Import-->>Frontend: TimelineResult
```

### 9.2 音高分析链路

```mermaid
flowchart TB
    subgraph Trigger["触发分析"]
        T1[Clip 添加/更新]
    end
    
    subgraph Dispatcher["调度层"]
        D1[检查缓存]
        D2{缓存存在?}
        D3[加入分析队列]
        D4[返回缓存数据]
    end
    
    subgraph Analyzer["分析层"]
        A1[选择分析器]
        A2[Harvest 分析]
        A3[DIO 分析]
        A4[生成 f0/tension]
    end
    
    subgraph Cache["缓存层"]
        C1[存储 pitch_orig]
        C2[更新 clip_pitch_cache]
    end
    
    subgraph Event["事件"]
        E1[emit progress]
        E2[emit complete]
    end
    
    T1 --> D1 --> D2
    D2 -->|"是"| D4
    D2 -->|"否"| D3 --> A1
    A1 --> A2
    A1 --> A3
    A2 --> A4
    A3 --> A4
    A4 --> C1 --> C2 --> E1
    E1 --> E2
```

### 9.3 合成渲染链路

```mermaid
sequenceDiagram
    participant Frontend as 前端
    participant Command as synthesize
    participant Renderer as Renderer
    participant Chain as ProcessorChain
    participant Vocoder as Vocoder
    participant Engine as AudioEngine

    Frontend->>Command: invoke('synthesize', clip_id)
    Command->>Renderer: synthesize(clip)
    
    Renderer->>Chain: build from clip params
    
    Note over Chain: 时间伸缩处理
    Chain->>Chain: TimeStretchStage.process()
    
    Note over Chain: 参数曲线应用
    Chain->>Chain: ParamCurveStage.process()
    
    Note over Chain: 声码器合成
    Chain->>Vocoder: synthesize(pitch_curve, audio)
    Vocoder-->>Chain: synthesized audio
    
    Chain-->>Renderer: output buffer
    Renderer-->>Command: audio data
    
    Command->>Engine: load for playback
    Command-->>Frontend: SynthesizeResult
```

### 9.4 工程保存链路

```mermaid
flowchart TB
    subgraph Trigger["触发保存"]
        T1[Ctrl+S / 菜单保存]
    end
    
    subgraph Serialize["序列化层"]
        S1[TimelineState → JSON]
        S2[ProjectState → JSON]
        S3[Track/Clip 元数据]
        S4[参数曲线数据]
    end
    
    subgraph Embed["嵌入资源"]
        E1{是否嵌入音频?}
        E2[Base64 编码音频]
        E3[引用外部文件]
    end
    
    subgraph Output["输出"]
        O1[写入 .hshp 文件]
        O2[更新最近项目列表]
    end
    
    T1 --> S1 --> S2 --> S3 --> S4 --> E1
    E1 -->|"是"| E2 --> O1
    E1 -->|"否"| E3 --> O1
    O1 --> O2
```

---

## 十、缓存策略

### 10.1 缓存架构

```mermaid
flowchart TB
    subgraph CacheSystem["缓存系统"]
        direction TB
        
        subgraph Memory["内存缓存"]
            M1[waveform_cache<br/>HashMap&lt;String, CachedPeaks&gt;]
            M2[clip_pitch_cache<br/>LRU Cache]
        end
        
        subgraph Disk["磁盘缓存"]
            D1[waveform_cache_dir<br/>sessionStorage 同步]
            D2[.hshp 工程文件]
        end
        
        subgraph Session["会话缓存"]
            S1[pitch_inflight<br/>去重分析任务]
            S2[pitch_timeline_snapshot<br/>增量刷新快照]
        end
    end
```

### 10.2 缓存策略

| 缓存类型 | 位置 | 淘汰策略 | 持久化 |
|----------|------|----------|--------|
| 波形 Peaks | 内存 + sessionStorage | LRU (512条) | 否 |
| Clip 音高缓存 | 内存 | LRU (100条) | 否 |
| 音高分析进度 | 内存 | 手动清除 | 否 |
| 工程状态 | 内存 | 无 | 是 (.hshp) |

---

## 十一、线程模型

### 11.1 线程架构

```mermaid
flowchart TB
    subgraph MainThread["主线程"]
        M1[Tauri 命令处理]
        M2[AppState 访问]
    end
    
    subgraph AudioThread["音频线程 (cpal)"]
        A1[音频输出回调]
        A2[Mixer::read_samples]
        A3[实时混音]
    end
    
    subgraph AnalysisPool["分析线程池 (rayon)"]
        P1[音高分析任务]
        P2[波形计算任务]
        P3[合成任务]
    end
    
    subgraph EventThread["事件线程"]
        E1[Tauri 事件发射]
        E2[进度更新通知]
    end
    
    M1 -->|"spawn"| AnalysisPool
    AnalysisPool -->|"emit"| EventThread
    M1 -->|"start"| AudioThread
    AudioThread -->|"独立运行"| A3
```

### 11.2 同步原语

```rust
// Mutex 用于独占访问
pub timeline: Mutex<TimelineState>,
pub project: Mutex<ProjectState>,

// RwLock 用于读写分离
pub ui_locale: RwLock<String>,
pub pitch_analysis_progress: RwLock<Option<...>>,

// AtomicBool 用于标志位
pub suppress_checkpoints: AtomicBool,

// OnceLock 用于单次初始化
pub app_handle: OnceLock<AppHandle>,
```

---

## 十二、性能优化

### 12.1 优化策略

```mermaid
flowchart TB
    subgraph Optimizations["优化策略"]
        direction TB
        
        subgraph Compute["计算优化"]
            C1[rayon 并行计算]
            C2[SIMD 指令]
            C3[缓存计算结果]
        end
        
        subgraph Memory["内存优化"]
            M1[Arc 共享数据]
            M2[零拷贝传输]
            M3[预分配缓冲区]
        end
        
        subgraph IO["I/O优化"]
            I1[异步文件读写]
            I2[批量操作]
            I3[延迟加载]
        end
        
        subgraph Audio["音频优化"]
            A1[cpal 低延迟回调]
            A2[环形缓冲区]
            A3[预解码音频]
        end
    end
```

### 12.2 关键优化点

| 优化点 | 方法 | 效果 |
|--------|------|------|
| 音高分析 | rayon 并行 | 多核加速 |
| 波形渲染 | 分段缓存 | 减少计算 |
| 合成过程 | ONNX GPU | 提速 10x |
| 音频播放 | cpal 回调 | 延迟 <10ms |
| 状态更新 | 批量操作 | 减少锁竞争 |

---

## 十三、扩展指南

### 13.1 新增 IPC 命令

1. 在 `commands/` 下创建或编辑命令文件
2. 添加 `#[tauri::command]` 函数
3. 在 `lib.rs` 的 `invoke_handler` 中注册
4. 在前端 `services/api/` 添加对应调用

```rust
// commands/my_commands.rs
#[tauri::command]
pub fn my_new_command(state: State<'_, AppState>, arg: String) -> Result<MyResult, String> {
    // 实现逻辑
    Ok(MyResult { ... })
}

// lib.rs
.invoke_handler(tauri::generate_handler![
    // ...existing commands...
    my_new_command,
])
```

### 13.2 新增声码器

1. 在 `vocoder/` 下创建新模块
2. 实现 `Vocoder` trait
3. 在 `renderer/vocoder_selector.rs` 添加选择逻辑
4. 更新 `PitchAnalysisAlgo` 枚举

### 13.3 新增处理 Stage

1. 在 `renderer/` 下创建 `my_stage.rs`
2. 实现 `Stage` trait
3. 在 `processor_chain.rs` 添加到链中
4. 暴露参数配置接口

---

*文档由 AI 自动生成，如有疑问请参考源代码或联系开发者。*