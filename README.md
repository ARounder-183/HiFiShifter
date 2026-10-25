# HachiShifter Next

HachiShifter 是用于录音调音、UTAU 音源合成与 DiffSinger 歌声合成的桌面编辑器。本仓库基于 [HiFiShifter](https://github.com/ARounder-183/HiFiShifter) 的项目历史继续开发，当前整合版采用 C++20 / JUCE 界面，并通过原生后端和 Python 桥接处理分析与合成。

当前公开版本：**0.25**。本机源码基线：**更新 152，2026-10-14**（中文 CVVC 配置拼写扩展与发音器工具栏入口）。外置 WCSNDM 版本仍为 **0.0803**；功能更新编号不属于公开版本号。`SOURCE_STATUS.md` 和源码包说明保留上次 GitHub 同步（更新 143）的发布记录。

## 从哪里开始

- 查实际功能对应的主接口、数据流和架构优化顺序：[功能接口与架构审计](docs/AI_INTERFACE_ARCHITECTURE.md)、[公开接口声明清单](docs/AI_API_REFERENCE.md)（2026-10-13 静态核对，未改运行时代码）。
- 阅读各模块职责、数据流和维护边界：[代码功能总览与维护指南](docs/AI_CODE_GUIDE.md)。
- 查找具体实现文件与诊断命令：[逐文件索引与命令入口](docs/AI_SOURCE_INDEX.md)。
- 查看每次修复与验证记录：[更新记录](docs/updates.md)。
- 查看本次源码同步范围和验证摘要：[源码状态](docs/SOURCE_STATUS.md)、[源码包说明](源码包说明.md)。
- AI 修改源码前先阅读 [AGENTS.md](AGENTS.md)。导航中保留的本机路径用于追溯开发环境，其他开发者按下方通用步骤构建。

## 主要功能与代码位置

| 部分 | 当前功能 | 主要实现 |
| --- | --- | --- |
| 编辑界面 | 多轨时间轴、钢琴卷帘、框选、音高工具、缩放滚动、源与渲染波形、多语言和主题 | `juce/src/MainComponent*`、`TimelineComponent*`、`PianoRollComponent*`、`Theme*`、`I18n*` |
| 工程与素材 | 撤销重做、保存与恢复、组合区域、复制粘贴、源文件与目标时间的映射 | `ProjectModel*`、`ProjectFileIO.h`、`ProjectRecovery.h`、`ClipParts.h` |
| 非 U 录音编辑 | 移动、拉伸、自由裁剪、向原文件扩展恢复隐藏内容、分割、粘连与断开、重叠控制 | `NativeNoteTiming.h`、`NativeAudio*`、`NativeTrimSource.h`、`NativeNoteJoin.h` |
| 原始音高分析 | GAME 音符划分、FCPE 连续 F0、native-hq 回退、源音高与目标音高分离 | `backend/AnalysisService*`、`GameAnalyzer*`、`FcpeAnalyzer*`、`NativeAnalyzer*` |
| 原生音高与响度 | 手绘、直线、贝塞尔标点、批量拉直及半音吸附、原音高还原、共享曲线、响度包络与颤音 | `ProjectModel.cpp`、`PianoRollComponent.cpp`、`NativeSharedEnvelope.h`、`backend/NativeEnvelope.h` |
| 原生 NSF 合成 | 多种时间拉伸、HiFiShifter Mel 线性拉伸、WCSNDM 非循环参数拉伸、可选音高突变平滑、谐波/噪声分离、清音保护 | `backend/NsfHifiganRenderer*`、`backend/WcsNonLoopFeatures*`、`SourceVoicing.h`、`NativeSourceTimeMap.h` |
| 标准 UTAU 与分区 | 普通/界/谋模式、音源和 OTO、辅音速度、先行发声与重叠、传统重采样器/HiFisampler 输出 | `backend/UtauRenderer*`、`VoicebankSettingsComponent*`、`OtoWaveformEditorComponent*` |
| 中文 CVVC | 标准 UTAU 的 `presamp.ini` 发音规则、VC/CV 编排、前后缀和句尾处理、别名回退 | `backend/ChineseCvvcPhonemizer.h`、[使用说明](docs/chinese-cvvc.md) |
| 包络与 FLAG | 响度、OTO 音头/音尾、高级混合分段、逐段直线/贝塞尔、FLAG 曲线与预设 | `AdvancedEnvelopePanel.h`、`backend/AdvancedEnvelope.h`、`MixedEnvelopeIO.h`、`HifisamplerFlags.h` |
| DiffSinger | 音源预检、读音、音素时长、pitch/variance/声学合成、参数曲线、局部重生成与后台复用 | `MainComponentDiffSinger.cpp`、`backend/DiffSinger*`、`juce/engines/diffsinger/` |
| 播放与渲染 | 播放附近任务优先、过期任务取消、NSF 局部块复用、轨道/区域增益、混音与 WAV 导出 | `AudioEngine*`、`backend/RenderService*`、`PlaybackRenderQueue.h`、`NsfRenderChunkCache.h` |
| HAMOOD | 调性与和声方案、生成声部、伴奏拍点/和弦分析及工程缓存 | `Hamood*`、`MainComponentHamood.cpp`、`tools/hamood_audio/` |
| 自动化与诊断 | 无界面 MCP、连接当前窗口的 MCP、工程和选区操作、专项回归入口 | `backend/McpServer*`、`LiveMcpBridge*`、`MainComponentMcp.cpp`、`Main.cpp`、`juce/src/tests/` |

## 相较上次上传的更新 108

- 非 U 裁剪可向内缩短或向外读取源素材；相邻分割点双击可恢复统一拉伸，长音符及包络时钟同步修正。
- 粘连素材共享连续音高和响度包络；默认编辑双击支持多选批量拉直与标准半音吸附，每个音符的主体保持独立。标点模式双击添加点，端点可按普通控制点左右拖动。
- 增加标准 UTAU 中文 CVVC 发音器、混合分段贝塞尔包络和非 U 响度包络、颤音入口；修复空轨切换 UTAU/NSF 时的阻塞。
- 增加 HiFiShifter Mel 线性拉伸、音高突变平滑开关和谐波/噪声分离开关；明确清音恢复源波形，减少辅音被错误变调。
- NSF 局部编辑复用未变化的推理块，取消已过期任务；限制分析线程争抢，未编辑录音继续保留原始 PCM。
- 未覆盖的源音频可生成灰色斜纹“无音高”片段，支持编辑或删除对应播放范围。这个标记不是背景杂音的语义识别结果。
- “气声虚线”独立显示判断到的带音高气声。没有实测 F0 的未编辑区域不画线；人为编辑进入这些区域时画圆点线“……”，与短划虚线区分，并保留清音状态。

## Windows 构建

需要 Visual Studio 2022 C++ 工具链、Windows SDK、Git 和 CMake 3.24 或更新版本。建议将仓库放到不含中文和空格的路径，例如 `C:/hachi-source`。

```powershell
cmake -S juce -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target HachiShifterNext --parallel 4
```

输出位于 `build/HachiShifterNext_artefacts/Release/`。也可运行 `./build-local.ps1 -Reconfigure`，默认输出到 `build-integrated/HachiShifterNext_artefacts/RelWithDebInfo/`。首次配置会获取 JUCE 8.0.8、ONNX Runtime 1.22.0 和 Windows DirectML 依赖；本项目所需的 JUCE 兼容补丁在 CMake 中自动应用。

源码不包含编辑器 EXE、编译缓存、用户音频/工程、音源库或大型分析与合成模型。神经功能需自行安装对应模型；DiffSinger 和 HAMOOD 另需相应 Python 运行环境。保留源码内的发音字典和配套 G2P 数据。参阅 [NSF 模型](docs/nsf-hifigan-model.md)、[输出引擎](docs/utau-output-engine.md)、[DiffSinger](docs/diffsinger.md) 和 [HAMOOD 伴奏分析](docs/hamood-audio.md)。

## 验证与当前边界

更新 143 的开发构建已通过 9 组、310 项检查，日常配布通过 3 组、96 项检查。本次同步另核对源文件哈希、本地 C++/CMake 引用、Python 语法和归档完整性；具体范围见 [源码状态](docs/SOURCE_STATUS.md)。这些是专项验证，不代表所有功能、所有音源或所有模型都经过完整测试。

中文 CVVC 当前针对同一区域内的一音符一音节，不是完整 OpenUtau 插件运行环境。气声/清音判断属于声学证据，不等同于完整音素或语音识别。NSF 局部复用主要减少声码器推理，源特征准备仍按整段进行。其他后端及模型兼容边界见维护指南和对应使用文档。

## 来源与许可

保留仓库原有作者、提交历史及 [LICENSE](LICENSE)。自有源文件保留原有 MIT 声明；启用 libllsm2 的组合构建适用 GPL v3-or-later 条款。JUCE、WORLD、Signalsmith、ciglet、HiFisampler、OpenUtau 规则及其他第三方内容按各自许可使用，完整许可和来源说明位于 `juce/third_party/`、`third_party/` 与相应资源目录。模型、音源和外置引擎的许可需分别核对。
