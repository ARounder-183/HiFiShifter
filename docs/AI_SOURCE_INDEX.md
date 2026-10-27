# HachiShifter 逐文件索引与命令入口

更新 249：juce/CMakeLists.txt 的 PROJECT_VERSION 更新为 0.27，供 JUCE_APPLICATION_VERSION_STRING 与 EXE 版本资源共用；运行功能沿用更新 248。

更新 248：JoinedVibratoBake.h 在草案中将既有音高求值拟合为普通可编辑贝塞尔标点，ProjectModel::bakeNoteVibratoIntoPitch 对颤音组统一规划、一次提交。JoinedVibratoBakeSmoke.h 覆盖短中间音节、弯曲基线、显示/合成请求、连接、撤销保存及实际 PCM 前后对比。

更新 247：JoinedVibrato.h 解析同轨、相接、同音高且共享参考起点/时长的持久颤音组。PianoRollComponent 统一控制音符、控制柄绘制/命中及成员预览，setNotesVibrato / setNotesVibratoRealLine 扩展至整组。NativeVibratoSmoke 增加单套控制柄、单成员编辑、全长结束拖动和旧工程参数统一回归。

更新 246：ProjectModel::setNotesVibrato 在每轨按绝对时间识别选中同音高连续段，复用 vibratoReferenceDurationSeconds / vibratoTimeOffsetSeconds 保存共同时间基准。NativeVibratoSmoke 增加连续段、相位/包络、合成请求、隔离及真实批量对话框回归。

更新 245：PianoRollComponent::pitchAnchorHasIncoming 使用 SharedPitchLines 的同线控制点判定原生上一段；setPitchAnchorShape 统一普通形状与贝塞尔菜单提交。NativeSparsePitchSmoke 增加六种形状、跨音符端点、移动重排、真实间隔隔离、保存撤销及显示/合成输入一致性检查。

更新 244：ProjectModel.cpp::nativeSharedPitchLines 全局连续区间使用 2 ms 正间隙容差；NativePointFirstSmoke 扩展跨微小分界点流唯一性，并在传入实际工程时检查各个极短分界左右是否使用同一条线。未改变音频时间或曲线存储。

更新 243：NativeNoteTiming::planNativeBoundaryStretch 校验相邻双音、计算共同可行位移，复用 planNativeNoteMove 的 sharedBoundary 模式仅移动中间分界。PianoRollComponent 的普通音符边界命中保留双选，预览与 ProjectModel::stretchNativeNoteBoundary 共用同一规划。新增 NativeBoundaryStretchSmoke.h / --smoke-native-boundary-stretch <output>，覆盖真实框选、双向边缘、同源/连接/跨片段、保存撤销、取消、极限和实际 PCM。

更新 242：ProjectModel::flattenNotePitch 统一生成连续同音高段的首尾控制点；NoteData::nativeSparsePitchControls 区分“有意无内部点”和“从原素材推导点”，进入工程序列化及音频缓存。PianoRollComponent::mouseDrag 与 PianoRollPointGroup.h 放开原生邻点时间约束并重排选中索引。新增 NativeSparsePitchSmoke.h / --smoke-native-sparse-pitch <output>，覆盖两段四点、跨点拖动、撤销保存、选区隔离、画笔与实际 PCM。

更新 241：ProjectModel.cpp::nativeSharedPitchLines 负责原生点先于线的连续区间构建、时间排序和重复时刻归一；PianoRollComponent::pitchAnchorsFor 读取同一规范控制点，AudioEngine::writeNativePitchContext 纳入共享线依赖。新增 NativePointFirstSmoke.h / --smoke-native-point-first <output> [工程文件]，覆盖跨录音/交错控制点、旧乱序点、显示/合成请求一致性和缓存失效；可只读验证实际工程。

更新 240：PianoRollComponent 的响度分支接入既有 marquee/selectedNotes；loudnessStartGains 保存整个手势选区的初值，displayedNativeGain 共用相同 dB 增量。ProjectModel::setNoteGains 在锁内验证版本及所有目标，一次提交；setNoteGain 委托此入口。NativeLoudnessSmoke 扩充真实框选、批量音频、选区隔离及撤销保存。

更新 239：MainComponent::nativeLoudnessButton → PianoRollComponent::setNativeLoudnessMode → ProjectModel::setNoteGain。新增 --smoke-native-loudness <output> / NativeLoudnessSmoke.h，取代已移除按钮的 native-clip-stretch 专项；原生时间规划回归仍用 native-gap-stretch/native-note-move。

更新 238：NativeAudioTrim::planNativeNoteTrim 优先识别同一源内无歧义的相邻分字，共用 repartitionNativeTrimNote 更新两字范围并保持合并结构；NativeNoteMovePlan::sharedTrimBoundary 让模型保留连接。NativeAudioTrimSmoke 扩展真实四方向拖动、合并增益、撤销重开与整段 PCM 比对。

更新 237：NativeNoteTiming::planNativeNoteMove 的连接素材代理接收 sourceWindows，区分真实源范围与空白，闭合空隙时保留代理映射双侧锚点。NativeGapStretchSmoke.h / --smoke-native-gap-stretch 覆盖真实拖动、源时钟、PCM 及只读实际工程验证。

更新 236：NativeUnpitchedRegions.h 的 nativeNoteRemovalRanges 统一选择范围减去未选音符保护范围；removeNativeNoteAudio 扩展到普通原生音符并支持 ripple。ProjectModel::removeNotes/removeNotesRippling 共用源切片实现；NativeUnpitchedRegionsSmoke 扩充普通删除与时序删除的实际 PCM/热缓存/时间线检查。

更新 234：ProjectModel::initializeNativeNotePitch 负责原始音高/作者标记/连接检查，通过既有 flattenNotePitch 单次事务初始化当前音符；PianoRollComponent 原生双击复用该入口。NativeNoteTiming.h::planNativeNoteMove 与 0.25/update185 函数一致，保留新版本原生素材/先行声查询辅助函数。TimelineComponent::nativeStretchEdgeAt 和 PianoRollComponent::setNativeStretchMode 统一按钮意图与边缘优先级。新增 --smoke-native-initial-pitch；既有 native-clip-stretch 增加波形端点和下方纯辅音相邻链实操。

更新 233：MainComponent::clipStretchButton/setToolButton 接通 TimelineComponent::setClipStretchMode。时间线在按下时固定本次操作模式，原生拉伸提交 ProjectModel::resizeClip，普通裁剪仍提交 trimClip；波形、音符、淡化和轨道响度包络提供比例预览，预览/提交共用 constrainNativeClipResize。新增 --smoke-native-clip-stretch <output>，覆盖真实按钮/鼠标、音频时钟、实际 WAV 尾部、撤销保存与模式隔离。

更新 232：ProjectData::materialBrowserState 序列化工程素材登记/隐藏记录，materialBrowserSessionId 为仅运行期的文档身份；ProjectModel::setMaterialBrowserState 与 MainComponent 回调接通。AssetManagerComponent::setProjectVoicebanks 统一加载工程浏览状态与自动引用，generation 隔离旧异步搜索/标注。新增 --smoke-project-material-memory <output>；素材右键/注册/搜索专项按工程持久化更新。

更新 231：`AudioEngine::refreshUtauWaveformSnapshot` 汇集有效 DS 历史与当前完成渲染，按 noteId 去重，供现有 PianoRoll 波形和音素区域显示。`--smoke-diffsinger-history-playback <bank> <output>` 扩展验证切换/等待/失败/取消选区/重复选择的持久显示、实参回填、PCM 不叠加以及改词后旧显示失效，并输出截图。

更新 230：`PianoRollComponent::batchLyricTargets` 限定 UTAU/DS 单选填词；`McpServer::resolveUtauRenderSelection` 为 live/headless 共同渲染范围解析入口。ChineseCvvcPhonemizer、UtauRenderer、NsfHifiganRenderer 排除静音上下文的缺失别名警告/家族拒绝。新增 `--smoke-utau-selection-scope <output>`：四种模式的非连续选区、远处重叠、参数/歌词/包络、保存撤销、MCP 空/失效选区，以及 CVVC/HiFisampler 实际 PCM 与告警验证。

更新 229：MainComponentDiffSinger 的 pronunciation 请求补 retake_ids，并保护 timing/pronunciation/pitch 的失效目标；showDiffSingerParameterLane 按当前选区检查实参。MainComponentMcp::ds_query_phonemes 在预测前设置 retake_ids。bridge.execute 在选中乐句内调用 Resolver.prepare；groups 的 check_overlap=False 仅用于允许编辑重叠音符的读音。现有 timing-selection UI 专项增加局部 pitch/实参、跨选区打开参数、发音预览/应用/保存/撤销和真实 MCP 查询。

更新 228：MainComponent::showDiffSingerPhonemeEditor 将现有 visibleIds 写入 timing 请求 retake_ids，复用 bridge.groups 的作用域检查，不删除上下文或更改音符时间。新增 --smoke-diffsinger-timing-selection <project> <output>，复用 DiffSingerUiSmoke 的音素窗口检查，覆盖远处重叠/正常打开和真实重叠警告；juce/tests/diffsinger_selection_overlap.py 增加真实 timing 作用域及结果一致性检查。

更新 227：AudioEngine::setUtauRenderNoteSelection/selectAllRenderedUtauNotes 区分精确选区与历史回放；rebuildLoadedClips 在 DS 历史分支使用 diffSingerHistoryAudio。DiffSingerPlaybackHistory 复用 RenderedClip 已完成 PCM；按完整渲染上下文校验，合并后压缩为每 Clip 一份累积 PCM，避免保存每次框选的整歌副本。新增 --smoke-diffsinger-history-playback <bank> <output>，从 MainComponent 空格键入口验证真实模型。

更新 226：MainComponent 的 volumeParamButton / resized / setEnvelopeLane 接通 DS 响度入口及既有预设；PianoRollComponent::amplitudeEnvelopeFor 的 DS 默认是线性平直，DS 底部包络绘制按渲染增益插值采样曲线。测试：TrackEditingInterfacesUiSmoke 覆盖按钮、鼠标加点拖动删除、保存撤销与预设；DiffSingerConsonantSmoke 补充真实模型单次增益、导出、缓存撤销与直线/曲线/直线切换。

更新 225：`PianoRollComponent::mouseDoubleClick` 将 DS 纳入歌词编辑入口，`ensureDefaultEnvelope` 跳过 DS。`engines/diffsinger/bridge.py::groups(notes, selected)` 按最大结束时间扫描重叠块；请求作用范围外的重叠块分隔上下文，范围内保持单声部错误。`execute` 对 render 使用 audible 范围，对局部生成使用 retake_ids。既有 `vowel-stretch-alignment` 覆盖 DS 双击；`juce/tests/diffsinger_selection_overlap.py` 验证嵌套重叠、上下文、真实模型音频及局部音高。更新需同时部署 EXE 和 bridge.py。

更新 224：将现有 `utauVoicebankName` 控件移至算法选择框右侧，在 UTAU／DS 下显示并隐藏 `trackEditingHeading`，移除音源选择按钮后面的名称布局项。`refreshTrackEditingInterface` 仍从当前轨道绑定读取名称；双击 OTO 入口限定 UTAU，DS 只显示模型名／完整路径。界面专项沿用名称位置、三模式、窄窗口和双击 OTO 检查，并核对 DS 头部模型名称。

更新 223：`pitchAlgorithm` 移至工具栏最左侧，`refreshTrackPitchItems` 提供全部七项并使用现有 Palette::accent 紫色主题色。移除 `trackEditingModeMenu/selectTrackEditingMode/showTrackEditingModeMenu`，标题恢复为只读轨道名。`setTrackEditingType` 接受可选算法／UTAU 子模式，在同一撤销事务内配置类型、兼容的拉伸算法及音源绑定；ComboBox 是唯一 UI 入口。界面专项覆盖跨家族切换、算法选择回显、单次撤销、窄窗口布局及 DS 合成家族。

更新 222：`MainComponent::trackEditingModeMenu/showTrackEditingModeMenu/selectTrackEditingMode` 将标题点击连接到 `ProjectModel::setTrackEditingType`。模型复用 `configureNewTrackType`，只修改类型配置和不兼容音源绑定；正常工程通知刷新音频及界面。异步菜单捕获轨道 ID，取消、相同类型、失效 ID 不写入。已有 `--smoke-track-editing-interfaces-ui` 扩展覆盖三种 UTAU 转 DS、反向转换、歌词／控制点保留、撤销重做和保存重开。

更新 221：新增 `UtauPointCurve.h`，以绝对时间排列连续自动过渡链上的控制点，沿用现有插值器与四点局部窗口缓存。`AudioEngine::projectTrackPitchImpl` 先生成共享曲线，再采样给各音符，保留控制点时间及自适应细化，跳过再次覆盖的名义过渡。缓存策略更新为 point-first v2。`--smoke-point-first-pitch <输出目录> <工程>` 验证显示、宿主投影和音符合成请求在控制点的一致性，覆盖三模式、越界交错点及真实工程。

更新 220：`backend/UtauPitchTimeline.h` 提取原 `CvvcPitchTimeline`，提供共同的不可变音高链构建、相对时间绑定和 CVVC 采样视图。`AudioEngine::makeUtauRequest` 按整轨投影绑定普通／界／谋音符，工作线程不再因外部重采样器而清除读取器，`renderKey` 加入策略版本。`projectTrackPitchImpl` 的 UTAU 基础作者曲线保留内部转折并自适应细化，音乐过渡及显式连接图仍走原入口。新命令 `--smoke-utau-overlap-pitch <输出目录> <重采样器> [Flags=K2]` 覆盖三模式、音高修改／恢复、PIT、HiFisampler F0、实际 PCM、后台导出、选区、跨 Clip、亚 5 ms 转折、贝塞尔与颤音、休止和间隙。

更新 219：`ChineseCvvcPhonemizer.h::CvvcPitchTimeline` 按连续非休止乐谱音符共享不可变宿主投影；音素的负时域／尾域读取实际时刻的音高，音源映射仍用名义 MIDI。`AudioEngine::renderKey` 增加 CVVC 音高策略版本。新入口 `--smoke-cvvc-pitch-project <输出目录> <工程> <重采样器> [Flags=K2]` 检查真实 peng→you→jin 段的原始／两种调音／恢复、数据包整数音分、HiFisampler F0 帧、实际混音时序及缓存恢复，支持 HF2 实际引擎验证；测试生成的音频和报告写到输出目录，原工程只读。

更新 218：`PianoRollComponent::amplitudeEnvelopeFor` 的 UTAU 默认分支改为 5/35 ms 且 `linearToNext=true`；不修改 `envelopePresets` 的显式预设。`UtauEnvelopeDefaultsSmoke.h` 通过实际行内填词与预设提交核对普通／界／谋模式；`CvvcLinearProjectSmoke.h` 扩展为五次直线／曲线交替、同一视图旧元数据失效及 WCSNDM 实际增益检查，`CvvcLegacyEnvelopeSmoke.h` 同时检查交叉区像素。原 `--smoke-cvvc-linear-project` 命令参数兼容保留；旧预设语义回归入口 `--smoke-envelope-shapes <重采样器>`，菜单截图入口 `--smoke-envelope-presets <PNG>`。

更新 217：`PianoRollComponent::ensureDefaultEnvelope` 在 CVVC 时序模式下不物化整音符柔起模板，避免填词把标准音素淡化变成叠乘。`ProjectModel::discardLegacyCvvcDefaultEnvelope` 共用于旧工程加载及手动→CVVC 切换，精确识别旧 5/15 ms 起音、35 ms 收尾、0 dB 平台。`--smoke-cvvc-linear-project <输出目录> <工程> <重采样器>` 用真实填词段检查载入、行内重填、保存重开及实际混音增益／响度包络截图，不写原工程。

更新 216：`MainComponent::openSelectedMaterialEditor` 的 UTAU 路径经 `resolveVoiceSample` → `OtoWaveformEditorComponent::forVoicebankSample` 打开 OTO 源参数；普通模式用精确来源条目，区域模式用与渲染器一致的 `SampleSettings::loadVoicebankOto` 模式参数，按解析后录音和别名匹配。既有保存器写对应 OTO 文件，不建立原生 OTTO。音源名称通过 Main 的鼠标监听复用 `showVoicebankSettings`。`UtauSourceEditorSmoke.h` 随 `track-editing-interfaces-ui` 检查模式、前后缀解析、直接参数页和保存目的地。

更新 215：`PianoRollComponent::mouseDoubleClick` 先复制音符命中，依据所属轨道 `effectiveTrackType == utau` 统一进入 `beginInlineAliasEdit`，不再判断歌词内容。UTAU 音符本体命中优先于 CVVC 音素标签选择，其余工具及原生对齐按原路径处理。`vowel-stretch-alignment` 覆盖三种 UTAU 模式、两种算法、是否绑定音源、七种歌词内容及多选情况下只改目标音符。

更新 214：`PianoRollComponent::mouseDoubleClick` 在音符工具命中后，根据 `effectiveTrackType == utau` 且空歌词分流到现有 `beginInlineAliasEdit`，跳过先行声音网格对齐。在选择回调前复制命中数据，避免回调刷新布局使迭代器失效。提交沿用 `commitInlineAlias`；`vowel-stretch-alignment` 覆盖真实双击序列、取消／提交、三种 UTAU 模式和原生空歌词不打开输入框。

更新 213：`MainComponent::utauVoicebankName` 在轨道界面刷新时读取 `TrackData::voicebankDirectory`；与音源选择按钮一起布局、仅 UTAU 界面显示，名称宽度上限 260，完整路径放在 tooltip。既有 `track-editing-interfaces-ui` 检查名称和按钮位置并输出绑定后的截图。

更新 212：`PanValueDialog.h` 提供声像输入草稿和有限数值／范围校验，未编辑直接应用保留原始浮点精度。`TrackListComponent::onTrackPanRequested` → `MainComponent::showTrackPanDialog` → 现有 `ProjectModel::setTrackPan`；不新增模型写入口。`--smoke-decibel-input` 同时覆盖声像点击、范围、独立目标、滑杆、撤销及保存，输出 `pan-dialog.png`。

更新 211：`DecibelGainDialog.h` 统一 dB 编辑草稿与校验。轨道数字经 `TrackListComponent::onTrackGainRequested` → `showTrackGainDialog` → `setTrackVolume`；区域数字短点击复用 `onClipGainRequested/showClipGainDialog/setClipGain`，拖动达到阈值后不打开输入框。区域增益 setter 改为精确相等才跳过，以保留极低音量的小数调整；显示只对真正的零标注负无穷。`--smoke-decibel-input` 验证实际点击命中、输入校验、精度、独立目标、撤销和保存。

更新 210：`TrackListComponent` 名称栏的 mouseDown/Drag/Up 和滚动定时器管理排序预览，`ProjectModel::moveTrackBefore(trackId,beforeTrackId)` 是唯一提交入口，空目标 ID 表示末尾。Main 切换轨道时清除旧区域的宿主选择，防止排序通知将焦点拉回旧轨道。`--smoke-track-reorder` 覆盖真实拖动、时间线顺序、取消、边缘滚动、控件手势隔离、内容保持、撤销重做和保存重开。

更新 209：`ProjectModel::setTrackPianoPreview` 是每轨钢琴试听开关的统一写入口。`MainComponent::pianoPreviewButton` 与发音器成组排列，界／谋模式独立显示；`AudioEngine` 将轨道值传入 `UtauRenderRequest::pianoPreviewEnabled`，传统和 NSF 渲染器仅在回退处分支。`piano-transport` 增加 UI 位置、关闭／重开、跨轨道、保存重开与撤销检查；`empty-lyric-piano` 增加实际 PCM 静音和正常歌声保持检查。

更新 208：`MainComponent::togglePlayback` 无视觉选区时传入同一工程快照，`AudioEngine::selectAllRenderedUtauNotes(project)` 合并历史与普通 UTAU 空歌词，避免首次播放被历史筛选清空。`tests/PianoTransportSmoke.h` / `--smoke-piano-transport <目录> [工程文件]` 经真实按钮、等待定时器和 PCM 混音验证；可选工程仅在内存副本取消旧独奏，哈希确认原文件未修改。

更新 207：`AssetManagerComponent::contextMenuOptions` 统一右键屏幕锚点，`removeFromBrowser` 是素材文件夹／录音移除入口；偏好 `assets.hiddenFolders`、`assets.hiddenAudio` 只控制浏览器视图。`tests/AssetContextSmoke.h`／`--smoke-asset-context` 覆盖坐标、缩放、搜索、同步、重开、恢复和文件保持；`--inspect-piano-project <工程> <输出目录>` 为此次实际工程读取与混音诊断，原工程不写入。

更新 206：`tests/EmptyLyricPianoSmoke.h`／`--smoke-empty-lyric-piano <输出目录>` 检查 UTAU 三模式及 HiFisampler 空歌词、错误歌词和休止的 PCM，以及同一 `AudioEngine` 预览／导出／恢复预览。`UtauRenderRequest::silenceEmptyLyrics` 由导出准备参数设置，`AudioEngine` 空歌词缓存按用途隔离；直接导出未准备的钢琴预览被拒绝，GUI、MCP 和 CLI 导出入口共用正确准备路径。

更新 205：删除 `MainComponent::bindDefaultUtauVoicebank` 及模式／偏好自动调用，保留 `chooseUtauVoicebank → selectVoicebank` 显式路径。`track-editing-interfaces-ui` 在真实有效旧默认目录存在时检查创建／应用设置／三模式切换／保存重开仍为空；旧选择及 NSF 测试改为显式绑定。钢琴行为入口 `backend/UtauRenderer.cpp::renderPianoPreview/render`，专项 `missing-alias-piano`。

更新 204：`MainComponent::utauVoicebankButton` 恢复连接 `chooseUtauVoicebank → selectVoicebank`；`resized` 将选择按钮和 `spliceButton` 同组放置，仅 UTAU 界面显示。`refreshTrackEditingInterface` 刷新翻译及路径提示。既有 `track-editing-interfaces-ui` 增加三种 UTAU 模式／三种宽度、路径绑定／撤销／保存检查和截图。

更新 203：音轨头 `showOutputEngineMenu` 增加 ID 108，经 `importMidiTrack → importMidiTrackFrom → addMidiTrackFrom` 传递捕获的目标 ID／时间。`ProjectModel::addMidiTrack` 新增可选目标及起点，共用原解析／歌词／音符转换；锁内重验目标，一次撤销追加片段。`midi-track-import` 覆盖三种轨道、速度变化、目标失效、保存和实际选择对话框。

更新 202：`TimelineComponent::mouseDown` 的 `onClipMenu` 时间参数改为 `playheadSeconds`；`MainComponent` 继续将该时间用于 `canSplitClip` 和异步回调的 `splitClip`。`clip-split` 的既有鼠标回归改为红线 1.75 秒、鼠标 2 秒，核对实际切点及拆分前后 PCM。

更新 201：`MainComponent::clipContextMenu/clipContextMenuItemChosen` 的 ID 8 读取当前片段 `muted`，调用现有 `setClipMuted` 切换；I18n 新增 `clip.muteToggle`，不改整轨静音或渲染逻辑。

更新 200：`MainComponent.cpp` 内部 `ClipEditDialog` 扩充现有区域长度窗口，`applyClipEditDialog` 先验证两个草稿，再调用 `applyClipLengthText`／`setTrackOutputEngine`。`UtauOutputEnginePanel` 复用为工具路径草稿编辑器，外部菜单路径不变。`clip-length` 验证取消、禁用轨道类型、保存重开及两个入口互相读取状态。

更新 199：`WaveformView::handleNear/mouseDown/paint` 以 `preferOuterRangeDrag` 为原生单音节源范围提供外边界命中和绘制优先级；`loadMaterialDetail` 按模式设置锁定。保存复用 `syncMaterialTiming → commitMaterialDetail → MaterialAnnotationSession::save`。`unified-material-editor` 新增真实双端鼠标拖动、保存重开和多音节锁定验证。

更新 198：`I18n.cpp` 登记单音节控件及提示文字翻译，`refreshMaterialVisibility` 以 `materialMode == 1` 统一控制总览和详情的原生专用控件可见性。回归检查中文控件及原生／三种 UTAU 视图切换。

更新 197：`OtoWaveformEditorComponent::bindMaterialViewPreferences/setSingleSyllableMaterial` 统一处理录音级单音节视图偏好、校验和切换。素材管理器及快速编辑入口绑定现有 PropertiesFile；普通音符右键定位通过原 `selectMaterialPronunciation/focusMaterialSourceRange` 尊重此开关。偏好键 `material.singleSyllable.<SHA256(规范化绝对路径)>`，附 `.region` 记忆所选稳定 ID；ID 仅在当前已解析标注版本内匹配。`unified-material-editor` 验证真实控件、冷热打开、独立录音、原始草稿和外边界；`main-material-variants` 验证实际右键窗口复用。

更新 196：`AssetManagerComponent::openMemberEditor` 对携带音符源范围的导航选择总览；`focusMaterialSourceRange` 支持详情／总览目的视图，`selectMaterialPronunciation(..., false)` 同时退出旧详情。`--smoke-main-material-variants` 覆盖真实右键、精确身份、源时间匹配、详情窗口复用返回总览和不自动发布。

更新 195：`MainComponent::openSelectedMaterialEditor` 通过 `nativeNoteSourceTiming/nativeClipClock` 将右键音符换算为源范围；`openMaterialEditor`、`AssetManagerComponent::editMaterial/openMemberEditor` 传递可选 `sourceRange`。已有 regionId 优先；无 ID 时 `OtoWaveformEditorComponent::focusMaterialSourceRange` 在当前会话内解析唯一发音并复用 `selectMaterialPronunciation`，无唯一结果则 `MaterialAnnotationEditor::selectSourceRange` 高亮原音频选区。窗口复用走同一路径。`--smoke-main-material-variants` 覆盖显式身份、未标注选区、拉伸合并片段、同名发音切换、详情框显及纯导航不写文件。

更新 194：原生音符右键菜单 ID 34“编辑素材标注…”经 `onOpenRegionEditor` → `MainComponent::showRegionEditorForNote` → `openSelectedMaterialEditor(false, noteId)` 打开现有素材编辑器；按点击音符 ID 解析实际源文件、合并片段及标注版本，复用既有保存和发布校验。`--smoke-main-material-variants` 覆盖真实菜单分派、多标注版本和未登记录音，不依赖宿主旧选中项。

更新 193：`MainComponent::addAnalysedAudioFile` 导入成功后显示素材面板；`AssetManagerComponent::setProjectVoicebanks` 统一调用 `updateProjectVoicebanks/updateProjectAudio`，后者从 clip/sourceFile 和嵌套 parts 派生临时工程文件视图，复用现有源标注编辑器。`MaterialFolder::projectAudio` 保证刷新时不递归扫描未登记目录，也不写全局素材偏好。`--smoke-main-material-variants` 增加真实 filesDropped、源列表去重／过滤／搜索、删除撤销、工程重开、原生标注保存和 PCM 保持验证。

更新 192：`MainComponent::importMaterialPronunciations` 在未指定 regionIds（文件行）时只使用 rows.front()，与素材列表顺序一致；明确指定 IDs 的子发音／多选不变。`AssetManagerComponent::audioDragDescription` 保留音频版本身份，未展开 OTO 在导入读取后决策；不依赖 UI 缓存完成时间。回归 `tests/MainMaterialVariantHandoffSmoke.h` / `--smoke-main-material-variants`。

## 区域数字长度（更新 191，修正 190）

`MainComponent::clipContextMenu` 的“编辑长度…”通过 `makeClipLengthDialog` / `applyClipLengthText` 接入 `ProjectModel::setClipEditableLength`；只改外框末端，不缩放、裁剪或删除音符，保留音频实际时间窗口，修改已有工程格式中的 durationSeconds / audioDurationSeconds。整体拉伸仍归 `resizeClip`、裁剪仍归 `trimClip`，菜单不调用这两者。测试 `juce/src/tests/ClipLengthSmoke.h`，命令 `--smoke-clip-length <输出目录>`。

## 显示菜单能力修复（2026-10-23）

旧配布 r 源码先行：`MainComponent::viewMenuItemEnabled(int, TrackType)` 为显示菜单和调用守卫的同一能力策略，`selectedEditorTrackType` 从模型选择轨道读取类型，不读算法下拉框。Hachi 灰禁拼字信息，UTAU／DiffSinger 使用实际音素显示；原生实际波形与辅音虚线仅 Hachi 开启。`Theme.cpp` 的弹出菜单禁用文字保持 opaque muted 灰色。`tests/EngineViewCapabilitiesSmoke.h` 由 MainComponent.cpp 单次包含，命令 `--smoke-engine-view-capabilities <输出目录>` 记录三轨菜单、守卫、偏好及深浅主题 PNG；测试结果由实际运行报告给出，尚不以入口存在声称通过。

## 开发包修复入口（2026-10-22）

当前修改基线为 `HachiShifter-0.2.6-dev/source`。别名发布由 `ProjectModel::applySourceSettingsWithReport` 同步显示元数据（局部调音快照仍保留）；`PianoRollComponent` 不用继承的“-”区段文字遮挡已命名发音。`AssetManagerComponent::refreshMaterialAnnotations` 供快速／精确编辑保存复用，`OtoWaveformEditorComponent::materialSourceSaved` 刷新共用发音外层。此修复按用户要求仅编译、不执行测试或渲染。Melodyne 导入将真实同录音连接组保存为 `ClipData::materialSourceUsageGroupId`，由 `ProjectMaterialRegistrationFacts::derive` 生成实际多行变体，`prepare` 合并相同完整标注集；旧工程缺失端点时不猜分组。

## 当前整合导航（2026-10-22，开发中，尚未整体验收）

公开版本仍为 **0.25**，WCSNDM 仍为 **0.0803**。下列是当前源码入口，不是发布记录。先完成所有非自动注音功能，再停下等待用户验收；日语 ASR／TIFA 自动注音不在本轮实施范围。历史更新段落保留原貌。

| 功能 | 当前权威入口与边界 |
| --- | --- |
| 三种轨道 | [ProjectModel.h](../juce/src/ProjectModel.h) 的 `TrackType`、`effectiveTrackType`、`addTypedTrack`；Hachi／UTAU／DiffSinger 与渲染器、伴奏标记分离。`addUstFile` 默认目的类型为 Hachi。[MainComponent.cpp](../juce/src/MainComponent.cpp) 的 `refreshTrackEditingInterface` 负责现有界面的类型分流，不另建工程模型。 |
| USTX 格式边界 | [UstxImporter.h](../juce/src/backend/UstxImporter.h) 的严格 YAML 读取进入既有 `UstProject` 公共映射。0.8 的 480 ticks、声部绝对位置、多轨道及不支持字段警告不等于完整 OpenUTAU 行为兼容。 |
| 音源家族保护 | [UtauRenderer.h](../juce/src/backend/UtauRenderer.h) 的 `RequestedVoicebankFamily`／`resolveRequestedVoicebankFamily`；请求明确声明 UTAU 或 DiffSinger，错误家族拒绝而不静默换引擎。真实 DiffSinger 神经模型正向验证仍缺资源。 |
| 同录音多套注音 | [OttoMaterialIO.h](../juce/src/OttoMaterialIO.h) 的 `listAudioVariants` 及带 `audioId` 的加载／保存／替换；同一路径可对应多个稳定音频 ID，物理文件不复制成多份。未指定 ID 且路径歧义时拒绝。 |
| 社区数据集 | [CommunityDatasetIO.h](../juce/src/CommunityDatasetIO.h)：openvpi `transcriptions.csv` 的读取、显式字典／分组、规划与原子发布；不是自动对齐器。[CommunityDatasetImportDialog.h](../juce/src/CommunityDatasetImportDialog.h) 是调用同一服务的导入选择向导；当前已进入宿主模拟 GUI 测试但尚未报告通过；生产 Main 整合未验收。 |
| 工程素材登记 | [ProjectMaterialRegistration.h](../juce/src/ProjectMaterialRegistration.h) 的 `prepare`／`makeCandidate`／`stage` 与租约发布；[ProjectMaterialRegistrationFacts.h](../juce/src/ProjectMaterialRegistrationFacts.h) 默认拒绝未确认事实，仅接受用户明确确认的现有摆放解释。模型由 `adoptRegisteredMaterialBindings` 一次撤销地收纳引用，不重分析、不重排原时钟。 |
| 当前音符局部参数 | [MaterialAnnotationSession.h](../juce/src/MaterialAnnotationSession.h) 的 `EditScope`／`NoteScopeBinding` 与 [ProjectModel.h](../juce/src/ProjectModel.h) 的 `getNoteMaterialLocalBinding`／`applyNoteMaterialLocalRow`／`restoreNoteMaterialLocalRow`。源发布和项目局部应用分离；完整两个 GUI 入口仍待验收。 |
| 重叠线与选择器 | [PianoRollComponent.h](../juce/src/PianoRollComponent.h) 的 `setShowOverlap` 默认关闭；现有 View 菜单 613／`ui.showOverlap` 只控制音符内黄色交叠线，不改工程。素材选择器仍只能右键显式打开一个，与此显示开关独立。 |

新增命令以 [Main.cpp](../juce/src/Main.cpp) 实际分支为准：

- 各取一个输出目录：`--smoke-track-types`、`--smoke-track-editing-interfaces-ui`、`--smoke-voicebank-family`、`--smoke-ustx-importer`、`--smoke-material-variants`、`--smoke-main-material-variants`、`--smoke-project-material-registration`、`--smoke-project-material-registration-model`、`--smoke-registration-facts`、`--smoke-note-material-local-override`、`--smoke-local-material-scope`、`--smoke-connection-pitch-ui`、`--smoke-overlap-selector`。
- `--smoke-community-dataset <输出目录> [NSF模型目录]`；模型缺省时不能据此声称神经音频通过。
- `--smoke-community-import-gui <输出目录> <NSF模型目录>` 是新向导测试分支，登记了命令不等于已经编译或通过。
- `--audit-project-material-registration <工程> <输出目录>` 只审计携带事实，不把每个音符／分析帧认作词或音节。
- `--smoke-registered-project-audio <输出目录> <工程> <NSF模型目录>` 用于登记前／收纳后／冷重开实际音频门槛；28b 真实 teio 的 23 项检查通过，31 个物理录音／81 套变体，三阶段实际 CPU ONNX 无警告／回退且解码 PCM 完全一致，见 [实测输出](<G:/hjs-tc/tmp/teio-reg28/stdout.txt>)；这不是 Main 登记 GUI 的验收。

当前结果及限制见 [实现状态](native-material-implementation.md)。接口清单／指纹待可变头文件稳定后统一再生成，本次未运行接口生成器。

## 原生素材基础导航（早期整合记录）

- `juce/src/OttoMaterialIO.{h,cpp}`: compact folder-level codec, stable IDs, strict validation and atomic conflict-protected writes.
- `juce/src/OttoMaterialMigration.{h,cpp}`: explicit all-or-nothing legacy migration and lossy path-aware standard OTO export.
- `juce/src/MaterialAnnotationSession.{h,cpp}`: shared non-destructive multi-pronunciation draft/history/publication service.
- `OtoWaveformEditorComponent::forMaterial`: the single precise editor extends the existing UTAU component; native overview/default and double-click detail share one draft. `MaterialAnnotationEditor` is only its internal overview child; there is no parallel compatibility editor action.
- [NativeExpression.h](../juce/src/backend/NativeExpression.h): one shared native gain/tension/breath DSP used by native-file and real NSF voicebank rendering. Native semitone formants are applied to final Mel before ONNX inference, not guessed FLAG units.
- [PronunciationTiming](../juce/src/PronunciationTiming.h): one P/C interpretation for all annotation views, default one C/one elastic V; `vowelSplitsExplicit` allows only explicitly named manual subdivisions. Source analysis epochs never imply authoring regions. Simplified consonant mode stores only its boolean intent; C-P is derived.
- `MaterialAnnotationSession::splitVowel/editVowel/mergeVowels`: atomic named authoring planners shared with the detailed editor. Draft normalization does not publish or rewrite the recording.
- `ProjectModel::alignNotePreutteranceToGrid`: default note-tool double-click translates the note onto the nearest feasible tempo grid without flattening pitch or changing selected duration/automation. The existing timing planner preserves C/pre-P and allocates stretch only to elastic V.
- `ProjectModel::setNotesPortamento`: explicit pitch-only graph edges; no source join, amplitude splice or note-control duplication. Graph-aware `sharedPitchLines(track, connections)` and `connectionPitchDigest` are shared editor/render contracts; protocol-note display also uses the canonical `AudioEngine::projectTrackNotePitch`. Compiled graph-handle, mixed-batch, strict global-seam and real forced-pitch audio fixtures have passed; this does not replace all-backend or final human acceptance.
- `PianoRollComponent::PitchScope`: focused versus whole-current-track editing. View item 612/preferences choose scope without musical undo or audio changes. Track strokes use the common `composeNativePitchStroke` and atomic `setPointCurves` pipeline.
- New acceptance commands: `--smoke-pronunciation-timing`, `--smoke-named-vowel`, `--smoke-adjacent-pronunciation`, `--smoke-vowel-stretch-alignment`, `--smoke-material-source-replacement`, `--smoke-forced-portamento`, `--smoke-track-pitch-scope` each take output; `--smoke-vowel-timing-audio <output> <nsfModel>` requires actual ONNX inference. Source/test presence alone is not a pass.
- `ProjectModel::applySourceSettingsWithReport(..., apply=false)`: publication preflight; default true applies one identity-based project transaction.
- Pure/model checks: `--smoke-otto-material`, `--smoke-material-annotation`, `--smoke-material-binding`, `--smoke-material-insertion`, `--smoke-native-connection-graph`, `--smoke-material-topology`, `--smoke-native-nsf-compatibility`, `--smoke-midi-export-scope` (each takes an output directory).
- Compiled UI captures: `--smoke-unified-material-editor`, `--smoke-material-asset-tree`, `--smoke-unified-draw-action`, `--smoke-single-entry-ui`. Capture existence is not visual acceptance: open and inspect the images.
- Real offline PCM fixture: `--smoke-material-topology-audio <output>` (two placements, source splits, save/reopen and exact stereo PCM). `AudioEngine(false)` does not open a device; normal GUI construction remains unchanged.
- Combined workflow: `--smoke-native-workflow-e2e <output> <nsfModelDirectory>`; requires isolated `HACHI_TEST_SETTINGS_DIR`, real GAME+FCPE resources and NSF model, and fails on analysis/fallback/automation/audio mismatch. This generated fixture does not replace the previous real-project references.
- Native controls: `--smoke-nsf-native-expression <output> [modelDirectory] [recording]`; without a model only DSP/mapping is tested, not real ONNX.
- `--render-project <project> <wav> [voicebank] [resampler] [nsfModel] [soloIndex] [splice|process|process-merge]` uses the same engine offline. Optional diagnostic ablations are explicitly labelled in logs and are not acceptance at unchanged settings. The old-project noise-protection default and decoder UV carrier are covered by `--smoke-native-nsf-compatibility`; new projects keep the explicit new feature defaults.
- Explicit CLI actions: `--migrate-otto-folder <folder> [classic|jie|mou] [--include-bare]`; `--export-otto-folder <folder> <output.ini> [--replace]`.
- Read [implementation status](native-material-implementation.md) for verified checkpoints and unfinished requirements. New annotation writes no longer use HJM CSV; historical entries below are retained for provenance, not current write policy.

更新 185：素材管理器增加声音文件名搜索框、清空按钮和无结果提示。搜索跨全部已注册及工程音源文件夹，包含未展开的子目录；文件名包含匹配忽略英文大小写，支持中文。结果平铺显示“文件名（音源／素材文件夹名字）”，同名录音保留各自来源；清空搜索恢复原展开状态。`AssetManagerComponent::ensureSearchFiles` 在单工作线程枚举尚未加载的文件列表，后续输入只筛选缓存，不解码音频或建立 OTO 索引。结果回主线程按仍存在的文件夹归并，使用最新搜索词，项目移除的音源不会重新出现。点击、Enter、右键、双击和拖拽继续使用原文件与所属音源 OTO 设置。新增 `--smoke-asset-search`，回归树视图、OTO 裁剪拆分、普通音频加载和大型音源注册；结果与界面截图见 `test-output/update185/`。

更新 184：工程文件记录素材管理器打开／关闭状态。`ProjectData::assetManagerVisible` 随工程保存及恢复快照写入，旧文件缺省为关闭。`MainComponent::setAssetManagerVisible` 统一应用界面状态，加载工程、新建与替换工程同步恢复；用户切换通过 `ProjectModel::setAssetManagerVisible` 标记工程待保存，不新增音乐撤销项或触发音频重同步。音乐撤销／重做与普通 replace 保持当前面板状态。新增保存开／关、不同工程切换、全新模型读取、旧工程缺省、恢复快照与撤销重做验证，构建及安装版工程安全专项通过。报告 `test-output/update184/verification.json`。

更新 183：OTO 别名选择菜单在录音拖放位置附近弹出。`itemDropped` 将落点由主界面坐标转为屏幕坐标，传给 `importAssetAudio`；菜单先指定主窗口以保持 DPI 缩放，再使用 withTargetScreenArea 指定该落点，不再锚定素材管理器整块面板的下沿。位置在读取音频与 OTO 前捕获，鼠标随后移动不会改变落点。右键／Enter 加载按调用时鼠标位置显示。导入时间、目标轨道、别名内容与取消行为不变。构建与安装版 OTO 专项共 66 项检查通过；专项验证导入流程，未模拟实体鼠标菜单操作。报告 `test-output/update183/verification.json`。

更新 182：修复素材拖拽时整块素材管理器跟随鼠标显示。`AssetManagerComponent::mouseDrag` 原先未指定拖拽图，JUCE 因而截图整个源组件；现显式提供 220×26 逻辑像素的录音文件名标签，使用 2 倍像素图保持高 DPI 清晰。拖拽源、文件路径、所属音源与 OTO 类型数据及导入流程保持不变。构建与安装版 `--smoke-oto-audio` 共 66 项检查通过，覆盖三种 OTO 拖入、裁剪拆分及导出音频；该专项验证导入逻辑，不模拟实体鼠标拖拽。报告 `test-output/update182/verification.json`。

更新 181：OTO 类型选择移入每个素材音源文件夹行的右下角，使用 100×18 小下拉菜单，保留 40 像素文件夹行高，移除顶部全局选择行。每个音源独立选择普通 OTO、界 OTO、谋 OTO 或不读取；按音源绝对路径记忆，重新排序、切换工程与重启均保留。已有安装首次沿用原全局选择，之后各音源互不影响。

主入口 `AssetManagerComponent::otoImportKind(fileOrFolder)` / `setOtoImportKind(folder, kind)`；路径归一后以 SHA256 作为 PropertiesFile 键。内部拖拽、右键加载与 Enter 均按源录音所属音源读取类型，沿用更新 180 的统一解析、裁剪和拆分入口。树视图专项新增多音源独立设置、非当前音源拖拽、重排与磁盘重开验证，并检查 320/220 像素面板截图；回归 OTO 实际音频输出、普通素材加载与真实大型音源导入。报告 `test-output/update181/`。

更新 180：素材管理器新增统一且持久记忆的“OTO 类型”下拉框：普通 OTO（二区）、界 OTO（四区）、谋 OTO（二至四区）以及不读取（完整录音），首次默认普通 OTO。素材内部拖入音轨、右键加载和 Enter 共用 `MainComponent::importAssetAudio`。按选定格式读取录音所属的最近 OTO，普通用 oto.ini，界优先独立 oto.jie.ini 并合并 oto4.ini，谋用 oto.ini 与 otomou.ini；不跨格式静默替代。一个录音有多个条目时列出别名供选择，取消不改工程。

`SampleSettings::loadAudioOto` 只读该录音的祖先文件夹中相关 OTO，不扫描整库；`nativeAudioFromOto` 将 offset／正负 cutoff 转成真实源范围，普通固定区按 consonant 拆分，界按三边界拆成四区，谋按 C/V/S 与有效边界数拆分；零长度区不生成音符。`ProjectModel::addAudioFile` 接收可选 OTO，直接生成原生 ClipData 与独立 NoteData，复用既有插入、一次撤销、异步音高分析和保存，不转换或覆盖 HJM。分析只补充源音高，保留作者边界。不读取模式保留更新 179 的普通音频/HJM 行为。内部拖拽携带文件、根目录与所选类型；外部拖拽沿用原文件/副本策略。同时修复谋 OTO 二区简写 CV,offset,b1 被四字段门槛误跳过的问题。

专项 `--smoke-oto-audio <输出目录>` 验证三种格式、二／三／四区、正负截尾、无效范围、单步撤销／重做、真实拖入处理、选择持久化、音高分析后边界、工程重开及导出 PCM。回归普通素材加载、文件夹树和 3219 文件音源注册，报告 `test-output/update180/`。公开版本保持 0.25。

更新 179：素材管理器在非 U 模式也可导入、展开 UTAU 文件夹。右键录音选择“加载为音频素材”，或选中后按 Enter，按播放头位置导入并聚焦素材。选中原生调音轨道时复用该轨道和算法；选中 UTAU、DiffSinger、伴奏等轨道时新建普通音频轨道。文件夹导入后自动展开。浏览只枚举文件，加载复用普通音频分析，不应用 OTO 别名／拆音或修改录音与 OTO。分析后沿用普通音频流程生成 HJM 注释，已有 HJM 侧文件沿用原导入规则，双击参数编辑与拖动保持。

主入口：`AssetManagerComponent::loadAudio` 经 `onLoadAudio` 委托 `MainComponent::addAnalysedAudioFile` → `ProjectModel::addAudioFile`／`scheduleAnalysis`；右键与 Enter 共用，异步菜单捕获原文件并用 SafePointer 检查组件生命周期。新增 `--smoke-asset-audio <输出目录> <真实录音>`，验证原生轨道、UTAU 选区分流、异步音高分析、保存重开和源文件不变；复用音源导入性能与树视图回归。

更新 178：HN-SEP 完整谐波波形磁盘缓存由内置 HiFiSampler、非 U 噪声保护与 WCSNDM HN1 共用。实际输入 PCM、模型和配置内容 SHA256 加分离算法版本／声学网格决定身份；实际 PCM 相同才互用，保留两边原有预处理。内容校验、跨进程互斥、原子发布、512 MiB 磁盘预算及模型字节快照会话详见 [HN 模式说明](wcsndm-hn.md)。入口为 `backend/HnsepDiskCache.inc` 与外置 `hf_backend/hn_cache.py`，既有 `hifiSeparate`／`hn_mel.separate` 接入。测试入口 `--smoke-hn-cache`、外置 `src/test-hn-cache.py`／`test-hn-cache-render.py`；报告 `test-output/update178/`。

更新 177：音高、响度与线性 FLAG 标点支持 Ctrl 加选／取消，可跨音符及 CVVC 音素。拖动任一已选点，以相同秒数和显示值位移整组平移；共享边界约束防止选中点越过未选中邻点，首尾保护沿用既有规则。Ctrl+R／多选右键菜单初始化所选点，保留时间：音高回音符音高、响度到 100%、FLAG 回参数默认值，DiffSinger 实参回模型参考值。Delete／Backspace 批量删除，连按不误删整音符；松开鼠标保留选择，普通点击新点替换整组。原生共享包络去重并同步预览。高级包络混合模式节点支持相同多选、平移、默认值与删除操作，仍为草稿，应用后保存。

主入口：`PianoRollPointGroup.h` 复用现有命中检测与参数读取，`PointGroupEdit.h` 计算整组位移边界；`ProjectModel::setPointCurves` 使用已有各参数提交函数并合并为一次撤销，不复制新的工程格式。高级包络使用 `AdvancedEnvelopePanel::editKnots`。回归：`--smoke-point-group`、`--smoke-envelope-point-selection`、`--smoke-pitch-point-selection`、`--smoke-advanced-envelope-panel`；报告 `test-output/update177/{build,installed}/verification.json`。

更新 176：高级包络混合模式的节点在点击／拖动并松开后保持选中高亮，Delete／Backspace 调用既有 deleteKnot 删除该节点；音头、音尾、原生音频及 CVVC 音素共用。选择与拖动状态分离，点击其他位置／面板控件、失焦、切换形状／预览、替换草稿时清除选择。连按 Delete 不继续删除其他节点，首尾边界点和贝塞尔控制柄沿用原有规则。删除只改草稿，应用时走原有提交入口，关闭不应用则不改工程。 入口 `AdvancedEnvelopePanel::Graph`，回归 `--smoke-advanced-envelope-panel <目录>`；报告 `test-output/update176/{build,installed}/verification.json`。

更新 175：响度包络和线性 FLAG 包络沿用音高标点的持久选择；点击或拖动后松开仍高亮，Delete／Backspace 删除所选标点，继续按键不会转而删除整音符。统一 PointSelection 记录类型、音符、点序号、CVVC 分片及 FLAG 键；点击其他位置、切换工具／FLAG／选区、失焦时清除。右键删除与快捷键共用 deleteEnvelopePoint，响度沿用已有选区／粘连组映射，FLAG 只改当前参数，CVVC 只改对应音素的对应包络；保留原来的响度／CVVC 两端与 FLAG 最后一个点。普通响度及 FLAG 纯点击不再提交工程编辑。 回归入口：`--smoke-envelope-point-selection <输出目录>`，复测 `--smoke-pitch-point-selection`；报告 `test-output/update175/{build,installed}/verification.json`。

更新 174：音高标点的选中状态独立于拖动状态；点击并松开或拖动后松开，标点继续高亮，Delete／Backspace 优先删除该点。点击其他位置、切换工具／轨道／片段、失去键盘焦点或重新选择音符时清除。删除使用与标点右键菜单相同的 deletePitchAnchor 入口，沿用端点保留和 DiffSinger 隐藏采样整理规则；通过既有 setNotePitchCurve 提交，支持撤销、重做及工程保存。删除完成或版本过期后保留按键消费状态直到下一次选择，防止连续 Delete 误删整音符。选择本身不改工程、不触发合成。 回归入口：`--smoke-pitch-point-selection <输出目录>`；测试文件 `juce/src/tests/PitchPointSelectionSmoke.h`，报告 `test-output/update174/{build,installed}/verification.json`。

更新 173：“自动交叉淡化”按钮增加右键直线／曲线菜单，默认直线，偏好 ui.autoCrossfadeCurve 记住选择。菜单只改后续操作方式，左键仍对同轨两个音符的已有发声重叠执行，不移动音符。PianoRollComponent::crossfadeSelectedNotes 将方式传给唯一提交入口 ProjectModel::crossfadeNotes；曲线复用 CVVC UtauMixRamp 的互补 smoothstep，以 32 段幅度直线保存到既有包络点格式，普通音符及 CVVC 分片共用，继续支持保存、撤销与重做。切回直线清除重叠区内的曲线中间点；无选区时仍可右键设置。

更新 172 补充核验：HN1 八类谐波参数在普通 UTAU 线性 FLAG 及带标注的二／三／四段分区中通过 145 次真实模型合成、660 项检查；曲线／分区继承、优先级、零长度段和短音已验证。普通 UTAU 的经典二区不暴露独立分区 FLAG。详见 [HN 模式说明](wcsndm-hn.md)。

更新 172：HN1 接入 g、Mo、QX/ME、Mr、Mq、Mt、Rd、NA 八类谐波参数及逐帧 FLAG 曲线。C 端复用现有声道编辑和 LF 模型生成相对中性谱的 dB 变化，Python 仅将功率增益作用于 HN-SEP 谐波；不重新合成 LLSM 谐波。源时间映射携带同帧增益，内部服务协议升级 HN2，公开开关仍只有 HN1。零参数保持上一版谐波功率完全一致；Mq 与普通路径共用实现。见 [wcsndm-hn.md](wcsndm-hn.md)。

更新 171：仅 HN1 选择 HN-SEP 谐波＋LLSM 非谐波神经内核，与 HF2、V1、V2、K/L/M 并列互斥；继续使用第一个有效内核优先。裸 HN、HN0、HN2 等不启用、不占内核位置，也不拦截后面的 HN1。编辑器分离导出资格识别同步采用相同规则；合成算法未改。见 [wcsndm-hn.md](wcsndm-hn.md)。

更新 170：WCSNDM 新增可选 HN/HN1 实验内核，HN-SEP 谐波波形的 STFT 功率按现有源时间映射拉伸，与 LLSM 非谐波功率相加后统一生成 log-Mel，目标 F0 交给 PC-NSF。HN0／不填保持默认 HF2；不混入 HN-SEP 噪声残差。独立版本化服务防止旧服务误处理；编辑器同步识别 HN 内核并禁止不支持的分离导出。详细范围、谐波 FLAG 限制与验证见 [wcsndm-hn.md](wcsndm-hn.md)。

更新 169：高级包络图表沿用主编辑器滚轮快捷键：滚轮纵向滚动，Shift＋滚轮横向滚动，Ctrl＋滚轮纵向缩放，Ctrl＋Shift＋滚轮横向缩放。共用 ZoomScrollBar.h 的修饰键映射；缩放锚定鼠标所在时间／响度，边缘按可见范围限位。支持横向触控板滚动和反向缩放事件，拖动控制点时忽略滚轮；视图操作不改包络草稿。图表悬停提示显示快捷键。

更新 168：高级包络图表增加横轴／纵轴独立缩放、倍率显示、滚动条与“适应全图”。时间轴最高 32 倍，响度轴最高 16 倍；图形、坐标刻度、控制柄与鼠标编辑共用视图变换。缩放和滚动仅属于面板视图，不写工程或预设；普通音符、音头／音尾、原生音频与 CVVC 单音素共用。

更新 167：包络预设下拉菜单的六个选项恢复绿色形状缩略图，左图右文，复用原预设数据和缩略图比例。菜单保留键盘选择和无障碍名称，应用仍走既有 applyEnvelopePreset；不修改包络或合成规则。

更新 166：工具栏六个包络预设按钮合并为“包络预设”下拉菜单，按原顺序提供标准、柔起、缓起、渐弱、句尾、短收。选择后仍调用 PianoRollComponent::applyEnvelopePreset，保留整音符和 CVVC 单音素选区语义；操作后清空菜单选中值，允许再次选择同一预设。移除旧的六按钮和重复标题，沿用原 UTAU 显示范围。

更新 165：中文 CVVC 的音符右键菜单将“辅音强制重置”改为“强制重置”。现有菜单动作按轨道发音方式分流，ProjectModel::resetCvvcBoundaries 批量清除所选音符的手动拆音位置和派生缓存，恢复发音器自动位置；一次撤销，多次点击无额外修改。菜单不依赖后台拼字布局是否已就绪。响度／FLAG 包络、辅音速度及父音符起止保留，普通 UTAU 与四区模式继续使用原辅音重置。

更新 164：CVVC 音符内部的 VC/CV、CV/尾音分界可左右拖动，固定相邻两片的外端点和父音符时长。NoteData::cvvcBoundaries 按右侧角色保存相对音符时长的位置；ProjectModel::setCvvcBoundary 是唯一提交入口，松手提交一次、支持保存及撤销，Esc 放弃草稿。CvvcTiming.h 的 applyCvvcBoundaries 统一发音规划和辅音速度预览的限位；上下包络复用实际分片布局，渲染键和派生拼字缓存随位置失效。分界手势不调用音源读取或合成，后台布局返回前保留已提交的位置。

2026-10-13 增补：[功能接口与架构审计](AI_INTERFACE_ARCHITECTURE.md) 回答每项功能应调用谁、数据在哪里多次复制、如何最小化改造；[实际接口声明清单](AI_API_REFERENCE.md) 为公开签名和协议命令目录，由 `tools/generate-interface-reference.py` 生成。扫描范围、位置及源码指纹在 [interface-audit-inventory.json](interface-audit-inventory.json)。本页继续作为逐文件导航，不重复维护另一份接口表。

更新日期：2026-10-15。实现基线：0.25 内部更新 166（包络预设下拉菜单）。先读 [代码功能总览](AI_CODE_GUIDE.md)。

更新 163：单独选中 CVVC 音素后，“高级包络”不再被提前返回拦截。PianoRollComponent::selectedCvvcEnvelope 提供当前音素的范围与曲线；MainComponent 复用 AdvancedEnvelopePanel 的整段编辑界面，应用仍统一调用 setCvvcEnvelope，仅提交该角色。包络点可保存可选贝塞尔控制柄，cvvcEnvelopeValue 与两个 UTAU 渲染器共用同一插值，曲线加入混音缓存键。打开／未改动应用不写工程；草稿应用、试听、修订保护、保存和撤销保留。恢复默认使用该音素标准梯形。

`tests/CvvcAdvancedEnvelopeSmoke.inc` 在中文 CVVC 专项中执行三个音素标签 → 高级包络按钮 → 应用／撤销／保存的实际交互，并检查两条渲染路径；沿用 `--smoke-chinese-cvvc`。

更新 162：修复旧 CVVC 工程保存的整音符默认分贝包络继续叠乘，造成句首弯曲起音和收尾弯曲淡出。ProjectModel::fromValueTree 仅对缺少 cvvcEnvelopeVersion、无高级包络及 UST 原始段的中文 CVVC 音符，识别四点 0 dB 平台、5/15 ms 起音、35 ms 收尾的历史默认形状，恢复为各音素独立标准梯形。新工程保存 cvvcEnvelopeVersion=1；现代显式曲线、非标准时长／音量、线性手绘和非 CVVC 曲线保持。音频与拼字混音缓存版本同步更新。真实“CVVC测试”工程专项覆盖修复前失败、保存重开、首尾像素位置、两种交叉方式逐采样增益及 HiFiSampler／WCSNDM 合成。

新增 `juce/src/tests/CvvcLegacyEnvelopeSmoke.h`，由 `HifisamplerEnvelopeSmoke.h` 的真实工程分支调用，命令仍为 `--smoke-hifisampler-envelope`。包含旧文件模板迁移、保留自定义曲线、工程副本往返与句首／句尾实际像素检查。

更新 161：CVVC 默认梯形与“标准”预设统一使用 standardCvvcEnvelope 的 5 ms／35 ms 线性幅度规则。自动混音 useStandardUtauEnvelope 在重叠边缘接管标准淡入／淡出，避免手动默认梯形与自动层重复相乘产生转折；保留非标准手动曲线。发音器菜单新增“默认交叉方式→直线／曲线”，TrackData::cvvcCurveCrossfade 随工程保存、撤销，并参与声音与混音元数据缓存键。曲线为互补 smoothstep，两端斜率为零；传统 UTAU 与 HiFiSampler 共享增益描述。replacesStandardEnvelope 标记让上下包络显示使用同一实际增益。整音符 CVVC 预设按音素批量提交 setCvvcEnvelopes，原单片接口委托此入口，一次撤销。

更新 160：CVVC 音符上方增加 VC／CV／尾音可点击标签，复用父音符 ID 加 envelopePart 的选区，不生成独立假音符。选中音素时响度预设只提交 setCvvcEnvelope，响度／FLAG 面板仅保留当前音素的编辑曲线和控制点，其他音素淡显实际曲线；点击主体或显式整音符选择恢复整音符操作。上方包络改为逐音素绘制实际自动混音增益，移除父音符包络的重复外形。音素选择时高级包络入口聚焦分片响度编辑，整音符 OTO 设置需点击主体。选择不写工程；曲线复用既有保存、撤销和渲染入口。

更新 159：`mixNsfUtauNotes` 的自动淡出完成后保持零增益，消除前一 VC/CV 在交叉区后恢复音量的错误；负 OTO 重叠按间隔提前结束前音并使用 3 ms 去爆音，长正重叠按相邻音符时序为前音补足尾部。`UtauMixEnvelope` 的显示元数据与实际混音使用同一持久淡出规则；手动包络仍先应用再乘自动层。HiFiSampler 音频缓存与 CVVC 混音元数据版本更新。专项 `--smoke-hifisampler-envelope <输出目录> [工程] [模型] [重采样器]` 覆盖多音重叠、短后音、零／负重叠及用户工程只读合成；现有 CVVC 专项增加真实模型淡出保持及跨节拍长重叠。

更新 158：CVVC 音头手柄按已解析的 VC／CV／收尾 OTO 与相邻音符时长构造纯时序求值器，通过二分反求整数辅音速度；拖动过程不读文件、不跑发音器或声码器。`cvvcTransitionDuration` 统一 VC 提前量规则，`scaledUtauPreutterance` 统一源时长封顶和固定辅音区缩放，移除 CVVC 独有的 0–200 限制；松手等待后台布局时保留已提交的预览音头位置。旧拼字与音频派生缓存按时序规则版本失效。`tests/CvvcConsonantDragSmoke.h` 随中文 CVVC 专项验证负值、快速辅音、饱和上限、实际渲染、撤销及松手前后位置。

更新 157：新增 `backend/CvvcPhonemeCache.h`（派生拼字缓存序列化）及 `tests/CvvcLayoutCacheSmoke.h`（真实消息循环下验证无选择、多片段后台补全和保存重开）。主入口 `ProjectModel::rememberCvvcPhonemes`；卷帘内 `requestCvvcLayouts` 调用现有发音器，`setUtauNoteWaveforms` 合并有效元数据。“显示”菜单第 9 项为拼字信息，显示偏好 `ui.showPhonemeInfo`；不改变声码器和合成算法。

更新 156：`MainComponent.cpp` 新增全模式“自动交叉淡化”按钮；`PianoRollCvvcEnvelope.h` 内的 `crossfadeSelectedNotes` 复用所有模式响度布局数据；`ProjectModel::crossfadeNotes` 是唯一生成／原子提交入口。测试 `tests/AutoCrossfadeSmoke.h`（`--smoke-auto-crossfade <目录>`），CVVC 实际渲染验证仍归 `tests/CvvcEnvelopeSmoke.h`。不新设混音器或工程字段。

更新 155：`backend::standardCvvcEnvelope` 统一生成各 CV／VC／收尾的默认四点直线梯形（5 ms 起音、35 ms 释放，短片段用三等分避免点序倒置）。`planChineseCvvc` 与 `PianoRollCvvcEnvelope.h` 复用该函数；仅在无手动父包络、无分片响度覆盖时使用。已有手动曲线及导入曲线保留，默认值不写回工程。`cvvcDefaultEnvelopeGain` 传递默认包络的基准音量，防止调节音量后被误判为手工平直曲线；CVVC 渲染键升级至 v3。

更新 154：`backend/UtauMixEnvelope.h` 保存渲染器实际使用的采样级自动淡化窗口，传统 UTAU 与 HiFisampler 混音、CVVC 响度可视化共享增益计算。`noteMixEnvelope` 经发音器折回 `UtauPhonemeSpan::mixEnvelope`，只读数据不写工程；实线显示手动包络乘自动淡化，虚线／圆点保留编辑值，VC／CV／收尾用不同颜色，交叉区着色。HiFisampler CVVC 波形回调移到已确定自动混音增益之后，避免预览缺少淡化。`utauPhonemeMixContextHash` 防止邻音时序或引擎变化后继续显示过期交叉曲线；包络／FLAG 编辑保留布局和有效淡化。

更新 153：`backend/CvvcEnvelope.h` 是分片曲线类型／存储格式；`PianoRollCvvcEnvelope.h` 为原卷帘的响度／FLAG 分片绘制和鼠标交互；`ProjectModel::setCvvcEnvelope` 是唯一提交入口，`ChineseCvvcPhonemizer.h` 是唯一合成映射入口。`tests/CvvcEnvelopeSmoke.h` 随 `--smoke-chinese-cvvc` 验证，未增加第二套发音器或独立音符模型。

更新 152：`MainComponent::phonemizerMenu/showPhonemizerMenu` 复用轨道发音器菜单；按钮位于拉伸算法下方。`ChineseCvvcRules::aliases` 统一配置别名展开；`tests/ChineseCvvcSmoke.h` 扩展分隔符、候选／代表元音、旧式配置优先级、手写覆盖和工具栏检查。沿用 `--smoke-chinese-cvvc`，不添加平行发音器接口。

更新 151：`AssetManagerComponent::importVoicebankDialog` 复用轻量 `registerFolder`；[tests/VoicebankImportSmoke.h](../juce/src/tests/VoicebankImportSmoke.h) 提供 `--smoke-voicebank-import 输出目录 音源目录`。`backend/ChineseCvvcPhonemizer.h` 加入受音源规则及 OTO 约束的等价拼写选择；`tests/ChineseCvvcSmoke.h` 增加元音继承、精确别名／替换优先及衔接回归。配置研究见 [chinese-cvvc.md](chinese-cvvc.md)。

更新 150：`MainComponent::selectVoicebank` 统一选择和默认音源的轻量绑定；[tests/VoicebankSelectionSmoke.h](../juce/src/tests/VoicebankSelectionSmoke.h) 提供 `--smoke-voicebank-selection 输出目录 音源目录`，验证真实音源选择及界面刷新。本页列出自有源码、测试和脚本；第三方供应代码按目录归类。

更新 148：素材管理器音频项改为 24 像素单行，只显示一次相对文件名；音源文件夹保持 40 像素名称／路径双行。`TreeRow` 缓存每行位置和高度，绘制、命中与滚动范围统一使用这些坐标，保持展开／收起、拖拽及参数编辑行为。

更新 147：素材管理器改为全宽可折叠文件列表；音源左侧箭头展开／收起，双击文件夹名称也可切换。展开后列出包含子目录在内的音频文件，显示文件名和相对路径；单击选中、拖拽使用素材、双击编辑参数。多个文件夹可同时展开，滚轮与纵向滚动条浏览长音源库。`AssetManagerComponent` 缓存已展开成员，绘制只遍历可见行；工程更新保留现存文件夹展开状态，换工程清除失效引用。扩展 `tests/ProjectVoicebankAssetsSmoke.h` 验证窄面板点击、展开／收起、子目录、滚动末尾和工程切换。

更新 146：`AssetManagerComponent.*` 新增工程音源文件夹同步、去重、工程标记和手动登记分离；`MainComponent.cpp` 的 `showAssetManager`／`changeListenerCallback` 调用同步。`tests/ProjectVoicebankAssetsSmoke.h` 由 `Main.cpp` 接入，命令 `--smoke-project-voicebank-assets 输出目录`，检查工程切换、保存重开、共用音源、合成类型过滤、全局登记隔离及缺失引用；输出实际面板截图。既有 `--smoke-asset-manager` 继续覆盖素材导入、参数及 OTO 导出。

更新 145：`AccompanimentTempo.*` / `AccompanimentTempoPanel.h` 和 `tools/hamood_audio/tempo_worker.py` 提供伴奏音频右键“识别曲速”。只运行已有 Beat This! small0，后台读取裁剪／合并片段，显示 BPM，不写工程；取消及析构回收进程和临时文件。新增 `tests/AccompanimentTempoSmoke.h`，命令 `--smoke-accompaniment-tempo 输出目录 [真实伴奏]`。

更新 144：新增 [backend/WcsNonLoopFeatures.cpp](../juce/src/backend/WcsNonLoopFeatures.cpp)／[WcsNonLoopFeatures.h](../juce/src/backend/WcsNonLoopFeatures.h)，非 U NSF 的 WCSNDM 非循环参数拉伸、物理 Mel、源特征缓存及分块取消。`RenderService` 传递源 MIDI，`NsfHifiganRenderer` 在既有噪声保护及可选 F0 平滑之后生成新特征。新增 [backend/WcsNonLoopSmoke.inc](../juce/src/backend/WcsNonLoopSmoke.inc)，由 NSF 渲染源包含；命令 `--smoke-wcs-nonloop 输出目录 NSF模型目录 [真实录音]`。`NsfPickerSmoke.h` 增加选择、撤销重做、工程保存、单段／整句请求、缓存失效和实际界面截图验证。值 6／菜单 ID 7／键 `wcsndm-nonloop` 追加，不改旧工程枚举含义。无需新模型，不启动外置引擎；没有增加发音时长保护。报告在 `test-output/update144/`。

更新 143：`NativePitchVoicingDisplay.*` 增加人为目标范围与真实圆点绘制，无 F0 处的人工曲线显示为圆点线，未编辑源曲线仍断开。`PianoRollComponent` 支持共享／独立曲线、边界外端点及手绘／直线预览；`ProjectModel::setNotePitchCurve` 保留无声帧目标形状而不改变清音掩码。扩展 `NativePitchVisibilitySmoke.h` 验证像素、圆点／短划线区别、滚动相位、手绘、撤销及无新增渲染 F0。报告在 `test-output/update143/`。

更新 142：`NativePitchVoicingDisplay.*` 增加真实有声区间与绘制裁切，`PianoRollComponent` 对普通／共享目标曲线、原始参考和颤音隐藏无 F0 区域，取消缺失音高的虚线参考。新增 `tests/NativePitchVisibilitySmoke.h`，命令 `--smoke-native-pitch-visibility 输出目录` 检查实际像素、特殊片段、普通音符内部清音、批量拉直、颤音、拉伸、占位和渲染不变；更新气声显示专项以验证其开关不会填补缺失 F0。报告在 `test-output/update142/`，无需新增模型。

更新 141：新增 `NativeUnpitchedRegions.h`，非 U 自动／右键补齐无音高片段，灰色斜纹区分；删除会切掉真实播放源范围，沿用裁剪／响度／剪贴板与撤销。`nativeUnpitched` 通过工程、HJM 与 MCP 保存，渲染不生成 F0。新增 `tests/NativeUnpitchedRegionsSmoke.h` 和 `--smoke-native-unpitched-regions 输出目录 [真实录音] [分析模型根目录]`，验证实际 PCM、源映射、保存、裁剪、响度与复制。无需新增模型。

更新 140：显示项更名“气声虚线”。新增 `backend/SourceBreathiness.h` 独立显示证据，使用周期抵消残差与多频段噪声分析，覆盖带音高气声；`NativePitchVoicingDisplay.*` 继续后台缓存和源时间映射，音频渲染 `SourceVoicing` 仍保守保护清音。新增 `tests/SourceBreathinessSmoke.h`，`--smoke-source-breathiness 输出目录 [真实录音]` 检查弱气声／低频气声、干净元音、滑音、颤音、边界、取消及实际录音；现有显示专项新增气声元音虚线验证。报告在 `test-output/update140/`，无需新增模型。

更新 139：`MainComponent::nativeVibratoButton` 提供非 U 颤音入口，`PianoRollComponent::showSelectedNativeVibratoDialog` 复用批量参数对话框并可选实际音高线显示；`ProjectModel::setNotesVibrato` 可将显示选择纳入同一步撤销。新增 `juce/src/tests/NativeVibratoSmoke.h`：`--smoke-native-vibrato 输出目录` 通过实际按钮、对话框和拖动验证界面及工程；`--smoke-native-vibrato-render 输出目录 NSF模型目录` 检查单段／整句／粘连目标、清音、缓存及真实音频 F0。报告在 `test-output/update139/`。

本页覆盖 `juce/src` 下的自有非测试 C++／头文件／包含单元，测试在后面单列。头文件也可能包含完整实现，`.inc` 文件通过其他源文件编译。

## 编辑器与工程模块

| 文件 | 功能 |
| --- | --- |
| [AccompanimentTempo.h](../juce/src/AccompanimentTempo.h) / [AccompanimentTempo.cpp](../juce/src/AccompanimentTempo.cpp) | 伴奏可见音频范围的只读曲速分析、后台解码、Beat This! 进程及取消回收 |
| [AccompanimentTempoPanel.h](../juce/src/AccompanimentTempoPanel.h) | 右键“识别曲速”的非模态进度／BPM 结果窗口，不写工程 |
| [AdvancedEnvelopePanel.h](../juce/src/AdvancedEnvelopePanel.h) | 原生音频、OTO 音头和音尾的高级包络草稿、曲线与预设界面 |
| [AssetManagerComponent.cpp](../juce/src/AssetManagerComponent.cpp) / [AssetManagerComponent.h](../juce/src/AssetManagerComponent.h) | 素材资源管理界面 |
| [AudioEngine.cpp](../juce/src/AudioEngine.cpp) / [AudioEngine.h](../juce/src/AudioEngine.h) | 音频设备、工程同步、合成缓存、波形、播放混音和导出 |
| [ClipParts.h](../juce/src/ClipParts.h) | 组合区域展开、源切片、坐标转换和父层增益继承 |
| [DiffSingerParameterCurves.h](../juce/src/DiffSingerParameterCurves.h) | DS 预测实参和偏移曲线 |
| [DiffSingerPhonemeDisplay.h](../juce/src/DiffSingerPhonemeDisplay.h) | DS 辅音元音和实际音素范围显示 |
| [DiffSingerPhonemeEditor.h](../juce/src/DiffSingerPhonemeEditor.h) | DS 音素时长和边界编辑窗口 |
| [DiffSingerPitchHandles.h](../juce/src/DiffSingerPitchHandles.h) | DS 音高控制点和拖动把手 |
| [DiffSingerPitchRestore.h](../juce/src/DiffSingerPitchRestore.h) | DS 原始预测音高恢复 |
| [DiffSingerPronunciationEditor.h](../juce/src/DiffSingerPronunciationEditor.h) | DS 读音覆盖编辑窗口 |
| [DiffSingerRequest.h](../juce/src/DiffSingerRequest.h) | 从工程音符组织 DS 请求 |
| [FlagCurveDrawing.h](../juce/src/FlagCurveDrawing.h) | FLAG 曲线绘制辅助 |
| [Hamood.cpp](../juce/src/Hamood.cpp) / [Hamood.h](../juce/src/Hamood.h) | 调性分析、声部方案和和声生成 |
| [HamoodAudio.h](../juce/src/HamoodAudio.h) | 伴奏分析进程、缓存查询和工程时间转换 |
| [HamoodProject.h](../juce/src/HamoodProject.h) | 工程内和声信息和持久化辅助 |
| [HamoodTimelinePanel.h](../juce/src/HamoodTimelinePanel.h) | 和声调性时间轴面板 |
| [I18n.cpp](../juce/src/I18n.cpp) / [I18n.h](../juce/src/I18n.h) | 多语言界面文本 |
| [Main.cpp](../juce/src/Main.cpp) | 程序启动、窗口生命周期、CLI 分支和内联诊断 |
| [MainComponent.cpp](../juce/src/MainComponent.cpp) / [MainComponent.h](../juce/src/MainComponent.h) | 主窗口布局、菜单工具栏、选择同步、设置与工程操作 |
| [MainComponentDiffSinger.cpp](../juce/src/MainComponentDiffSinger.cpp) | 主界面 DS 推理任务、发音和参数操作 |
| [MainComponentHamood.cpp](../juce/src/MainComponentHamood.cpp) | 主界面和声配置、分析与生成 |
| [MainComponentMcp.cpp](../juce/src/MainComponentMcp.cpp) | 窗口实时 MCP、选区、修订校验、事务和后台任务 |
| [ModelessWindows.h](../juce/src/ModelessWindows.h) | 非模态窗口共用封装 |
| [NativeAudioClipboard.h](../juce/src/NativeAudioClipboard.h) | 原生分段复制粘贴的素材和参数组织 |
| [NativeAudioDisconnect.h](../juce/src/NativeAudioDisconnect.h) | 关联音频分离成独立区域 |
| [NativeUnpitchedRegions.h](../juce/src/NativeUnpitchedRegions.h) | 生成无音高片段与实际播放范围删除 |
| [NativeAudioFocus.h](../juce/src/NativeAudioFocus.h) | 重叠音频显示聚焦、亮度和绘制次序 |
| [NativeAudioLink.h](../juce/src/NativeAudioLink.h) | 不同素材按原时钟组装为粘连区域 |
| [NativeAudioOverlap.h](../juce/src/NativeAudioOverlap.h) | 碰撞、重叠许可与关闭时的裁切处理 |
| [NativeAudioTrim.h](../juce/src/NativeAudioTrim.h) | 自由端点裁剪计划、源映射和音高包络保留 |
| [NativeTrimSource.h](../juce/src/NativeTrimSource.h) | 隐藏源时间映射、完整源音高参考、扩展音高恢复及持久化 |
| [NativeNoteJoin.h](../juce/src/NativeNoteJoin.h) | 相邻同源粘连段的非破坏合并、统一拉伸与曲线保留 |
| [NativeNoteTiming.h](../juce/src/NativeNoteTiming.h) | 移动、拉伸、分段联动和源范围绑定 |
| [NativePitchIdentity.h](../juce/src/NativePitchIdentity.h) | 真实源 F0 已知和未编辑音高判断 |
| [NativeSharedEnvelope.h](../juce/src/NativeSharedEnvelope.h) | 原生粘连音频的公共响度包络、插值、分段存储与断开／复制时物化 |
| [NativeSourceTimeMap.h](../juce/src/NativeSourceTimeMap.h) | 源目标时间互换和有效音频窗口转换 |
| [NoteDanceAnimation.h](../juce/src/NoteDanceAnimation.h) | 音符舞动显示动画 |
| [OtoRegionGuides.h](../juce/src/OtoRegionGuides.h) | OTO 分区参考位置与绘制辅助 |
| [OtoWaveformEditorComponent.cpp](../juce/src/OtoWaveformEditorComponent.cpp) / [OtoWaveformEditorComponent.h](../juce/src/OtoWaveformEditorComponent.h) | OTO 波形、频谱和分段边界编辑 |
| [PianoRollComponent.cpp](../juce/src/PianoRollComponent.cpp) / [PianoRollComponent.h](../juce/src/PianoRollComponent.h) | 钢琴卷帘、音符音高包络 FLAG 绘制、命中、拖动和菜单 |
| [PianoRollDiffSingerOffset.h](../juce/src/PianoRollDiffSingerOffset.h) | 卷帘 DS 音高偏移辅助 |
| [Pinyin.cpp](../juce/src/Pinyin.cpp) / [Pinyin.h](../juce/src/Pinyin.h) | 中文拼音转换 |
| [PinyinTable.inc](../juce/src/PinyinTable.inc) | 生成的拼音映射数据包含单元 |
| [ProjectFileIO.h](../juce/src/ProjectFileIO.h) | 工程二进制完整性、临时写入、备份和安全替换 |
| [ProjectModel.cpp](../juce/src/ProjectModel.cpp) / [ProjectModel.h](../juce/src/ProjectModel.h) | 数据结构、正式编辑、快照、撤销重做、保存和加载 |
| [ProjectRecovery.h](../juce/src/ProjectRecovery.h) | 异步自动恢复快照和租约 |
| [RenderedWaveformPeaks.h](../juce/src/RenderedWaveformPeaks.h) | 已渲染音频峰值数据 |
| [SampleSettings.cpp](../juce/src/SampleSettings.cpp) / [SampleSettings.h](../juce/src/SampleSettings.h) | 素材设置与注释数据 |
| [SettingsComponent.cpp](../juce/src/SettingsComponent.cpp) / [SettingsComponent.h](../juce/src/SettingsComponent.h) | 用户设置、算法、模型路径和推理选项 |
| [SourceWaveformPreview.h](../juce/src/SourceWaveformPreview.h) | 按时间映射绘制拉伸后的原音近似波形 |
| [NativePitchVoicingDisplay.cpp](../juce/src/NativePitchVoicingDisplay.cpp) / [NativePitchVoicingDisplay.h](../juce/src/NativePitchVoicingDisplay.h) | 真实有声区间的绘制遮罩、人为曲线的圆点引导；气声虚线后台证据缓存、源／目标时钟映射和分区绘制；不改变合成 |
| [StartupLog.h](../juce/src/StartupLog.h) | 启动诊断日志 |
| [TailFadePresetStore.h](../juce/src/TailFadePresetStore.h) | 自定义高级包络预设保存、读取和删除 |
| [Theme.cpp](../juce/src/Theme.cpp) / [Theme.h](../juce/src/Theme.h) | 配色、控件外观和工具图标 |
| [TimelineComponent.cpp](../juce/src/TimelineComponent.cpp) / [TimelineComponent.h](../juce/src/TimelineComponent.h) | 上方轨道区域、框选、跨轨移动和波形预览 |
| [TimelineGainEnvelope.h](../juce/src/TimelineGainEnvelope.h) | 区域增益包络辅助 |
| [TrackGainEnvelope.h](../juce/src/TrackGainEnvelope.h) | 轨道增益包络和求值 |
| [TrackListComponent.cpp](../juce/src/TrackListComponent.cpp) / [TrackListComponent.h](../juce/src/TrackListComponent.h) | 轨道名称、引擎标识和混音控件 |
| [UtauOutputEnginePanel.h](../juce/src/UtauOutputEnginePanel.h) | 每轨重采样器与 wavtool 配置 |
| [VoicebankSettingsComponent.cpp](../juce/src/VoicebankSettingsComponent.cpp) / [VoicebankSettingsComponent.h](../juce/src/VoicebankSettingsComponent.h) | 音源库与 OTO 设置 |
| [WavExportOptions.h](../juce/src/WavExportOptions.h) | WAV 格式、范围相关选项与分量类型 |
| [ZoomScrollBar.h](../juce/src/ZoomScrollBar.h) | 带端点缩放的滚动条 |

## 分析 合成与自动化后端

| 文件 | 功能 |
| --- | --- |
| [AdvancedEnvelope.h](../juce/src/backend/AdvancedEnvelope.h) | 高级包络曲线和效果求值 |
| [AmplitudeEnvelopeCurve.h](../juce/src/backend/AmplitudeEnvelopeCurve.h) | 响度包络求值和时长适配 |
| [AnalysisService.cpp](../juce/src/backend/AnalysisService.cpp) / [AnalysisService.h](../juce/src/backend/AnalysisService.h) | GAME、FCPE、native-hq 统一分析和源 F0 附着 |
| [AudioFileReader.cpp](../juce/src/backend/AudioFileReader.cpp) / [AudioFileReader.h](../juce/src/backend/AudioFileReader.h) | 音频读取与转码缓存 |
| [DiffSingerOptions.h](../juce/src/backend/DiffSingerOptions.h) | DS 设备和预览导出质量配置 |
| [DiffSingerProjectCache.h](../juce/src/backend/DiffSingerProjectCache.h) | DS 工程缓存 |
| [DiffSingerRenderer.cpp](../juce/src/backend/DiffSingerRenderer.cpp) / [DiffSingerRenderer.h](../juce/src/backend/DiffSingerRenderer.h) | DS C++ 请求、后台进程和合成桥 |
| [DiffSingerTiming.h](../juce/src/backend/DiffSingerTiming.h) | DS 时长上下文与边界覆盖 |
| [FcpeAnalyzer.cpp](../juce/src/backend/FcpeAnalyzer.cpp) / [FcpeAnalyzer.h](../juce/src/backend/FcpeAnalyzer.h) | FCPE 连续 F0 ONNX 分析 |
| [GameAnalyzer.cpp](../juce/src/backend/GameAnalyzer.cpp) / [GameAnalyzer.h](../juce/src/backend/GameAnalyzer.h) | GAME 音符 ONNX 分析 |
| [HifisamplerDsp.inc](../juce/src/backend/HifisamplerDsp.inc) | 内置 HiFisampler DSP 包含实现 |
| [HifisamplerFlags.h](../juce/src/backend/HifisamplerFlags.h) | 文本 FLAG、参数范围、曲线和优先级 |
| [HifisamplerSmoke.inc](../juce/src/backend/HifisamplerSmoke.inc) | HiFisampler 专项验证包含实现 |
| [LegacyTextCodec.h](../juce/src/backend/LegacyTextCodec.h) | 旧文本编码兼容 |
| [LiveMcpBridge.cpp](../juce/src/backend/LiveMcpBridge.cpp) / [LiveMcpBridge.h](../juce/src/backend/LiveMcpBridge.h) | 实时窗口发现、会话绑定和进程通信 |
| [Llsm2Renderer.cpp](../juce/src/backend/Llsm2Renderer.cpp) / [Llsm2Renderer.h](../juce/src/backend/Llsm2Renderer.h) | LLSM2 合成适配 |
| [McpServer.cpp](../juce/src/backend/McpServer.cpp) / [McpServer.h](../juce/src/backend/McpServer.h) | 独立 MCP 工具 schema 和执行 |
| [MelodyneImporter.cpp](../juce/src/backend/MelodyneImporter.cpp) / [MelodyneImporter.h](../juce/src/backend/MelodyneImporter.h) | 独立 MPD 解析和工程转换 |
| [MelodyneProvider.cpp](../juce/src/backend/MelodyneProvider.cpp) / [MelodyneProvider.h](../juce/src/backend/MelodyneProvider.h) | 安装发现、VST3 探测和能力开关 |
| [Mld3Renderer.cpp](../juce/src/backend/Mld3Renderer.cpp) / [Mld3Renderer.h](../juce/src/backend/Mld3Renderer.h) | MLD3 兼容和自有渲染实现 |
| [Mld5Renderer.cpp](../juce/src/backend/Mld5Renderer.cpp) / [Mld5Renderer.h](../juce/src/backend/Mld5Renderer.h) | MLD5 兼容和自有渲染实现 |
| [NativeAnalyzer.cpp](../juce/src/backend/NativeAnalyzer.cpp) / [NativeAnalyzer.h](../juce/src/backend/NativeAnalyzer.h) | 无神经模型的原生音频分析 |
| [NsfHifiganRenderer.cpp](../juce/src/backend/NsfHifiganRenderer.cpp) / [NsfHifiganRenderer.h](../juce/src/backend/NsfHifiganRenderer.h) | NSF 模型、声学特征、帧对齐和 ONNX 推理 |
| [NsfRenderChunkCache.h](../juce/src/backend/NsfRenderChunkCache.h) | 非 U NSF 会话内核心音频复用；完整输入指纹、LRU 内存限制及不可变共享读者 |
| [OrtExecution.cpp](../juce/src/backend/OrtExecution.cpp) / [OrtExecution.h](../juce/src/backend/OrtExecution.h) | 原生 ONNX 环境、执行后端和设备支持 |
| [OtoAudioAnalysis.cpp](../juce/src/backend/OtoAudioAnalysis.cpp) / [OtoAudioAnalysis.h](../juce/src/backend/OtoAudioAnalysis.h) | OTO 音频与频谱分析 |
| [PlaybackRenderPriority.h](../juce/src/backend/PlaybackRenderPriority.h) | 按播放位置计算合成优先级 |
| [PlaybackRenderQueue.h](../juce/src/backend/PlaybackRenderQueue.h) | 任务排队、并发隔离、取消和过期清理 |
| [RenderService.cpp](../juce/src/backend/RenderService.cpp) / [RenderService.h](../juce/src/backend/RenderService.h) | 原音保真判断、调度和各合成后端调用 |
| [NativeEnvelope.h](../juce/src/backend/NativeEnvelope.h) | 原生独立包络参数、序列化、求值、显示和贝塞尔裁剪 |
| [MixedEnvelopeIO.h](../juce/src/backend/MixedEnvelopeIO.h) | 混合包络分段点与各段形状的工程／预设 JSON 编解码和合法性检查 |
| [SourceBreathiness.h](../juce/src/backend/SourceBreathiness.h) | 显示用气声／噪声证据，周期抵消残差、多频段平坦度和短时确认；带音高气声不抹去 F0 |
| [SourceVoicing.h](../juce/src/backend/SourceVoicing.h) | 渲染／分析用的保守清音保护，独立于气声显示，不把有声气声元音整体归为清音 |
| [TailFadeSettings.h](../juce/src/backend/TailFadeSettings.h) | 高级音头音尾包络参数与形状 |
| [UstExchange.h](../juce/src/backend/UstExchange.h) | UST 导出、兼容字段和警告 |
| [UstImporter.cpp](../juce/src/backend/UstImporter.cpp) / [UstImporter.h](../juce/src/backend/UstImporter.h) | UST 导入解析 |
| [UstText.h](../juce/src/backend/UstText.h) | UST 文本结构和辅助 |
| [UtauOtoOverride.h](../juce/src/backend/UtauOtoOverride.h) | 单音 OTO 覆盖 |
| [ChineseCvvcPhonemizer.h](../juce/src/backend/ChineseCvvcPhonemizer.h) | 中文 CVVC presamp 规则、严格别名编排、父音符时钟和波形聚合；标准 UTAU 两种后端共用 |
| [UtauRenderer.cpp](../juce/src/backend/UtauRenderer.cpp) / [UtauRenderer.h](../juce/src/backend/UtauRenderer.h) | 音源与 OTO、传统重采样器、内置 NSF 句段、缓存和 HF 进程管理 |
| [UtauMixEnvelope.h](../juce/src/backend/UtauMixEnvelope.h) | 混音与 CVVC 包络显示共用的采样级自动淡化窗口和增益；只读渲染元数据，不保存到工程 |
| [WorldRenderer.cpp](../juce/src/backend/WorldRenderer.cpp) / [WorldRenderer.h](../juce/src/backend/WorldRenderer.h) | WORLD 合成适配 |

## DiffSinger Python 模块

位于 `juce/engines/diffsinger/`，由构建复制到运行资源目录。

| 文件 | 功能 |
| --- | --- |
| [bridge.py](../juce/engines/diffsinger/bridge.py) | 音源和模型装载、分组、时长、pitch、variance、声学和声码器操作 |
| [compatibility.py](../juce/engines/diffsinger/compatibility.py) | 采样输入兼容和音源预检 |
| [english_g2p.py](../juce/engines/diffsinger/english_g2p.py) | 英文读音预测 |
| [expressions.py](../juce/engines/diffsinger/expressions.py) | 能力和参数采样、variance、音色混合 |
| [inference.py](../juce/engines/diffsinger/inference.py) | 推理配置、执行后端、Context 和 Session |
| [pronunciation.py](../juce/engines/diffsinger/pronunciation.py) | 多语言前缀、词典、Resolver 和发音覆盖 |
| [runtime.py](../juce/engines/diffsinger/runtime.py) | 后台请求服务、缓存指纹、父进程存活和退出 |
| [timing.py](../juce/engines/diffsinger/timing.py) | 时长上下文与边界覆盖应用 |
| [variance_retake.py](../juce/engines/diffsinger/variance_retake.py) | 局部重生成 mask、结构键和预测 |

## 原生测试文件

检查项摘录仅用于定位，不表示已执行通过。完整运行参数在 Main.cpp 中确认。

| 文件 | 检查项摘录 |
| --- | --- |
| [SourceBreathinessSmoke.h](../juce/src/tests/SourceBreathinessSmoke.h) | 弱／低频气声、滑音和颤音隔离、静音、边界、取消及真实录音显示证据 |
| [AccompanimentSmoke.h](../juce/src/tests/AccompanimentSmoke.h) | `source_written`；`source_sidecar_written`；`independent_track_type` |
| [AccompanimentTempoSmoke.h](../juce/src/tests/AccompanimentTempoSmoke.h) | 实际右键入口与非模态窗口、已有节拍模型、裁剪／合并、任务取消、缺资源及工程快照不变 |
| [AdvancedEnvelopePanelSmoke.h](../juce/src/tests/AdvancedEnvelopePanelSmoke.h) | `settings_validate`；`reversed_range_rejected`；`nonfinite_level_rejected` |
| [AdvancedEnvelopeSmoke.h](../juce/src/tests/AdvancedEnvelopeSmoke.h) | `voice_fixture_written`；`oto_fixture_loaded`；`oto_regions_written` |
| [ClassicFlagUiSmoke.h](../juce/src/tests/ClassicFlagUiSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [ClipEditorScopeSmoke.h](../juce/src/tests/ClipEditorScopeSmoke.h) | `split_created_two_regions`；`left_click_focuses_only_left_notes`；`select_all_is_region_local` |
| [ClipGainKnobSmoke.h](../juce/src/tests/ClipGainKnobSmoke.h) | `source_written`；`export_succeeded`；`export_length_unchanged` |
| [ClipMergeSmoke.h](../juce/src/tests/ClipMergeSmoke.h) | `source_written`；`export_succeeded`；`export_length_unchanged` |
| [ClipSplitSmoke.h](../juce/src/tests/ClipSplitSmoke.h) | `source_written`；`export_succeeded`；`right_click_mute_badge_opens_menu_without_muting` |
| [ClipTrimSmoke.h](../juce/src/tests/ClipTrimSmoke.h) | `source_written`；`export_succeeded`；`export_keeps_outer_clip_length` |
| [ComponentExportSmoke.h](../juce/src/tests/ComponentExportSmoke.h) | `component_fixture`；`world_component_capability`；`llsm_component_capability` |
| [ContinuousFlagSmoke.h](../juce/src/tests/ContinuousFlagSmoke.h) | `existing_point_mode_toolbar`；`mode_preference_roundtrip`；`right_click_menu_checked` |
| [CrossRegionEditingSmoke.h](../juce/src/tests/CrossRegionEditingSmoke.h) | `mouse_down_retains_cross_region_selection`；`ui_moves_both_regions_by_same_time_and_pitch`；`ui_keeps_selection_and_hidden_region_unchanged` |
| [DiffSingerCacheSmoke.h](../juce/src/tests/DiffSingerCacheSmoke.h) | `project_saved`；`real_render_ok`；`expected_disk_hit` |
| [DiffSingerConsonantSmoke.h](../juce/src/tests/DiffSingerConsonantSmoke.h) | `user_project_render`；`user_project_export`；`real_zhang_wang_render` |
| [DiffSingerExpressionsSmoke.h](../juce/src/tests/DiffSingerExpressionsSmoke.h) | `ordinary_U_DF_eligible`；`ordinary_UTAU_WCSNDM_eligible`；`DF_namespace_rejected_in_WCSNDM` |
| [DiffSingerInferenceSmoke.h](../juce/src/tests/DiffSingerInferenceSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [DiffSingerParameterRetakeSmoke.h](../juce/src/tests/DiffSingerParameterRetakeSmoke.h) | `retake_completed`；`DS_button_named_FLAG`；`FLAG_click_opens_actual_without_enabling_offsets` |
| [DiffSingerParametersSmoke.h](../juce/src/tests/DiffSingerParametersSmoke.h) | `accept_prediction`；`stale_prediction_rejected`；`only_selected_note` |
| [DiffSingerPitchHandlesSmoke.h](../juce/src/tests/DiffSingerPitchHandlesSmoke.h) | `generation_applied`；`full_prediction_and_reference_preserved`；`generation_has_2_to_16_key_handles` |
| [DiffSingerPitchOffsetSmoke.h](../juce/src/tests/DiffSingerPitchOffsetSmoke.h) | `DS_offset_mode_available`；`DS_offset_mode_enabled`；`default_line_click_does_not_edit` |
| [DiffSingerPitchPersistenceSmoke.h](../juce/src/tests/DiffSingerPitchPersistenceSmoke.h) | `fresh_process_keeps_loaded_reference`；`fresh_process_keeps_reference_provenance`；`fresh_process_keeps_manual_curve_separate` |
| [DiffSingerPitchReferenceSmoke.h](../juce/src/tests/DiffSingerPitchReferenceSmoke.h) | `generation_applied`；`one_undo_for_generation_and_reference`；`reference_matches_prediction` |
| [DiffSingerPitchRestoreSmoke.h](../juce/src/tests/DiffSingerPitchRestoreSmoke.h) | `DS_reference_enables_mode`；`restore_mode_enabled`；`anchor_context_menu_also_offers_restore` |
| [DiffSingerSmoke.h](../juce/src/tests/DiffSingerSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [DiffSingerTimingSmoke.h](../juce/src/tests/DiffSingerTimingSmoke.h) | `legacy_DS_migrated_to_mou`；`DS_stays_in_mou`；`real_duration_prediction` |
| [DiffSingerUiSmoke.h](../juce/src/tests/DiffSingerUiSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [EmptyTuningClipSmoke.h](../juce/src/tests/EmptyTuningClipSmoke.h) | `non_backing_menu_enabled`；`backing_menu_disabled`；`outside_lanes_has_no_entry` |
| [GameDefaultsSmoke.h](../juce/src/tests/GameDefaultsSmoke.h) | `default_medium`；`portable_game_and_fcpe_ready`；`portable_runtime_ready` |
| [HamoodPersistenceSmoke.h](../juce/src/tests/HamoodPersistenceSmoke.h) | `opening_hamood_does_not_dirty_project`；`manual_key_is_stored_without_generating_harmony`；`closing_and_reopening_retains_manual_keys` |
| [HamoodUtauSmoke.h](../juce/src/tests/HamoodUtauSmoke.h) | `real source sample copied into isolated bank` |
| [IndependentZoomSmoke.h](../juce/src/tests/IndependentZoomSmoke.h) | `upper_wheel_leaves_lower_scale_and_position_unchanged`；`upper_wheel_changes_upper_scale`；`lower_wheel_leaves_upper_scale_and_position_unchanged` |
| [IntegratedSmoke.h](../juce/src/tests/IntegratedSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [ModelessOtoSmoke.h](../juce/src/tests/ModelessOtoSmoke.h) | `modeless_audio_fixture`；`bank_window_is_modeless`；`main_transport_and_roll_are_not_blocked` |
| [MouToJieSmoke.h](../juce/src/tests/MouToJieSmoke.h) | `fixture_cp932_written`；`conversion_succeeds`；`all_four_region_rows_converted_once` |
| [NativeUnpitchedRegionsSmoke.h](../juce/src/tests/NativeUnpitchedRegionsSmoke.h) | 自动／手动补齐、实际 PCM 删除、撤销、非线性时钟、保存、响度、裁剪与剪贴板 |
| [NativePitchVisibilitySmoke.h](../juce/src/tests/NativePitchVisibilitySmoke.h) | 源曲线断线／人为曲线圆点的实际像素、线型区别、滚动相位、局部手绘、边界外端点、批量拉直、撤销与零 F0 |
| [NativeAudioDisconnectSmoke.h](../juce/src/tests/NativeAudioDisconnectSmoke.h) | `source_written`；`single_selection_cannot_disconnect`；`right_marquee_selects_three` |
| [NativeAudioLinkSmoke.h](../juce/src/tests/NativeAudioLinkSmoke.h) | `single_selection_cannot_link`；`multi_source_selection_offers_link`；`three_sources_linked_without_unselected_neighbor` |
| [NativeAudioOverlapFocusSmoke.h](../juce/src/tests/NativeAudioOverlapFocusSmoke.h) | `chosen_source_stays_bright`；`other_overlapping_source_dims`；`nonoverlapping_source_keeps_brightness` |
| [NativeAudioOverlapSmoke.h](../juce/src/tests/NativeAudioOverlapSmoke.h) | `two_sources_written`；`default_prohibits_overlap`；`note_move_stops_at_neighbor` |
| [ChineseCvvcSmoke.h](../juce/src/tests/ChineseCvvcSmoke.h) | 发音规则、边界／编码／多音高、上下文试听、两种实际合成、菜单／工程保存和缓存 |
| [NativeEnvelopeSmoke.h](../juce/src/tests/NativeEnvelopeSmoke.h) | `--smoke-native-envelope 输出目录`；原生响度通道鼠标加点／拖动／删除／多选及高级包络、持久化、裁剪、实际导出验证 |
| [NativeAudioTrimSmoke.h](../juce/src/tests/NativeAudioTrimSmoke.h) | `source_written`；`subpixel_left_trim_ignores_coarse_grid_and_drag_threshold`；`undo_restores_full_audio` |
| [NativeNoteCopyPasteSmoke.h](../juce/src/tests/NativeNoteCopyPasteSmoke.h) | `source_written`；`three_native_segments_created`；`head_moved_and_middle_stretched` |
| [NativeNoteMoveSmoke.h](../juce/src/tests/NativeNoteMoveSmoke.h) | `source_written`；`split_creates_three_contiguous_notes`；`utau_drag_fixture_saved` |
| [NativeRenderedWaveformSmoke.h](../juce/src/tests/NativeRenderedWaveformSmoke.h) | `real_waveform_is_an_optional_view`；`real_waveform_toggle_does_not_change_source_toggle`；`real_waveform_option_persists` |
| [NativeShutdownSmoke.h](../juce/src/tests/NativeShutdownSmoke.h) | `cpu_model_available`；`fixture_written`；`oto_written` |
| [NativeIncrementalRenderSmoke.h](../juce/src/tests/NativeIncrementalRenderSmoke.h) | 真实 NSF 局部音高／Mel、重叠边界、缓存清理、取消恢复和过期任务回写验证；记录冷／热耗时与推理块数 |
| [NativeNoiseOptionSmoke.h](../juce/src/tests/NativeNoiseOptionSmoke.h) | 谐波／噪声开关的真实模型与 RenderService、清音 PCM、非线性时间映射、关闭后的核心复用和原音直通 |
| [NativeSourcePitchRestoreSmoke.h](../juce/src/tests/NativeSourcePitchRestoreSmoke.h) | `native_context_offers_restore`；`right_marquee_selects_two_fragments`；`restore_keeps_timeline_and_nonlinear_source_clock` |
| [NativeSourcePitchSmoke.h](../juce/src/tests/NativeSourcePitchSmoke.h) | `fixture_written`；`fixture_written`；`real_analysis_models_available` |
| [NativeWaveformPreviewSmoke.h](../juce/src/tests/NativeWaveformPreviewSmoke.h) | `fixture_written`；`fixture_written`；`background_source_peaks_loaded` |
| [NormalDisplaySmoke.h](../juce/src/tests/NormalDisplaySmoke.h) | `normal_display_starts_disabled`；`empty_tuning_region_can_enable`；`backing_region_cannot_enable` |
| [NoteDancePreview.h](../juce/src/tests/NoteDancePreview.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [NoteHintsSmoke.h](../juce/src/tests/NoteHintsSmoke.h) | `start_hint_menu_on_note_region`；`backing_and_empty_regions_disabled`；`menu_enables_only_target_region` |
| [NsfPickerSmoke.h](../juce/src/tests/NsfPickerSmoke.h) | `--smoke-nsf-picker`：空轨算法往返、响应时间、默认音源无整库转换、原生／旧版 NSF 区分、撤销重做与保存重开 |
| [NsfProjectNoteSmoke.h](../juce/src/tests/NsfProjectNoteSmoke.h) | `actual_project_native_render_succeeds`；`saved_short_envelope_reproduces_reported_silence`；`fitted_envelope_preserves_actual_shang_latter_half` |
| [NsfRegionsSmoke.h](../juce/src/tests/NsfRegionsSmoke.h) | `recursive_oto_and_independent_jie_timing`；`resolver_preserves_jie_regions_and_mou_classes`；`mou_oto_loads_two_and_three_region_entries` |
| [OtoContinuitySmoke.h](../juce/src/tests/OtoContinuitySmoke.h) | `line_scanner_preserves_unicode_and_mixed_line_endings`；`continuity_audio_fixture`；`real_voicebank_dialog_opens` |
| [OtoOverlapSmoke.h](../juce/src/tests/OtoOverlapSmoke.h) | `snapshot` |
| [OtoSpectrumSmoke.h](../juce/src/tests/OtoSpectrumSmoke.h) | `source_fixture_written`；`background_analysis_finishes`；`spectrum_and_fcpe_loaded` |
| [OutputEngineSmoke.h](../juce/src/tests/OutputEngineSmoke.h) | `duplicate_nsf_jie_mou_algorithms_removed`；`ordinary_mou_mode_visible`；`region_menu_has_three_output_engine_choices` |
| [PlaybackRenderSmoke.h](../juce/src/tests/PlaybackRenderSmoke.h) | `sounding_before_future`；`future_before_past`；`nearest_past_first` |
| [ProjectSafetySmoke.h](../juce/src/tests/ProjectSafetySmoke.h) | `ordinary_edit_remains_undoable`；`ordinary_edit_redo`；`new_document_clears_undo_and_redo` |
| [ScrollZoomSmoke.h](../juce/src/tests/ScrollZoomSmoke.h) | `custom_scrollbar_installed`；`endpoint_changes_zoom_in_correct_direction`；`opposite_endpoint_stays_anchored` |
| [TempoChangeSmoke.h](../juce/src/tests/TempoChangeSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [TimelineNotePreviewSmoke.h](../juce/src/tests/TimelineNotePreviewSmoke.h) | `preview_written`；`notes_visible_without_source_or_render`；`note_start_and_duration_align_with_timeline` |
| [TimelinePitchSmoke.h](../juce/src/tests/TimelinePitchSmoke.h) | 查看文件的 run／diagnostic 实现和 Main.cpp 调用 |
| [TimelineSelectionSmoke.h](../juce/src/tests/TimelineSelectionSmoke.h) | `cross_track_marquee_selects_only_intersections`；`marquee_does_not_seek_or_change_project`；`marquee_preview_written` |
| [TrackGainEnvelopeSmoke.h](../juce/src/tests/TrackGainEnvelopeSmoke.h) | `source_written`；`preview_written`；`empty_gap_does_not_create_points` |
| [UstFidelitySmoke.h](../juce/src/tests/UstFidelitySmoke.h) | `ust_`；`ust_`；`ust_` |
| [WavExportSmoke.h](../juce/src/tests/WavExportSmoke.h) | `modal_callback_completed`；`fixture_stereo_tones_written`；`write_stereo_` |

## Main.cpp 命令行入口

以下扫描实际 `arguments[0] == ...` 分支，参数数量含命令本身。行号可能随代码移动，可按命令字符串重新搜索。执行前阅读完整分支；有的需要模型、音源、WAV 或探针 EXE，有的会启动窗口或写出文件，不要批量执行全部命令。

| 命令 | 参数数量条件 | 分支位置 |
| --- | --- | --- |
| `--audit-hamood-keys` | `size >= 3` | [Main.cpp L108](../juce/src/Main.cpp#L108) |
| `--smoke-hamood-utau` | `size >= 4` | [Main.cpp L127](../juce/src/Main.cpp#L127) |
| `--smoke-hamood` | `size >= 2` | [Main.cpp L133](../juce/src/Main.cpp#L133) |
| `--smoke-diffsinger-pitch-offset` | `size >= 2` | [Main.cpp L140](../juce/src/Main.cpp#L140) |
| `--smoke-accompaniment` | `size >= 2` | [Main.cpp L147](../juce/src/Main.cpp#L147) |
| `--smoke-wav-export-options` | `size >= 2` | [Main.cpp L154](../juce/src/Main.cpp#L154) |
| `--smoke-diffsinger-pitch-persistence` | `size >= 3` | [Main.cpp L164](../juce/src/Main.cpp#L164) |
| `--smoke-diffsinger-pitch-handles` | `size >= 2` | [Main.cpp L172](../juce/src/Main.cpp#L172) |
| `--smoke-diffsinger-pitch-restore` | `size >= 2` | [Main.cpp L178](../juce/src/Main.cpp#L178) |
| `--smoke-diffsinger-pitch-reference` | `size >= 2` | [Main.cpp L185](../juce/src/Main.cpp#L185) |
| `--smoke-continuous-flag` | `size >= 2` | [Main.cpp L191](../juce/src/Main.cpp#L191) |
| `--smoke-classic-flag-ui` | `查看分支` | [Main.cpp L198](../juce/src/Main.cpp#L198) |
| `--smoke-diffsinger-inference` | `size >= 3` | [Main.cpp L207](../juce/src/Main.cpp#L207) |
| `--smoke-diffsinger-parameter-retake` | `size >= 3` | [Main.cpp L213](../juce/src/Main.cpp#L213) |
| `--smoke-diffsinger-cache` | `size >= 4` | [Main.cpp L220](../juce/src/Main.cpp#L220) |
| `--smoke-diffsinger-parameters` | `size >= 2` | [Main.cpp L225](../juce/src/Main.cpp#L225) |
| `--smoke-diffsinger-expressions` | `查看分支` | [Main.cpp L230](../juce/src/Main.cpp#L230) |
| `--smoke-diffsinger-consonants` | `size >= 3` | [Main.cpp L237](../juce/src/Main.cpp#L237) |
| `--smoke-diffsinger-timing` | `size >= 3` | [Main.cpp L243](../juce/src/Main.cpp#L243) |
| `--smoke-diffsinger-ui` | `size >= 3` | [Main.cpp L250](../juce/src/Main.cpp#L250) |
| `--smoke-diffsinger` | `size >= 3` | [Main.cpp L259](../juce/src/Main.cpp#L259) |
| `--smoke-chinese-cvvc` | `size >= 2`，可选模型目录 | [Main.cpp L266](../juce/src/Main.cpp#L266) |
| `--smoke-output-engine` | `size >= 4` | [Main.cpp L274](../juce/src/Main.cpp#L274) |
| `--smoke-native-incremental` | 输出目录、NSF 模型目录 | [Main.cpp](../juce/src/Main.cpp) |
| `--smoke-native-noise-option` | 输出目录、NSF 模型目录 | [Main.cpp](../juce/src/Main.cpp) |
| `--smoke-native-shutdown` | `size >= 3` | [Main.cpp](../juce/src/Main.cpp) |
| `--smoke-hifisampler` | `size >= 3` | [Main.cpp L291](../juce/src/Main.cpp#L291) |
| `--smoke-nsf-regions` | `size >= 3` | [Main.cpp L298](../juce/src/Main.cpp#L298) |
| `--smoke-nsf-picker` | `查看分支` | [Main.cpp L306](../juce/src/Main.cpp#L306) |
| `--smoke-cross-region-editing` | `size >= 2` | [Main.cpp L315](../juce/src/Main.cpp#L315) |
| `--smoke-hamood-persistence` | `size >= 2` | [Main.cpp L315](../juce/src/Main.cpp#L315) |
| `--smoke-mou-to-jie` | `size >= 2` | [Main.cpp L322](../juce/src/Main.cpp#L322) |
| `--smoke-oto-overlap` | `size >= 2` | [Main.cpp L328](../juce/src/Main.cpp#L328) |
| `--smoke-modeless-oto` | `size >= 2` | [Main.cpp L334](../juce/src/Main.cpp#L334) |
| `--smoke-oto-continuity` | `size >= 2` | [Main.cpp L339](../juce/src/Main.cpp#L339) |
| `--smoke-advanced-envelope` | `size >= 2` | [Main.cpp L344](../juce/src/Main.cpp#L344) |
| `--smoke-advanced-envelope-panel` | `size >= 2` | [Main.cpp L344](../juce/src/Main.cpp#L344) |
| `--smoke-playback-priority` | `size >= 3` | [Main.cpp L349](../juce/src/Main.cpp#L349) |
| `--smoke-ust-fidelity` | `size >= 2` | [Main.cpp L350](../juce/src/Main.cpp#L350) |
| `--smoke-project-safety` | `size >= 2` | [Main.cpp L351](../juce/src/Main.cpp#L351) |
| `--smoke-normal-display` | `size >= 2` | [Main.cpp L359](../juce/src/Main.cpp#L359) |
| `--smoke-note-hints` | `size >= 2` | [Main.cpp L367](../juce/src/Main.cpp#L367) |
| `--smoke-clip-editor-scope` | `size >= 2` | [Main.cpp L375](../juce/src/Main.cpp#L375) |
| `--smoke-empty-tuning-clip` | `size >= 2` | [Main.cpp L383](../juce/src/Main.cpp#L383) |
| `--smoke-track-gain-envelope` | `size >= 2` | [Main.cpp L391](../juce/src/Main.cpp#L391) |
| `--smoke-native-rendered-waveform` | `size >= 2` | [Main.cpp L399](../juce/src/Main.cpp#L399) |
| `--smoke-native-note-move` | `size >= 2` | [Main.cpp L406](../juce/src/Main.cpp#L406) |
| `--smoke-native-source-pitch-restore` | `size >= 2` | [Main.cpp L412](../juce/src/Main.cpp#L412) |
| `--smoke-native-audio-overlap-focus` | `size >= 2` | [Main.cpp L418](../juce/src/Main.cpp#L418) |
| `--smoke-native-audio-overlap` | `size >= 2` | [Main.cpp L424](../juce/src/Main.cpp#L424) |
| `--smoke-native-audio-link` | `size >= 2` | [Main.cpp L430](../juce/src/Main.cpp#L430) |
| `--smoke-native-audio-trim` | `size >= 2` | [Main.cpp L436](../juce/src/Main.cpp#L436) |
| `--smoke-native-audio-disconnect` | `size >= 2` | [Main.cpp L442](../juce/src/Main.cpp#L442) |
| `--smoke-native-note-copy-paste` | `size >= 2` | [Main.cpp L448](../juce/src/Main.cpp#L448) |
| `--smoke-native-source-pitch` | `size >= 3` | [Main.cpp L455](../juce/src/Main.cpp#L455) |
| `--smoke-native-waveform-preview` | `size >= 2` | [Main.cpp L463](../juce/src/Main.cpp#L463) |
| `--smoke-timeline-notes` | `size >= 2` | [Main.cpp L471](../juce/src/Main.cpp#L471) |
| `--smoke-clip-gain-knob` | `size >= 2` | [Main.cpp L479](../juce/src/Main.cpp#L479) |
| `--smoke-clip-merge` | `size >= 2` | [Main.cpp L487](../juce/src/Main.cpp#L487) |
| `--smoke-timeline-selection` | `size >= 2` | [Main.cpp L495](../juce/src/Main.cpp#L495) |
| `--smoke-clip-trim` | `size >= 2` | [Main.cpp L503](../juce/src/Main.cpp#L503) |
| `--smoke-clip-split` | `size >= 2` | [Main.cpp L512](../juce/src/Main.cpp#L512) |
| `--smoke-independent-zoom` | `查看分支` | [Main.cpp L521](../juce/src/Main.cpp#L521) |
| `--smoke-scroll-zoom` | `size >= 2` | [Main.cpp L530](../juce/src/Main.cpp#L530) |
| `--preview-note-dance` | `size >= 2` | [Main.cpp L539](../juce/src/Main.cpp#L539) |
| `--smoke-tempo-change` | `查看分支` | [Main.cpp L547](../juce/src/Main.cpp#L547) |
| `--smoke-timeline-pit` | `size > 1` | [Main.cpp L556](../juce/src/Main.cpp#L556) |
| `--mcp-live` | `查看分支` | [Main.cpp L562](../juce/src/Main.cpp#L562) |
| `--smoke-native-envelope` | `输出目录` | [Main.cpp](../juce/src/Main.cpp) |
| `--mcp-list-sessions` | `查看分支` | [Main.cpp L562](../juce/src/Main.cpp#L562) |
| `--smoke-nsf-project-note` | `size >= 5` | [Main.cpp L570](../juce/src/Main.cpp#L570) |
| `--mcp` | `查看分支` | [Main.cpp L576](../juce/src/Main.cpp#L576) |
| `--smoke-oto-playback` | `size >= 1` | [Main.cpp L590](../juce/src/Main.cpp#L590) |
| `--smoke-oto-editor` | `size >= 3` | [Main.cpp L790](../juce/src/Main.cpp#L790) |
| `--smoke-splice-envelope` | `size >= 2` | [Main.cpp L913](../juce/src/Main.cpp#L913) |
| `--smoke-dialog-enter` | `查看分支` | [Main.cpp L1023](../juce/src/Main.cpp#L1023) |
| `--smoke-source-edit-view` | `size >= 2` | [Main.cpp L1081](../juce/src/Main.cpp#L1081) |
| `--smoke-consonant-hold` | `size >= 1` | [Main.cpp L1143](../juce/src/Main.cpp#L1143) |
| `--smoke-mou-panel-counts` | `size >= 2` | [Main.cpp L1363](../juce/src/Main.cpp#L1363) |
| `--smoke-mou-consonant-ticks` | `size >= 3` | [Main.cpp L1454](../juce/src/Main.cpp#L1454) |
| `--smoke-mou-boundary-drag` | `size >= 3` | [Main.cpp L1607](../juce/src/Main.cpp#L1607) |
| `--smoke-prefix-map` | `size >= 3` | [Main.cpp L1696](../juce/src/Main.cpp#L1696) |
| `--repair-mou-oto` | `size >= 2` | [Main.cpp L1737](../juce/src/Main.cpp#L1737) |
| `--smoke-mou-oto-file` | `size >= 2` | [Main.cpp L1758](../juce/src/Main.cpp#L1758) |
| `--smoke-utau-mode` | `size >= 2` | [Main.cpp L1957](../juce/src/Main.cpp#L1957) |
| `--smoke-lane-marquee` | `size >= 2` | [Main.cpp L2097](../juce/src/Main.cpp#L2097) |
| `--smoke-flag-lane` | `size >= 3` | [Main.cpp L2198](../juce/src/Main.cpp#L2198) |
| `--smoke-keyboard-labels` | `size >= 2` | [Main.cpp L3059](../juce/src/Main.cpp#L3059) |
| `--smoke-lane-labels` | `查看分支` | [Main.cpp L3113](../juce/src/Main.cpp#L3113) |
| `--smoke-export-targets` | `查看分支` | [Main.cpp L3141](../juce/src/Main.cpp#L3141) |
| `--smoke-audition-position` | `size >= 2` | [Main.cpp L3247](../juce/src/Main.cpp#L3247) |
| `--smoke-lyric-envelope` | `size >= 4` | [Main.cpp L3289](../juce/src/Main.cpp#L3289) |
| `--smoke-lyric-timing` | `size >= 5` | [Main.cpp L3361](../juce/src/Main.cpp#L3361) |
| `--smoke-vibrato-bake` | `size >= 2` | [Main.cpp L3460](../juce/src/Main.cpp#L3460) |
| `--smoke-mou-note-regions` | `size >= 4` | [Main.cpp L3573](../juce/src/Main.cpp#L3573) |
| `--smoke-region-guides` | `size >= 2` | [Main.cpp L3859](../juce/src/Main.cpp#L3859) |
| `--smoke-rest-playback` | `size >= 4` | [Main.cpp L3933](../juce/src/Main.cpp#L3933) |
| `--smoke-rest-lyric` | `size >= 3` | [Main.cpp L4029](../juce/src/Main.cpp#L4029) |
| `--smoke-missing-alias-piano` | `size >= 1` | [Main.cpp L4192](../juce/src/Main.cpp#L4192) |
| `--smoke-flag-curve-mode` | `size >= 1` | [Main.cpp L4319](../juce/src/Main.cpp#L4319) |
| `--smoke-flag-point-value` | `size >= 2` | [Main.cpp L4504](../juce/src/Main.cpp#L4504) |
| `--smoke-flag-overlap` | `size >= 3` | [Main.cpp L4594](../juce/src/Main.cpp#L4594) |
| `--smoke-load-timing` | `size >= 2` | [Main.cpp L4709](../juce/src/Main.cpp#L4709) |
| `--smoke-flatten-pitch-line` | `size >= 1` | [Main.cpp L4822](../juce/src/Main.cpp#L4822) |
| `--smoke-note-across-tracks` | `size >= 1` | [Main.cpp L4958](../juce/src/Main.cpp#L4958) |
| `--smoke-clip-across-tracks` | `size >= 1` | [Main.cpp L5114](../juce/src/Main.cpp#L5114) |
| `--smoke-reference-track` | `size >= 1` | [Main.cpp L5263](../juce/src/Main.cpp#L5263) |
| `--smoke-native-pitch-seam` | `size >= 1` | [Main.cpp L5411](../juce/src/Main.cpp#L5411) |
| `--smoke-native-pitch-points` | `size >= 1` | [Main.cpp L5483](../juce/src/Main.cpp#L5483) |
| `--smoke-paste-at-pointer` | `size >= 1` | [Main.cpp L5707](../juce/src/Main.cpp#L5707) |
| `--smoke-note-menu-split` | `size >= 1` | [Main.cpp L5850](../juce/src/Main.cpp#L5850) |
| `--smoke-lyric-gate` | `size >= 1` | [Main.cpp L6032](../juce/src/Main.cpp#L6032) |
| `--smoke-lyric-tab` | `size >= 1` | [Main.cpp L6129](../juce/src/Main.cpp#L6129) |
| `--smoke-syllable-cuts` | `size >= 1` | [Main.cpp L6329](../juce/src/Main.cpp#L6329) |
| `--smoke-single-syllable` | `size >= 1` | [Main.cpp L6511](../juce/src/Main.cpp#L6511) |
| `--smoke-deleted-notes-silent` | `size >= 1` | [Main.cpp L6618](../juce/src/Main.cpp#L6618) |
| `--smoke-delete-clip` | `size >= 1` | [Main.cpp L6765](../juce/src/Main.cpp#L6765) |
| `--smoke-draw-unit` | `size >= 1` | [Main.cpp L6862](../juce/src/Main.cpp#L6862) |
| `--smoke-new-track-draw` | `size >= 1` | [Main.cpp L6955](../juce/src/Main.cpp#L6955) |
| `--smoke-draw-drag` | `size >= 1` | [Main.cpp L7062](../juce/src/Main.cpp#L7062) |
| `--smoke-draw-overlap` | `size >= 1` | [Main.cpp L7233](../juce/src/Main.cpp#L7233) |
| `--smoke-dropdown-arrow` | `size >= 1` | [Main.cpp L7417](../juce/src/Main.cpp#L7417) |
| `--smoke-pitch-line` | `size >= 1` | [Main.cpp L7508](../juce/src/Main.cpp#L7508) |
| `--smoke-native-unpitched-regions` | 输出目录；可选真实录音与 GAME／FCPE 模型根目录 | [Main.cpp](../juce/src/Main.cpp) |
| `--smoke-source-breathiness` | `size >= 2`；输出目录，可选真实录音 | [Main.cpp](../juce/src/Main.cpp#L507) |
| `--smoke-native-consonant-display` | `size >= 2`；输出目录，可选真实录音 | [Main.cpp](../juce/src/Main.cpp) |
| `--smoke-view-menu` | `size >= 1` | [Main.cpp L7799](../juce/src/Main.cpp#L7799) |
| `--inspect-melodyne-tracks` | `size >= 2` | [Main.cpp L7898](../juce/src/Main.cpp#L7898) |
| `--smoke-melodyne-provider` | `size >= 1` | [Main.cpp L7909](../juce/src/Main.cpp#L7909) |
| `--probe-melodyne-vst3-instance` | `size >= 1` | [Main.cpp L7934](../juce/src/Main.cpp#L7934) |
| `--smoke-native-timing` | `size >= 1` | [Main.cpp L7946](../juce/src/Main.cpp#L7946) |
| `--smoke-integrated-ui` | `size >= 2` | [Main.cpp L8003](../juce/src/Main.cpp#L8003) |
| `--smoke-integrated` | `查看分支` | [Main.cpp L8012](../juce/src/Main.cpp#L8012) |
| `--smoke-native-hjm` | `size >= 1` | [Main.cpp L8018](../juce/src/Main.cpp#L8018) |
| `--smoke-batch-lyrics` | `size >= 1` | [Main.cpp L8085](../juce/src/Main.cpp#L8085) |
| `--smoke-envelope-lanes` | `size >= 1` | [Main.cpp L8177](../juce/src/Main.cpp#L8177) |
| `--smoke-wheel` | `size >= 1` | [Main.cpp L8236](../juce/src/Main.cpp#L8236) |
| `--smoke-waveform-toggle` | `size >= 1` | [Main.cpp L8277](../juce/src/Main.cpp#L8277) |
| `--smoke-envelope-base` | `size >= 1` | [Main.cpp L8387](../juce/src/Main.cpp#L8387) |
| `--smoke-amplitude-waveform` | `size >= 1` | [Main.cpp L8708](../juce/src/Main.cpp#L8708) |
| `--smoke-vibrato-end` | `size >= 1` | [Main.cpp L9166](../juce/src/Main.cpp#L9166) |
| `--smoke-hanzi-pinyin` | `size >= 1` | [Main.cpp L9426](../juce/src/Main.cpp#L9426) |
| `--smoke-shared-pitch-line` | `size >= 1` | [Main.cpp L9565](../juce/src/Main.cpp#L9565) |
| `--smoke-waveform-alignment` | `size >= 3` | [Main.cpp L10223](../juce/src/Main.cpp#L10223) |
| `--smoke-note-waveform` | `size >= 3` | [Main.cpp L10422](../juce/src/Main.cpp#L10422) |
| `--smoke-utau-waveform` | `size >= 3` | [Main.cpp L10728](../juce/src/Main.cpp#L10728) |
| `--smoke-voicebank-index` | `size >= 1` | [Main.cpp L10902](../juce/src/Main.cpp#L10902) |
| `--smoke-ust-pitch` | `size >= 1` | [Main.cpp L11368](../juce/src/Main.cpp#L11368) |
| `--smoke-hf-daemon-owner` | `size >= 3` | [Main.cpp L11792](../juce/src/Main.cpp#L11792) |
| `--smoke-hf-prewarm` | `size >= 2` | [Main.cpp L11805](../juce/src/Main.cpp#L11805) |
| `--smoke-ust-vibrato` | `size >= 1` | [Main.cpp L11938](../juce/src/Main.cpp#L11938) |
| `--smoke-mcp-roots` | `size >= 1` | [Main.cpp L12183](../juce/src/Main.cpp#L12183) |
| `--smoke-mcp-schema` | `size >= 1` | [Main.cpp L12290](../juce/src/Main.cpp#L12290) |
| `--smoke-dragged-transition` | `size >= 1` | [Main.cpp L12455](../juce/src/Main.cpp#L12455) |
| `--smoke-forward-bend` | `size >= 1` | [Main.cpp L12678](../juce/src/Main.cpp#L12678) |
| `--smoke-midi-track-import` | `size >= 1` | [Main.cpp L12850](../juce/src/Main.cpp#L12850) |
| `--smoke-midi-export` | `size >= 1` | [Main.cpp L13179](../juce/src/Main.cpp#L13179) |
| `--smoke-ust-import` | `size >= 1` | [Main.cpp L13472](../juce/src/Main.cpp#L13472) |
| `--smoke-ust-encoding` | `size >= 2` | [Main.cpp L13613](../juce/src/Main.cpp#L13613) |
| `--smoke-ust` | `size >= 2` | [Main.cpp L13765](../juce/src/Main.cpp#L13765) |
| `--smoke-active-resampler` | `size >= 1` | [Main.cpp L13970](../juce/src/Main.cpp#L13970) |
| `--smoke-bundled-resampler` | `size >= 1` | [Main.cpp L14015](../juce/src/Main.cpp#L14015) |
| `--smoke-anchor-frequency` | `size >= 1` | [Main.cpp L14073](../juce/src/Main.cpp#L14073) |
| `--smoke-note-oto` | `size >= 1` | [Main.cpp L14259](../juce/src/Main.cpp#L14259) |
| `--smoke-prefix-jie` | `size >= 1` | [Main.cpp L14689](../juce/src/Main.cpp#L14689) |
| `--smoke-prefix-note` | `size >= 1` | [Main.cpp L14843](../juce/src/Main.cpp#L14843) |
| `--smoke-utau-overlap` | `size >= 1` | [Main.cpp L15200](../juce/src/Main.cpp#L15200) |
| `--smoke-note-stp` | `size >= 1` | [Main.cpp L15515](../juce/src/Main.cpp#L15515) |
| `--smoke-track-toggle-tips` | `size >= 1` | [Main.cpp L15754](../juce/src/Main.cpp#L15754) |
| `--smoke-track-area-menu` | `size >= 1` | [Main.cpp L16001](../juce/src/Main.cpp#L16001) |
| `--smoke-stretch-items` | `size >= 1` | [Main.cpp L16102](../juce/src/Main.cpp#L16102) |
| `--smoke-render-order` | `size >= 1` | [Main.cpp L16154](../juce/src/Main.cpp#L16154) |
| `--smoke-llsm2-length` | `size >= 1` | [Main.cpp L16286](../juce/src/Main.cpp#L16286) |
| `--smoke-glide-seam` | `size >= 1` | [Main.cpp L16338](../juce/src/Main.cpp#L16338) |
| `--smoke-native-envelope-base` | `查看分支` | [Main.cpp L16402](../juce/src/Main.cpp#L16402) |
| `--smoke-nsf-utau-phrase` | `查看分支` | [Main.cpp L16486](../juce/src/Main.cpp#L16486) |
| `--smoke-nsf-utau-synth` | `查看分支` | [Main.cpp L16518](../juce/src/Main.cpp#L16518) |
| `--smoke-nsf-utau-f0` | `查看分支` | [Main.cpp L16566](../juce/src/Main.cpp#L16566) |
| `--smoke-nsf-utau-mix` | `查看分支` | [Main.cpp L16611](../juce/src/Main.cpp#L16611) |
| `--smoke-nsf-utau-plan` | `查看分支` | [Main.cpp L16676](../juce/src/Main.cpp#L16676) |
| `--smoke-render-capability` | `查看分支` | [Main.cpp L16733](../juce/src/Main.cpp#L16733) |
| `--smoke-oto-hjm-discipline` | `size >= 2` | [Main.cpp L16784](../juce/src/Main.cpp#L16784) |
| `--smoke-asset-register-dir` | `size >= 2` | [Main.cpp L16854](../juce/src/Main.cpp#L16854) |
| `--smoke-asset-manager` | `查看分支` | [Main.cpp L16898](../juce/src/Main.cpp#L16898) |
| `--smoke-gap-menu` | `size >= 2` | [Main.cpp L17018](../juce/src/Main.cpp#L17018) |
| `--smoke-insert-gap` | `size >= 2` | [Main.cpp L17137](../juce/src/Main.cpp#L17137) |
| `--smoke-ripple-delete` | `size >= 2` | [Main.cpp L17253](../juce/src/Main.cpp#L17253) |
| `--smoke-paste-notes` | `size >= 2` | [Main.cpp L17342](../juce/src/Main.cpp#L17342) |
| `--smoke-consonant-edge` | `size >= 2` | [Main.cpp L17518](../juce/src/Main.cpp#L17518) |
| `--smoke-consonant-reset` | `size >= 2` | [Main.cpp L17674](../juce/src/Main.cpp#L17674) |
| `--smoke-region-split` | `size >= 3` | [Main.cpp L17781](../juce/src/Main.cpp#L17781) |
| `--smoke-vibrato-presets` | `查看分支` | [Main.cpp L17873](../juce/src/Main.cpp#L17873) |
| `--smoke-follow-rule` | `查看分支` | [Main.cpp L17943](../juce/src/Main.cpp#L17943) |
| `--smoke-play-until` | `size >= 2` | [Main.cpp L18015](../juce/src/Main.cpp#L18015) |
| `--smoke-flags-field` | `size >= 3` | [Main.cpp L18074](../juce/src/Main.cpp#L18074) |
| `--smoke-vertical-drag` | `size >= 2` | [Main.cpp L18139](../juce/src/Main.cpp#L18139) |
| `--smoke-scroll-sync` | `size >= 2` | [Main.cpp L18232](../juce/src/Main.cpp#L18232) |
| `--smoke-edit-follow` | `size >= 2` | [Main.cpp L18313](../juce/src/Main.cpp#L18313) |
| `--smoke-note-drag-threshold` | `size >= 2` | [Main.cpp L18413](../juce/src/Main.cpp#L18413) |
| `--smoke-roll-drag` | `size >= 2` | [Main.cpp L18488](../juce/src/Main.cpp#L18488) |
| `--smoke-playhead-band` | `size >= 2` | [Main.cpp L18619](../juce/src/Main.cpp#L18619) |
| `--smoke-roll-cull` | `size >= 2` | [Main.cpp L18672](../juce/src/Main.cpp#L18672) |
| `--smoke-roll-timing` | `size >= 2` | [Main.cpp L18745](../juce/src/Main.cpp#L18745) |
| `--smoke-envelope-shapes` | `size >= 1` | [Main.cpp L18791](../juce/src/Main.cpp#L18791) |
| `--smoke-envelope-presets` | `size >= 2` | [Main.cpp L19114](../juce/src/Main.cpp#L19114) |
| `--smoke-piano-roll` | `size >= 2` | [Main.cpp L19152](../juce/src/Main.cpp#L19152) |
| `--render-project` | `size >= 3` | [Main.cpp L19200](../juce/src/Main.cpp#L19200) |
| `--inspect-project` | `size >= 2` | [Main.cpp L19277](../juce/src/Main.cpp#L19277) |
| `--inspect-settings` | `查看分支` | [Main.cpp L19313](../juce/src/Main.cpp#L19313) |
| `--smoke-oto-spectrum` | `size >= 2` | [Main.cpp L19340](../juce/src/Main.cpp#L19340) |
| `--smoke-game-defaults` | `size >= 2` | [Main.cpp L19347](../juce/src/Main.cpp#L19347) |
| `--inspect-analysis` | `查看分支` | [Main.cpp L19353](../juce/src/Main.cpp#L19353) |
| `--inspect-fcpe` | `size >= 3` | [Main.cpp L19375](../juce/src/Main.cpp#L19375) |
| `--smoke-export` | `size >= 3` | [Main.cpp L19401](../juce/src/Main.cpp#L19401) |
| `--smoke-mld5` | `size >= 4` | [Main.cpp L19431](../juce/src/Main.cpp#L19431) |
| `--smoke-utau-selection` | `size >= 2` | [Main.cpp L19524](../juce/src/Main.cpp#L19524) |
| `--smoke-utau-voicebank` | `size >= 2` | [Main.cpp L19638](../juce/src/Main.cpp#L19638) |
| `--smoke-utau` | `查看分支` | [Main.cpp L19721](../juce/src/Main.cpp#L19721) |
| `--inspect-midi` | `size >= 2` | [Main.cpp L19807](../juce/src/Main.cpp#L19807) |
| `--inspect-audio` | `size >= 2` | [Main.cpp L19830](../juce/src/Main.cpp#L19830) |
| `--inspect-mpd` | `size >= 2` | [Main.cpp L19865](../juce/src/Main.cpp#L19865) |

## Python 集成测试

位于 `juce/tests/`。运行前读脚本参数；真实模型、外置引擎和运行中窗口要求不同。

- [advanced_envelope_live.py](../juce/tests/advanced_envelope_live.py)
- [advanced_envelope_settings_live.py](../juce/tests/advanced_envelope_settings_live.py)
- [consonant_untouched_smoke.py](../juce/tests/consonant_untouched_smoke.py)
- [diffsinger_expressions.py](../juce/tests/diffsinger_expressions.py)
- [diffsinger_inference.py](../juce/tests/diffsinger_inference.py)
- [diffsinger_parameters.py](../juce/tests/diffsinger_parameters.py)
- [diffsinger_pronunciation_retake.py](../juce/tests/diffsinger_pronunciation_retake.py)
- [diffsinger_runtime.py](../juce/tests/diffsinger_runtime.py)
- [diffsinger_smoke.py](../juce/tests/diffsinger_smoke.py)
- [diffsinger_timing.py](../juce/tests/diffsinger_timing.py)
- [diffsinger_variance_retake.py](../juce/tests/diffsinger_variance_retake.py)
- [envelope_stretch_smoke.py](../juce/tests/envelope_stretch_smoke.py)
- [envelope_tail_smoke.py](../juce/tests/envelope_tail_smoke.py)
- [export_tracks_smoke.py](../juce/tests/export_tracks_smoke.py)
- [flag_curve_smoke.py](../juce/tests/flag_curve_smoke.py)
- [flag_precedence_smoke.py](../juce/tests/flag_precedence_smoke.py)
- [flag_rerender_smoke.py](../juce/tests/flag_rerender_smoke.py)
- [game_default_smoke.py](../juce/tests/game_default_smoke.py)
- [game_fcpe_smoke.py](../juce/tests/game_fcpe_smoke.py)
- [hamood_context_live.py](../juce/tests/hamood_context_live.py)
- [hamood_live_smoke.py](../juce/tests/hamood_live_smoke.py)
- [hifisampler_curve_smoke.py](../juce/tests/hifisampler_curve_smoke.py)
- [mcp_live_smoke.py](../juce/tests/mcp_live_smoke.py)
- [mcp_smoke.py](../juce/tests/mcp_smoke.py)
- [melodyne_import_smoke.py](../juce/tests/melodyne_import_smoke.py)
- [mou_mixed_counts_smoke.py](../juce/tests/mou_mixed_counts_smoke.py)
- [mou_oto_smoke.py](../juce/tests/mou_oto_smoke.py)
- [native_pitch_smoke.py](../juce/tests/native_pitch_smoke.py)
- [native_source_pitch_smoke.py](../juce/tests/native_source_pitch_smoke.py)
- [native_unedited_audio_smoke.py](../juce/tests/native_unedited_audio_smoke.py)
- [nsf_hifigan_model_smoke.py](../juce/tests/nsf_hifigan_model_smoke.py)
- [project_safety_live.py](../juce/tests/project_safety_live.py)
- [region_flag_smoke.py](../juce/tests/region_flag_smoke.py)
- [rest_lyric_smoke.py](../juce/tests/rest_lyric_smoke.py)
- [splice_smoke.py](../juce/tests/splice_smoke.py)
- [ust_fidelity_live.py](../juce/tests/ust_fidelity_live.py)
- [utau4_mode_smoke.py](../juce/tests/utau4_mode_smoke.py)

## 构建与通用工具

| 文件 | 用途 |
| --- | --- |
| [CMakeLists.txt](../juce/CMakeLists.txt) | C++ 目标、依赖、资源和版本 |
| [build-local.ps1](../build-local.ps1) | 本机 MSVC 增量构建 |
| [build-windows-native.ps1](../build-windows-native.ps1) | MSYS2 clang-cl 与 xwin 路线 |
| [windows-build.yml](../.github/workflows/windows-build.yml) | Windows 无模型构建与启动验证 |
| [.gitlab-ci.yml](../.gitlab-ci.yml) | GitLab 构建与配布 |
| [check_mcp_roots.py](../tools/check_mcp_roots.py) | MCP 根路径检查 |
| [check_mcp_schemas.py](../tools/check_mcp_schemas.py) | MCP schema 检查 |
| [integration-package-smoke.py](../tools/integration-package-smoke.py) | 配布包检查 |
| [integration-regression.py](../tools/integration-regression.py) | 整合回归驱动 |
| [make_fixtures.py](../tools/make_fixtures.py) | 测试素材生成 |
| [make_mcp_doc.py](../tools/make_mcp_doc.py) | MCP 文档生成 |
| [make_pinyin_table.py](../tools/make_pinyin_table.py) | 拼音表生成 |
| [package-current-distribution.py](../tools/package-current-distribution.py) | 历史配布归档；默认仍指向旧 0.2.2，先核对路径 |
| [organize-distribution.py](../tools/organize-distribution.py) | 配布根目录整理；默认预览，`--apply` 将验证报告和旧 EXE 备份归入子目录，校验内容并记录迁移 |
| [prepare-diffsinger-pronunciation.py](../tools/prepare-diffsinger-pronunciation.py) | DS 发音数据准备 |
| [prepare-note-dance.py](../tools/prepare-note-dance.py) | 音符舞动资源准备 |
| [prepare-nsf-model.py](../tools/prepare-nsf-model.py) | NSF 模型准备 |
| [regress.sh](../tools/regress.sh) | 回归驱动 |
| [run_breaks.sh](../tools/run_breaks.sh) | 边界／分段开发检查 |
| [verify_midi.py](../tools/verify_midi.py) | MIDI 验证 |

## HAMOOD 音频工作流脚本

运行时核心是 worker.py、runtime.json 和 client.py；其余为分析、KARA2、连续和声、东京泰迪熊等任务与配布验证入口。带具体歌曲名称的脚本可能硬编码工程和输出路径，不应直接用于新工程。

- [analyse_kara2.py](../tools/hamood_audio/analyse_kara2.py)
- [analyse_reference.py](../tools/hamood_audio/analyse_reference.py)
- [apply_reference.py](../tools/hamood_audio/apply_reference.py)
- [arrange_continuous_harmony.py](../tools/hamood_audio/arrange_continuous_harmony.py)
- [build_continuous_harmony.py](../tools/hamood_audio/build_continuous_harmony.py)
- [build_kara2_project.py](../tools/hamood_audio/build_kara2_project.py)
- [client.py](../tools/hamood_audio/client.py)
- [continuous_harmony_artifacts.py](../tools/hamood_audio/continuous_harmony_artifacts.py)
- [corroborate_kara2.py](../tools/hamood_audio/corroborate_kara2.py)
- [deploy_local.py](../tools/hamood_audio/deploy_local.py)
- [export_tokyo.py](../tools/hamood_audio/export_tokyo.py)
- [export_tokyo_cached.py](../tools/hamood_audio/export_tokyo_cached.py)
- [finish_tokyo_audio.py](../tools/hamood_audio/finish_tokyo_audio.py)
- [generate_tokyo.py](../tools/hamood_audio/generate_tokyo.py)
- [generate_tokyo_final.py](../tools/hamood_audio/generate_tokyo_final.py)
- [kara2-model.json](../tools/hamood_audio/kara2-model.json)
- [kara2.cmd](../tools/hamood_audio/kara2.cmd)
- [kara2.py](../tools/hamood_audio/kara2.py)
- [kara2_artifacts.py](../tools/hamood_audio/kara2_artifacts.py)
- [package_full.py](../tools/hamood_audio/package_full.py)
- [prepare_full_package.py](../tools/hamood_audio/prepare_full_package.py)
- [prepare_reference.py](../tools/hamood_audio/prepare_reference.py)
- [prepare_tokyo.py](../tools/hamood_audio/prepare_tokyo.py)
- [recheck_chorus_artifacts.py](../tools/hamood_audio/recheck_chorus_artifacts.py)
- [recheck_chorus_filters.py](../tools/hamood_audio/recheck_chorus_filters.py)
- [recheck_chorus_report.py](../tools/hamood_audio/recheck_chorus_report.py)
- [recheck_chorus_separation.py](../tools/hamood_audio/recheck_chorus_separation.py)
- [reference_polyphony.py](../tools/hamood_audio/reference_polyphony.py)
- [render_continuous_baseline.py](../tools/hamood_audio/render_continuous_baseline.py)
- [render_continuous_harmony.py](../tools/hamood_audio/render_continuous_harmony.py)
- [render_kara2.py](../tools/hamood_audio/render_kara2.py)
- [render_reference.py](../tools/hamood_audio/render_reference.py)
- [runtime.json](../tools/hamood_audio/runtime.json)
- [validate_tokyo.py](../tools/hamood_audio/validate_tokyo.py)
- [verify_installed.py](../tools/hamood_audio/verify_installed.py)
- [verify_kara2_portable.py](../tools/hamood_audio/verify_kara2_portable.py)
- [verify_portable_live.py](../tools/hamood_audio/verify_portable_live.py)
- [worker.py](../tools/hamood_audio/worker.py)
- [tempo_worker.py](../tools/hamood_audio/tempo_worker.py)：只运行现有 Beat This! small0 的伴奏曲速任务，不写工程。

## 资源和依赖

- `juce/resources/`：编辑器资源，由 CMake 引用。
- `juce/third_party/`：WORLD、LLSM2、ciglet 等依赖实现与许可。
- `.deps/` 和构建目录 `_deps/`：获取的 JUCE、ONNX Runtime 等依赖。
- `third_party/hifisampler/`：当前为 LICENSE 和 NOTICE，适配实现位于 backend。
- `models/`：本地权重及来源信息，不保证随源码包提供。
- `juce/engines/diffsinger/pronunciation-data/`：词典和供应的读音依赖。
- `tools/hamood_audio/deps/`、`vendor/`、`reference-deps/`：分析依赖与验证用第三方代码。

## 专题文档

- [chinese-cvvc.md](chinese-cvvc.md)：标准 UTAU 中文 CVVC 使用方法、规则、实现及当前边界。

部分专题记录历史设计或验证范围，当前行为先看总览与源码。

- [accompaniment.md](accompaniment.md)：伴奏轨道
- [advanced-envelope.md](advanced-envelope.md)：原生音频与 OTO 高级包络（0.25 / 内部更新 116）
- [audio-export.md](audio-export.md)：WAV 音频导出（0.25，文档更新 034）
- [diffsinger-audit.md](diffsinger-audit.md)：DiffSinger 使用完整性检查 · 0.2.3
- [diffsinger.md](diffsinger.md)：DiffSinger 兼容（0.25）
- [game-analysis.md](game-analysis.md)：GAME 默认识别（0.25 内部更新 060）
- [hamood-audio.md](hamood-audio.md)：HAMOOD 伴奏分析与和声 MCP
- [hamood.md](hamood.md)：HAMOOD 自动和声（0.25 / 内部更新 055）
- [hifisampler-flags.md](hifisampler-flags.md)：HiFisampler（PC-NSF-HiFiGAN）
- [integration-notes.md](integration-notes.md)：UTAU / Melodyne Integration Notes
- [kara2.md](kara2.md)：KARA2 主唱 / 背景和声分离
- [local-integration-20260924.md](local-integration-20260924.md)：HachiShifter 本机整合版 0.2.1
- [mcp-live.md](mcp-live.md)：MCP 连接当前编辑器窗口（0.25）
- [mcp.md](mcp.md)：HachiShifter Next 的 MCP 接口
- [nsf-hifigan-model.md](nsf-hifigan-model.md)：Native NSF-HiFiGAN model installation
- [pitch-timeline-fix-20260921.md](pitch-timeline-fix-20260921.md)：2026-09-21 跨音符音高时间线修复
- [playback-rendering.md](playback-rendering.md)：按播放位置优先渲染
- [tempo-change-options.md](tempo-change-options.md)：局部曲速的音符处理选项（0.2.2）
- [unified-editor-design.md](unified-editor-design.md)：Unified Native Editor — Design
- [updates.md](updates.md)：编辑器更新记录
- [ust-fidelity.md](ust-fidelity.md)：音源编码与 UST 数据保真
- [utau-feature-parity.md](utau-feature-parity.md)：UTAU Feature Parity — Checklist
- [utau-linear-flags.md](utau-linear-flags.md)：WCSNDM 线性 FLAG（0.25 · 更新 011）
- [utau-output-engine.md](utau-output-engine.md)：UTAU 输出引擎
- [windows-build.md](windows-build.md)：Native Windows Build (MSYS2 + xwin)

## 索引维护

新增源码补齐职责和链接；新增 CLI 核对参数并更新入口表。功能语义变化同步更新 AI_CODE_GUIDE.md。测试记录放到本次输出目录，不把“有入口”写成“已通过”。

更新 121：`juce/src/tests/NativeBatchFlattenSmoke.h` 由 `--smoke-flatten-pitch-line` 调用，验证原生多选框选后双击批量拉平与最近半音吸附（普通／标点工具），跨区域、选区保留、未选中隔离和整组撤销重做。

更新 122 新增：`juce/src/tests/NativeLinkedPitchSmoke.h`，入口 `--smoke-native-linked-pitch <输出目录>`；检查原生粘连音频的共享线、渲染一致性、控制点交互及缓存依赖。实现位于 `ProjectModel.cpp::nativeSharedPitchLines`、`ClipParts.h::expandProjectClipParts` 与 `AudioEngine.cpp::applyNativeSharedPitch/writeNativePitchContext`。

更新 123 新增：`juce/src/NativeSharedEnvelope.h`（原生组包络合并、插值、切片、物化）；`juce/src/tests/NativeLinkedEnvelopeSmoke.h`，入口 `--smoke-native-linked-envelope <输出目录>`，检查跨切口编辑、显示／请求一致性、源音高保留、撤销、保存重开、复制和断开。

更新 124 新增：`NativeTrimSource.h`；`NativeTrimExpandSmoke.h`，入口 `--smoke-native-trim-expand <输出目录>`，覆盖向外恢复原文件、分离后删除恢复、双向文件边界、非线性时钟、重叠限制、保存重开、旧工程后台音高补齐和实际 PCM 原音对比。

更新 125 新增：`NativeNoteJoin.h` 和 `NativeNoteJoinSmoke.h`；入口 `--smoke-native-note-join <输出目录>`，验证真实双击事件、无连接误切换、统一拉伸、源时钟、曲线、单步撤销、同源子素材、保存重开和实际 PCM。

更新 126：`NativeNoteTiming.h` 修正跨音符控制点的分段时钟；`ProjectModel::resizeClip` 修正原生整段拉伸的源锚点、响度与颤音。沿用上述合并专项及 `--smoke-native-note-move <输出目录>`，加入长拉伸、共享曲线、区域增益、子素材和源映射检查；设置 `HACHI_TEST_NSF_MODEL_DIR` 增加真实 NSF 长拉伸音频检查。

## 更新 127：原生导入性能复测

`--smoke-native-import-performance <测试音频副本.wav> <models目录>`：使用 GAME＋FCPE 实际分析，记录导入、源音高附着、渲染请求、完整离屏截图及可见区域软件重绘耗时；检查分析前原声逐采样一致和显式音高编辑不被跳过。请传入独立测试目录的音频与同名 HJM 副本，避免诊断流程接触正式素材标注。完整离屏截图包含图像后端开销，不能直接换算成实际编辑器帧率。实现：[NativeImportPerformanceSmoke.h](../juce/src/tests/NativeImportPerformanceSmoke.h)。

## 更新 128：拉直后的原生音高独立

`NativeBatchFlattenSmoke.h`（`--smoke-flatten-pitch-line`）新增实际显示曲线／渲染目标逐点独立平直及共享响度保持检查；`NativeLinkedPitchSmoke.h`（`--smoke-native-linked-pitch <输出目录>`）检查保存重开、后续单段编辑不牵动邻段，并生成 `independent-pitch.png`。

## 更新 129：原生独立音高的最小间隔连接

`NativeLinkedPitchSmoke.h`（`--smoke-native-linked-pitch <输出目录>`）检查边界控制点相隔 1 ms、短直线连续、主体平直、合成目标一致、空隙／重叠不连接和后续邻段独立。使用现有 `SharedPitchLine::Piece` 表达分段过渡；`nativeBoundaryAnchors` 仅用于派生显示和标点操作，不替代持久化原始控制点。

更新 130：`backend/HiFiShifterMelSmoke.inc` 是 `NsfHifiganRenderer.cpp` 包含的专项测试，覆盖上游参考插值、压缩／扩展／单帧、非均匀时钟／源跳接、尾部长度对齐与真实 ONNX 的 0.5／1／2／8 倍时长。CLI：`--smoke-hifishifter-mel <输出目录> <模型目录>`。

更新 131：`backend/SourceVoicing.h` 独立源周期性与清音保护掩码；`backend/NativeNoiseSmoke.inc` 覆盖清辅音／混合气声元音、FCPE 判定、真实 HN-SEP 与用户录音、源噪声时钟、未调音辅音 PCM 和音高变换。整句零 F0 回归见 `tests/NativeLinkedPitchSmoke.h`。

更新 132：`--smoke-flatten-pitch-line` 覆盖默认工具批量规整与标点工具加点／不规整／撤销；更新 121 中普通与标点都双击规整的行为已被替换。

更新 133：`NativeLinkedPitchSmoke.h` 增加端点水平拖动、无重生控制点、越过音频边缘、插值渲染及保存撤销回归。实际端点状态 `NoteData::nativePitchHandlesPlaced` 在 `ProjectModel` 写入和保存，`AudioEngine` 纳入缓存，`PianoRollComponent` 提供统一拖动约束。

更新 134：`backend/NsfPitchTransitionsSmoke.inc`（由 NSF 渲染源包含）检查突变平滑开关的数值和真实模型；`tests/NsfPickerSmoke.h` 检查非 U NSF 复选框、按轨道设置、撤销重做、保存重开、单段／整句请求和缓存失效。新命令 `--smoke-nsf-pitch-transitions <输出目录> <nsf模型目录>`。

更新 135：`NativePitchVoicingDisplay.*` 负责可选清辅音显示的后台源分析、缓存、非线性源时钟和曲线虚线分区；`tests/NativeConsonantDisplaySmoke.h` 由 `--smoke-native-consonant-display <输出目录> [真实录音]` 调用，覆盖旧工程 F0、气声元音、普通／标点视图、缺失 F0 断线（更新 142）、显示恢复、UTAU 隔离、工程不变、裁剪／拉伸及真实录音缓存。`--smoke-view-menu` 验证菜单第 8 项和 `ui.consonantPitchDashed` 设置的往返与默认关闭。
