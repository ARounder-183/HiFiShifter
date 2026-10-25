# HachiShifter 代码功能总览与维护指南

更新 215：UTAU 音符工具双击任意音符均进入歌词／别名编辑，不再限制空歌词。已有中文、带空格别名、休止符、未解析别名均可改；打开时回显并全选现有内容，Enter 提交、Esc 取消。原生 OTTO 和其他编辑工具不变。此行为替代更新 214 的“仅空歌词”限制。

更新 214：音符工具双击 UTAU 空歌词（含纯空白）音符恢复行内歌词输入，普通／界／谋模式、未选音源均可用。原生 OTTO 的双击对齐行为和其他工具保持不变。

更新 213：UTAU 编辑栏的“选择音源库…”右侧显示当前选中区域所属轨道绑定的音源文件夹名称，未绑定显示“未选择音源”，悬停显示完整路径。切换轨道或恢复工程时读取当前绑定，不使用全局默认值。

更新 212：轨道声像 `P` 数字支持单击输入 -1 到 1 的小数，-1 最左、0 居中、1 最右。Enter 应用、Esc 取消；每轨独立，支持撤销和工程保存，原声像滑杆保留。

更新 211：左侧轨道音量 dB 和区域增益数字支持单击输入。`DecibelGainDialog` 共用精确回显和严格数值校验，Enter 应用、Esc 取消，支持小数及 `-inf` 静音；空值、非数值和超范围输入不可应用。轨道上限沿用线性 2、区域上限沿用 4，有限负值最低 -120 dB。区域数字原拖动调节保留。

更新 210：左侧轨道名称栏支持拖动排序。5 像素阈值启动，显示插入线，松开提交一次撤销事务；边缘自动滚动，Esc 或横向移出面板取消。`ProjectModel::moveTrackBefore` 按稳定 ID 改变轨道向量顺序，轨道内容和工程序列化保持原有结构。

更新 209：UTAU 工具栏在发音器右侧提供“钢琴试听”，普通／界／谋轨道各自记住 `TrackData::utauPianoPreview`（默认 true），工程保存和撤销均保留。关闭仅静音空歌词及缺失样本的钢琴回退，正常样本合成不变；HiFisampler 同样生效，导出空歌词仍静音。状态参与渲染缓存身份。

更新 208：未选中音符时直接播放，`selectAllRenderedUtauNotes(project)` 将普通 UTAU 空歌词纳入钢琴预览范围，不再要求先框选以建立渲染历史。非空歌词合成继续使用已有历史；显式框选仍是精确试听范围，DiffSinger 家族验证和导出空歌词静音不变。播放入口专项见 `--smoke-piano-transport`，验证状态见更新记录。

更新 207：素材树右键菜单用鼠标屏幕坐标构造 1×1 锚点，修正误用双对角点矩形导致左上角错位。文件夹、工程音源及录音菜单提供“移除”，仅持久隐藏浏览项／取消手动登记，不删除磁盘文件、不解绑轨道；工程同步和搜索遵守隐藏记录，显式重新导入文件夹恢复其中录音。

更新 206：UTAU 空歌词在已加载音源时同样播放内置钢琴预览，显式休止仍静音；导出准备阶段将空歌词保持静音。`syncProject(..., forExport)` 区分预览／导出，空歌词片段缓存身份包含用途，GUI／MCP 导出结束恢复预览；HiFisampler 共用同一个钢琴音生成器。专项 `--smoke-empty-lyric-piano` 覆盖实际播放混音、WAV 导出和重复切换。

更新 205：移除 UTAU 模式切换和应用偏好时的默认音源自动绑定。未手动选音源的轨道保持空，历史 `algorithm.utauVoicebank` 仅作为显式选择目录时的初始位置。工程已有音源引用继续恢复，不清除作者绑定。钢琴预览仍是渲染器自动行为，空歌词在无音源时预览、有音源时休止；未解析的非空别名按原规则回退。

更新 204：恢复 UTAU 工具栏“选择音源库…”按钮，位于强制连接右侧；普通／界／谋三种模式共用原文件夹选择和轨道绑定入口。两按钮作为同组换行，窄窗口也保持相邻；悬停显示当前音源路径。

更新 203：左侧旋律音轨头右键增加“导入 MIDI…”，选择文件后确认一条 MIDI 轨道，在打开菜单时的播放线位置新增区域。保留已有音符、音源、引擎和工程速度；MIDI 开头空白保留，节拍从插入点按工程速度图换算，SMPTE 文件保留秒时间。原新建音轨导入入口保留。

更新 202：区域右键“拆分”按打开菜单时红色播放线的位置执行，不再按鼠标横坐标。菜单可用性与实际拆分使用同一捕获时间；红线不在可拆分范围内时禁用。

更新 201：区域右键菜单增加带勾选状态的“静音”，仅切换被点击片段，复用 `ProjectModel::setClipMuted`，与原 M 按钮、播放、撤销及工程保存共用状态。

更新 200：区域右键“编辑…”提供原长度字段及整轨输出引擎选择，原外部引擎菜单保留。重采样器配置沿用现有面板，先编辑草稿，主窗口应用时经验证写入；长度继续只改变可编辑范围。

更新 199：原生单音节详情允许直接拖两端紫线，重合标记时外边界优先命中并置顶绘制；显式保存到 OTTO 后重开保持范围。原生多音节详情始终锁定外边界，不再按当前行数临时解锁。

更新 198：单音节音素控件和提示文字补齐 I18n 翻译；仅原生标注总览／详情显示，三种 UTAU 视图均隐藏，录音偏好保持。

更新 197：共用素材编辑器顶栏提供每录音独立记忆的“单音节音素”，勾选跳过总览，取消恢复总览；已标注录音仅切换视图，不合并发音。未标注录音显式勾选后建立单发音草稿，保存仍经原 OTTO 会话和预检。偏好保存于应用设置而非源标注。

更新 196：音符右键“编辑素材标注…”定位后停留在原生发音总览，直接提供起止范围编辑；已有详情窗口也返回总览并选择新目标。内部元音字段未改动时不做重新命名校验，保留默认占位名。

更新 195：从原生音符右键进入素材标注会自动定位到原录音发音区域；已有标注 ID 优先，旧工程未绑定 ID 时使用实际源时间映射在当前标注版本中定位唯一发音。无法唯一匹配时显示原音频选区，保持草稿、工程和 OTTO 不被自动改写；复用窗口也会跟随新的右键目标。

更新 194：原生音符右键的“编辑素材标注…”直接复用素材管理器原生编辑器。菜单携带被点击音符的 ID，经现有源录音解析保留合并片段归属及标注版本／发音身份；普通未标注录音同样支持，无需先在素材树中查找。

更新 193：外部录音导入后自动打开素材管理器，并以“工程素材”显示实际引用的录音，可右键“编辑素材标注…”直接使用原生编辑器。`AssetManagerComponent::setProjectVoicebanks` 同时同步音源和工程录音（含嵌套合并 parts），引用不变时不重建列表。未登记的父目录仅显示实际引用文件，不扫描其他录音、不复制 PCM、不自动创建 OTTO 或写入全局登记；已有素材根复用。列表从工程源引用恢复，删除／撤销／重新打开自动同步。标注仅在用户保存时通过既有 `MaterialAnnotationSession` 写入 `otto.ini`。

更新 191：区域右键“编辑长度…”仅调整可编辑外框末端，音符位置／时长、曲线及录音时钟不变。主入口 `ProjectModel::setClipEditableLength`，与 `resizeClip` 拉伸、`trimClip` 裁剪区分。缩短时保留区域外内容，保存重开不裁掉显式音频窗口，可再次延长恢复范围。专项 `--smoke-clip-length`。

**Active implementation (2026-10-20):** Native annotations are moving from per-audio HJM CSV writes to compact folder-level `otto.ini`. Read [implementation status](native-material-implementation.md) and [format specification](otto-format.md) before changing material loading, source-update or asset insertion paths. Version stays 0.25. Legacy sections below describe historical behavior and are not authority for new annotation writes. The complete approved consolidation is not finished yet.

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

更新 176：高级包络混合模式的节点在点击／拖动并松开后保持选中高亮，Delete／Backspace 调用既有 deleteKnot 删除该节点；音头、音尾、原生音频及 CVVC 音素共用。选择与拖动状态分离，点击其他位置／面板控件、失焦、切换形状／预览、替换草稿时清除选择。连按 Delete 不继续删除其他节点，首尾边界点和贝塞尔控制柄沿用原有规则。删除只改草稿，应用时走原有提交入口，关闭不应用则不改工程。

更新 175：响度包络和线性 FLAG 包络沿用音高标点的持久选择；点击或拖动后松开仍高亮，Delete／Backspace 删除所选标点，继续按键不会转而删除整音符。统一 PointSelection 记录类型、音符、点序号、CVVC 分片及 FLAG 键；点击其他位置、切换工具／FLAG／选区、失焦时清除。右键删除与快捷键共用 deleteEnvelopePoint，响度沿用已有选区／粘连组映射，FLAG 只改当前参数，CVVC 只改对应音素的对应包络；保留原来的响度／CVVC 两端与 FLAG 最后一个点。普通响度及 FLAG 纯点击不再提交工程编辑。

更新 174：音高标点的选中状态独立于拖动状态；点击并松开或拖动后松开，标点继续高亮，Delete／Backspace 优先删除该点。点击其他位置、切换工具／轨道／片段、失去键盘焦点或重新选择音符时清除。删除使用与标点右键菜单相同的 deletePitchAnchor 入口，沿用端点保留和 DiffSinger 隐藏采样整理规则；通过既有 setNotePitchCurve 提交，支持撤销、重做及工程保存。删除完成或版本过期后保留按键消费状态直到下一次选择，防止连续 Delete 误删整音符。选择本身不改工程、不触发合成。

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

2026-10-13 增补：[实际功能主入口、数据流与架构审计](AI_INTERFACE_ARCHITECTURE.md) 按业务能力对应权威接口，记录重复快照、音频回调、大锁、缓存键、请求采样和 MCP 草稿问题及最小改造顺序；[接口声明清单](AI_API_REFERENCE.md) 列出核心公开方法、MCP 和启动参数。此次为静态核对，不代表性能优化已经实施；下面更新 148 的行为记录继续有效。

更新日期：2026-10-15。实现基线：公开版本 **0.25**、内部更新 **176**、外置 WCSNDM **0.0803**。本指南覆盖当前整合版的界面、工程数据、音频编辑、UTAU、HiFisampler、DiffSinger、分析、和声、MCP、构建与验证。

更新 163：单独选中 CVVC 音素后，“高级包络”不再被提前返回拦截。PianoRollComponent::selectedCvvcEnvelope 提供当前音素的范围与曲线；MainComponent 复用 AdvancedEnvelopePanel 的整段编辑界面，应用仍统一调用 setCvvcEnvelope，仅提交该角色。包络点可保存可选贝塞尔控制柄，cvvcEnvelopeValue 与两个 UTAU 渲染器共用同一插值，曲线加入混音缓存键。打开／未改动应用不写工程；草稿应用、试听、修订保护、保存和撤销保留。恢复默认使用该音素标准梯形。

更新 162：修复旧 CVVC 工程保存的整音符默认分贝包络继续叠乘，造成句首弯曲起音和收尾弯曲淡出。ProjectModel::fromValueTree 仅对缺少 cvvcEnvelopeVersion、无高级包络及 UST 原始段的中文 CVVC 音符，识别四点 0 dB 平台、5/15 ms 起音、35 ms 收尾的历史默认形状，恢复为各音素独立标准梯形。新工程保存 cvvcEnvelopeVersion=1；现代显式曲线、非标准时长／音量、线性手绘和非 CVVC 曲线保持。音频与拼字混音缓存版本同步更新。真实“CVVC测试”工程专项覆盖修复前失败、保存重开、首尾像素位置、两种交叉方式逐采样增益及 HiFiSampler／WCSNDM 合成。

更新 161：CVVC 默认梯形与“标准”预设统一使用 standardCvvcEnvelope 的 5 ms／35 ms 线性幅度规则。自动混音 useStandardUtauEnvelope 在重叠边缘接管标准淡入／淡出，避免手动默认梯形与自动层重复相乘产生转折；保留非标准手动曲线。发音器菜单新增“默认交叉方式→直线／曲线”，TrackData::cvvcCurveCrossfade 随工程保存、撤销，并参与声音与混音元数据缓存键。曲线为互补 smoothstep，两端斜率为零；传统 UTAU 与 HiFiSampler 共享增益描述。replacesStandardEnvelope 标记让上下包络显示使用同一实际增益。整音符 CVVC 预设按音素批量提交 setCvvcEnvelopes，原单片接口委托此入口，一次撤销。

更新 160：CVVC 音符上方增加 VC／CV／尾音可点击标签，复用父音符 ID 加 envelopePart 的选区，不生成独立假音符。选中音素时响度预设只提交 setCvvcEnvelope，响度／FLAG 面板仅保留当前音素的编辑曲线和控制点，其他音素淡显实际曲线；点击主体或显式整音符选择恢复整音符操作。上方包络改为逐音素绘制实际自动混音增益，移除父音符包络的重复外形。音素选择时高级包络入口聚焦分片响度编辑，整音符 OTO 设置需点击主体。选择不写工程；曲线复用既有保存、撤销和渲染入口。

更新 159：`mixNsfUtauNotes` 的自动淡出完成后保持零增益，消除前一 VC/CV 在交叉区后恢复音量的错误；负 OTO 重叠按间隔提前结束前音并使用 3 ms 去爆音，长正重叠按相邻音符时序为前音补足尾部。`UtauMixEnvelope` 的显示元数据与实际混音使用同一持久淡出规则；手动包络仍先应用再乘自动层。HiFiSampler 音频缓存与 CVVC 混音元数据版本更新。专项 `--smoke-hifisampler-envelope <输出目录> [工程] [模型] [重采样器]` 覆盖多音重叠、短后音、零／负重叠及用户工程只读合成；现有 CVVC 专项增加真实模型淡出保持及跨节拍长重叠。

更新 158：CVVC 音头手柄按已解析的 VC／CV／收尾 OTO 与相邻音符时长构造纯时序求值器，通过二分反求整数辅音速度；拖动过程不读文件、不跑发音器或声码器。`cvvcTransitionDuration` 统一 VC 提前量规则，`scaledUtauPreutterance` 统一源时长封顶和固定辅音区缩放，移除 CVVC 独有的 0–200 限制；松手等待后台布局时保留已提交的预览音头位置。旧拼字与音频派生缓存按时序规则版本失效。`tests/CvvcConsonantDragSmoke.h` 随中文 CVVC 专项验证负值、快速辅音、饱和上限、实际渲染、撤销及松手前后位置。

更新 157：`NoteData::cvvcPhonemeCache` 保存经哈希校验的派生拼字布局；`backend/CvvcPhonemeCache.h` 负责有界 JSON 读写。`ProjectModel::rememberCvvcPhonemes` 静默存储、不增加 revision／undo；工程和恢复文件序列化包含缓存，内容指纹与音频缓存键不包含。`PianoRollComponent::requestCvvcLayouts` 只在后台调用现有发音器补缺，`rebuildUtauSoundSpans` 优先恢复已保存布局，再接收有效渲染元数据。“显示→拼字信息”对应偏好 `ui.showPhonemeInfo`，开启后所有音符显示实际音素别名，关闭后仅选中项。验证见 `--smoke-cvvc-layout-cache <测试音源> <输出目录>`、`--smoke-chinese-cvvc`、`--smoke-view-menu`。

更新 156：“自动交叉淡化”全模式按钮 → `PianoRollComponent::crossfadeSelectedNotes` 从现有发声布局收集未乘自动混音的曲线 → `ProjectModel::crossfadeNotes` 按重叠区统一写入普通响度／CVVC 分片响度。只选同轨两个前后重叠音符，一次撤销，修订号校验，无重叠不移动音符，重复执行不累积衰减。原有基准音量和 OTO 自动混音层保持生效。验证入口 `--smoke-auto-crossfade` 和 `--smoke-chinese-cvvc`，详见更新记录 156。

更新 155：`backend::standardCvvcEnvelope` 统一生成各 CV／VC／收尾的默认四点直线梯形（5 ms 起音、35 ms 释放，短片段用三等分避免点序倒置）。`planChineseCvvc` 与 `PianoRollCvvcEnvelope.h` 复用该函数；仅在无手动父包络、无分片响度覆盖时使用。已有手动曲线及导入曲线保留，默认值不写回工程。`cvvcDefaultEnvelopeGain` 传递默认包络的基准音量，防止调节音量后被误判为手工平直曲线；CVVC 渲染键升级至 v3。

更新 154：`backend/UtauMixEnvelope.h` 保存渲染器实际使用的采样级自动淡化窗口，传统 UTAU 与 HiFisampler 混音、CVVC 响度可视化共享增益计算。`noteMixEnvelope` 经发音器折回 `UtauPhonemeSpan::mixEnvelope`，只读数据不写工程；实线显示手动包络乘自动淡化，虚线／圆点保留编辑值，VC／CV／收尾用不同颜色，交叉区着色。HiFisampler CVVC 波形回调移到已确定自动混音增益之后，避免预览缺少淡化。`utauPhonemeMixContextHash` 防止邻音时序或引擎变化后继续显示过期交叉曲线；包络／FLAG 编辑保留布局和有效淡化。

更新 153：标准中文 CVVC 的 VC、CV／VCV、收尾具有独立响度／FLAG 曲线。`NoteData::cvvcEnvelopes` 按稳定角色与参数存储归一化曲线，`ProjectModel::setCvvcEnvelope` 负责修订校验和一次撤销提交；`backend/CvvcEnvelope.h` 提供共用类型及持久化。`planChineseCvvc` 将曲线映射到实际生成片段，两个输出后端共用。`PianoRollCvvcEnvelope.h` 是同一钢琴卷帘的片段绘制和鼠标实现，不创建假音符 ID 或另一份工程；`UtauNoteWaveform::phonemeLayoutHash` 让包络修改期间保留有效布局。分片继承原曲线，只有编辑过的角色产生覆盖。具体使用和边界见 [chinese-cvvc.md](chinese-cvvc.md)。

更新 152：标准 UTAU 的拉伸算法下方增加发音器按钮，`phonemizerMenu` 是按钮与右键菜单的共同菜单定义，选择仍走 `outputEngineItemChosen`／`setTrackChineseCvvc`。中文 CVVC 扩展 `ChineseCvvcRules::aliases`，统一生成配置指定的 CV、VC、VCV、句首、收尾候选，真实 OTO 决定是否可用；支持分隔符、代表元音、旧式 ENDTYPE 与逐音节 PRIORITY。不要在界面复制发音器状态，不按录音表品牌增加平行算法。完整范围见 [chinese-cvvc.md](chinese-cvvc.md)。

更新 151：素材管理器导入音源直接复用 `AssetManagerComponent::registerFolder`，只登记目录并枚举成员；编辑／渲染按需读 OTO。不再使用只为统计而逐 WAV 重读 OTO 的 `SampleSettings::importVoicebank`；该旧接口仅保留在独立 MCP 库存命令／历史测试中，不应接回 GUI。导入完成提示改为真实音频数量，不声称生成 HJM。`--smoke-voicebank-import 输出目录 音源目录` 验证同一登记入口耗时、去重、持久化及不构建音源索引。

中文 CVVC 的 `ChineseCvvcRules::spellingVariants/lookup` 和 `planChineseCvvc` 在本音源定义及真实 OTO 的约束下兼容 nve/nue、lve/lue 与 ü 输入，精确别名及显式 REPLACE 优先。不得强制合并各录音方案的元辅音标记；兼容边界、公开样本来源和测试记录见 [chinese-cvvc.md](chinese-cvvc.md)。

更新 150：手动选择与默认音源通过 `MainComponent::selectVoicebank` 绑定。普通 UTAU 不调用整库 `SampleSettings::importVoicebank`，仅更新轨道和默认目录，OTO 仍按需使用后台索引；选择状态栏只显示名称。真实音源回归：`--smoke-voicebank-selection 输出目录 音源目录`，包含消息线程响应、撤销重做、重复选择与轨道隔离；不写音源。

更新 148：素材管理器音频项改为 24 像素单行，只显示一次相对文件名；音源文件夹保持 40 像素名称／路径双行。`TreeRow` 缓存每行位置和高度，绘制、命中与滚动范围统一使用这些坐标，保持展开／收起、拖拽及参数编辑行为。

更新 147：素材管理器改为全宽可折叠文件列表；音源左侧箭头展开／收起，双击文件夹名称也可切换。展开后列出包含子目录在内的音频文件，显示文件名和相对路径；单击选中、拖拽使用素材、双击编辑参数。多个文件夹可同时展开，滚轮与纵向滚动条浏览长音源库。`AssetManagerComponent` 缓存已展开成员，绘制只遍历可见行；工程更新保留现存文件夹展开状态，换工程清除失效引用。扩展 `tests/ProjectVoicebankAssetsSmoke.h` 验证窄面板点击、展开／收起、子目录、滚动末尾和工程切换。

更新 146：素材管理器自动列出当前工程轨道使用的 UTAU 音源文件夹，置顶并标记“工程音源”。`AssetManagerComponent::setProjectVoicebanks` 读取工程快照，按路径去重；`MainComponent` 在面板打开和工程变更通知时同步。包含 UTAU／HiFisampler／原生 NSF 的音源库合成轨道，排除伴奏、非 U 录音和 DiffSinger。工程引用不写全局 `assets.materialFolders`，也不复制音源或生成 HJM；手动登记过的同路径素材保留。新建／打开工程、切换音源和移除最后一个使用轨道时自动更新，缺失路径仍显示并提示重新定位。使用中的工程音源不可直接从面板移除，应在轨道上变更音源。列表不变时不重新扫描成员，其他文件夹变化保留当前选中素材。验证入口：`--smoke-project-voicebank-assets 输出目录`，另运行既有 `--smoke-asset-manager` 回归。

先读本页确定功能归属，再用 [逐文件索引](AI_SOURCE_INDEX.md) 找到实际源文件和命令分支。历史变更在 [updates.md](updates.md)。不要将历史评估中的“待实现”直接当成当前缺失项，也不要把一个测试文件存在等同于测试已通过。

更新 144：非 U NSF 拉伸菜单新增“WCSNDM 非循环拉伸”，`StretchAlgorithm::wcsNonLoop` 追加为值 6（菜单 ID 7，工程／MCP 键 `wcsndm-nonloop`）。`backend/WcsNonLoopFeatures.*` 对源 PCM 和源 F0 作 LLSM layer-1 分析，按既有单调时间映射插值 Rd、声道 dB 包络、噪声 PSD 与残差；在采样后的目标 F0 上重建谐波功率谱，再生成物理 Mel，交给已有 NSF ONNX。它不是给旧 Mel 拉伸换名，也不先合成一遍 PCM。没有新增 OTO／C-V 识别、固定辅音头或发音时长保护；普通整段均匀拉长会同时拉长声学变化，手动分段、裁剪和源跳跃仍遵循项目时钟。源零 F0 不插值成有声，原有谐波／噪声保护、音高突变平滑、响度、颤音及原音直通继续有效。

源参数分析使用 256 帧核心、两侧 16 帧上下文，块间和输出帧检查取消；同一源 PCM／F0／模型特征配置的参数保存于 64 MiB、最多 4 项缓存，目标音高改动可复用，程序退出时清空。超过 24000 源或目标模型帧显式报错，不替换用户选择的算法。物理 Mel 按原素材 STFT 标定电平，不将每次拉伸输出强行对齐固定响度。此路径在 C++ 内运行，不启动外置 WCSNDM EXE／Python daemon，不添加模型。专项命令 `--smoke-wcs-nonloop 输出目录 NSF模型目录 [真实录音]`；菜单、撤销、保存及单段／整句路由验证加入 `--smoke-nsf-picker`。

## 先确定目录和版本

| 位置 | 当前用途 |
| --- | --- |
| 本仓库根目录 | `E:/和声合成新UI/HachiShifter-integrated`，当前维护的整合版源码 |
| `juce/src/` | 自有 C++ 编辑器、工程模型、音频引擎、后端适配和原生测试 |
| `juce/engines/diffsinger/` | DiffSinger 的 Python 推理桥、后台运行时、发音和参数处理 |
| `juce/tests/` | Python 集成、音频、MCP 和真实模型测试 |
| `tools/` | 回归、模型准备、打包和开发辅助脚本；部分是历史快照脚本 |
| `tools/hamood_audio/` | 和声分析工作进程、伴奏分析、KARA2 与具体歌曲验证脚本 |
| `models/` | 本地模型资源，不保证包含在源码归档中 |
| `juce/third_party/`、`third_party/` | 算法依赖与来源许可；不是所有同名外部项目的完整源码 |
| `build-integrated/`、`.deps/` | 本地构建产物与依赖缓存，不作为业务源码修改入口 |
| `test-output/`、`integration-tests/` | 测试产物、报告、截图、音频与夹具；不要当作正式源素材 |
| `../HachiShifter-整合版-0.25-DiffSinger/` | 现用便携配布，EXE、运行库、模型、说明和构建信息 |
| `../HachiShifter-github-0.2.3-20261006/` | 曾用于团队仓库发布的独立工作副本；目录名不等于当前内容版本 |

当前用户要求保持公开版本 0.25，普通修复只增加文档内部更新号。版本约束见 [AGENTS.md](../AGENTS.md)。旧目录中的 0.2.3 名称、旧打包脚本默认值、旧日志不应驱动当前版本回退。

团队发布目标曾明确为 `openhachimi/HachiShifter` 的 `next`，不是用户个人 fork。每次发布仍需核对实际工作树、远端、分支与本次授权；本指南不是自动推送、删分支或发布的授权。

## 总体结构与调用方向

```text
Main.cpp                         程序启动、窗口、CLI、诊断命令
  MainComponent                  界面组合、菜单、工具选择、设置和任务协调
    Timeline / TrackList         上方轨道与区域
    PianoRoll                    下方音符、音高、包络、FLAG 和拖动预览
    ProjectModel                 正式编辑、撤销、快照、保存、加载
    AudioEngine                  渲染请求、缓存、试听、混音、波形、导出
      AnalysisService            GAME / FCPE / native-hq 原音分析
      RenderService              排队、取消、后端路由
        Native renderers         WORLD / LLSM2 / NSF 等录音调音
        UtauRenderer             OTO、传统重采样器、内置 HiFisampler
        DiffSingerRenderer       C++ 请求与 Python 后台进程桥
          diffsinger/*.py         音素、时长、pitch、variance、声学、声码器

MCP headless -> McpServer -> 独立 ProjectModel 与 AudioEngine
MCP live -> LiveMcpBridge -> MainComponentMcp -> 当前窗口真实工程
```

编辑通常走“鼠标或命令 → 预览草稿 → ProjectModel 一次提交 → change listener → AudioEngine 同步”。预览不得反复写入正式工程。新增声音相关字段一般要同时核对：数据结构、保存加载、复制粘贴、撤销、渲染请求、缓存键与波形失效。

更新 138：非 U NSF 的“谐波/噪声分离”按轨道控制 `TrackData::nsfNoiseProtection`，默认 true；旧工程缺字段也为 true，以延续更新 131 的声音。开启时只对 HN-SEP 分出的谐波变调，源噪声按源时间映射混回，明确清音恢复原 PCM；关闭时跳过这三项，对完整素材进入声码器。缺少 HN-SEP 模型时开启只提供明确清音保护。该值参与保存、撤销、单段／整句请求、音频与波形缓存，以及 MCP `nsf_noise_protection`。UTAU 不使用这个轨道开关，未编辑原音在两种状态下都保持 PCM 直通。

更新 139：非 U 参数栏“响度包络”旁增加普通按钮“颤音”，只有当前选中有效原生音符时可用，复用原有右键颤音对话框、预设及可拖动范围／深度／周期控制。支持多选，默认导入仍不开启。非 U 对话框增加实际音高线显示复选框，新颤音默认勾选；参数和显示作为同一次撤销写入 `setNotesVibrato(..., optionalRealLine)`，拖动等旧调用不覆盖显示选择。音高基线、源 F0、时间与响度不被覆写；单段和共享曲线／整句请求已在密集目标 MIDI 上叠加一次 `vibratoCentsAt`，保留零 F0 掩码。保护开启时清辅音沿用源波形保护。关闭颤音恢复基线，原音无其他编辑时恢复 PCM 直通。测试入口 `--smoke-native-vibrato 输出目录` 和 `--smoke-native-vibrato-render 输出目录 NSF模型目录`。

`RenderService` 非 U 请求独立传入 `NsfHifiganRenderer::render` 的新增末尾 `incrementalNativeRender=true`；即使关闭噪声保护，更新 137 的局部核心复用仍有效。直接旧调用的默认值为 false，原 `protectNativeNoise=true` 也保留核心复用；UTAU 默认两者均 false，维持旧推理块。该开关与 `nsfSmoothPitchTransitions` 相互独立，界面提示不能再写“清辅音保护始终有效”。验证 `--smoke-native-noise-option 输出目录 NSF模型目录`、扩展 `--smoke-nsf-picker`，分别检查实际声音／映射／PCM／缓存，以及界面／工程／撤销／旧工程。

更新 134：`TrackData::nsfSmoothPitchTransitions` 默认 true，在非 U NSF 的“音高突变平滑”复选框中按轨道编辑。`Mld5FileRenderRequest` 单段／整句传递到 `NsfHifiganRenderer::render` 末尾 `smoothPitchTransitions` 参数；false 时 `conditionNsfPitchTransitions` 不改写采样后的 F0，并取消邻音上下文 F0 过渡。清音保护和幅度去脉冲／边缘保护是独立处理，不随此开关关闭。字段需参与工程保存、撤销、音频／波形缓存及 MCP `nsf_smooth_pitch_transitions`。UTAU 默认调用仍为 true。验证入口 `--smoke-nsf-pitch-transitions` 和 `--smoke-nsf-picker`。

更新 133：非 U 手工曲线通过 `nativePitchHandlesPlaced` 区分实际控制点与初次拉直／旧工程的默认边界点。`setNotePitchCurve`（含拖动预览）使用真实点，`nativeSharedPitchLines` 在实际相邻端点之间连接，不得刷新后在音频边界补点。此字段需要持久化和参与缓存；再次拉直重置为初始 1 ms 过渡。原生点的水平拖动以同线相邻控制点排序约束，不能以音符起止为硬边界。`--smoke-native-linked-pitch` 检查首尾往返拖动、不增点、曲线与渲染一致、撤销及保存重开。

更新 132：`PianoRollComponent::mouseDoubleClick` 的非 U 拉直／标准半音吸附只允许 `Tool::note`；`Tool::points` 对原生及 UTAU 曲线均使用加点路径，沿共享曲线判定时间归属，禁止抢先批量拉直。真实事件回归见 `--smoke-flatten-pitch-line` 的 `NativeBatchFlattenSmoke.h`。

更新 131：`SourceVoicing.h` 独立于 FCPE bin confidence，结合归一化自相关、高频能量与频谱平坦度判断明确非周期区域；FCPE 导入清音掩码保留到源曲线。`AudioEngine::mergedRequestFor` 不再沿用清音前的 F0。`NsfHifiganRenderer::render` 末尾参数 `protectNativeNoise` 只由非 U `RenderService` 开启：默认模型有 HN-SEP 时仅谐波经过变调，源噪声按同一分段时钟混回，明确清音插值回原 PCM；缺失 HN-SEP 时保留独立清音保护。原生清音帧不得被 F0 防爆音逻辑重新发声；神经直流／孤立脉冲修复必须先于源噪声和清音 PCM 混回，最终仅做边缘渐变。未编辑 PCM 直通、UTAU FLAG 合成不变。旧工程不必重做调音即可获得渲染保护，更新原始音高显示需重新分析源音。专项命令 `--smoke-native-noise <输出目录> <nsf模型目录> <原始录音>`。

更新 130：非 U NSF 增加 `StretchAlgorithm::hifiShifterMel`（值 5，菜单 ID 6，持久化／MCP 键 `hifishifter-mel`）。`NsfHifiganRenderer.cpp::stretchHiFiShifterMel` 使用上游 HiFiShifter 的目标帧数取整与首尾对齐线性插值规则；`alignHiFiShifterTail` 按两 Hop 尾部渐弱对齐实际样本数。多段编辑保留源时间映射。它只对齐拉伸方法，继续使用本地 F0 防爆音、模型上下文及分块。真实模型与数值验证入口 `--smoke-hifishifter-mel <输出目录> <模型目录>`，GUI 持久化检查并入 `--smoke-nsf-picker`。

更新 129：独立原生音高在同一素材／显式粘连组内精确相邻时，用交界两侧各 0.5 ms 的端点和 1 ms 直线过渡连接，音符内部仍是独立插值分段。`SharedPitchLine` 按主体—短桥—主体分段，`nativeBoundaryAnchors` 为编辑器提供与实际曲线一致的边界标点；不跨空隙、重叠或其他穿越交界的声部建立连接，极短音符缩短窗口。恢复整组原始 F0 不创建额外过渡；响度分组不变。

更新 128：非 U 双击拉直／批量拉直将每个目标音符标记为 `nativeIndependentPitch`，分别吸附标准音高并关闭自动过渡和颤音。共享音高分组在这些音符处断开，显示、单素材渲染和整句渲染均不可重新跨段插值；后续编辑某段也不能牵动邻段。响度包络分组使用 `sharedPitchLines` 的 `nativeEnvelopeGrouping` 模式忽略音高独立标记，保持原有响度、时间粘连和源映射。标记随工程保存、撤销及音符复制保留，参与渲染／波形缓存键。未拉直的粘连片段仍共享原有音高。

## 工程数据与时间坐标

核心类型定义在 [ProjectModel.h](../juce/src/ProjectModel.h)，主要实现位于 [ProjectModel.cpp](../juce/src/ProjectModel.cpp)。

| 类型或字段 | 职责与容易混淆的地方 |
| --- | --- |
| `ProjectData` | 全部轨道、曲速／拍号等工程信息、原生连接关系；工程快照的顶层 |
| `TrackData` | 区域列表、调音算法、伴奏／创作状态、音源、输出引擎覆盖、重叠开关及轨道混音参数 |
| `nativeNsfAudio` | 明确选择原生 NSF 后置为 true 并保存，避免记住的 UTAU 音源路径把模式重新判定为旧版 NSF＋OTO；旧数据缺省为 false，保留兼容迁移 |
| `ClipData` | 时间轴区域位置和长度、原文件、源截取、音频有效窗口、时间映射、子区域、音符与区域增益 |
| `NoteData` | 音符位置、长度、歌词、MIDI 中心、原始 F0、目标音高控制、包络、颤音、OTO、分段、FLAG、DS 状态 |
| `PitchPoint` | 密集原始相对音分、去颤音曲线、有声状态以及可选的手动目标 |
| `PitchCurveEditPoint` | 稀疏目标 MIDI 控制点及其连接形状／贝塞尔参数，不能当成原始 F0 |
| `sourceOffsetSeconds` / `sourceDurationSeconds` | 实际源文件读取范围，与目标时间轴长度不同 |
| `sourceTimeMap` | 目标局部时间到源局部时间的映射；拉伸、裁剪、波形和音高应使用同一映射 |
| `audioStartSeconds` / `audioDurationSeconds` | 区域内实际有音频的窗口，可区别于区域外框 |
| `parts` / `clipPartId` | 组合区域的子素材；父层音符列表通过 ID 指向对应子素材 |
| `nativeSourceStartSeconds` / `nativeSourceEndSeconds` | 音符对应的真实源区间，分离和裁剪后要重新绑定 |
| `nativeConnections`、连接标记、`nativeAudioLinked` | 显式音符连接、兼容标记与多素材组合关系，不能只清除一个标记就假定完全独立 |

时间轴音符位置通常是 `clip.startSeconds + note.startSeconds`。音高点、音符包络等一般使用音符内秒数；子区域和 `forView` 展开有专门转换。先检查 [ClipParts.h](../juce/src/ClipParts.h) 的 `expandedClipParts` / `expandProjectClipParts`，不要给已展开的坐标重复加父层偏移或重复乘父层增益。

`ProjectModel` 管理操作原子性、修订号与撤销历史；`ProjectFileIO.h` 负责保存时的临时文件、写后读回、完整性验证、文件替换与备份保护；`ProjectRecovery.h` 负责异步自动恢复快照和进程间租约。`.hjpx` 是 JUCE ValueTree 二进制工程，不是可以随意文本替换的 JSON。`.hjm` 是素材注释侧文件，读写与素材设置相关逻辑还要查 `SampleSettings` 和模型导入路径。

## 界面和交互

更新 143：无 F0 区间仅对明确人为目标显示圆点线（2.2 px 圆点、6 px 间隔），与“气声虚线”的短划线区分；未编辑源曲线及原始参考继续留空。`nativeAuthoredPitchDisplayRanges` 使用真实控制点范围／逐帧手动目标范围，`PianoRollComponent` 按共享曲线合并、缓存并对拖动失效；`strokeNativeAuthoredPitchDots` 沿完整曲线取圆点，保留相位，跳过屏外点。标点可伸出音频范围，手绘／直线预览与提交同样显示圆点。非 U `setNotePitchCurve` 现在保留无声帧的手动目标几何，仍保持 `voiced=false`，不改变声码器源 F0／清音；无需新工程字段或新模型。验证扩展 `--smoke-native-pitch-visibility 输出目录`，检查实线／短划线／圆点区别、普通与共享曲线、范围外端点、手绘、撤销及实际渲染 F0。

更新 142：非 U 音高线只在 `PitchPoint::voiced` 连续有效的源区间绘制，`nativeUnpitched` 与待分析的中性占位曲线不画音高。`NativePitchVoicingDisplay::nativePitchDisplayRanges` 用当前音符时钟构造可见区间，`PianoRollComponent` 随快照缓存、拖动预览失效，并将粘连成员的有效区间并入同线遮罩；不能用稀疏手动目标点填补无 F0 区域。目标曲线、原始参考、分离显示的颤音均遵循遮罩，标点仍可编辑。移除缺失 F0 的虚线补线，带实际 F0 的气声继续由“气声虚线”切换线型。只改绘制，不改项目或渲染请求；旧工程已有的无声掩码立即生效，检测器误报的有声音高仍需重新分析／修正。专项 `--smoke-native-pitch-visibility 输出目录`，并更新气声显示验证。

更新 113：`MainComponent::bindDefaultUtauVoicebank` 自动绑定普通 UTAU 默认音源时仅设置路径，渲染器按需读取 OTO；不得在算法切换的消息线程上调用整库 `SampleSettings::importVoicebank`。显式音源导入入口仍保留。`AudioEngine` 对无区域轨道跳过音源缓存依赖扫描。选择原生 NSF 必须经模型 setter 设置 `nativeNsfAudio`；音源路径可以保留，模式判定应统一使用 `trackUsesVoicebankSynthesis`。

| 模块 | 当前职责 |
| --- | --- |
| `Main.cpp` | JUCE 应用生命周期、主窗口、命令行分流、大量内联诊断及 smoke 入口 |
| `MainComponent` | 所有面板的协调层；菜单、参数工具栏、选择同步、工程操作、设置、导入导出、后台任务 |
| `TimelineComponent` | 上方区域外框、缩略音符与波形、播放线、区域选择／框选、跨轨拖动及区域操作 |
| `TrackListComponent` | 左侧轨道选择、名称、混音控件与引擎标识 |
| `PianoRollComponent` | 下方音符、源音高和目标音高、波形、包络、分区、FLAG、颤音、工具命中与拖动预览 |
| `SettingsComponent` | 默认算法、模型与工具路径、推理、音频和界面等用户设置 |
| `Theme` / `I18n` | 配色、控件和工具图标；中简、中繁、日、韩、英文本 |
| `ZoomScrollBar` | 滚动条端点缩放与独立视图缩放交互 |
| `ModelessWindows` | 非模态工具窗口相关共用实现 |
| `AssetManagerComponent` / `SampleSettings` | 素材管理、素材参数和注释数据 |

目前工具排列是选择、自由绘制、标点、直线、扳手、裁剪、连接／分离，再到各增减按钮。图标在 `Theme.cpp`，创建与排列在 `MainComponent.cpp`，行为枚举和事件分派在 `PianoRollComponent`。不要仅交换图标而漏掉按钮位置、提示文本、状态或命中逻辑。

显示开关包括歌词、音高、原始音高虚线、波形等。非 U 模式的歌词在波形上方；选中重叠素材时其他重叠素材降低显示亮度，**只改变显示，不改变音量**。

更新 141：非 U 导入分析完成后，以灰色斜纹“无音高”片段覆盖源音频中未被音符占用的空隙（至少 5 ms，包含静音）。旧工程在上方音频右键菜单选择“标记无音高片段”；不重新分析、不覆盖现有调音。该标记表示缺少可编辑的音高对象，不是人声／背景噪声分类，漏识别的人声也可能落入其中。`NativeUnpitchedRegions.h` 按实际音频窗口补齐、保持源时钟并防止重复添加。`NoteData::nativeUnpitched` 使显示音高仅作为行位置；音高绘制、规整、颤音和源音高刷新不会赋予 F0，渲染请求中该片段 F0 为零。沿用裁剪、响度包络和源音频剪贴板。删除这种片段时，`removeNotes` 使用 `slicedClipParts` 将实际播放源范围分割并移除对应时间段，前后素材绝对位置与源映射不变，原文件不改，单步撤销可恢复；重叠未选音符保留自己的范围。普通音符删除语义沿用原行为。工程保存、HJM CSV 末列 `native_unpitched` 与 MCP 属性 `native_unpitched` 保存类型；MIDI 不导出虚假音高，转入 UTAU 时作为休止、DiffSinger 不生成歌词。无新增模型。专项 `--smoke-native-unpitched-regions <输出目录> [真实录音] [分析模型根目录]`，报告位于 `test-output/update141/` 与配布 `verification/update141/`。

更新 140：“显示 → 气声虚线”默认关闭，仍用 `ui.consonantPitchDashed` 保存，兼容原“虚线辅音”设置。`NativePitchVoicingDisplay.*` 单个后台线程按源文件路径／大小／修改时间缓存 `SourceBreathiness` 的气声／噪声证据；40 ms 窗／10 ms 步长结合归一化周期性、分数周期抵消残差和分频段频谱平坦度，覆盖带音高的弱气声、低频气声及清摩擦音，排除静音、孤立帧和正常滑音／颤音的窄带残差。它是显示用的声学启发式，不是语义音素识别或精确分离，粗糙嗓音／录音噪声也可能被标记。带音高的气声仍保留 F0；渲染继续使用更保守的 `SourceVoicing` 清音保护，两者不得混用。区间按 `nativeSourceTimeMap` 跟随裁剪／拉伸，粘连组共享曲线，重叠声部独立；只改变线型，不改控制点、工程音高或声音。更新 142 起，缺失 F0 处不画参考线，也不补写基频；后台完成前也不能填补已测清音空隙。不得在绘制中同步解码或重新分析原音。本次无需新增模型。专项入口 `--smoke-source-breathiness <输出目录> [真实录音]`、`--smoke-native-consonant-display`；开发版及配布报告见 `test-output/update140/` 和 `verification/update140/`。

## 非 U 原生音频编辑

适用范围用 `trackShowsAllNativeRegions` / `nativeNoteTimingEnabled` 等实际条件判断，不能简单写成“不是普通 UTAU”；DiffSinger 和伴奏有不同语义。

### 导入和未编辑原音

更新 127：GAME／FCPE 分析线程限制为最多 4 个、低核数机器取逻辑核心数的一半；ONNX 会话关闭等待时忙轮询，避免多个常驻模型线程池争抢界面和音频线程。带纯时间标注的 HJM 初始占位音高可保留原音播放，不在分析完成前启动不必要的整段合成；显式音高编辑、拉伸及非中性音色参数仍受正常渲染判定约束。

`AnalysisService` 汇总 GAME 音符识别、FCPE 连续 F0 与 `NativeAnalyzer` 回退。默认 GAME Medium + FCPE；模型缺失或失败可回退 native-hq。`applySourcePitch` 将真实 F0 按源时间映射附着到已有音符，避免覆盖用户划分和目标控制点。

直接拖入的原音，在没有调音操作时应保留原始音高与平直的 100% 响度包络。显示或分析原始 F0 本身不是变调操作。`NativePitchIdentity.h`、`AudioEngine.cpp` 中的 `makeRenderRequest` 和 `RenderService::canPreserveNativeSource` 控制原音保真路径；有真实变调、拉伸、共振峰或其他需要重合成的编辑时才走对应后端。

### 移动 拉伸 分离与裁剪

| 文件 | 行为 |
| --- | --- |
| `NativeSourceTimeMap.h` | 统一源／目标时间转换，以及音频窗口到预览区域的转换 |
| `NativeNoteTiming.h` | 移动、左右拉伸、源区间绑定；处理普通、组合和粘连素材 |
| `NativeAudioTrim.h` | `planNativeNoteTrim`，向内裁短、向外读取隐藏源素材，保留已有时钟，供预览和提交共用 |
| `NativeAudioClipboard.h` | 将选中分段及所属源素材、映射、参数组织为可粘贴内容 |
| `NativeAudioDisconnect.h` | 将选中的关联素材分离为独立区域并维护源范围 |
| `NativeUnpitchedRegions.h` | 未覆盖源音频补成无音高编辑对象；删除时切掉实际播放源范围 |
| `NativeAudioLink.h` | `assembledLinkedAudio`，保留不同文件／源时钟的同时组装多素材 |
| `ClipParts.h` | 子素材展开、切片、父层增益继承与坐标转换 |

普通工具的端点拖动仍是拉伸；原生连接分段的移动／拉伸有联动规则。首段、尾段移动可改变中间跨度；中间左右拉伸应移动外侧片段而不是压缩外侧片段。多个选中片段整体移动应保持相对位置。具体规则及老工程兼容集中在 `NativeNoteTiming.h` 与 `NativeNoteMoveSmoke.h`。

更新 126：原生音符的控制点（含负时间和跨相邻段的共享点）使用录音整体分段时钟 `warp(note.startSeconds + point.timeSeconds) - newStart`，不能按所属音符长度比例处理范围外的点。响度曲线跨时钟斜率变化处补插值支持点，保持淡入／淡出位置。上方区域拉伸的 `resizeClip` 在修改辅音速度前固定原生源时钟，避免 UI 的 0.05–20 倍速度范围改变大比例拉伸的源边界；原生响度、区域增益及继承层、分割颤音时钟随区域长度一起缩放。检查入口 `--smoke-native-note-join` 与 `--smoke-native-note-move`；后者设置 `HACHI_TEST_NSF_MODEL_DIR` 可加入真实 NSF 模型长拉伸播放与尾部发声检查。

裁剪工具位于连接工具左侧，仅原生模式启用。端点向内拖动裁去首尾，向外拖动读取原文件中的隐藏内容（更新 124），可以恢复曾经裁去或分离后删除的发音范围。双向均不使用网格、最小音符单位或普通 6 像素启动阈值。扩展上限为源文件边界、时间轴零点与禁止重叠时的相邻音频，不改变已有部分的源时间映射。源文件不修改。内部片段裁剪保留空隙、邻段位置和长度，解除切口的显式连接。

裁剪保留源时间映射、包络和音高插值；密集源 F0 在新边界插值截断，不能继续画到被删除区域。颤音通过 `vibratoReferenceDurationSeconds` / `vibratoTimeOffsetSeconds` 保留原相位及渐变时钟；相关字段已纳入工程保存与渲染键。DSP 对自由裁剪的半采样位置保留分数对齐，不能因独立取整相差一个采样就误触发声码器。

### 音高 波形与重叠

更新 122：`sharedPitchLines` 对原生同一录音的分段及 `nativeAudioLinked` 多素材组生成一条公共目标曲线，内部切口不做独立起头和重复尾端点。`ClipData.nativePitchGroupId` 只在 `expandProjectClipParts` 的读取副本中标记来源组，不写工程；实际素材编辑通过 `expandedClipParts` 不携带该临时关系。`PianoRollComponent` 的普通／标点显示和预览、`AudioEngine` 的逐素材及 NSF 整句请求共用曲线；未编辑原音目标和清音掩码保持。渲染及真实波形缓存通过 `writeNativePitchContext` 纳入同组邻段依赖。普通合并的独立子源、断开的素材和重叠声部不自动粘连。

- 默认编辑工具下双击原生音频将音符中心吸附到最近整数 MIDI 半音，并将目标音高线拉直（更新 118／121；双击已选中音符时批量处理全部选中音符，各自吸附最近半音，整组共用一次撤销；双击未选中音符只处理该音符）；标点工具双击则添加控制点，不规整音高（更新 132）。原始 F0 虚线仍可独立显示。右键“还原为原始素材音高”保留时间位置和拉伸，只还原音高相关编辑。
- `SourceWaveformPreview.h` 给出按当前源映射拉伸后的原音波形近似；`RenderedWaveformPeaks.h` 与 AudioEngine 的波形快照显示当前渲染音频。二者含义不同，不能把近似源波形标成真实合成结果。
- `NativeAudioOverlap.h` 负责默认禁止重叠、移动／拉伸／跨轨／粘贴碰撞约束，以及关闭允许重叠时的裁切。前段长度保留；后段裁去被覆盖的头部，结束位置不变；完全覆盖的后段可被移除。
- `NativeAudioFocus.h` 负责重叠的显示聚焦、绘制次序和提示；钢琴卷帘有重叠区间提示与两条素材预览。显示聚焦不能改合成数据。
- 原生模式同轨区域默认可相互看到；点击某个区域不应使编辑区只剩该区域唯一音符。聚焦、选区、可见范围和可编辑范围需要分开处理。

## UTAU OTO 与输出引擎

`PitchAlgorithm::utau` 下有普通、界、谋三个 `UtauMode`。**调音模式与输出引擎是两条独立维度**：普通／界／谋决定 OTO 与分段语义，轨道输出覆盖决定使用传统重采样器还是内置 HiFisampler。不要重新添加独立的“NSF 界”“NSF 谋”编辑模式。

主要文件：`backend/UtauRenderer.*`、`backend/UtauOtoOverride.h`、`VoicebankSettingsComponent.*`、`OtoWaveformEditorComponent.*`、`OtoRegionGuides.h`、`backend/OtoAudioAnalysis.*`、`UtauOutputEnginePanel.h`。

OTO 路径包括音源选择、别名解析、全局／单音 OTO、先行发声、重叠、固定段／可拉伸段、界／谋的可变段数与 C/V/S 分类、手动分界、辅音速度和区域 FLAG。不要假设所有音符固定四段，也不要用 DS 音素边界冒充 OTO。

轨道／区域右键的“选择输出引擎”提供跟随设置、HiFisampler、重采样器配置。轨道覆盖优先于全局默认；跟随设置需要真正撤销轨道覆盖。显式工具缺失不能悄悄换后端。引擎名显示在轨道和区域标题。兼容序列化键仍使用 `pc-nsf-hifigan`。

传统路径调用外置 resampler，支持 wavtool 配置并有内置混音；外置 wavtool 处理已应用编辑器包络和交接的句段，不应重复套包络。UST 文本、编码、导入和导出在 `UstText.h`、`LegacyTextCodec.h`、`UstImporter.*`、`UstExchange.h`。参考 [输出引擎](utau-output-engine.md)、[UST 保真](ust-fidelity.md)、[线性 FLAG](utau-linear-flags.md)。

### 标准 UTAU 中文 CVVC（更新 112）

轨道／区域右键的“发音器”可选择手动别名或中文 CVVC，按轨道保存为 `utauPhonemizer`。旧工程默认手动；目前仅标准 UTAU 启用。`backend/ChineseCvvcPhonemizer.h` 在两个合成后端之前读取 presamp、编排 VC/CV/收尾音并聚合原音符波形；`UtauRenderer::mappedAliasTiming` 严格查询 OTO，已解析别名不能再次套前后缀。

选区保留同一区域内邻音上下文，未选片段只保留衔接参数、不合成；父音符的音高、包络与 FLAG 时钟随生成片段平移。音符 ID 和歌词不拆改。发音方式、presamp、prefix.map 和邻音变更参与渲染缓存。生成片段使用共享只读音素显示，不得给普通 UTAU 音符启用 DiffSinger 专属编辑语义。详见 [中文 CVVC 使用与边界](chinese-cvvc.md)。

## 内置 HiFisampler

这是 PC-NSF-HiFiGAN 的 C++／ONNX 内置音源合成路径，不等于启动上游 Python 程序，也不等于外置 WCSNDM 的 HF 服务。关键文件：

| 文件 | 职责 |
| --- | --- |
| `backend/NsfHifiganRenderer.*` | 原生神经渲染、模型和特征、源目标帧映射、推理和取消 |
| `backend/UtauRenderer.cpp` | OTO 句段请求、`renderNsfUtauPhrase` 与音源合成的连接 |
| `backend/HifisamplerFlags.h` | 支持的 FLAG、解析、范围、优先级和曲线处理 |
| `backend/HifisamplerDsp.inc` | 内置 HiFisampler 的 DSP 实现，作为包含单元使用 |
| `backend/HifisamplerSmoke.inc` | 相应 DSP 和模型相关验证入口 |
| `third_party/hifisampler/` | Apache-2.0 LICENSE 与来源 NOTICE；完整源逻辑已适配进上述 C++ 文件 |

已移植上游实际处理的 `g Hb Hv Ht HG P t A G He`。数值曲线使用 `HIFI:` 前缀与 WCSNDM 曲线隔离；有曲线时覆盖同名文本参数，分区覆盖音符、音符覆盖全局。`G` 强制刷新；`He` 才显式开启可伸缩段的反射循环。默认按照 OTO 把每个源段读取一次并拉伸到目标长度，不能重新回到默认循环补长度。

`P` 未填写时不做响度归一化；裸 `P` 是 100。Hb/Hv/Ht 按需要启用 HN-SEP 分离模型，默认路径不必加载它。完整录音提供真实声学上下文；目标 Mel 使用半 hop 中心。音高、FLAG、帧和音符时间轴对齐要共同检查。

文件大小／修改时间参与音源刷新；源音频同大小同时间原位替换时可用 G 强制重算。长音的包络尾部要按当前音符时长和交接位置适配，不能留在旧结束点导致后半段静音。详细范围、模型、优先级和限制见 [hifisampler-flags.md](hifisampler-flags.md)。

## 响度包络 FLAG 与预设

更新 117：用户所需的“响度包络”是钢琴卷帘下方的标点通道。非 U 工具栏的“响度包络”及音符右键入口现在切换 `Tool::amplitude`，不再打开高级包络弹窗；“音量”仍调整整体音量。通道共用 UTAU 的百分比坐标、双击加点、拖动、右键删内部点及多选按时长比例映射。绘制、命中与提交允许 `nativeNoteTimingEnabled` 的音符；原生默认两个端点 100%，新通道按线性振幅插值，旧有包络保留其插值标记。编辑写入 `NoteData::amplitudeEnvelope`，不写 `nativeEnvelope`。刷新工程与切换原生轨道不应关闭通道。原生源波形按拉伸时钟显示在通道背景，修改点位即时预览；已有高级包络开启时以虚线显示叠加结果。原生高级弹窗仍可通过右键“音量”打开。

更新 116 增加了独立的原生高级包络弹窗（当前通过右键“音量”进入）。`NoteData::nativeEnvelope` 独立于 OTO 参数，默认关闭且增益为 1；启用后按整段音频归一化位置定义线性、贝塞尔或混合包络，可上升和下降。`backend/NativeEnvelope.h` 负责参数编解码、求值、显示采样及裁剪；原生包络乘在基础响度包络上，进入 AudioEngine 的逐帧 noteGain 和缓存键，纯响度编辑仍可保留原音路径。分离、自由裁剪和关闭重叠后的自动裁剪以 de Casteljau 截取三次曲线，拉伸沿归一化时间缩放；旧式幂 S 曲线裁剪使用线性密集近似。原生预设单独保存为 `native-envelope-presets.json`。多选一次应用、撤销重做、工程保存支持完整参数；合并音符延续原有清空音符曲线的语义，也清空此独立包络。

`AdvancedEnvelopePanel.h` 是原生音频与 OTO 高级音头／音尾共用界面；`backend/AdvancedEnvelope.h`、`TailFadeSettings.h` 与 `AmplitudeEnvelopeCurve.h` 提供效果参数及曲线求值；`TailFadePresetStore.h` 保存、读取与删除用户预设；`FlagCurveDrawing.h` 是 FLAG 绘制辅助。

界面采用草稿，应用后成为工程编辑，关闭未应用的草稿不应改变工程。原始包络与实际效果曲线分开，预览的密集采样不应反写成大量控制点。音头和音尾按对应 OTO 区域比例定位，自定义曲线／贝塞尔与预设必须同渲染求值一致。DS 没有 OTO，不能启用 OTO 专属效果。

更新 114：高级包络新增“混合”，音头／音尾独立保存分段点（范围内归一化时间、0–200% 响度）和各段直线／贝塞尔曲线。`mixed` 与 `knots` 位于 `TailFadeSettings`／`HeadEnvelopeSettings`，混合使用原有 mode=2 并由 mixed 标记区分；`MixedEnvelopeIO.h` 统一 JSON 编解码，工程属性 `utauTailMixed`／`utauHeadMixed` 和 v4 用户预设保存完整形状，兼容旧预设。包络预览、混音与导出共享同一求值函数，节点及段型参与混音缓存键。更新 115：每个分段独立保存 `EnvelopeBezier`，首段用 `firstBezier`，内部点用 `EnvelopeKnot::bezier` 表示到下一点的曲线；统一按局部贝塞尔横坐标反解再求响度。缺少控制柄的旧混合数据使用等效 S 曲线默认值；v1–v3 预设可继续读取。选中曲线段共用图中方形控制柄及四个数值框，修改仅影响该段。操作见 [高级包络](advanced-envelope.md)。

## DiffSinger

入口在 `MainComponentDiffSinger.cpp`；音源检测和 C++ 后台桥在 `backend/DiffSingerRenderer.*`。选择受支持的 `dsconfig.yaml` 音源后走 DS 乐句合成，沿用编辑器的音符界面，不走普通 OTO 分区。

| 模块 | 功能 |
| --- | --- |
| `DiffSingerRequest.h` | 从工程／音符组合 DS 请求和上下文 |
| `DiffSingerPhonemeEditor.h` / `DiffSingerTiming.h` | 音素时长草稿、预测边界与手动覆盖、保持元音拍点 |
| `DiffSingerPhonemeDisplay.h` | 主界面显示实际辅音／元音及提前发声范围 |
| `DiffSingerPronunciationEditor.h` | 逐音读音覆盖及发音编辑 |
| `DiffSingerParameterCurves.h` | 模型预测实参和用户参数偏移分层 |
| `DiffSingerPitchHandles.h` / `DiffSingerPitchRestore.h` / `PianoRollDiffSingerOffset.h` | pitch 参考、手动偏移、恢复、控制点与拖动 |
| `backend/DiffSingerOptions.h` | auto／CPU／DirectML、设备、预览与导出质量、步数和 depth |
| `backend/DiffSingerProjectCache.h` | 工程相关的预测缓存管理 |
| `engines/diffsinger/bridge.py` | 模型／音源加载、词元、时长、pitch／variance／声学／声码器请求 |
| `runtime.py` | 常驻后台、请求缓存、指纹、进程存活与退出 |
| `compatibility.py` / `inference.py` | 音源预检、旧／新采样接口、输入兼容、推理后端与会话 |
| `pronunciation.py` / `english_g2p.py` | 多语言前缀、词典与中英文发音处理、英文预测 |
| `timing.py` / `variance_retake.py` / `expressions.py` | 时长覆盖、局部重生成掩码、模型能力与表达曲线 |

DS 保留原始预测、实际编辑、用户偏移等不同层；重新生成所选区域时保留完整上下文和未选区编辑。生成结果提交前检查工程修订，避免覆盖预测期间的新编辑。提前发声的辅音归所属音符，前音符尾部静音或 DYN 不能误削掉它。音源不支持的参数不能仅因 UI 存在就宣称生效。

当前已存在 GPU／DirectML 与质量设置、模型后台复用、局部 retake 等代码，不能照抄 `diffsinger-audit.md` 中早期的“尚未实现”。另一方面，不应把当前兼容层描述成完整 OpenUtau 发音器插件运行环境或完整 USTX 工程交换。功能细节见 [diffsinger.md](diffsinger.md)。

## 播放 混音 缓存与导出

`AudioEngine` 管理音频设备、原音试听、工程同步、渲染请求、播放位置、混音、音量声像、源和渲染波形、WAV 导出。`RenderService` 分派后端；`PlaybackRenderQueue.h` 与 `PlaybackRenderPriority.h` 按播放位置给待处理任务排序，隔离需串行运行的同类后端并清理过期任务。

播放只先等附近所需区域，遇到未就绪区域时所有轨道一起等待，不能让伴奏独自继续。导出等待所需范围完整渲染。不要为了提高优先级切断 DS 的乐句预测上下文。新优先级不表示必须打断正在执行的任务。

更新 137 借鉴 HachiTune 的局部合成及过期任务取消思路：`PlaybackRenderQueue::endUpdate` 对不在当前渲染键集合中的运行任务发出取消信号，异步等待其自行退出并释放 scheduled 状态，不在界面线程等待。仍有效的其他轨道任务保留；只移动播放头不会取消任务。ONNX Run、外部重采样和 DS 继续使用原有取消传播。

非 U NSF 的 `infer` 使用固定 512 帧核心块（44.1k/512 时约 5.94 秒），保留 32 帧前后上下文与 16 帧重叠混合。`NsfRenderChunkCache.h` 按完整 Mel、F0、输入形状及裁剪范围的 SHA256 复用已完成核心；缓存属于具体模型／执行配置的会话，LRU 上限 64 MiB／256 项，关闭会话时释放。改音高、共振峰或时间映射时，仅输入真正改变的块重新推理。取消／失败不发布半成品，已完成的有效核心可以供下一次编辑复用。UTAU 继续使用原 4096 帧路径，DS 保留完整乐句上下文。

此优化集中于声码器推理；当前源解码、清辅音判定与 Mel 特征仍按整段请求准备，HN-SEP 复用既有源缓存。未编辑原音仍走 PCM 直通。清辅音／气声保护、音高突变平滑开关、时间映射和后置响度包络保持原有规则。诊断 `inferredChunks`／`reusedChunks` 只用于验证，不进入工程文件。检查入口：`--smoke-native-incremental 输出目录 NSF模型目录`、`--smoke-playback-priority 输出目录 探针EXE`、`--smoke-native-shutdown`；真实录音复用与气声回归见 `--smoke-native-noise`。

`writeNoteRenderFields` 是音符渲染键和波形失效签名的重要共用入口。参数已经改动但声音／波形仍旧时，先检查参数是否进入请求与签名、文件指纹及任务会话，而不是只清空所有缓存。

`TrackGainEnvelope.h` 和 `TimelineGainEnvelope.h` 处理轨道／区域增益。`WavExportOptions.h` 及 AudioEngine 的导出路径处理导出范围、采样率、位深、声道和分量；默认立体声 24-bit 等行为需查当前配置。伴奏按原音播放，不进入人声调音合成。

## 进程和资源退出

更新 108 的相关入口包括 `MainComponent` 析构、AudioEngine 生命周期、`RenderService::cancelAll`、`DiffSingerRenderer::shutdown`、`UtauRenderer::shutdownHfDaemon` 及 `NsfHifiganRenderer` 的取消检查。

`UtauRenderer.cpp` 中 `OwnedHfDaemon` / `HfDaemonLifetime` 管理编辑器自己启动的 WCSNDM/HF 后台进程；Windows Job 使用 `KILL_ON_JOB_CLOSE` 回收其子进程。已有的其他服务不应被接管或按进程名批量结束。内置 NSF/HN-SEP 推理有取消传播，外置重采样器等待会分段检查取消；取消结果不能发布到当前工程。ONNX 会话与环境的销毁顺序也会影响能否完整退出。

## HAMOOD 和声与伴奏分析

更新 145：伴奏音频右键“识别曲速”调用 `AccompanimentTempoTask` 与 `tools/hamood_audio/tempo_worker.py`，仅运行已有 Beat This! small0，不运行 BTC 和弦模型。`AccompanimentTempoPanel.h` 在非模态窗口中显示 BPM、估计限制及变速提示，只有取消／关闭按钮。分析持有片段副本，读取可见裁剪范围；合并区域按各子音频偏移拼出分析输入。后台逐块解码，取消／关闭窗口／程序退出回收自己的子进程和临时目录，不接触 ProjectModel、工程 BPM、位置、长度或撤销状态。不下载模型；运行环境沿用 `HACHI_HAMOOD_RUNTIME` 或配布 `engines/hamood/runtime.json`。单个编辑器窗口只允许一个曲速结果窗口，重复点击将其置前。`--smoke-accompaniment-tempo 输出目录 [真实伴奏]` 验证实际菜单、模型、裁剪／合并、取消、错误和工程不变。

`Hamood.*` 是原生调性／和声计划与生成；`HamoodProject.h` 保存工程中的和声分析信息；`HamoodTimelinePanel.h` 显示时间轴相关结果；`MainComponentHamood.cpp` 负责菜单、任务与界面操作。生成通常创建新声部，继承需要保留的音源和唱法信息，原旋律不应被隐式改写。

`HamoodAudio.h` 与 `tools/hamood_audio/worker.py` 对接伴奏音频分析。beat-this、BTC 与低频音级统计提供拍点、和弦等旁路证据；结果缓存到工程旁 `.hamood-cache`。`kara2.py` 等脚本提供更具体的分析／验证工作流；东京泰迪熊等命名脚本是具体任务用例，不是所有工程的默认处理。

和弦分数不是准确率；分析 BPM 不自动覆盖工程 BPM。复杂拉伸音频不一定符合伴奏分析接口条件。详见 [hamood.md](hamood.md)、[hamood-audio.md](hamood-audio.md)、[kara2.md](kara2.md)。

## Melodyne 兼容与其他后端

`MelodyneImporter.*` 保留独立的 MPD 读取／兼容转换；`MelodyneProvider.*` 隔离安装发现、VST3 描述和实例化尝试，以及是否真正开放原生导入／渲染。当前 `nativeImportAvailable()`、`nativeRenderAvailable()` 返回 false，`experimentalSelfImportEnabled()` 返回 true，实验合并渲染关闭。发现已安装 Melodyne 不等于获得完整 ARA／工程访问能力。

`Mld3Renderer`、`Mld5Renderer`、`WorldRenderer`、`Llsm2Renderer`、`NsfHifiganRenderer` 是不同算法路径，名称和历史兼容字段不能证明其等同于商业 Melodyne 内部算法。界面隐藏的实验路径不要未经验证恢复为默认。`OrtExecution` 统一原生 ONNX 的设备选择、环境与执行支持；Python DS 有自己的 inference 层。

## MCP 自动化入口

| 入口 | 作用 |
| --- | --- |
| `--mcp` | `backend/McpServer.*`，独立无界面工程会话 |
| `--mcp-live` | `backend/LiveMcpBridge.*`，连接现有编辑器窗口的 stdio 代理 |
| `--mcp-list-sessions` | 枚举当前可连接窗口；多窗口不能猜测目标 |
| `MainComponentMcp.cpp` | 在真实窗口上读取选区、编辑工程、排队后台任务和管理修订 |

真实窗口查询优先 `editor_status` / `editor_selection`；空选区不能扩成整轨。编辑使用实际稳定 ID 和 `expected_revision`；长任务接收成功不等于完成，应继续查询任务及 `result.isError`。`editor_batch` 对允许的普通编辑合并撤销；文件、播放和长推理不是普通批量事务。结果未知的写操作先查询状态，不盲目重发。

MCP 参数以当前 `tools/list`、`McpServer` 的 schema 和 Live handler 为准。不要假定新增 GUI 裁剪按钮已经自动增加了专门的 MCP 方法。完整协议见 [mcp.md](mcp.md) 与 [mcp-live.md](mcp-live.md)。

## 构建和配布

`juce/CMakeLists.txt` 是 C++20／JUCE 8.0.8 构建入口，公开版本定义为 0.25，并管理 ONNX／DirectML、引擎资源和第三方算法依赖。`build-local.ps1` 是当前本机 MSVC 整合版构建脚本，读取已有 CMakeCache，支持 ASCII junction；`build-windows-native.ps1` 和 `juce/cmake/` 是另一条 MSYS2 clang-cl/xwin 构建路线，不要混用同一缓存。

本机最近成功构建使用：

```powershell
# 在 HachiShifter-integrated 根目录执行。先检查该缓存仍指向当前源码。
& 'F:/VS/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build `
  'C:/Users/Asuna/.codex/visualizations/2026/09/24/01a0d2fe-79ca-70f0-88a0-1ae8129f7682/hachi-integrated/build-integrated' `
  --config RelWithDebInfo --parallel 2
```

这是本机现有路径，不是其他开发机的前置条件。产物位于 `build-integrated/HachiShifterNext_artefacts/RelWithDebInfo/HachiShifter Next.exe`。构建可能需要 SDK 目录权限；通过正常权限流程处理，不改系统安全设置。链接前确认目标 EXE 是否在运行，不擅自结束用户未保存的编辑器。

源码归档、无模型 CI 包和本机完整配布是不同产物。`.github/workflows/windows-build.yml` 当前在 new／next 触发无模型 Windows 构建并打包运行库、引擎资源与许可，不等于包含全部本机权重和用户音源。`tools/package-current-distribution.py` 仍带历史 0.2.2 默认路径，不能凭文件名直接运行发布当前版本。先读脚本参数、路径与排除规则。

2026-10-08 已整理当前配布根目录：116 份历史 JSON 验证报告位于 `verification/reports/`，旧版 `HachiShifter Next.previous-update065.bin` 位于 `backups/previous-executables/`；主程序、DLL、`build-info.json`、`mcp-live.json`、引擎和模型位置不变。历史报告中的原路径是当时的记录，现位置查 `verification/directory-layout.json`；配布的 `docs/目录结构.txt` 提供用户导航。新的测试输出统一写入 `verification/update编号/` 或 `verification/reports/`，不要再堆积根目录。`tools/organize-distribution.py` 默认预览，`--apply` 才移动，检查目标、内容校验和碰撞并保留迁移记录。本次目录整理不更新 EXE 或功能基线。

部署应核对 EXE 校验值、运行库、引擎资源与 `build-info.json`，并在目标配布环境验证；不能只依据开发构建成功宣称日常使用版已更新。更新 117 的验证入口是 `test-output/update117/verify.py`，包含原生包络 43 项（含实际 PCM 导出、原音路径、分离／裁剪／去重叠、界面与持久化）、OTO 包络 219 项（含各段贝塞尔、混合分段、旧曲线、预设、持久化、混音与界面截图）、空轨算法切换 22 项、中文 CVVC 48 项（含 HiFisampler 和外部重采样器实际合成），以及 `--smoke-integrated`。部署结果查 `test-output/update117/deployment.json` 的 `verified` 与 `installed_checks`，配布报告放在 `verification/update117/`。更新 112 的 OTO 重叠与编辑连续性证据保留在原测试目录。这类本地产物可能不在源码归档中。

## 验证入口和已有证据

测试有三层：`Main.cpp` 内联 CLI 分支、`juce/src/tests/*Smoke.h` 原生验证、`juce/tests/*.py` / `tools/` 集成验证。逐项命令见 [索引](AI_SOURCE_INDEX.md)。执行前读命令分支，某些测试还需要模型、音源、输入 WAV 或探针 EXE，不是所有命令都只有输出目录一个参数。

| 修改范围 | 优先查看的验证 |
| --- | --- |
| 裁剪 | `NativeAudioTrimSmoke.h`，`--smoke-native-audio-trim` |
| 非 U 移动拉伸 | `NativeNoteMoveSmoke.h`，`--smoke-native-note-move` |
| 重叠／显示聚焦 | `NativeAudioOverlapSmoke.h`、`NativeAudioOverlapFocusSmoke.h` |
| 粘连、断开、复制 | `NativeAudioLinkSmoke.h`、`NativeAudioDisconnectSmoke.h`、`NativeNoteCopyPasteSmoke.h` |
| 原音 F0／保真／还原／波形 | `NativeSourcePitch*`、`NativeWaveformPreviewSmoke.h`、`NativeRenderedWaveformSmoke.h` 与相应 Python 测试 |
| 原生音频包络 | `NativeEnvelopeSmoke.h`，`--smoke-native-envelope 输出目录` |
| OTO／包络／FLAG | `AdvancedEnvelope*`、`Oto*`、`ContinuousFlagSmoke.h`、`hifisampler_curve_smoke.py` |
| DS | `DiffSinger*Smoke.h` 与 `juce/tests/diffsinger_*.py`；替身测试不能取代真实音源验证 |
| 排队、导出、退出 | `PlaybackRenderSmoke.h`、`WavExportSmoke.h`、`NativeShutdownSmoke.h` |
| 保存和 MCP | `ProjectSafetySmoke.h`、`project_safety_live.py`、`mcp_live_smoke.py` |
| 工具栏和布局 | `--smoke-integrated-ui` 输出截图；同时核对浅／深色及不同宽度 |

功能更新 110 的本地记录：最终裁剪 smoke **22 项通过**，已安装 EXE 再次通过；同轮普通移动／拉伸 **116 项**、重叠 **45 项**和 integrated layout 通过。裁剪最后的 F0 显示边界修正后重跑裁剪测试；此前回归的二进制校验值保留在 `validation.json` 的嵌套记录中，不应声称所有历史检查都在最终哈希上重跑。

文档整理不需要重新执行神经模型或全套音频测试。以修改范围选验证；声音变化必须检查输出音频或指标，不能只看截图；交互变化要检查实际命中／拖动，不能只调用底层模型函数。

## 按问题定位

| 现象或需求 | 首先追踪 |
| --- | --- |
| 按钮位置、图标、提示不对 | `MainComponent::resized`、按钮 onClick、`Theme`、`I18n` |
| 预览正确但松手后位置错误 | `PianoRollComponent::mouseDrag/mouseUp` 与共用 plan，再查 ProjectModel |
| 拉伸变成循环、长音后半段无声 | OTO 分区映射、He 判定、当前包络适配、源读取范围和输出帧长度 |
| 只显示原始 F0 却改变原音 | `NativePitchIdentity`、请求中的 source/target MIDI、`canPreserveNativeSource` |
| 裁剪后声音变化 | 源／目标映射、采样取整、分数偏移、包络支持点和颤音参考时钟 |
| 改参数仍播放旧声音 | `writeNoteRenderFields`、请求键、文件／模型指纹、任务修订和缓存发布 |
| 波形不随修改更新 | 源近似还是渲染波形、波形 generation/signature、拖动期间旧缓存隐藏 |
| 单独点击区域后其他音符消失 | 原生编辑 scope、焦点／选区、`trackShowsAllNativeRegions`、子区域展开 |
| 关闭后进程残留 | 进程所有权、Job 句柄、取消传播、线程退出、ORT 生命周期 |
| DS 重生成覆盖手调 | 完整上下文、选区 mask、reference／actual／offset 层及修订校验 |
| 保存后恢复错误 | ProjectModel 序列化双向字段、默认值、ID／clipPartId、路径解析与工程缓存 |

## 后续维护这份导航

增加或拆分模块时更新本页对应职责，并更新逐文件索引；命令变化时重新核对 `Main.cpp` 与测试脚本参数。优先记录当前行为和关键不变量，历史过程留在 updates 与专门的评估文档。修改前用 `rg` 找实际符号，避免依赖会漂移的旧行号。保留已有用户改动，不把整个脏工作区当成本次修改。

更新 118 的定向回归：`test-output/update118/verify.py` 检查双击最近半音吸附、普通／标点工具、一次撤销恢复、原始音高参考与时间位置保留，并回归 UTAU 歌词双击和整合功能。结果与部署状态见 `test-output/update118/deployment.json`。右键拉直继续调用默认 `snapNativeToSemitone=false`，仅原生双击请求吸附。

更新 119：“响度包络／高级包络”入口使用普通 `juce::TextButton`，去掉下拉箭头，文字按按钮全宽居中；真正打开显示菜单的按钮仍使用 `DropdownButton`。验证与按钮截图查 `test-output/update119/`。

更新 121 的定向回归：`--smoke-flatten-pitch-line` 包含 `NativeBatchFlattenSmoke.h`，通过真实框选及完整双击按下／抬起序列，检查普通工具、标点控制点、跨区域分离音符批量吸附与拉直、多选保留、未选中音符隔离、源音高与时序保留，以及整组撤销／重做。结果与安装验证见 `test-output/update121/deployment.json`。

更新 122 的定向回归：`--smoke-native-linked-pitch <目录>` 验证三素材共享、接缝几何和 F0、控制点拖动预览、显示与请求逐帧一致、邻段渲染／波形缓存刷新、源音高保留、清音掩码、断开、撤销和工程重开，并生成普通／标点工具截图。另回归音频粘连、批量拉平、UTAU 共享线和原生波形预览。报告见 `test-output/update122/deployment.json`。

更新 123：`NativeSharedEnvelope.h` 按原生共享音高线的素材成员关系组织响度包络，以绝对时间合并真实控制点并移除内部切口的普通端点。`PianoRollComponent` 的包络面板、波形及拖动预览和 `AudioEngine` 的逐素材／整句 `noteGain` 共用同一曲线。编辑时切片存回各音符，合成的切口端点不成为新的组控制点；`AmplitudeEnvelopePoint.nativeSeamAnchor` 标记手动切口控制点并保存，避免其在重开时丢失。复制和断开前物化当前共享形状。独立重叠声部与 UTAU 不参与。回归入口 `--smoke-native-linked-envelope <目录>`，报告见 `test-output/update123/deployment.json`。

更新 124：`NativeTrimSource.h` 保存绝对源文件 F0 参考与可包含当前范围外锚点的 `ClipData.nativeTrimClock`。裁短保留隐藏时钟；扩展读取原文件并恢复源 F0，已有音高／响度控制点和颤音相位保持绝对时间位置。完整源 F0 通过不可变共享引用保留、二进制持久化；旧工程缺少隐藏 F0 时标记 `nativeSourcePitchPending`，由 MainComponent 后台分析源文件补齐，不在鼠标拖动中运行分析。时间轴拉伸、音符时间编辑、分离、复制与重开同时维护隐藏时钟。验证入口为 `--smoke-native-trim-expand <输出目录>`；测试报告在 `test-output/update124/` 与配布 `verification/update124/`。

更新 124 的开发版和日常配布均通过 12 组回归检查；新增向外扩展专项 41 项通过，包括裁剪恢复及剪切删除恢复两种实际 PCM 原音比对。证据：`test-output/update124/development/results.json`、`test-output/update124/deployment.json` 和配布 `verification/update124/`；部署报告已标记 `verified: true`。

更新 125：连接／分离工具中，双击同一原始录音的相邻粘连分割点调用 `joinNativeNotes`，删除该分界、保持总时间跨度，并改为一个统一拉伸音符。`NativeNoteJoin.h` 保留源时钟、源 F0、目标音高、共享响度与源参考；高级包络和颤音折入当前曲线。原有通用 `mergeNotes`（清空曲线）不用于此交互。分割点附近单击只选中，避免双击前先切换连接关系。支持同一源内部及连续同源子素材；有间隙、重叠、断开的独立区域不会隐式合并。音色参数不相容或源文件／源选区不连续时不移除分界，以免丢失素材信息。测试入口 `--smoke-native-note-join <输出目录>`。
