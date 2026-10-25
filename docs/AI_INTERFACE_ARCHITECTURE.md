# 实际功能主入口、数据流与架构审计

## 当前整合任务（0.25，尚未整体验收）

以下历史更新不是新素材编辑规则的权威；当前进度见 [实现状态](native-material-implementation.md) 和 [源码索引](AI_SOURCE_INDEX.md)。

- `MaterialPronunciationCatalog::index/resolve` 是只读发音目录与选择时重验证入口；复用 `OttoMaterialMigration::convertOtoEntry` 的原有转换和稳定 ID，不另写 OTO 解析器。OTTO 优先且缺失音频节不回退，重复别名保留物理来源；浏览不登记、不迁移、不写注音或 PCM。
- `PronunciationTiming` 是 P/C、默认单 C/V、显式命名 V 与保护区伸缩的共同权威。分析帧不成为作者切点。
- `ProjectModel::setPointCurves` 的共享规划器同时供提交与 `previewPointCurves` 使用；音符及显式连接图可原子混合编辑，一次撤销。连接曲线只存在连接图，不复制进音符。
- `AudioEngine::projectTrackNotePitch` 共用既有宿主音高规划：逐音符基础曲线、名义自动过渡、最后显式连接窗口。UI 与实际 UTAU 数据包读同一结果；原生文件渲染策略独立保留。显式 `pitchSupportAuthored` 仅由实际越界编辑设置，不按来源或旧控制点形状推断，也不随视图开关改变。
- 同一个 `MaterialAnnotationSession` 按保存目的地分流源文件发布与当前音符回调；局部保存必须在会话层阻止源发布，不能只拦截某一个 Save 按钮。目前目的地回调专项已通过，完整项目局部适配及入口整合仍待验收。
- 实际测试结果和未通过项须分开报告。模型状态 ready 不证明此次分析使用了该模型；需记录实际 worker 后端及回退警告。当前停止条件是完成非自动注音功能后等待用户验收；ASR／TIFA 自动注音不在本轮实施范围。公开版本仍为 0.25、WCSNDM 仍为 0.0803。

### 新模块与事务边界（2026-10-22，仍在集成）

- [ProjectModel.h](../juce/src/ProjectModel.h) 的 `TrackType` 明确区分 Hachi／UTAU／DiffSinger；`effectiveTrackType` 兼容旧工程，新建／UST 导入缺省 Hachi。轨道类型、伴奏与输出引擎是不同维度；[MainComponent.cpp](../juce/src/MainComponent.cpp) 只在既有编辑器上切换类型界面，不能靠选择另一个轨道提交旧轨未确认文本。`cancelPendingEditingForTrackChange` 是无模型写入的放弃入口。
- [UtauRenderer.h](../juce/src/backend/UtauRenderer.h) 的 `RequestedVoicebankFamily` 是请求契约，`resolveRequestedVoicebankFamily` 是调度／工作线程的共同家族检查。禁止把错误家族的银行交给另一个渲染器来掩盖失败。家族拒绝专项通过不等于真实 DiffSinger 模型正向神经测试通过。
- [UstxImporter.h](../juce/src/backend/UstxImporter.h) 只负责严格 YAML 格式边界，进入既有 `UstProject` 和公共工程映射，不新增平行谱面模型／渲染器。保留不支持字段及警告不代表完整 OpenUTAU 保真。
- [OttoMaterialIO.h](../juce/src/OttoMaterialIO.h) 的音频 ID 是“注音登记变体”身份，不是物理文件身份。同路径可以多 ID；必须显式传所选 ID，缺少 ID 的歧义查找拒绝。模型／目录／拖放都应保持这层身份，不能靠文件名或行序重绑定。
- [CommunityDatasetIO.h](../juce/src/CommunityDatasetIO.h) 是社区 CSV 读取、显式分组、P/C 确认、规划与冲突保护发布的唯一业务权威。[CommunityDatasetImportDialog.h](../juce/src/CommunityDatasetImportDialog.h) 只是导入选择向导，不是第三个注音编辑器；任务由宿主管理的 worker 执行，发布前由宿主取得修订门闩并调用既有预检。新向导当前宿主模拟 GUI 测试在运行、未报告通过，生产 Main 接线也未验收；已通过的是服务与合成数据集真实渲染链。
- [ProjectMaterialRegistration.h](../juce/src/ProjectMaterialRegistration.h) 规划／暂存／发布文件，[ProjectMaterialRegistrationFacts.h](../juce/src/ProjectMaterialRegistrationFacts.h) 只在用户确认携带摆放解释后提供事实；默认拒绝，不由音符数或 F0 帧猜音节。复制一次物理录音，按独立变体生成稳定 ID。`ProjectModel::adoptRegisteredMaterialBindings` 只改引用与绑定，一次撤销，保留音乐、源时钟和原始分区；`registrationMusicSourceFingerprint` 是这种保留的校验，不能替代实际导出音频比较。
- 登记事务边界不是分布式原子提交：先预检修订与来源指纹，再发布租约、模型收纳；收纳成功即 `retainAfterAdoption()`，之后偏好登记失败仅警告／重试，不允许析构回滚删除项目已采用的录音。未收纳的暂存目录才能按所有权凭据安全清理。
- 局部编辑以 [MaterialAnnotationSession.h](../juce/src/MaterialAnnotationSession.h) 的 `EditScope::thisNote`／`NoteScopeBinding` 及模型局部应用／恢复接口为权威，必须同时检查稳定 ID、修订与源指纹；源范围、局部基线 `-1` 哨兵及第三方调音字段不得自行归一化。28b 已验证基线哨兵保留／指纹修正及真实 teio 登记三阶段实际 PCM 一致（[23 项输出](<G:/hjs-tc/tmp/teio-reg28/stdout.txt>)），但不代表局部两个生产入口或 Main 登记控制器已全面验收。
- View 菜单的 `ui.pitchScope`／`ui.showOverlap` 只影响编辑范围或显示，不做音乐撤销或重渲染。重叠显示缺省关闭，只有音符内黄色交叠线，无整高竖带；素材选择器仍由右键显式打开一个目标，与显示开关独立。

最新已运行与待运行事项分别见 [实现状态](native-material-implementation.md)，命令和模块见 [逐文件索引](AI_SOURCE_INDEX.md)。接口声明清单和源码指纹待可变头文件稳定后重生成；本次不以静态清单作为运行通过证据。


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

更新 163：单独选中 CVVC 音素后，“高级包络”不再被提前返回拦截。PianoRollComponent::selectedCvvcEnvelope 提供当前音素的范围与曲线；MainComponent 复用 AdvancedEnvelopePanel 的整段编辑界面，应用仍统一调用 setCvvcEnvelope，仅提交该角色。包络点可保存可选贝塞尔控制柄，cvvcEnvelopeValue 与两个 UTAU 渲染器共用同一插值，曲线加入混音缓存键。打开／未改动应用不写工程；草稿应用、试听、修订保护、保存和撤销保留。恢复默认使用该音素标准梯形。

更新 162：修复旧 CVVC 工程保存的整音符默认分贝包络继续叠乘，造成句首弯曲起音和收尾弯曲淡出。ProjectModel::fromValueTree 仅对缺少 cvvcEnvelopeVersion、无高级包络及 UST 原始段的中文 CVVC 音符，识别四点 0 dB 平台、5/15 ms 起音、35 ms 收尾的历史默认形状，恢复为各音素独立标准梯形。新工程保存 cvvcEnvelopeVersion=1；现代显式曲线、非标准时长／音量、线性手绘和非 CVVC 曲线保持。音频与拼字混音缓存版本同步更新。真实“CVVC测试”工程专项覆盖修复前失败、保存重开、首尾像素位置、两种交叉方式逐采样增益及 HiFiSampler／WCSNDM 合成。

核对日期：2026-10-13。对象是本机 `HachiShifter-integrated` **当前工作树，包括未提交源码**；公开版本 0.2.4，导航基线更新 148。不是 GitHub 某个提交或已配布 EXE 的等价性证明。本次只整理文档和可再生成的接口索引，没有重构运行时代码。

先读本页理解“一个功能由谁负责”，再查 [实际接口声明清单](AI_API_REFERENCE.md) 获取完整签名／源位置、[代码指南](AI_CODE_GUIDE.md) 获取行为细节、[逐文件索引](AI_SOURCE_INDEX.md) 找外围模块。历史更新不是另一份接口定义。

后续落地（0.25／更新 150–151）：手动选择和默认 UTAU 音源绑定统一使用 `MainComponent::selectVoicebank`，素材管理器显式导入复用 `AssetManagerComponent::registerFolder`。两个 GUI 入口均去掉只为统计数量而调用的整库 `SampleSettings::importVoicebank`；OTO 仍由编辑器／既有后台渲染索引按需读取。旧库存接口只留在独立 MCP 命令与历史测试，未作接口删除。本页其余条目保持原静态审计语境。

## 结论与使用原则

后续落地（0.25／更新 157）：拼字布局缓存复用 `UtauPhonemeSpan`，由 `CvvcPhonemeCache.h` 有界序列化，唯一写入口 `ProjectModel::rememberCvvcPhonemes` 带 revision 保护；它属于派生数据，不推入撤销、不改变渲染键。背景补全只调用 `planChineseCvvc`，禁止为显示标签调用完整声码器。`toValueTree` 的 `includeDerived` 区分实际文件保存与内容指纹，递归片段遵循相同规则。

后续落地（0.25／更新 156）：自动交叉淡化的唯一写入入口为 `ProjectModel::crossfadeNotes`，输入为已解析发声区及原始包络、预期工程修订号。GUI `PianoRollComponent::crossfadeSelectedNotes` 只收集已有布局，不在按钮事件中扫描音源；结果复用普通／CVVC 响度曲线，合成、保存、撤销沿用现有链路。原来的 `spliceSelectedNotes`／“强制连接”用途保留，不能替代本动作。

更新 155：`backend::standardCvvcEnvelope` 统一生成各 CV／VC／收尾的默认四点直线梯形（5 ms 起音、35 ms 释放，短片段用三等分避免点序倒置）。`planChineseCvvc` 与 `PianoRollCvvcEnvelope.h` 复用该函数；仅在无手动父包络、无分片响度覆盖时使用。已有手动曲线及导入曲线保留，默认值不写回工程。`cvvcDefaultEnvelopeGain` 传递默认包络的基准音量，防止调节音量后被误判为手工平直曲线；CVVC 渲染键升级至 v3。

更新 154：`backend/UtauMixEnvelope.h` 保存渲染器实际使用的采样级自动淡化窗口，传统 UTAU 与 HiFisampler 混音、CVVC 响度可视化共享增益计算。`noteMixEnvelope` 经发音器折回 `UtauPhonemeSpan::mixEnvelope`，只读数据不写工程；实线显示手动包络乘自动淡化，虚线／圆点保留编辑值，VC／CV／收尾用不同颜色，交叉区着色。HiFisampler CVVC 波形回调移到已确定自动混音增益之后，避免预览缺少淡化。`utauPhonemeMixContextHash` 防止邻音时序或引擎变化后继续显示过期交叉曲线；包络／FLAG 编辑保留布局和有效淡化。

更新 153：CVVC 分片包络直接归属于 `NoteData::cvvcEnvelopes`，按角色／参数保存，鼠标工具统一提交 `ProjectModel::setCvvcEnvelope`，规划器统一转换为合成片段时间。界面仍用父音符 ID 加角色标识，避免构造虚拟工程再写回父音符。布局校验与声音校验分开：`utauNotePhonemeHash` 不包含包络／FLAG，编辑时保持可抓取的布局；最终声音仍按完整渲染键失效。没有新增音源扫描或 UI 线程模型推理。

更新 152：工具栏和右键菜单共用 `MainComponent::phonemizerMenu`，选择仍统一进入 `outputEngineItemChosen`。`ChineseCvvcRules::aliases` 是中文 CVVC 配置模板展开的唯一入口；`UtauRenderer` 在既有音源索引中增加大小写精确项，自动发音的查询与已解析片段的合成均通过 `indexedAlias`。人工别名的原查询语义保留，不增加重复扫描；中文 CVVC 缓存键包含规则修订标识。

主要问题是**同一份工程内容反复复制、全量通知后重新推导，以及播放线程与重建流程共享大锁**。不是算法后端数量多本身导致慢。应先缩小数据与失效范围，再整理职责；把大文件拆成很多类而保留原数据往返，不会解决这些问题。

“一个功能一个接口”指一个业务动作只有一个权威实现。按钮、右键、快捷键、MCP 可以是多个入口，但应调用同一动作；单对象调用可以包装批量入口。不是把所有功能塞进一个 `execute(string, JSON)`，也不是让一个动作只允许一个调用者。

最小设计约束：

1. 正式工程修改只由 `ProjectModel` 提交；交互预览保留局部草稿，完成手势提交一次。
2. 源音频／源 F0 是事实，目标曲线／时间映射是编辑意图，PCM 是派生结果。三者不互相冒充。
3. 同一次修订的只读数据能共享就共享；后台任务只捕获所需轨道、区域、源资源及修订，不无条件复制整工程。
4. 需要不同结果的能力保留参数或策略；不为菜单位置、显示样式或输入格式复制业务实现。
5. 用户明确选择的合成后端失败应报告失败，不私自换算法。兼容旧工程在读取边界转换，不能让旧语义持续扩散。
6. 取消、撤销、原子保存、修订校验属于正确性，不应作为“兜底冗余”删除。去除的是重复执行和隐藏替代路径。

## 核对范围

静态清点 `juce/src` 的自有非测试 `.cpp/.h/.inc` 共 **143 个文件、91,708 行**，含注释和空行；`Main.cpp` 仍包含内联诊断，因此并非纯产品代码行数。另核对 DiffSinger Python 桥、HAMOOD/曲速任务入口、现有维护文档与测试入口。第三方算法实现按适配边界归类，不逐个列其内部函数。

[接口清单](AI_API_REFERENCE.md) 覆盖 11 个边界类的 **342 个公开声明**（重载和生命周期分别计数），其中 `ProjectModel` 162 个；列出 **71 个 MCP 注册命令**及 Main.cpp 中 **243 个启动参数字面量**。数字不是功能数，也不代表 243 项测试已通过。源码指纹和统计口径见 [扫描记录](interface-audit-inventory.json)。UI 私有事件和数学辅助函数不作为额外产品接口；重要辅助规则在下表标出唯一归属。

## 功能到主入口的对应

以下 `PM` 表示 [ProjectModel](../juce/src/ProjectModel.h)，`AE` 表示 [AudioEngine](../juce/src/AudioEngine.h)。每行是一项业务能力；“主入口”是当前实际存在且应优先复用的实现，不是已经完成接口收敛的承诺。为避免重复目录，详细字段、重载和辅助查询集中在接口清单。

### 工程、交换格式与轨道

| 功能 | 当前主入口 | 输入 → 输出／副作用 |
| --- | --- | --- |
| 读取工程 | `PM::snapshot` | 当前模型 → 值类型 ProjectData；现为整工程复制 |
| 判断过期结果 | `PM::revisionNumber` | 模型 → 修订号；异步提交前比对 |
| 新建／打开后切换文档 | `PM::resetDocument` | 新 ProjectData → 清除旧文档撤销历史 |
| 提交整份编辑草稿 | `PM::replace` | ProjectData → 一次工程替换及撤销 |
| 撤销 | `PM::undo` | 恢复上一状态并通知 |
| 重做 | `PM::redo` | 恢复后一状态并通知 |
| 保存 HJPX | `PM::save` | 路径 → 成功／错误；ProjectFileIO 原子写入和校验 |
| 读取 HJPX／兼容工程 | `PM::load` | 文件 → 工程；格式转换集中在边界 |
| 自动恢复 | `ProjectRecovery::tick` | 模型修订／工程文件 → 后台恢复快照；`saveRecoverySnapshot` 写盘 |
| 导入普通音频 | `PM::addAudioFile` | 文件、时长、位置、轨道 → clip ID；UI/MCP 负责发起分析 |
| 导入 MIDI | `PM::addMidiFile` | 文件 → 音符与曲速；选轨入口 `addMidiTrack` 复用 MIDI 领域规则 |
| 导出 MIDI | `PM::writeMidiFile` | ProjectData → MIDI |
| 导入 UST | `PM::addUstFile` | UST + 导入模式 → 添加轨道或替换工程 |
| 导出 UST | `PM::exportUst` | 单轨 → UST；保留可往返的原格式数据 |
| 导入 Melodyne MPD | `backend::MelodyneImporter::importProject` | MPD → ProjectData；`SampleSettings::convertMelodyneProject` 转原生注释，不另建一套编辑器 |
| 新建轨道 | `PM::addTrack` | 名称、创作／素材／伴奏属性 → track ID |
| 删除轨道 | `PM::removeTrack` | track ID → 删除及撤销 |
| 轨道重命名 | `PM::setTrackName` | ID、名称 → 工程显示信息 |
| 创作状态 | `PM::setTrackCompose` | ID、开关 → 音符编辑／合成状态 |
| 素材轨试听语义 | `PM::setTrackReferenceOnly` | ID、开关 → 是否只在工作轨试听 |
| 静音 | `PM::setTrackMuted` | ID、开关 → 混音可听状态 |
| 独奏 | `PM::setTrackSolo` | ID、开关 → 混音可听状态 |
| 轨道音量 | `PM::setTrackVolume` | ID、线性增益 → 混音增益 |
| 轨道声像 | `PM::setTrackPan` | ID、声像 → 混音声道权重 |
| 全局曲速／拍号 | `PM::setTempo` | BPM、分子、分母 → 曲速结构 |
| 曲速变化点 | `PM::setTempoChange` | 四分音符位置、BPM → 曲速映射；按参数同步音符 |
| 时间／拍位置转换 | `ProjectData::secondsForQuarterPosition` | 拍 → 秒；逆向 `quarterPositionForSeconds`，不能用固定 BPM 在调用处重算 |
| 网格显示 | `PM::setGridDivision` | 分度 → 工程网格 |
| 音符编辑最小单位 | `PM::setNoteEditDivision` | 分度 → 编辑约束；原生自由裁剪不使用该量化 |
| 基准调 | `PM::setBaseScale` | 调名 → 工程基准调 |

### 区域、音符和源时间

| 功能 | 当前主入口 | 输入 → 输出／副作用 |
| --- | --- | --- |
| 创建区域 | `PM::addClip` | 轨道／范围 → clip ID |
| 创建空调音区域 | `PM::addEmptyTuningClip` | 轨道／范围 → 可编辑空区域；由 canAdd… 检查重叠 |
| 区域移动 | `PM::moveClips` | IDs、时间偏移、轨道偏移 → 批量移动；单区域另有 moveClip |
| 区域整体拉伸 | `PM::resizeClip` | 新起点／长度 → 统一调整曲线和源时钟 |
| 可编辑区域长度 | `PM::setClipEditableLength` | 新长度 → 只改外框末端；音符和源音频时钟保留，单次撤销 |
| 区域裁剪 | `PM::trimClip` | 新可见范围 → 保留素材时钟，区别于拉伸 |
| 区域分割 | `PM::splitClip` | ID、绝对切点 → 新 clip ID；canSplitClip 是可用性查询 |
| 区域合并 | `PM::mergeClips` | IDs → 组合区域，保留源 parts |
| 区域复制 | `PM::duplicateClip` | 源 ID、位置 → 新 ID 与深复制的编辑内容 |
| 区域删除 | `PM::removeClips` | IDs → 删除；单项 removeClip 尚有独立实现 |
| 区域增益 | `PM::setClipGain` | ID、线性增益 → 混音参数 |
| 区域增益包络 | `PM::setClipGainEnvelope` | ID、点集 → 区域自动化 |
| 区域淡入淡出 | `PM::setClipFades` | ID、两端秒数 → 混音包络 |
| 区域静音 | `PM::setClipMuted` | ID、开关 → 播放状态 |
| 创建音符 | `PM::addNote` | 区域、时间、音高 → note ID；addNoteFrom 用模板内容 |
| 普通音符复制 | `PM::duplicateNotes` | IDs、目标区域、时间偏移 → 新 note IDs |
| 粘贴原生切片 | `PM::insertNativeAudioClips` | NativeAudioClipboard 组织的素材／源映射 → 新区域及音符 |
| 普通音符删除 | `PM::removeNotes` | IDs → 删除；无音高录音对象还要删除对应源播放范围 |
| 波纹删除 | `PM::removeNotesRippling` | IDs → 删除并收拢后续内容；不同于普通删除 |
| UTAU 音符移动 | `PM::moveUtauNotes` | IDs、时间／音高差 → 保留合成音符时序规则 |
| 原生音符移动 | `PM::moveNativeNotes` | IDs、差值 → NativeNoteTiming 计算源映射及联动 |
| 原生端点拉伸 | `PM::resizeNativeNoteEdge` | ID、差值、左／右 → 调整发音速度和关联段 |
| 原生端点自由裁剪／拓展 | `PM::trimNativeNoteEdge` | ID、差值、左／右 → `planNativeNoteTrim`，受源文件边界约束 |
| 普通音符长度 | `PM::resizeNote` | ID、新起点／时长 → UTAU 等普通音符时长编辑 |
| 音符分割 | `PM::splitNote` | ID、切点 → 两部分 |
| 普通音符合并 | `PM::mergeNotes` | IDs → 一个音符 |
| 原生相邻切点消除 | `PM::joinNativeNotes` | 两个可合并 ID → 总长度不变、统一拉伸的音符 |
| 音符连接关系 | `PM::setNotesConnection` | IDs、连接开关 → 相邻连接；toggleNoteConnection 是单项入口 |
| 跨素材粘连音频 | `PM::linkNativeAudio` | IDs → nativeAudioLinked 组合，保留不同源时钟 |
| 断开音频 | `PM::disconnectNativeAudio` | IDs → 独立区域，物化共享曲线；不是只清一个连接标志 |
| 允许／关闭音频重叠 | `PM::setNativeAudioOverlap` | track ID、开关 → 关闭时按既有规则裁掉后段重叠头部 |
| 插入音符前空隙 | `PM::insertGapBeforeNote` | ID、秒数 → 后移内容 |
| 收拢音符前空隙 | `PM::closeGapBeforeNote` | ID、秒数 → 前移内容 |
| 补齐无音高片段 | `PM::markNativeUnpitchedRegions` | clip ID → 可编辑无音高对象，不是语义噪声分类器 |
| 更新隐藏源分析 | `PM::setNativeTrimSourceAnalysis` | 源文件、分析音符 → 裁剪拓展用源音高事实 |
| 解析源时间映射 | `nativeSourceTimeMap` | ClipData → 单调目标／源秒数映射；正逆查表复用 NativeSourceTimeMap.h |

`splitClip`、`splitNote`、`joinNativeNotes` 的编辑对象和源时钟后果不同，不是应该删掉的三个同义接口。`linkNativeAudio` 与 `setNotesConnection` 也不完全等价；应共用必要的关系计算，不把所有关系压成一个 bool。

### 音高、表达、包络和显示

| 功能 | 当前主入口 | 输入 → 输出／副作用 |
| --- | --- | --- |
| 整体移调 | `PM::transposeNotes` | IDs、半音差 → 平移中心与目标；transposeNote 已包装批量入口 |
| 指定中心音高 | `PM::setNotesMidi` | IDs、MIDI → 指定音高 |
| 取平均音高 | `PM::averageNotesMidi` | IDs → 平均中心 |
| 吸附音高 | `PM::quantizeNotesMidi` | IDs、步长 → 标准音高网格 |
| 双击拉直并吸附 | `PM::flattenNotePitch` | IDs、snapNativeToSemitone → 每个音独立拉直；默认工具启用，标点工具不调用 |
| 还原原始录音音高 | `PM::restoreNativeSourcePitch` | IDs → 去除目标音高编辑，保留位置／拉伸 |
| 手绘／直线／标点曲线 | `PM::setNotePitchCurve` | ID、稀疏目标 MIDI 点 → 同一目标曲线；工具只是生成点集方式 |
| 共享音高求值 | `sharedPitchLines` | TrackData → SharedPitchLines；显示和单段／整句渲染共用 |
| 漂移 | `PM::setNoteDrift` | ID、强度 → 目标音高表达 |
| 音高起伏 | `PM::setNoteModulation` | ID、强度 → 目标音高表达 |
| 共振峰 | `PM::setNoteFormant` | ID、半音差 → 共振峰参数 |
| 气声 | `PM::setNoteBreath` | ID、强度 → 合成参数，非显示判断 |
| 张力 | `PM::setNoteTension` | ID、强度 → 合成参数 |
| 音符音量 | `PM::setNoteGain` | ID、线性增益 → 音符增益 |
| 辅音／Attack 时钟 | `PM::setNoteAttack` | ID、辅音时长、速度 → 源目标关系；只改速度可用 setNoteAttackSpeed |
| 颤音参数 | `PM::setNotesVibrato` | IDs、参数、可选真实线显示 → 一次提交；`vibratoCentsAt` 求值 |
| 颤音烘焙到音高 | `PM::bakeNoteVibratoIntoPitch` | ID → 将效果写进曲线并关动态颤音，避免叠加两次 |
| 响度包络点编辑 | `PM::setNotesAmplitudeEnvelopes` | 每音符点集 → 响度曲线；单音符入口应为批量的薄包装 |
| 包络基准百分比 | `PM::setNotesAmplitudeEnvelopeBase` | IDs、百分比 → 按线性幅度缩放，保留静音 |
| 原生高级包络 | `PM::setNotesNativeEnvelope` | IDs、NativeEnvelopeSettings → 区间淡变／混合分段 |
| OTO 高级音头／尾包络 | `PM::setNotesTailFade` | IDs、mode、TailFadeSettings → OTO 区间形状；不等同于响度点编辑 |
| 共享响度求值 | `nativeSharedEnvelopes` | 轨道 → 粘连包络；切片及断开时 materializeNativeSharedEnvelope |
| 歌词批量设置 | `PM::setNoteLabels` | ID/文本列表 → 同一步撤销；单项 setNoteLabel 尚有独立实现 |
| 歌词转拼音 | `PM::convertTrackLyricsToPinyin` | 轨道 → 原音符歌词转换 |
| 音高突变平滑 | `PM::setTrackNsfSmoothPitchTransitions` | 轨道、开关 → NSF 目标 F0 条件处理 |
| 谐波／源噪声保护 | `PM::setTrackNsfNoiseProtection` | 轨道、开关 → 非 U HN-SEP／清音 PCM 路由 |
| 渲染稳健音高参数 | `PM::setNoteRobustPitchCurve` | 音符、开关 → 逐音符过滤域；不同于全轨突变平滑 |
| 无音高隐藏／人为圆点线 | `NativePitchVoicingDisplay` | 源有声掩码、人工目标范围 → 绘制裁切，不改 F0 |
| 气声虚线 | `NativePitchVoicingDisplay` + `SourceBreathiness` | 后台声学证据 → 已有音高线的短划样式，不决定合成清音 |
| 原始参考音高线 | `PianoRollComponent` 的参考线显示路径 | 源 F0 → 参考虚线，独立开关，不复制一套目标曲线 |
| 原始／渲染后波形 | `SourceWaveformPreview` / `AE::nativeClipWaveforms` | 源映射预览／实际 PCM 峰值 → 两种不同证据；不能把预览冒充渲染结果 |
| 重叠聚焦和文字 | `NativeAudioFocus` / PianoRoll 绘制 | 当前选中关系 → 层级、亮度、波形上方歌词；不改声音 |
| 音符提示／正常显示 | `PM::setClipNoteHints` / `setClipNormalDisplay` | 区域显示状态 → 布局行为；应产生显示失效而非全音频同步 |

### UTAU、音源与 DiffSinger

| 功能 | 当前主入口 | 输入 → 输出／副作用 |
| --- | --- | --- |
| 选择轨道音源 | `PM::setTrackVoicebankDirectory` | 轨道、目录 → 音源引用，不复制音源 |
| 普通／界／谋 UTAU | `PM::setTrackUtauMode` | 轨道、枚举 → OTO 解释策略；全局 setUtauMode 作用域不同 |
| 自动中文 CVVC 开关 | `PM::setTrackChineseCvvc` | 标准 UTAU 轨道、开关 → 启用发音器 |
| 中文 CVVC 编排 | `backend::planChineseCvvc` | UtauRenderRequest → CV/VC/句尾计划；UI 不单独拆音 |
| 选择输出引擎 | `PM::setTrackOutputEngine` | 轨道、类型／路径 → 引擎覆盖；AE 保存全局默认 |
| 音源 OTO 读取 | `SampleSettings::loadVoicebankOto` | 音源根目录 → OTO 条目；合成查找由 UtauRenderer 索引承接 |
| OTO 波形与参数编辑 | `SampleSettings::updateVoicebankOtoEntry` | 原／新条目 → 写回文件；UI 用 OtoWaveformEditorComponent |
| 界 OTO 写回 | `SampleSettings::updateJieVoicebankOtoEntry` | 条目 → 界格式文件，保留格式语义 |
| 谋 OTO 写回 | `SampleSettings::updateMouOtoEntry` | 条目 → 谋格式文件 |
| 创建界分区 | `SampleSettings::createJieOto` | 音源目录 → 界 OTO；createJieOtoFromMou 是有来源转换的另一操作 |
| 创建谋分区 | `SampleSettings::createMouOto` | 音源目录 → 谋 OTO |
| 音符局部 OTO 覆盖 | `PM::setNoteUtauOto` | note ID、override → 工程覆盖，不改原音源文件 |
| 还原音符 OTO | `PM::clearNotesUtauOto` | IDs → 删除工程局部覆盖 |
| 先行／重叠覆盖 | `PM::setNoteUtauTimingOverrides` | ID、秒数 → 工程级时序 |
| STP | `PM::setNotesUtauStp` | IDs、秒数 → 起始裁剪 |
| 辅音速度 | `PM::setNotesUtauConsonantVelocity` | IDs、速度 → 音符覆盖；轨道继承由 setTrackUtauConsonantVelocity 管理 |
| FLAG 文本 | `PM::setNotesUtauFlags` | IDs、文本 → FLAG；轨道公共项由 setTrackUtauGlobalFlags 管理 |
| FLAG 连续曲线 | `PM::setNoteUtauFlagCurves` | flag、多音符点集 → 同次提交；setNoteUtauFlagCurve 包装单音符 |
| FLAG 启停／清除 | `PM::setNotesUtauFlagCurveEnabled` / `resetNotesUtauFlagCurves` | IDs → 控制使用或恢复；并非另一渲染实现 |
| 界／谋分区覆盖 | `PM::setNotesUtauJieSplit` | IDs、分区参数 → 音符分区；clearNotesUtauJieSplit 还原 |
| 音源合成解析 | `UtauRenderer::resolveVoiceSample` | 音源、别名、音高 → 样本及 OTO；别名查找集中处理 |
| 标准重采样／内置 HiFisampler | `UtauRenderer::render` | UtauRenderRequest → PCM、时序／音素回调、错误 |
| UTAU 原生 NSF 整句 | `RenderService::renderNsfUtau` | UTAU 请求、模型 → NSF 音源合成，不是非 U 录音调音 |
| DS 音源配置 | `PM::setDiffSingerOptions` | 轨道、语言、声库 speaker → 工程配置 |
| DS 能力查询 | `DiffSingerRenderer::invoke`，operation=`inspect` | 音源 → 模型／参数支持信息 |
| DS 读音覆盖 | `PM::applyDiffSingerPronunciation` | 预期修订、读音表 → 提交覆盖并校验过期 |
| DS 音素推断 | `DiffSingerRenderer::invoke`，operation=`timing` | 整句上下文 → 音素边界 |
| DS 音素边界编辑 | `PM::applyDiffSingerTiming` | 预期修订、上下文及起点 → 时长覆盖 |
| DS 音高预测 | `MainComponent::generateDiffSingerPitch` | 选区／整轨 → 后台预测；PM::applyDiffSingerPitch 负责提交 |
| DS 参数预测 | `MainComponent::generateDiffSingerParameters` | 选区／整轨 → 后台预测；PM::applyDiffSingerParameters 提交 |
| DS 独立音高偏移 | `PM::setDiffSingerPitchOffset` | ID、半音偏移点 → 保留预测基准 |
| DS 参数实际值 | `PM::setDiffSingerParameterCurve` | ID、参数键、点集 → ABS 实参；与偏移层区分 |
| DS 预测缓存回收 | `PM::rememberDiffSingerParameters` | 渲染回传的预测曲线 → 工程可用预测；不当作新人工参数 |
| DS 单点恢复 | `PM::restoreDiffSingerPitchPoint` | 修订、点位置 → 还原预测参考 |
| DS 合成 | `DiffSingerRenderer::render` | UtauRenderRequest → C++/Python 桥 → PCM；没有必要再建一套通用 RPC 层 |

DS Python 桥的实际操作是 `inspect / phonemize / timing / pitch / parameters / render`，见 [bridge.py](../juce/engines/diffsinger/bridge.py#L427)。同一个 `invoke` 是进程适配边界，操作枚举有业务区别，不能因“只有一个接口”而隐藏这些区别。

### 分析、播放、导出、素材与自动化

| 功能 | 当前主入口 | 输入 → 输出／副作用 |
| --- | --- | --- |
| 原录音分析 | `AnalysisService::analyse` | 文件、配置 → GAME 音符 + FCPE F0／实际后端说明 |
| 附着源 F0 | `AnalysisService::applySourcePitch` | ClipData、源分析 → 保留用户区域和目标编辑 |
| 工程源音高重分析 | `AnalysisService::reanalyseProjectSourcePitch` | 工程、配置 → 更新源事实；不是改用户目标 |
| 变调算法选择 | `PM::setTrackPitchAlgorithm` | track ID、枚举 → 渲染路由 |
| 拉伸算法选择 | `PM::setTrackStretchAlgorithm` | track ID、枚举 → 特征时间映射策略 |
| 先处理／先拼接 | `PM::setTrackRenderOrder` | track ID、顺序 → 单段或成组请求 |
| 播放工程同步 | `AE::syncProject` | ProjectData → loadedClips、缓存、待渲染任务 |
| 原生录音渲染调度 | `RenderService::renderMld5File` | Mld5FileRenderRequest → RenderedAudio；名字历史遗留，实际不只 MLD5 |
| NSF 原生合成 | `NsfHifiganRenderer::render` | PCM、目标 F0、映射、选项 → PCM 与实际后端信息 |
| WCSNDM 非循环拉伸 | `backend::buildWcsNonLoopMel` | 源 PCM/F0 → 源 LLSM 特征缓存 → 按目标映射重建物理 Mel；非外置 EXE 调用 |
| 保留未编辑原音 | `backend::canPreserveNativeSource` | 请求及采样对齐信息 → 是否直通 PCM；不是失败后偷偷兜底 |
| 任务优先级／过期取消 | `PlaybackRenderQueue` | schedule.key、播放位置、有效键集 → 排队／取消；一个队列管理各 lane |
| 开始播放 | `AE::play` | 已同步播放内容 → transport 状态 |
| 停止播放 | `AE::stop` | transport → 停止，区别于销毁全部推理资源 |
| 跳转位置 | `AE::setPosition` | 秒数 → 播放游标和优先级 |
| 素材文件试听 | `AE::setAuditionFile` | 文件 → 试听 reader，不插入工程 |
| WAV 导出 | `AE::exportWav` | 文件、范围／轨道、格式选项 → 离线混音；组件类型由统一选项约束 |
| 检测导出分量能力 | `AE::componentExportIssue` | 工程、引擎、导出范围 → 明确拒绝不支持的分量 |
| 合成后波形更新 | `AE::refreshNativeWaveforms` / `refreshUtauWaveforms` | 渲染代际 → 共享峰值快照；两类音频时序不同 |
| 伴奏 BPM 识别 | `AccompanimentTempoTask` | 捕获的 ClipData → Beat This! 结果；不修改工程 BPM |
| 伴奏和弦／节拍分析 | `hamoodaudio::analyse` | 工程区域、配置 → 缓存的时间戳证据，不直接调音 |
| 和声预览 | `hamood::analyse` | 工程、Options → Plan，无工程写入 |
| 和声生成 | `hamood::generate` | 草稿、Options、Plan → 新轨道 IDs；PM 一次提交 |
| 和声工程元信息 | `PM::setHamoodState` | JSON 状态 → 持久化元信息 |
| 素材 HJM 读写 | `SampleSettings::load` / `save` | 源文件、分段行 → 同格式注释读写，非项目快照 |
| 工程音源列表 | `AssetManagerComponent::setProjectVoicebanks` | ProjectData → 去重的工程音源文件夹，不复制音源 |
| 音源展开／文件选择 | `AssetManagerComponent` 的树行缓存及鼠标处理 | 已展开文件夹 → 可见行／选区；不另存一套工程素材状态 |
| OTO/HJM 素材设置传播 | `PM::applySourceSettings` | 源文件、设置 → 关联工程素材更新 |
| 自动化 headless | `McpServer::executeTool` | 命令／参数 → PM 或服务调用；协议输入只在这一层解析 |
| 自动化 live | `MainComponent::handleLiveMcp` | 窗口命令 → 选区／任务／修订检查 → 复用 McpServer 或 PM；实现于 MainComponentMcp.cpp |
| 连接真实窗口 | `LiveMcpBridge::runProxy` | 会话 ID → 本地文件信箱，不把 GUI 对象跨线程共享 |
| 推理进程退出 | `DiffSingerRenderer::shutdown` / `UtauRenderer::shutdownHfDaemon` | 所属后台资源 → 停止及回收；两个不同进程，不能只调用其中一个 |

显示面板的尺寸、颜色、图标、菜单位置由 `MainComponent / Theme / I18n` 等维护，属于视图职责，不增加领域业务接口。设置文件属于应用配置，工程声学参数属于 ProjectData，不应双写两份权威值。

### GUI 与 MCP 覆盖差异

自动化接口不是 GUI 功能的完整镜像。已核对 McpServer 的注册和分发，不应因为 `set_note` 描述写“全部参数”就推断它覆盖所有后加功能。

| 能力 | 当前 MCP 接入 | 应如何收敛 |
| --- | --- | --- |
| 移动区域／拉伸区域／复制区域 | move_clip / resize_clip / duplicate_clip | 继续复用对应 PM 动作 |
| 音高批处理 | edit_notes_pitch 的 transpose/set/average/quantize | 不等于原生 flattenNotePitch 的独立分段与吸附语义 |
| 画目标音高 | set_pitch_curve | 手绘／直线／标点共用点集，无需三套工具接口 |
| 普通响度包络／FLAG | set_note 的 amplitude_envelope / flag_curve 等字段 | 聚合参数须一次提交；不能循环多个 setter 形成多次撤销 |
| NSF 平滑／噪声保护 | set_track 的 nsf_smooth_pitch_transitions / nsf_noise_protection | 保留轨道级参数复用 |
| DS 读音／音素／参数／偏移 | Live 的 ds_set_* 与 ds_generate_* | 复用 PM 提交和真实窗口生成逻辑 |
| 原生自由裁剪／拓展、粘连、断开、消除切点 | 未发现对应专用 MCP 命令 | 若将来需要暴露，薄适配现有 PM；不能用 resize_note 或 toggle_note_connection 假装同义 |
| 原生批量拉直、原音高还原、动态颤音设置 | 未发现等价的专用 MCP 命令／完整参数组 | 不让自动化自行拼凑近似效果；先复用现有领域动作 |
| 伴奏右键 BPM 识别 | 未发现 AccompanimentTempoTask 的独立 MCP 命令；有和声伴奏分析 | 两者产物和成本不同，不应自动替换 |
| 工程音源树展开、显示样式、缩放 | GUI 视图状态 | 没有自动化需求就不新增协议 API |

这张表记录当前覆盖缺口，不是要求本次添加所有 MCP 工具；“一个功能一个主实现”不意味着每个 UI 状态都必须对外暴露。

## 数据的权威归属和传递

| 数据 | 权威位置 | 消费者 | 约束 |
| --- | --- | --- | --- |
| 原始 PCM | 源文件／解码资源 | 分析、原音播放、声码器 | 编辑不回写源文件 |
| 原始 F0／voiced | `NoteData::pitch` 等源分析字段、裁剪源参考 | 显示、目标求值、清音保护 | 不能由目标线反推“这里有声” |
| 目标音高 | PitchCurveEditPoint、音符中心／表达 | 求值器、请求采样 | 统一 MIDI／音分换算；无 F0 的人工线只是意图 |
| 音符／区域位置 | ProjectData 层级及稳定 ID | 所有视图、编辑和渲染 | UI 像素坐标不能进入模型 |
| 时间映射 | `sourceTimeMap`／NativeSourceTimeMap | 裁剪、伸缩、波形、合成 | 目标局部秒与源局部秒只能在明确边界转换 |
| 组合源 parts | ClipData + clipPartId | 编辑／展开读取 | 展开副本不是另一个可提交工程 |
| 响度 | 音符包络、区域增益、轨道混音 | 对应的合成／混音阶段 | 继承增益不能重复乘；分开是不同作用域，不是重复功能 |
| OTO 原文件 | SampleSettings／音源目录 | OTO 编辑、音源解析 | 工程 override 不直接覆盖音源文件 |
| 渲染缓存 | AudioEngine + 后端缓存 | 播放与峰值显示 | 只缓存派生结果，必须有明确失效键 |
| 选区／焦点／缩放 | GUI 组件 | 用户交互、Live MCP | 不把选区变化当作音高编辑 |
| DS 预测／偏移／手工值 | DS 参数层与工程缓存 | DS 编辑／合成 | 三者有不同语义，不能为了少字段相互覆盖 |

当前主要路径：

```text
鼠标/快捷键 ──> 组件预览 ──> ProjectModel 提交 ──> ChangeBroadcaster
                                              ├─> Timeline: snapshot
                                              ├─> TrackList: snapshot
                                              ├─> PianoRoll: snapshot -> 展开 parts
                                              └─> Main: snapshot -> syncAudio
                                                   -> AudioEngine 写锁
                                                   -> 可选再复制/展开
                                                   -> 求共享曲线/造缓存键/采样请求
                                                   -> RenderService 队列 -> 后端
                                                   -> PCM/峰值 -> 播放/显示

Live MCP 普通编辑 -> 整工程草稿 -> McpServer -> ProjectModel 方法
                 -> fingerprint 比较 -> 再次 snapshot -> 正式 replace

DS -> UtauRenderRequest -> JSON request -> Python worker
   -> JSON result + WAV -> C++ PCM -> AudioEngine
```

每个边界存在有理由；问题在边界传递的**体量、次数、线程和失效范围**。C++ 到 Python 需要格式转换，GUI 到 C++ 服务不需要先转 JSON 再转回来。

## 已确认的架构问题

下面“已确认”表示代码机制存在，不表示已测出占用多少毫秒或它就是用户每次卡顿的唯一原因。P1 优先处理可听卡顿／交互关键路径；P2 处理随工程规模增长的浪费；P3 整理维护成本。

### A01 · P1 · 音频回调等待工程重建使用的同一把锁

证据：[AudioEngine.cpp:1853](../juce/src/AudioEngine.cpp#L1853) 的 `syncProject` 持有 `ScopedWriteLock(renderLock)` 调用整个 `rebuildLoadedClips`；[2787](../juce/src/AudioEngine.cpp#L2787) 的实时回调获取 `ScopedReadLock`。重建内包含排序、共享曲线、文件 reader 创建、缓存键构造及任务组织。读锁是等待型，不是实时安全的无等待读取。

最小改法：在非音频线程构建下一份播放计划和 reader/PCM 引用，准备好后短暂发布；回调只持有本轮不可变播放计划。旧计划和大 buffer 的释放也放在非音频线程，避免“用了 shared_ptr 就认为实时安全”。不在回调里加 sleep、重试、超时兜底。

验收：记录回调超时次数与 p95/p99 时长，边播放边改曲线／拖区域；48 kHz、256 样本时单块预算约 5.33 ms。先测本机基线，再对比，不承诺固定提速倍数。

### A02 · P1 · 实时路径仍有磁盘读取和动态分配

证据：[AudioEngine.cpp:2826](../juce/src/AudioEngine.cpp#L2826) 素材试听 `setSize + reader->read`；[2858](../juce/src/AudioEngine.cpp#L2858) 每块创建重叠归一用 unordered_map/vector 并 assign；[3001](../juce/src/AudioEngine.cpp#L3001) 普通源播放也 `scratch.setSize + reader->read`。已有 capacity 可能让部分 setSize 不分配，但容器构造和 I/O 仍在回调中。

最小改法：准备阶段按设备 block 上限预分配混音 scratch；以已知轨道索引替代每块字符串哈希表；普通源播放加后台预读，回调只消费已就绪样本。WAV 导出可以继续离线读取，不必把实时限制强加给离线路径。

验收：回调路径不进行文件 I/O／堆分配；缓存冷启动、大伴奏、多重叠声部与设备块长变化均验证。不是靠把线程优先级调高来掩盖。

### A03 · P1 · 一次变更多份整工程快照，通知不区分影响范围

证据：[ProjectModel.cpp:1071](../juce/src/ProjectModel.cpp#L1071) 返回 ProjectData 值；[MainComponent.cpp:3723](../juce/src/MainComponent.cpp#L3723)、[PianoRollComponent.cpp:4984](../juce/src/PianoRollComponent.cpp#L4984)、[TimelineComponent.cpp:213](../juce/src/TimelineComponent.cpp#L213)、[TrackListComponent.cpp:112](../juce/src/TrackListComponent.cpp#L112) 分别读取。Main 内部刷新控件还会调用其他 snapshot。不是每次鼠标移动都一定发生，但每个被提交并广播的工程变更会进入这些路径。

证据：[AudioEngine.cpp:2010](../juce/src/AudioEngine.cpp#L2010) 在存在音频窗口或 parts 时另拷贝工程；[ClipParts.h:17](../juce/src/ClipParts.h#L17) 再复制 parts、分发 notes、改写坐标。轨道名称和显示开关也走同一工程通知，缓存可避免重推理，但无法避免前面的同步工作。

最小改法：每修订发布一份只读快照，视图共用；变更附带受影响 ID 和简单的 metadata/geometry/mix/render/source 分类。在现有 sync 路径做选择性更新，不另造多个消息总线。正式编辑用局部草稿，不能把共享快照改成多个消费者都可写。

验收：重命名／显示切换不造声学请求；仅音量变化只更新混音；一次音高编辑只更新依赖区域及其必要上下文。统计 snapshot 深复制次数，而非只统计 render 次数。

### A04 · P2 · 全工程撤销成本与重复 no-op 提交

证据：[ProjectModel.cpp:1083](../juce/src/ProjectModel.cpp#L1083) 每次 `pushUndoLocked` 复制完整工程；上限 [ProjectModel.h:1277](../juce/src/ProjectModel.h#L1277) 为 24，不是无限增长。密集源 F0／多曲线随之重复保存。某些 setter 无变化检查，如 [setTrackVolume:2652](../juce/src/ProjectModel.cpp#L2652) 在查 ID 前先 pushUndo；[setNoteModulation:4900](../juce/src/ProjectModel.cpp#L4900) 找到对象便提交相同值也会产生修订。

最小改法：先统一“找到对象且规范化后的值确实不同才提交”；保留批量一次撤销。下一步只把大、不可变源分析块共享，再按实际瓶颈考虑局部轨道／区域历史。不要第一步重写整套通用命令／事件溯源框架，也不要为省内存取消撤销。

### A05 · P1/P2 · 缓存键重复序列化整轨依赖

证据：[AudioEngine.cpp:824](../juce/src/AudioEngine.cpp#L824) 每个音源 clip 的 `renderKey` 都遍历轨道所有 clip/notes 并写入完整音符渲染字段；[906](../juce/src/AudioEngine.cpp#L906) 返回的是序列化字节构成的 string，不是固定长度摘要。设 C 为区域数，S 为全轨这些字段字节数，该段建键工作约 O(C×S)，并在缓存 key 中持有这些字节。

原生路径 [writeNativePitchContext:785](../juce/src/AudioEngine.cpp#L785) 与 [nativeClipWaveformHash:1726](../juce/src/AudioEngine.cpp#L1726) 又做共享上下文／字段遍历；音频键和波形键应共享依赖描述，但不能简单判定两种键完全等价。

最小改法：一次更新先计算每个音符／共享组／语句的声学摘要，clip 键组合源版本、编辑摘要和必要上下文摘要。跨区共享音高、CVVC、DS 整句上下文仍须计入；不能为了缓存命中只保留 clip 自身字段。仅把整个长字符串最终 hash 一次只能省存储，不能消除每 clip 重复遍历。

验收：单次同步对同一上下文只编码一次；局部编辑使应变的缓存失效、不相干轨道保持；覆核缓存字段覆盖测试和跨区域音高依赖。

### A06 · P2 · 目标采样器按“每帧扫所有音符”工作

证据：[AudioEngine.cpp:186](../juce/src/AudioEngine.cpp#L186) 分配 5 ms 密集曲线；[197](../juce/src/AudioEngine.cpp#L197) 外层 frame、内层 clip.notes 搜索，基础定位为 O(F×N)，连接邻居另有遍历。无音高间隔仍要扫描；这是前端请求生成，后端块缓存命中也不能省掉全部成本。

最小改法：为已排序不重叠音符使用游标；允许重叠时使用活动区间列表并保留当前覆盖优先规则。必要的密集 F0 留到渲染准备阶段生成，UI 继续用稀疏编辑点。不要用同一算法强行处理不重叠和多声部重叠。

验收：有声／无声、1 ms 交界桥、伸出音符的控制点、颤音、重叠、自由裁剪，生成目标数组与旧实现一致或在既定容差内。

### A07 · P2 · 合成整句随后复制成各区域 buffer

证据：[AudioEngine.cpp:2083](../juce/src/AudioEngine.cpp#L2083) `sliceInto` 给每个目标重新分配 buffer、copyFrom、扫描可听边界并计算峰值。整句声学上下文必要，但每片段都持有重复 PCM 不一定必要。

最小改法：区域持有共享整句 PCM + 样本起点／长度，边界和峰值按 slice 求；后处理会改写样本时才物化副本。不要把这一优化误写成“取消整句渲染”，否则破坏拼接和相位连续性。统计峰值生成次数和相同 PCM 的实际持有字节数。

### A08 · P2 · Live MCP 的小操作也经过整工程事务草稿

证据：[MainComponentMcp.cpp:637](../juce/src/MainComponentMcp.cpp#L637) 创建 draft 并 replace(snapshot)，再做 contentFingerprint；[688](../juce/src/MainComponentMcp.cpp#L688) 每条命令又 snapshot 校验目标；[728](../juce/src/MainComponentMcp.cpp#L728) 批量允许 128 命令；[747](../juce/src/MainComponentMcp.cpp#L747) 比较指纹，再 snapshot 回写。草稿 ProjectModel 的普通 setter 也会走全量撤销，仅正式文档最终一次撤销不等于草稿无成本。

最小改法：直接只读查询用同一共享快照；短小写命令在消息线程校验修订后走已有模型动作；长操作捕获最小输入并提交结果。原子 batch 保留事务语义，但草稿阶段禁用内部历史、一次验证 ID 集、最后提交一次。不要另写一份与 GUI 不同的编辑算法。

### A09 · P2/P3 · 单项／批量与多字段 setter 不一致

已正确复用的例子：[transposeNote:3707](../juce/src/ProjectModel.cpp#L3707) 调用 transposeNotes，`setNoteUtauFlagCurve` 调用批量曲线入口。未统一例子：[setNoteLabel:5239](../juce/src/ProjectModel.cpp#L5239) 与 setNoteLabels、[setNoteUtauFlags:5261](../juce/src/ProjectModel.cpp#L5261) 与 setNotesUtauFlags 有独立遍历／提交；removeNote(s)、removeClip(s) 也应逐项核对差异后收敛。

最小改法：每个动作保留一份批量实现，单项只包装；相关的多字段一次用户操作在同一事务中提交。`set_track`／`set_note` 是协议聚合入口，不能因 schema 是一个调用就假定内部只有一次扫描／撤销。不同作用域（全工程、轨道默认、音符覆盖）不直接合并。

### A10 · P3 · 职责集中与命名／边界耦合

静态体量：Main.cpp 20,219 行（大量内联诊断）、PianoRollComponent.cpp 10,467 行、ProjectModel.cpp 8,491 行、MainComponent.cpp 6,191 行。大文件不直接证明运行慢，但编辑事件、曲线几何、菜单、模式条件、存储转换混在一起，增加重复修复概率。

`Mld5FileRenderRequest` 实际服务多个原生后端；`UtauRenderRequest` 同时携带 DS 字段；ProjectModel.h 直接引入 backend 中的 OTO／包络设置类型。是职责与命名债务，不能将简单改名宣传成性能优化。

最小整理：先按已有边界移出纯函数（曲线求值／源时钟／渲染计划），再把内联诊断移至测试编译单元。共享领域数据可放小型无 UI、无进程依赖的类型头；无需每类数据再加 factory/provider/manager。是否把 DS 请求拆成独立类型，以真实字段和生命周期差异为依据。

## 有成本但不应先删除的设计

| 当前设计 | 必要性／边界 | 处理意见 |
| --- | --- | --- |
| `PlaybackRenderQueue` 的 lane | 外置 UTAU、DS、LLSM、神经后端有不同并发约束 | 保留一个调度器；已有播放优先和过期键取消，不再叠加任务中心 |
| NSF 局部推理缓存／WCS 源参数缓存 | 对重复改音高的推理成本有效 | 保留；前端造请求成本独立测量 |
| 波形 generation 与共享峰值 | 避免每个 timer tick 重新复制峰值 | 保留，见 AE::refreshUtauWaveforms / refreshNativeWaveforms 的早退出 |
| 双击、右键、按钮共用功能 | 是可达性，不自动构成重复业务 | 审查是否落到同一 PM 方法，不按 UI 个数删功能 |
| 不同拉伸算法 | 音色／相位／声学假设不同 | 作为策略，不能因名称相近只保留一个实现 |
| 源音高、目标音高、voiced、气声证据 | 分别描述测量、意图、有无周期性、音色显示 | 不混用；同一段可有气声但仍有 F0 |
| 干波形／包络后的波形 | 编辑包络需看到未被自己压没的底图 | 可共享底层峰值与按需求值，不直接删其中一份语义 |
| C++/Python 文件信箱与 WAV | 进程隔离、模型依赖与请求确认 | 先测 JSON/落盘/解码占比；不要无证据换共享内存或引入 RPC 框架 |
| 自动恢复、原子文件写入 | 防止用户工程丢失 | 保留，与算法失败后偷偷替代无关 |
| `canX` + `doX` | UI 可用性与实际提交时可能状态已变 | 共用一份计划/校验规则，不能直接删除提交时校验 |

## 回退与兼容收敛

实际代码 [AnalysisService.cpp:234](../juce/src/backend/AnalysisService.cpp#L234) 在 GAME 无结果时会走 NativeAnalyzer，包括模型缺失和 GAME 返回空音符。该策略存在后端／warning 输出，不能描述为一定静默；但“合法空结果”和“模型失败”共用条件值得明确。

建议让分析结果明确区分 success-with-notes、success-empty、unavailable、failed、cancelled；自动模式才按一个明确策略选可用分析器，显式选择失败就返回错误。不要 GAME→FCPE→native→再造占位音高这种多层补猜。

当前 [RenderService.cpp:915](../juce/src/backend/RenderService.cpp#L915) NSF 缺模型直接返回错误，LLSM2 失败也拒绝替换；末尾还保留 `nsf-hifigan-fallback` 等文案，并不能据字符串证明实际走过该路径。应以分支可达性清理历史文案。

`AE::resolveUtauResampler` 对不存在的配置路径选择随包 WCSNDM，是便携迁移策略。若要严格遵循“不兜底”：区分“未配置，可用默认”与“用户显式路径失效，报错”；不要一刀切移除随包默认，导致正常完整包无法首次使用。

旧格式字段在 load/import 边界转换为当前数据；不能直接删旧枚举值、旧工程字段和来源注释。当前 `NativeSegment/NativeMaterialAnnotation` 已表达共同原生语义，继续沿这一方向收敛输入格式。解析、校验、保存完整性属于必要步骤。

## 最小目标结构与实施顺序

目标只收敛成三个职责层，不增加通用中间平台：

```text
界面 / MCP / CLI 适配
        ↓ 强类型动作或查询
ProjectModel：权威状态、事务、修订、一次变更通知
        ↓ 共享只读快照 + 受影响 IDs/类别
AudioEngine：准备播放计划 → RenderService/现有后端 → 共享 PCM
        ↓ 准备完成后发布
实时回调：读取已就绪计划、混音，不读盘、不做工程转换
```

改动依赖顺序：

| 顺序 | 一次解决的问题 | 最小改动及退出条件 |
| --- | --- | --- |
| 1 | 先建立可信基线 | 给 snapshot、syncProject、缓存建键、请求采样和回调做计数／时长记录；同工程重复测 |
| 2 | 无变化也提交、单项重复 | setter no-op 检查、单项包装批量；相同值不改 revision、不加撤销 |
| 3 | 重复复制与失效范围 | 共用只读快照和受影响范围；显示／混音变化不进声学生成 |
| 4 | 音频线程阻塞和分配 | 锁外准备播放计划、后台预读、预分配；回调无读盘／动态建容器，旧资源后台释放 |
| 5 | 大工程请求准备 | 一次依赖摘要、音符时间索引、成组 PCM slice；保持声音和曲线语义 |
| 6 | MCP 草稿放大 | 最小任务输入、无历史 batch 草稿、统一提交；仍有 revision 冲突保护 |
| 7 | 维护结构 | 根据已稳定边界移动纯函数、请求命名、测试编译单元；避免与声音变更混在一个提交 |

这里的阶段是迁移次序，不是建议新增七个系统。不要同时全面替换存储、撤销、MCP、渲染器和 UI，否则无法定位行为变化。

## 后续验收清单

本次完成静态审计和文档核对，**没有运行音频性能基准，也没有声称上述问题已修复**。建议在同一构建、同设备缓冲、同模型缓存状态下记录：

- 中长度真实录音（叮咚鸡素材）、多段密集 F0、大伴奏、跨素材粘连、UTAU CVVC、DS 长句各一份；冷／热缓存分别测。
- 一次框选拉直／移动／裁剪／重命名／音量变化对应的快照复制次数、复制字节、造键时间、目标采样时间、实际推理块数、峰值内存。
- 播放中的回调 p95/p99、最大时长、超预算块；UI 响应时延与消息线程阻塞分别看，不能用平均 FPS 代替。
- `NativeImportPerformanceSmoke`、`NativeIncrementalRenderSmoke`、`PlaybackRenderSmoke`、原生源时钟／粘连／响度／无音高／颤音专项、DS 时序与参数专项、ProjectSafety 和 MCP live/batch 回归；具体启动参数见 [命令索引](AI_SOURCE_INDEX.md)。
- 输出长度、源时间映射、voiced 掩码、邻接曲线、撤销重做和保存重开应保持；确定性链路可比较 PCM，神经链路按既定数值／听感容差复核。

维护要求：新增一个功能先在本页找到归属，能复用现有动作和一个参数就不增加平行实现；新增真正的领域动作再增加一行。接口签名变动后重跑生成脚本，更新源码指纹；本页分析结论由人工复核，不能用自动目录替代。

本次文档校验已核对本地链接／源码行号范围、表中 PM 方法名与头文件声明、ProjectModel 公开方法提取覆盖及 143 个扫描源文件的 SHA-256 未变化。没有构建 EXE，没有改动合成或编辑行为。
