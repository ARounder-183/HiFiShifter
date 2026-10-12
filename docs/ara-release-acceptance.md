# ARA 插件交付与验收记录

中文交付记录，2026-10-05。当前分支为 `feature/ara-plugin`，开发工作区为
`E:\code\HiFiShifter\.worktrees\ara-plugin`；主工作区 `develop` 未修改。

## 验收结论

用户使用 `release-delivery-02` 的双轨工程副本后确认“验收通过”，随后再次确认
“我测试过了，ok”。以用户实际操作作为集中 GUI 验收证据，不冒称代理逐项复跑了
所有宿主矩阵。保留原独立 App，插件使用原 GUI，同工程多轨共享工作区、逐轨输出。

本次交付包含跨轨曲线/气声/音高自动应用、宿主渐变显示、播放控制与播放头同步、
编辑状态保存恢复、正向线性拉伸及 HiFiGAN 缓存/分块处理。相关源码与模型证据见
`probe/ara/EDITOR-SYNC-FINDINGS.md`、`SOURCE-AUTHORITY-FINDINGS.md`、
`NEURAL-CACHE-FINDINGS.md`、`MULTITRACK-RESOURCE-FINDINGS.md`。

用户已要求停止重复全量测试/新增 review。本次整理与推送不重新执行测试或构建。
最后一次集中回归已完成 frontend 2729、kernel App配置587/插件配置585、App245及
plugin156项；后续Verify流水线被中止，不能称整条Verify成功。

## 交付位置

已构建的 Release 位于 `.build-tmp/deliveries/release-delivery-02/`：

- 独立应用：`app/HiFiShifter.exe`，及模型/运行依赖。
- 插件：整个 `HiFiShifter.vst3` 文件夹，包括 loader、engine、原GUI和模型。
- 构建身份：`build-manifest.json`，构建源码commit为 `5ea3df47`，共45个产物文件。

产物与用户工程位于ignored目录，不随Git分支推送。manifest记录的是构建时状态，
`verificationRequested=false`、`nativeAcceptance=false`不回写篡改；后续用户验收
结论由本文记录。没有重建或替换用户当前REAPER加载的模块。

## 已确认的边界

> **2026-10-11 起部分边界已改变**，见文末《后续更新》一节；本节保留当时（2026-10-05）
> 的原始结论以便对照。

- 只做正向线性拉伸；倒放、非线性stretch markers不在本次范围。
  （**已更新**：倒放已支持，含方向识别与镜像坐标复原；非线性 stretch markers 仍不支持，
  但现在会**显式**在片段浮标里说明，而不是静默按线性处理 —— 见文末。）
- 用户允许HiFiShifter自己的渐变示意曲线；普通渐变声音仍由REAPER应用一次。
- HiFiGAN分块，HNSEP整段处理。三分钟HNSEP曾实测瞬时峰值约13.3GB，已有资源预检；
  不能保证普通16GB机器无条件运行长素材，也不能把PCM额度当模型总内存。
- GPU VRAM、任意自定义模型的资源严格上界及其它宿主矩阵没有完整测量证据。
- **DLL 一旦打开过编辑器就无法卸载（硬约束）**：WebView2 的环境创建没有取消 API，
  迟到的 COM 回调仍会进入本 DLL，因此 `pin_callback_code`（`editor/webview.rs`）用
  `GET_MODULE_HANDLE_EX_FLAG_PIN` 把本模块钉到宿主退出。`ExitDll` 在有活窗口时返回
  `false`（关窗后可卸载，但模块仍被 PIN 住）。**开发升级必须完整退出宿主**，不能靠
  "卸载插件再重载"热替换；这不是可以绕过的缺陷，是 WebView2 生命周期的直接后果。
- Linux/macOS插件本次不要求实现；独立App原跨平台路径保留。

## 后续使用与开发

用户操作见用户手册的插件章节（[简体中文](i18n/USERMANUAL.md#八vst3--ara-插件reaper)，
另有英/日/韩/繁中四语）；构建入口见 `scripts/build/build-hifishifter.ps1`，打包见
`scripts/release/package-vst3.ps1` 与 `scripts/build/pack-portable.ps1`。只维护同一份frontend/kernel，
宿主权限与生命周期留在适配层。
用户已明确授权把当前分支改名为 `feature/ara-plugin` 并推送origin；不改名或删除其它
远程分支，不强推，不自动创建PR。

## 后续更新（2026-10-11，`develop`）

以下内容取代上文中与之冲突的旧结论。

### 倒放（已支持）

- **方向识别与 App 逐条对齐**：`play_rate < 0` **或** `SECTION MODE 1` 都判为倒放；
  此前插件只认 SECTION 方向位，只用负速率表达倒放的 take 会被**静默按正放**渲染。
- **镜像坐标复原收口到共享解算**：`hifishifter_kernel::take_geometry` 同时服务独立 App 的
  `.rpp` 导入与插件的编辑器/渲染两条路径，`[source_start, source_end]` 两侧逐位一致。
- **编辑器路径不再被覆盖回镜像坐标**：`present_host_inventory` 写镜像坐标，
  其后必须紧跟一次事实投影（`project_host_facts`）；片段身份解析按 `ara-item-` 后缀，
  对带 GUI namespace 前缀的编辑器 id 同样成立。
- **倒放参与参数/音高曲线投影**：源时间映射允许**严格降序**源轴（项目时间增 ⇒ 源位置减），
  正放/倒放共用同一份区间插值。

### 明确不渲染（现在会**说出来**，而不是静默）

- **非线性 stretch markers**：不物化为 `ClipTake::stretch_markers`，源时间映射按线性处理。
  片段浮标会显示"不支持非线性拉伸"，分割/拖边仍被后端拒绝并给出明确原因。
- **Take 包络（VOLENV/PANENV/MUTEENV）**：原样搬运但不消费；浮标会显示"不渲染 Take 包络"。

### ARA-only 宿主（Studio One / Cubase / Nuendo / Logic）

- 内容淡化缺宿主几何时**逐 clip 降级**（保留 ARA 声明的淡化），不再让整条 renderer 失败；
  此前只要有交叉淡化就**整轨静音**。
- 状态横幅按 `HostKind` 区分"只读（ARA-only）"与"未识别"，不再对能正常出声的 ARA-only
  宿主显示"本宿主尚不支持"。
- 倒放、拉伸、委托淡化、宿主事实未就绪时**强制走内核路径**，杜绝静默直通。

### 宿主兼容与稳定性

- **5.1/7.1 轨道不再把插件弹出插槽**：接受任意非零总线布局，内部按前两声道工作。
- **拉伸算法设置生效**：改走内核的 `resolved_external_stretch_algorithm()`，不再硬编码。
- **会话终态可传播**：前端按 `errorKind` 区分瞬时/终态，终态停止轮询（此前会无限重试并刷日志）。
- **日志不再被单一错误刷满**：前端按 `(method, cause)` 去重，后端按原因分桶限流；
  剪贴板**探测**改走只读、无重试、无锁路径，`available()` 永不失败。
- **宿主重复 `bindDocumentToExtension` 不再致命**：companion 层先拒绝第二次绑定时，
  幂等复用既有扩展实例并留痕，而不是返回 null 让宿主认为插件未绑定。
- **region→sequence 归属缺失不再整份失败**：按声明数量顺序归属，多出的 region 归首条序列。

### 仍未做（诚实声明）

- **非线性 stretch markers 的完整物化**、**take 包络消费**：需要内核侧真正消费
  `SourceTimeMap::piecewise` 与 `ClipTakeEnvelopeSet`，属独立后续任务。
- **512 MiB PCM 预算的按实例配额 + LRU 淘汰**：多实例下仍是顺序相关；需要产品侧先定
  每实例份额。
- **Cubase 重绑的实际触发时机、5.1 总线在真实宿主上的行为**：需要实机取证，
  相关修复只保证"不再致命/不再弹出"，不改变绑定语义。
