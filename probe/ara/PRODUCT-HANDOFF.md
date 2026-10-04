# ARA 产品开发当前交接

中文工作记录，更新于 2026-10-05。此文件记录产品分支，历史探针仍见 HANDOFF.md。

## 最新状态：一期核心真实验收通过，二期改为内嵌GUI与自动应用

原GUI手绘音高提交r1/m2，REAPER真实输出四个窗口220.5Hz→393.75Hz，gap=0。
正常关GUI、保存/关REAPER后无GUI重开，输出PCM maxdiff=0；恢复GUI连接r2/m3，
用户确认曲线显示。证据见`captures/forward-gui-output.json`和同名三份WAV。
首次重开126加载失败已定位flat模块依赖搜索受项目CWD影响，启动脚本改为进程局部PATH；
失败WAV保留。完整规范bundle、同路径改源/seek/采样率声道边界尚未全部宿主复测。

用户新目标是整个原GUI嵌入REAPER插件、不再另开app或手动提交，同时保留独立app。
当前源码`IEditController::createView`仍返回null；一期成果不能证明二期已完成。
二期规格/计划为`2026-10-05-ara-embedded-editor`，替代一期A5中“禁止任何WebView”的
UI边界，仅允许插件原生WebView2，不引入Tauri app/事件循环/设备音频。下文旧状态保留
历史排错意义，最新状态以本节为准。

本轮起点只读进程检查无REAPER主进程/HiFiShifter，仍有两个reaper_host32辅助进程；
未终止它们。所有变更仍仅在ara-plugin worktree，不push。用户要求集中最终测试，
不逐小改动重跑验证；必要编译检查不冒称行为验收。

二期当前源码检查点：plugin原生IPlugView/WebView2（createView不再null）、前端pluginHost
命令与hostEvents事件适配、独立app原params完整实现迁入kernel/editor，AppState薄hook。
native cargo check --tests exit0（含CSP/token/防同步重入和参数迁移），前端通信迁移
tsc-b exit0，迁移后独立app cargo check exit0。最新模块pin增量未重新编译，新增测试未运行，不能宣称宿主
已经显示或能用。WebView当前只接受ping/日志，实际编辑请求明确报session未绑定。
下一步Task22：正确processor/controller连接、实例会话、宿主PCM分析/波形及原API分发；
随后原App能力适配、自动应用与state保存、bundle部署、最后集中REAPER实测。
前端入口是plugin.html，bundle必须带它（不是独立app的index.html）；暂未生成发行包。

## 一期历史排错记录

用户明确要求本轮先不做倒放，继续到原 GUI 正向全流程可用。此授权覆盖下方历史方向停止条件，
不代表倒放已修复。当前规格/计划为 `2026-10-04-ara-gui-forward`。

工作树仍是 `E:/code/HiFiShifter/.worktrees/ara-plugin`，`codex/ara-plugin`。
Task16 PCM注入口提交 c836bd65；Task18 原GUI客户端及嵌入dist/Low临时目录修正为
82942a25、49bda828、6da16345。Task17 plugin/IPC基础检查点为4800705。
最新修复为8eba3ae5/0b8f4dd4，中文doc8c6cb394。controller新鲜验证plugin68/IPC4/
kernel28/app ARA18/params10/project21/frontend8通过；前端生产构建也通过。
四项集中审查已修，首轮窄复审仅F2的false-checkpoint并发dirty留P1，现已定向修正，
最终窄复审F2/R1为ADDRESSED，无新问题。最终独立target GUI构建38.55秒正常exit0，
产物92661760 bytes。真实导出/重开仍未完成，不能称完整链路验收通过。

本机WebView2创建0x800700AA：隔离WEBVIEW2_USER_DATA_FOLDER后实际GUI正常显示；
独立debug app需要custom-protocol feature。Low GUI到Medium REAPER的管道已仅在本应用
对象设置Low标签，保留同用户DACL/token/remote拒绝；大PCM必须16KiB分块，回归已通过。
未改系统设置或用户原profile，未停止其他应用。

真实GUI已下载宿主PCM并显示两片段及原音高编辑器。用户截图记录手绘曲线提交时
`Conflict: host model changed; refresh`；最新成功Snapshot之后仅有samples_access=false，
而clear_renderers无条件revision++，已定位为访问开关与模型版本混淆。已分离版本，
实际授权回调/同快照提交/真实源改变拒绝均有回归。相关报告在本plan的SDD目录。
随后用户自行操作产生真实`GUI commit ready revision=1 model=8`；这仅证明一次提交被接受，
未作REAPER音频导出/重开比对，不能代替修音验收。

用户用物理Escape停止了Computer Use。当前GUI/隔离REAPER仍运行且可能有用户未保存曲线，
不得为重建/部署直接终止它们、刷新覆盖或发送脚本到已有实例。新GUI只在独立target构建，
startup脚本在当前窗口还活着时明确拒绝部署启动。最终验收仍待继续。
旧开发版非空v1 state没有稳定轨道身份，新v2明确拒绝猜测迁移；保留现有GUI编辑。

## 历史检查点：完整 v1 被真实倒放输出阻塞

Task13真实controller身份/销毁、editor sequence通知与展开已补齐。Task14已实现普通/
裁切的host PCM -> VST3 process链路，绝不按persistentID读源文件。真实REAPER输出
正常/裁切maxdiff=5.960464e-8、间隙0；官方倒放action/section已确认，最终输出仍为正向，
反向oracle差0.5003815。遵照Task15停止条件，**不进入Phase3b/4/5，不宣称完整修音/GUI联动**。
下一轮先读 `captures/phase3a-FINDINGS.md`，旧“下一步按顺序”部分仅作历史，不能跳过方向阻塞。

用户要求减少review，本批只一次集中审查；两个P2（96k协商/sequence迁移）已补红绿回归。
验证器6条正确/变异测试通过，真实倒放采集依旧应FAIL。源/快照共享512MiB硬预算，retired
快照保留到owner释放；time stretch/fades先导明确拒绝。日志/源码/输出WAV/JSON/截图均保留。
真实host PCM首轮缺ARA绑定是误将controllerRef当instance，已按锁定header/shim纠正，
旧失败产物保留 `.build-tmp`。隔离REAPER已关闭。main workspace不改，SDK不改，不push。

最终验证：构建成功，55条插件测试通过，两套验证器回归各6条通过，diff检查通过。
真实输出验证仍exit 1（倒放输出为正向）；这不是完整Phase3a/v1通过。内核/app全套
测试本批未重跑。后续必须先收敛方向契约，不能直接执行下方历史待办。

## 工作位置与授权

- 仅 `E:\code\HiFiShifter\.worktrees\ara-plugin`，分支 `codex/ara-plugin`。
- 用户授权按建议持续分批推进，无需中途选择执行方式；不 push，不 git add -A。
- 主工作区 develop、用户 REAPER 工程均不改；SDK 检出不修改。
- 当前 v1 是原本独立 app 图形界面联动插件，不是把完整 Tauri UI 嵌入 REAPER FX 窗口。

## 已完成

Phase 1 已抽内核并保持 app/内核测试合计。Phase 2 Task 10/11 提交 `a9d80b06`：
真实 REAPER UI 移动/切片、实际拉伸、官方 action 41051 倒放并读 ARA PCM；原始证据
已进 git（两个 *.log 用 explicit force-stage）。U1 仍是当前映射的方向表达缺口。

Task 12 提交 `e36889be`：完整 VST3 音频 ABI 原生布局 oracle、安全输出初始化、非法
输入/格式/布局拒绝、process/setProcessing 日志守卫。当前 process 只输出零。
上位设计已更正不存在的 storeAudioSourceContent，head/tail 查询不是预渲染调度。

Task 13 的部分检查点：RegionOwners、真实 region key 注册/撤销、扩展 add/remove
通知、entry builder Arc owner（无 Box::leak）、Processor 工厂初始 COM 引用修复。
完整插件构建及 38 测试通过；采集验证器 6 通过。审查无部分检查点阻塞，**并未批准
Task 13 全部完成或产品发布**。新 DLL 尚未部署到 REAPER。

## 下一步，按顺序

1. 阅读 Phase 3a spec/plan 的 Task 13 “当前部分检查点”。实际 destroy_document 只撤销
   RegionOwners，没有通知 ExtensionControllerLease 或清空 ExtensionOwner.assignments。
   必须找真实 controller -> document -> owner 的关联，不使用“最后一个文档”猜测。
2. 补 bound-extension teardown、observer 重入，以及 editor sequence assignment 通知。
   独立 FFI teardown 测试销毁手动 lease，不能证明产品接线；native entry lifetime
   测试目前是不带文档绑定的 entry。
3. Task 13 完成后才做 Task 14：scope 内宿主 PCM、不可变快照、项目 sample 时间读取。
   不直接按 persistentID 打开文件，不在 process 读 host reader/推理/IO/等待锁。
4. Task 15 在隔离 REAPER 输出波形验证普通/裁切/seek 与真实倒放；未测输出不关闭 U1。
5. 其后 Phase 3b 完整内核注入式修音与供音/cache 风险，Phase 4 原 app 参数通道及持久化，
   Phase 5 打包。不要宣称插件已经能修音或 GUI 联动。

## 环境与验证

每个 PowerShell 独立构建环境：点源 tools/msvc-env.ps1，之后设置 TEMP/TMP 到
.build-tmp/cl；两 SDK 变量指向 probe/ara/rust-path/.third-party 下对应检出。
全套命令见 Phase 3a plan；cargo 均 --offline --jobs 1。
产物为 backend/target/debug/hifishifter_plugin.dll，不是 crate 自己的 target 目录。

最后已停止 task10-clean 隔离 REAPER。重启前检查进程与命令行；不向已有用户实例发
脚本，不使用 -nonewinst。REAPER 会补扫系统 VST3；用已有隔离扫描缓存避免激活弹窗。
38 条插件测试与 6 条验证脚本回归无失败；本批不改 kernel/app，没重跑其完整测试，
四条既有 Windows /tmp 失败不修。既有未用 macro/mut/sequence index 警告保留。
