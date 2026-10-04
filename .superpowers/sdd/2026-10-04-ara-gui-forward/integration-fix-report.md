# ARA 正向 GUI 集中修复报告

中文记录，2026-10-04。工作树 `E:/code/HiFiShifter/.worktrees/ara-plugin`，分支 `codex/ara-plugin`。
基础实现 checkpoint `4800705`；控制任务后续仅提交验收记录。没有操作、关闭或部署到运行中的 GUI/REAPER，没有修改 SDK、registry、ACL、OS 或 push；未启动子代理。

## 根因和修复

### 新增真实授权假冲突

`forward-gui-plugin.log` 的成功 Snapshot 后是两次 `samples_access enable=false`、MissingSource 和两次 Commit false，中间没有 begin_editing 或 clip 几何通知。源码链为授权回调→`refresh_source_pcm`→`clear_renderers`→无条件 model revision++。后续日志中的用户手动成功 revision=1/model=8 是独立观察，不推翻此前假冲突。

将快照撤销与模型版本前进分开：授权开关仍清实时发布指针、只在授权 scope 内读取并同步释放 reader，保留有界 edit_sources；纯开关不推进版本。内容、源属性、deactivate/destroy 和区域几何仍推进版本。宿主事务中的授权回调不能提前把完整图置 ready；end_editing 后再准备。冲突日志仅记录原因和 base/current 编辑/模型 revision，不记录 token/PCM。

### Finding 1：局部视图覆盖共享编辑

`EditState::merge` 曾直接用客户端局部 params/tracks 替换整张文档。现在只替换授权轨道并保留视图之外记录；host/client 的轨道 ID 必须是唯一且完全相等的集合。空分配 renderer 不获得整文档轨道。双 renderer 回归覆盖 A/B 顺序提交、B 刷新最新 revision、两轨曲线/volume、实际 PCM 和 encode/restore。

### Finding 2：忽略非参数变化后清 dirty

连接时保存完整时间线中受支持参数投影之外的基线；提交前拒绝 clip 移动/crop/gain、轨道名字及其他未支持字段变化，错误明确解释应在 REAPER 修改且本地仍未保存。不触发自动刷新。只有送出后 timeline_version 未变、未支持投影与本地工程基线均未变才清 dirty；备注、附件、本地工程名称/保存路径/设置及IPC期间新编辑继续保留确认保护。选择、播放位置和派生分析缓存不造成假拒绝。

### Finding 3：按会话序号恢复会丢编辑或串轨

经 controller 明确确认的 Ruling：每条编辑轨道保存真实 region→sequence 边上完整、排序去重的 `(modification persistentID, source persistentID)` 集合；同源但不同 modification 可区分。不用轨道名称、会话序号或臆造 sequence persistentID。仅唯一完整集合可恢复，重复/跨轨同 pair/空集合/缺失明确错误。live 图中已绑定轨道成员增删时刷新当前集合，保留其编辑；恢复在完整宿主图 ready 后关联，逐对象创建中不套用旧序号。保存前再次核对完整图。

state 版本变为 v2。旧开发版 v1 非空 state 没有归属证据，明确拒绝；旧空 state/空 bytes 仍合法。这是开发版兼容限制，不自动迁移运行中 GUI 的现有曲线，不关闭当前进程。

### Finding 4：compose-off 手工音高被错误门禁跳过

host PCM helper 曾在共用判定前以 compose=false 且 adjustment=false 返回原声。现在服从既有 `does_clip_need_processor_render`，手工 `pitch_edit_user_modified=true` 与独立 GUI 语义一致。

## RED/GREEN 证据

每次 cargo 均先 dot-source `tools/msvc-env.ps1`，TEMP/TMP 固定工作树 `.build-tmp/cl`；ARA_VST3_SDK_DIR/ARA_SDK_DIR 固定 `probe/ara/rust-path/.third-party/vst3sdk` 与 `ARA_SDK`，使用 offline/jobs1。下列 `cargo` 命令均在工作树根执行。

- 授权 RED：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1 source_grant_version_change_and_revocation_refresh_only_host_pcm -- --nocapture --test-threads=1`。预期断言失败：授权不改变宿主模型版本，实际 `left:1 right:0`；0 passed/1 failed，退出1。最小修复后同命令 1 passed/0 failed，退出0。扩展真实回调回归还检查同快照撤权后 Commit 成功、实时旧指针撤销、reader 创建/释放数量一致、源几何/content/区域移动后旧 Commit 拒绝。
- Finding 1 RED：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1 state_channel::tests -- --nocapture --test-threads=1`。初版 fixture 使用 Default frame period=0 被合法校验拒绝，修正为 5ms 后真实 RED：A 曲线 `no entry found for key`、重复ID `is_err()` 断言失败；2 passed/2 failed，退出1。修复后曲线/volume/state 回归通过，双 renderer 音频每样本为源×A0.5/B0.25。
- Finding 3 RED：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1 persisted_b_edits_follow_host_identity -- --nocapture --test-threads=1`。B-first 还原音量实际1.0、期望0.25，0 passed/1 failed，退出1。修复后删A→只B、新建B/A顺序均保留B曲线、volume和PCM；恢复B源0.2的实际输出逐样本0.05，不串到A。
- Finding 4 RED：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-kernel --offline --jobs 1 compose_off_manual_pitch_changes_real_host_pcm -- --nocapture --test-threads=1`。真实 WORLD oracle `rms=0.146789, mean_abs_diff=0.000000`，差异断言失败，0 passed/1 failed，退出1。GREEN 包含在下列28项中：`rms=0.101196, mean_abs_diff=0.146199`，真实非静音且与原声不同。
- Kernel GREEN：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-kernel --offline --jobs 1 mixdown::tests -- --nocapture --test-threads=1`，28 passed/0 failed，退出0；WORLD原有和absolute-time oracle同为 mean_abs_diff=0.146199。
- Plugin阶段GREEN：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1 -- --nocapture --test-threads=1`，47 lib +13 mapping +1 no-Tauri +5 assignments +1 exports =67 passed/0 failed，退出0。最终补充事务收口/零授权/保存身份一对一保护后，同完整命令 **48 lib +13 mapping +1 no-Tauri +5 assignments +1 exports =68 passed/0 failed**，退出0。
- App原target尝试 `cargo test --manifest-path backend/Cargo.toml -p HiFiShifter --features custom-protocol --lib --offline --jobs 1 commit_rejects_local_move_crop_gain_and_track_name_but_accepts_parameters -- --nocapture --test-threads=1` 在 build.rs/Tauri DLL复制因运行GUI的os error32失败，未到测试，不算RED。没有修源码或关闭GUI。改用独立 `--target-dir backend/target/ara-fix-app`；曾仅中断本任务自己的cargo来先完成plugin身份RED，之后继续独立target缓存编译。独立target正常RED为移动clip被接受，`未支持编辑必须拒绝` 断言失败，0 passed/1 failed、退出1（依赖完整编译6m40s）。
- App GREEN：`cargo test --manifest-path backend/Cargo.toml --target-dir backend/target/ara-fix-app -p HiFiShifter --features custom-protocol --lib --offline --jobs 1 ara_bridge::tests -- --nocapture --test-threads=1`，12 passed/0 failed、退出0；同前置的 `project_tests` 21 passed/0 failed、退出0。拒绝move/crop/gain/name且正常参数放行；参数成功可清dirty，notes、本地path/name/scale及新版本编辑仍要求刷新确认。
- 补充身份一对一RED：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1 duplicate_saved_track_identities -- --nocapture --test-threads=1`，两个保存记录同身份压到一个重建轨道时错误接受，断言失败，0 passed/1 failed、退出1。增加映射目标唯一校验后随最终plugin68项GREEN。
- IPC最终验证：`cargo test --manifest-path backend/Cargo.toml -p hifishifter-ara-ipc --offline --jobs 1 -- --nocapture --test-threads=1`，4 passed/0 failed、退出0；包括真实管道、大于kernel pipe buffer的PCM传输，约0.08s。

构建输出仍有已有 unused warnings；原target SoundTouch/VSLIB DLL复制锁警告未阻止plugin/kernel测试。WORLD运行出现已有ONNX初始化回退诊断，但音频oracle和28项测试退出0。上述结果不代表真实REAPER GUI全流程验收完成。

## 最终检查和提交

`git diff --check` 无输出，未变更控制任务路径。本轮源码路径：

- `backend/hifishifter-kernel/src/mixdown.rs`
- `backend/hifishifter-plugin/src/ara/model.rs`
- `backend/hifishifter-plugin/src/render/document.rs`
- `backend/hifishifter-plugin/src/render/extension.rs`
- `backend/hifishifter-plugin/src/state_channel.rs`
- `backend/hifishifter-plugin/src/vst3.rs`
- `backend/src-tauri/src/ara_bridge.rs`

源码本地提交：**`8eba3ae5`**（`fix(ara): preserve GUI edits across access changes and host rebuilds`），只含以上七个源码路径；报告为后续独立文档提交。没有add -A或push，提交前cached diff check无输出。

独立插件生产构建：`cargo build --manifest-path backend/Cargo.toml -p hifishifter-plugin --offline --jobs 1`，退出0，`Finished dev profile ... in 9.79s`。不与app同cargo构建，避免kernel vslib feature统一。没有把新二进制部署到运行中的REAPER或GUI；app使用独立target编译测试库，最终app可运行产物重建交controller。

## 仍未验收

controller负责重建部署、真实宿主音高导出差异、保存并重开工程后的曲线及输出；本任务未验收、未部署。不承诺倒放、stretch/fades、vslib插件。旧v1非空开发state需重新获取宿主图并人工重新提交，不能猜测迁移其轨道归属。

## F2/R1 剩余 P1 的窄修复追加

本次仅承接 `integration-re-review.md` 的 R1，不重新开展全功能wave；F1/F3/F4和授权版本边界没有改动。复审正确指出：前次 dirty 回归手动 bump_timeline_version，只覆盖 checkpoint 版本变化；真实 `commands/params.rs::set_param_frames(checkpoint=false)` 尾块/异步平滑不推进版本，也不会在成功清 dirty 后重新标脏。`restore_param_frames`、`set_static_param`、`stretch_track_linked_params` 的非checkpoint成功写入同源。

修复：`submit` 直接从本次 `Request::Commit` 的实际 timeline JSON 保存受支持参数投影（全部持久曲线/静态参数、track ID/volume/muted/solo/compose_enabled/pitch_analysis_algo）。成功后在 timeline 锁内比较当前受支持投影与实际发送投影，再结合既有版本、未支持字段、project基线决定是否清 dirty。四个真实参数写入入口在同一 timeline 锁内标 dirty，因此响应前写入使投影不等、响应后尾块重新标脏。checkpoint继续只控制undo，不增加每块undo，不为了通过回归制造额外timeline版本。

无GUI回归直接调用这四个命令的原实现：内部入参改为 `&AppState`，Tauri命令门面外部签名保持原样并借用转发；`cfg(test)`别名仅供测试，不替换参数写入行为。AppState字段装配提取为私有共享initializer；Default仍用真实 `AudioEngine::new`。显式 `cfg(test)` AppState fixture注入无worker/无设备的真实AudioEngine对象；没有global/TLS/ambient模式，没有产品worker生命周期改动。fixture选 `PitchAnalysisAlgo::None`，dirty契约不需要推理，避免无关FCPE预热。

### 新 RED/GREEN 与实际调用路径

每条 cargo 前置与上文一致（MSVC dot-source、工作树TEMP/TMP、锁定两SDK、offline/jobs1）。新增回归实际调用：

- `commands.rs` 的测试别名→`commands/params.rs::set_param_frames`：首块 `Some(true)` 后捕获真实 Commit 载荷，再以 `Some(false)` 写尾块或平滑，成功响应后仍 dirty；另一用例在成功清dirty以后才写 false 尾块，必须重新标脏。
- 对实际 `restore_param_frames`、`set_static_param`、`stretch_track_linked_params` 分别从 clean 状态执行 `Some(false)`，断言真实曲线恢复/静态值/关联映射确已写入，dirty=true，undo仍为0，version不变。
- 所有 false 写入测试都断言真实 timeline_version 与写前相同、undo深度相同；没有手动 bump 模拟这些场景。既有正向用例仍验证当前参数等于实际提交时可清dirty；另覆盖所有被接受的track controls不同于实际载荷时不能清dirty。

正常 RED 命令：

`cargo test --manifest-path backend/Cargo.toml --target-dir backend/target/ara-fix-app -p HiFiShifter --features custom-protocol --lib --offline --jobs 1 ara_bridge::tests::noncheckpoint -- --nocapture --test-threads=1`

在显式fixture与None算法保持不变的情况下，暂移除仅本P1的参数相等检查和4处markdirty，5条均失败在预期dirty断言：`成功后的false尾块必须重新标脏`、`IPC中的未发送尾块/平滑仍须确认刷新`、restore/static/linked真实false写入必须标脏；**0 passed/5 failed，正常退出1**。版本和undo未变化断言已先通过。随后恢复全部修复。

最终 GREEN：

- 同前置、同独立target，filter `ara_bridge::tests`：**18 passed/0 failed，正常退出0**，约0.02s；包括原12项、上述真实命令5项以及受支持track controls比较。
- 同前置、同独立target，filter `commands::params::`：**10 passed/0 failed，正常退出0**，保留参数换算/值域/选区语义回归。
- `git diff --check` 无输出。只验证本P1受影响边界，没有重跑不相关plugin/kernel/整工程大套件；产品GUI重建/部署仍交controller。

### 测试运行历史与隔离

最初真实命令5条获得正确失败摘要，修复后的17/18项获得成功摘要，但测试进程在摘要后仍未正常退出；仅中断本任务自己的cargo/测试进程，这些摘要没有当作完成证据。先前以私有引擎shutdown尝试隔离CPAL不充分。进一步阅读实际路径发现：fixture默认算法为NsfHifiganOnnx，`ensure_params_for_root→pitch_analysis::build_root_pitch_key→fcpe_onnx::is_available→ensure_background_prewarm` 会派ONNX预热线程。采用显式无设备fixture与None测试算法后取得上列正常退出1/0，未修改生产分析或引擎生命周期。

一次隐藏Start-Process输出重定向未继承工作树TEMP/TMP，原临时目录4项PermissionDenied（14 passed/4 failed），不计为本P1回归或GREEN；没有更改ACL/OS，后续均恢复规范cargo前置。原用户GUI/REAPER及曲线未操作、未关闭、未部署，未启动子代理。

本追加源码路径：`backend/src-tauri/src/ara_bridge.rs`、`commands.rs`、`commands/params.rs`、`state/app.rs`、`audio_engine/engine.rs`、`audio_engine/mod.rs`。后二者仅增加 `cfg(test)` 工厂/再导出，生产引擎构造及worker代码没有改动。追加源码本地提交：**`0b8f4dd4`**（`fix(ara): retain dirty state for unsent parameter writes`），只含上述六个路径。报告单独提交。控制任务的 `probe/ara/FORWARD-GUI-RUN.md`、`start_forward_gui.ps1` 现有修改予以保留，不纳入提交。

报告追加提交 `89bab2fa`。完工前按用户硬约定补齐params.rs中文文件头及四个成功写入入口、detached_engine、with_audio_engine/显式AppState构造器的中文doc；最后补充仅注释和报告，不改变行为，diff check无输出。
