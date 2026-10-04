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
