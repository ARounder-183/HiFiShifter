# 当前内容缓存/共享GUI源码的REAPER验证

中文实测记录，2026-10-05。用户明确恢复Computer Use并授权启动REAPER。
只做针对性验收，不是最终集中review，不代表五项目标完成。

## 环境与资产保护

- worktree：`E:\code\HiFiShifter\.worktrees\ara-plugin`，源码 `5e7a90da`。
- release build正常exit0，Rust耗时2m05s；前端生产构建正常exit0，旧警告保留。
- 新独立bundle：`.build-tmp/embedded-content-acceptance-01/HiFiShifter.vst3`。
- bundle引擎与实际release源DLL SHA256均为
  `A42E55C2B9EA5F49BF6AFF74363FCD96443775620CDC90B7D6649216F97797C7`。
- 起始确认无REAPER，使用独立profile/vstpath、一次性工程副本，未用-nonewinst。
- 新scratch：`.build-tmp/embedded-content-acceptance-probe`。
- 工程由归档`gui-keyboard-edited.RPP`复制；初始SHA256
  `229F801AA6F49DA399508595B38281FA9AC74BCB612DD314FB0835A6D9DB2E3F`。
- 原`.build-tmp/embedded-probe/embedded-editor.RPP`前后SHA256保持
  `4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3`。
- CPU仅对验收进程设置，不改系统/用户设备设置；未触碰旧测试PID49800。
- 两次实例24824/42604均使用Computer Use正常关闭，最后进程检查无REAPER。

## 已实测

1. 新版被REAPER加载并按ARA建图；一个原FX GUI显示同工程两条轨道/四个clip。
   源88200帧、两sequence；GUI和宿主BPM都为180，位置为1秒/3秒。
2. 原始v2归档第二轨已编辑状态恢复。该轨宿主Solo，实际新导出与归档已编辑
   `gui-keyboard-edited.wav`前5秒逐样本maxdiff=0；新6秒导出的额外1秒全零。
   这是既有WORLD曲线恢复，不是本轮新手绘或native HiFiGAN通过。
3. Computer Use按实际截图关闭编辑器，保留轨上FX，再导出整6秒，PCM相对
   编辑器打开时maxdiff=0；正常Ctrl+S保存副本40678 bytes，正常退出。
4. 冷重开副本、不打开任何GUI，在后台就绪后再次导出：整6秒PCM相对前次
   maxdiff=0，RMS=0.08688376956019962。保存的参数没有丢失。

## 真实失败，必须修复

### 冷启立即离线导出静音

同一冷启实例42604，脚本在启动加载完成后立即发起首个导出，而未等待后台快照。
`acceptance-cold-b.wav`全零，RMS=0，相对参考maxdiff=0.3567821979522705。
随后plugin-cold.log才出现role1/revision2/model4后台snapshot ready。
**不重开、不打开GUI、不改参数**，仅在就绪后定向再导出`acceptance-cold-ready-b.wav`，
maxdiff=0。因此证据指向offline导出与后台准备的竞态，不是“保存编辑丢失”的证明。
当前offline首个输出完成门未过；不能让宿主成功输出一份看似有效的静音文件。

### REAPER只读扩展/游标权威未取得

实际所有初始化日志为`REAPER host extension available=false`；GUI诊断
`reaper_position_authority=false`。两次原生播放、没有GUI seek/循环命令的过程中，
`backwards_while_playing`升到265，last_mode=1且大量prefetch/writer切换。
不能仅凭库合同宣布游标平稳；目前仍由旧context回退，播放头/普通fade完整同步不通过。
官方锁定SDK仍明确IReaperHostApplication可从IHostApplication取得；需加分阶段诊断，
区分QI失败/初始化时parent(project)缺失，不能猜宿主永不支持或把null当当前project。

### Computer Use网页输入限制

已按用户提示先点击原FX标题，并将宿主最大化使FX位于父窗口范围内；实际坐标标题
点击成功。工具直接网页控件click仍报`element ... not available in cached app state`，
另一次原生Add索引错误定位到WebView并被安全边界拒绝。未用猜HWND/PowerShell UIA/
自制输入代理/JS注入/脚本参数提交绕过。
原生窗口操作和宿主空格播放可执行，但没有足够证据证明本轮网页播放按钮、HiFiGAN
算法切换或新曲线编辑成功。当前真实GUI数据可见，不等于全部交互验收。

## 可复查产物（ignored scratch，不覆盖旧证据）

4份WAV均44.1kHz/stereo/6秒；文件hash不同来自REAPER元数据，比较使用独立RIFF
解析得到的PCM，未重排或修改捕获音频。

| 文件 | SHA256 |
| --- | --- |
| acceptance-restored-b.wav | 2FDFD00F60883A88A282FCE539B72A1E349793FE939CA62C012E2E0CB249D4E4 |
| acceptance-closed-b.wav | B1BEC181A384FA8F048150013B1C8FCCC9681CBC67CDF4B4DBDF66DA846F2A6F |
| acceptance-cold-b.wav | BA0A9DCC77D71A74C1F8B01C7526B7CB5073F8CDD77CA097E90145D80EA8B2E2 |
| acceptance-cold-ready-b.wav | 949FFDA11191BBEE78EEBDD6D63179BEE5F1ACDF3CE3DE5F326B9177C312C2A6 |

同目录`plugin.log`、`plugin-cold.log`、`acceptance-script.log`、`host-state.txt`与保存RPP。
采集脚本`probe/ara/content_acceptance_probe.lua`只读宿主状态/打开窗口/导出；不写任何
HiFiShifter参数，不修改clip几何，启动时验证工程绝对路径。

下一批先修offline就绪契约与host扩展初始化，之后恢复正常长素材、fade与native
HiFiGAN矩阵；不重跑本批已通过的旧WORLD链路，不把此局部结果当最终验收。

## 离线就绪修复与初始化诊断（后续批）

### 首次离线导出修复：已实测

锁定VST3 SDK `ivstaudioprocessor.h`明确setupProcessing在UI线程/禁用状态调用，
切换kOffline须经过该入口。只在该离线setup等待现有worker；process/setProcessing
不等待、不查盘。发布记录包含model/edit/授权epoch/scope/真实keys，旧快照或空闲
队列不能冒充最新就绪；缺快照/上下文的offline process返回失败而非成功静音。
Condvar的worker和离线等待者有不同条件，request使用notify_all防止唤醒丢失。

8项离线/实时/队列定向回归正常exit0（包含真实SDK子对象、参数版本变化后重新准备、
超时/失败/关闭、实时零分配与纯editor透传）。初次编译只有测试audio_ptr遗漏unsafe，
补齐后通过，未弱化行为断言。

实测包`.build-tmp/embedded-offline-preflight-01/HiFiShifter.vst3`：release构建1m53s/exit0。
引擎SHA256 `7E374701CEEF6D559CBDBB3F7213F0D0C1D9DA19FF184A691B188F5C9652681A`。
新scratch`.build-tmp/embedded-offline-preflight-probe`，复制保存副本（初始RPP SHA256
`EA5E7CD43A327777CB517269FB2DC5E63D621E79145DA334C4ABF7AFBA79B95F`）。
进程42280，不打开GUI、不加人为延迟；脚本第一轮立即导出，实际setup日志含mode=2。
首份`acceptance-cold-immediate.wav`整6秒/44.1k/stereo，PCM相对正确参考maxdiff=0、
RMS=0.08688376956019962，264600帧。WAV SHA256
`89DC1334BB3CDD8C9198A7C0B9E6E9C812D34A1A01DA33198A3AB8568A44B295`。
这是同一旧v2/WORLD夹具上的真实RED→GREEN，不外推长源/HiFiGAN全部native通过。

### 宿主扩展根因：已实测；延迟绑定修复：待native

分阶段日志证明每次QI成功，但initialize的parent(project)为空。不是REAPER不支持
扩展，也不是IID未找到；旧构造函数因此丢弃可用拥有引用。
新构造保留该接口，第一次model/UI查询只从同一个直接parent(3)绑定非空project，
绑定后不随活动tab换project，不用NULL参数的“当前工程”语义；线程/重入/活性及
ValidatePtr2检查继续保留。2新增合同及旧几何/初始化/离线共21定向回归正常exit0。
新包`embedded-host-parent-late-01`与7E37的当前运行包分开，尚未实测新游标/几何。

Computer Use恢复截图时遇到一次monitor capture 0x80070057，重选返回窗口后恢复。
随后工具报告用户正在输入，按技能停止自动键鼠，保留用户当前42280实例（标题已modified），
不关闭、不热替换、不向当前实例发送新采集命令。其仍加载7E37离线修复包，不是延迟
project修复包。原用户RPP SHA256仍为4AD35908…，没有push或修改主develop。
