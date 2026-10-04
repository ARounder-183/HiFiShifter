# 正向 ARA 原 GUI 开发版运行说明

这是本地先导工作流，不是已完成验收的发行包。仅 Windows x64、REAPER 7.81；
每源最多30秒，44.1/48k、mono/stereo、普通/裁切片段。倒放不支持，stretch/content fades不承诺。
独立 app 不变；插件不内嵌 WebView，原 GUI 作为独立客户端。

## 工作位置

全部命令在 `E:\code\HiFiShifter\.worktrees\ara-plugin` 执行，`codex/ara-plugin`。
不要在主仓库 develop 下执行。不 push，不 add -A，不修改SDK。

## 构建和启动

当前GUI若有未保存曲线，先保留它，不能为了升级强制关闭。
以下构建脚本会拒绝重建仍运行的本worktree GUI，不会替用户终止进程。

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-plugin
.\probe\ara\build_forward_gui.ps1
# 确认所有REAPER已正常关闭后；只部署到本worktree隔离vst目录。
.\probe\ara\start_forward_gui.ps1
```

构建脚本加载MSVC，再设干净TEMP/TMP；先构建GUI的custom-protocol嵌入dist版本，
然后单独构建plugin，避免一次workspace构建让app的vslib feature进入plugin。
启动脚本会设置隔离WebView2 profile，绕过本机原profile的0x800700AA；没有删除原profile。
若REAPER已运行，启动脚本明确拒绝；绝不向已有实例送脚本，绝不使用-nonewinst。

## GUI工作流

原app上方`ARA / REAPER`面板：刷新实例、选取目标、连接。已有未保存工程须显式确认替换。
宿主提供PCM，GUI为既有波形/音高管线生成私有WAV，不从宿主persistentID打开原文件。
在原参数编辑器画音高/音量曲线，然后点`提交到REAPER`。提交是显式操作，不会自动覆盖本地编辑。

宿主位置、裁切、源替换等几何在REAPER中操作，不属于GUI参数提交范围。
发生真正版本Conflict时，当前本地曲线保留；不要在没有保留编辑前直接确认刷新替换。
当前截图的授权开关误Conflict已修正源码（8eba3ae5），新增回归通过；运行中的旧二进制不会自动更新。
为不关闭当前GUI，修正版app另构建到`backend/target/ara-fix-app/debug/HiFiShifter.exe`。
启动脚本会在两份本worktree产物中选最新版本，并拒绝重复启动仍有编辑的GUI。

## 验收边界

成功连接/提交不是修音完成。必须真实导出`forward-gui-edited.wav`，与基线比较音高；
保存/关GUI/重开一次性RPP后导出`forward-gui-reopened.wav`，再检查输出保持。
隔离Lua脚本监听worktree scratch命令文件，但不得往用户正在用的REAPER发送脚本。

```powershell
.\probe\ara\test_forward_gui_output.ps1
.\probe\ara\verify_forward_gui_output.ps1 -ReopenedPath .\probe\ara\captures\forward-gui-reopened.wav
```

验证器只适用于本轮220Hz周期合成夹具，拒绝把gain变化冒充pitch变化。当前真实普通/裁切/
间隙基线maxdiff=5.96e-8；edited/reopened实际导出尚未完成。

## 开发版持久化兼容性

正在修正临时轨道序号造成的恢复错配。新关联基于实际宿主modification/source身份，
只能唯一匹配时恢复；共用/空/缺失身份必须明确拒绝，不能按轨道名称或新序号猜。
旧开发版非空v1编辑state缺少持久身份，不能静默迁移；升级前保留当前GUI编辑。
这不是正式版工程格式的兼容承诺。

当前证据与未完成项见`captures/forward-gui-FINDINGS.md`和`PRODUCT-HANDOFF.md`。
