# REAPER 内嵌原 GUI 开发入口

中文说明，2026-10-05。仅本地先导，不是完整发行包或二期最终验收。

在`E:/code/HiFiShifter/.worktrees/ara-plugin`执行：

```powershell
.\probe\ara\build_embedded_editor.ps1
.\probe\ara\start_embedded_editor.ps1
# 正常保存并关闭一次性工程后才可重开：
.\probe\ara\start_embedded_editor.ps1 -Reopen
```

构建输出：`.build-tmp/embedded-vst3/HiFiShifter.vst3`。`Contents/x86_64-win`内为
薄入口`HiFiShifter.vst3`、Rust引擎`HiFiShifterEngine.dll`及邻接依赖；资源为
`Contents/Resources/frontend/plugin.html`及原前端assets。薄入口只用Win32，按自身
绝对路径与DLL_LOAD_DIR加载引擎，不修改系统/启动进程PATH，不依赖项目CWD。
入口编译显式`/utf-8`避免本机CP936把中文注释解释成续行，`engine_exit`避免CRT
全局terminate符号冲突。

启动只打开隔离REAPER与一次性Lua工程，**不启动HiFiShifter.exe**。profile和RPP在
`.build-tmp/embedded-probe`，已有REAPER主进程时脚本拒绝，不-nonewinst、不强杀。
首次隔离profile可能出现设备选择/评估提示，只正常处理，不改用户原profile/许可。
已有一次性扫描缓存避免重复第三方插件激活。原独立app入口与外部ARA客户端保留。

GUI保留原DockRoot、时间线、参数编辑器；外部连接/手动提交栏换为自动应用状态。
文件/轨道几何/录音/独立播放在插件中由宿主接管；支持的参数仍在原界面编辑。
“重新载入宿主”会在有未应用编辑时确认替换，冲突不能自动覆盖本地曲线。
首次WebView激活会将引擎映射保持到REAPER退出，开发升级需正常退出REAPER。

一次性脚本监听`.build-tmp/embedded-probe/command.txt`：`save`、`show`、`close-view`、
`baseline-fresh`、`edited`、`reopened`、`source-changed`。不要把这个脚本送到其它实例。
真实输出记录在`captures/embedded-editor-*`，当前只有未编辑基线，**不是自动修音通过**。

已实测宿主加载、正确processor消息关联及FX内原GUI。剩余自动pitch输出、关闭FX供音、
保存重开、双实例及资源/采样率/声道/seek回归见二期plan；不能把显示通过当全流程通过。
