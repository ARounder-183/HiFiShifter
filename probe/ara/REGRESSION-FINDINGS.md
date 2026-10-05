# Windows集中回归与模型查询生命周期

中文一次性证据记录，2026-10-05；工作区仅为`ara-plugin`。不是最终REAPER/原GUI验收报告。

## 修正与边界

1. App快照的四个测试写死Unix `/tmp/*.aiff`，Windows上创建失败。改为系统TEMP下UUID
   唯一路径，时间线、source存在性夹具和decode cache使用同一个路径；没有改快照产品算法。
2. 原快照五个断言通过后，进程曾以`0xc0000005`退出，CPU诊断也曾在退出阶段停留。
   源码确认三个ONNX `is_available()`会隐式启动后台ORT会话预热。现查询只读结果，App
   `src-tauri/src/lib.rs`的三个显式启动预热调用保留，实际分析/合成仍按需加载。
   修正后同一五项正常exit0，新增三个轮询护栏通过，真实HiFiGAN/HNSEP也正常exit0。
   未取得匹配旧EXE的有效符号栈，不能把原生崩溃的精确故障地址归因当作已证明。
3. 复制轨道/保存范围夹具只有四个PCM样本，本来用原声逐样本验证gain，却默认选HiFiGAN。
   完成原线门禁会正确保留pending。夹具明确`pitch_analysis_algo=none`，保留两轨不同曲线、
   复制/restore、gain和debounce断言；产品pending门禁没有放宽。真实修音由独立WORLD/
   HiFiGAN用例验证，不能用这个旁路夹具证明复制轨道的神经修音已在宿主通过。
4. 实际IBStream保存用例旧断言写死v2，source basis自动建立后应为v3。新断言同时检查
   atlas只含一个region且root属于当前组件；短读/短写和setState回读保持。

仅终止了权威命令行确认的本批快照诊断15992；CIM返回0，原会话30152最终exit1。
未处理上午旧进程49800/39748/48260，未启动/终止REAPER。移除了两个未能成功定位旧
符号的一次性诊断源码，原诊断输出仍在ignored `.build-tmp`中。

## 实测结果

| 阶段 | 结果与证据 |
| --- | --- |
| 快照修正 | 5项/exit0；`.build-tmp/snapshot-fixed-8cd36625db2748caa41c7799c1e3252f/app-snapshot.log` |
| 可用性轮询 | 3项/exit0；同目录`availability.log`；不启动预热、不创建共享模型会话 |
| frontend全量 | 315文件、2724项通过；`.build-tmp/combined-regression-02.log` |
| kernel App配置 | 587通过/7显式ignored，正常退出；同上 |
| kernel插件配置 | 585通过/7显式ignored，正常退出；同上 |
| App全量 | 245通过、0失败、正常退出；同上 |
| plugin首次全量 | 154通过/2失败/2ignored；上面两项旧夹具，不记为绿 |
| plugin修正过程 | 第二次全量155通过/1失败；编辑误落到相似夹具后已撤回，不记为绿 |
| 最终scope/state定向 | bound_tests 22项及实际IBStream 1项/exit0；`.build-tmp/scope-final-6677aaa182cb4902a99aaf1719aad4c8/scope-state.log` |
| 真实短神经链 | 1项/exit0；cold796ms，HiFiGAN冷1次，移动额外0次/PCM完全相同，改pitch新增1次，HNSEP整段1次；48k从44.1k派生逐样本一致 |
| 插件原生/依赖合同 | ARA mapping13、renderer assignments5、导出1、依赖隔离1，均exit0 |
| IPC | 4项/exit0，含大于pipe buffer的真实PCM传输与错误token拒绝 |

最后三行原始输出：`.build-tmp/remaining-contracts-09aac66329fa4c2e8713c1a9c0f2e521/contracts.log`。
最后一处夹具修正只定向重测，未再次全量重跑plugin；不得称一次完整`-Verify`已成功。
三分钟模型诊断不重复运行，本轮短链仅验证查询生命周期改动没有阻断实际模型加载。

## 交付与未过门

统一All Debug `regression-fixed-build-03`已exit0；输出在
`.build-tmp/deliveries/regression-fixed-build-03`，原始构建输出在
`.build-tmp/regression-fixed-build-03.log`。App和规范VST3 bundle、三模型、运行DLL均生成；
45个manifest文件摘要全复核，0差异。源码/模型指纹为
`B4E0ABC3AA260808A8AF231153A992BAB151D25D18FFF32E2D516E8774A11E26`；构建捕获
`27e567d0 + workingTreeModified=true`，不是伪装成该旧commit的干净二进制。
dumpbin确认App含vslib_x64.dll导入、插件engine不含；本轮未安装、不启动REAPER。
打包未再请求Verify，所以manifest的verificationRequested/nativeAcceptance均保持false。

Release、完整最终Verify、独立App原GUI及REAPER四BUG/渐变精确形状/宿主几何/保存冷恢复
仍未最终验收，不能由这份Debug产物关闭完整目标。
源RPP SHA256仍为`4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3`。
没有把本批库测试外推成真实宿主通过，没有修改主develop或SDK/registry，没有push。
