# 多短clip闪退修复

中文记录，2026-10-06。仅`feature/ara-plugin`工作区；未更改用户工程/音频、SDK或registry。

## 根因与证据

用户流程为：空母轨先加插件，再把三个子轨的item全部搬到母轨。实际工程副本118个
item，其中114个ARA音频region、36个源。普通100个英文路径WAV，以及空母轨搬100个
英文路径WAV都正常，说明不是简单的clip数量上限。

Windows转储4次相同访问异常：REAPER RVA `0x58efa`读取空指针+`0x7c`。离线展开栈
显示故障在宿主的源注册/窗口流程，不是一个直接的模型OOM栈。安装的旧engine与原
Release SHA256一致，PDB CodeView也匹配，未用不匹配符号猜函数。

用用户素材与布局的四轨隔离副本复现，添加空母轨插件并搬118个item后，旧代码明确
记录`createAudioSource property rejection: persistent ID must be nonempty ASCII`，
随后宿主崩溃。原始日志位于ignored `.build-tmp/actual-parent-118-01`。

SDK确实要求ARAPersistentID为七位ASCII，但REAPER传入含中文路径的UTF-8 ID。
桥接层拒绝后返回null，REAPER继续使用这个空音频源对象。数量多只增加遇到此类
素材的机会，不是应通过限制clip数量解决的资源问题。

## 修正

- 仅宿主AudioSource/AudioModification的FFI输入接受有界、非空、有效UTF-8 ID。
  保留原字节，不转码/归一化/生成替代身份，避免保存和同源关联漂移。
- 插件自身生成的ID、Factory/Archive ID等既有ASCII要求不放宽。
- null、空ID、无效UTF-8仍拒绝；源创建拒绝原因进入现有日志。
- 没有提高内存额度、取消授权校验或增加clip上限，没有改变DSP算法。

## 定向验证与产物

`host_persistent_ids`两个边界用例exit0，覆盖中文/emoji/组合字符、原字节身份、
空/null/无效UTF-8及输出ASCII防线。未跑整套回归。

新Release bundle：`.build-tmp/many-clips-fix-01/HiFiShifter.vst3`，仅构建插件，exit0。

- 实际副本118个item：全部搬移、保存、正常退出；36源/39 modification/114 region。
  `.build-tmp/actual-parent-118-fixed-01`，PID23920已退出，无该PID新崩溃事件。
- 200个中文路径、双声道短WAV：空母轨先加插件，三子轨全部搬入，保存、正常退出；
  200源/200 modification/200 region，无注册拒绝。
  `.build-tmp/many-utf8-parent-200-fixed-01`，PID17844已退出。

早先隐藏的空白复现实例9136只在用户明确允许后结束；其余成功实例正常退出。
辅助转储工具仅离线读数据，不启动转储目标或改系统调试设置。用户项目和素材不进Git，
不会推送。现有D:\VST安装目录未覆盖；使用新包时请关闭REAPER后替换整个bundle。
