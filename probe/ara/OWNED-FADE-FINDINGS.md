# HFS自有渐变：真实REAPER责任门

中文实测记录，2026-10-06，未完成报告。

## 本批已实现的源码链路

HFS shape/dir使用原kernel公式，REAPER保留manual/auto长度及默认方式；私有形状按item
GUID保存到组件state v4，仍读旧v2/v3。GUI、波形与kernel采用相同形状族。宿主明确
委托的head/tail逐端处理，未委托的一端不能重复烘焙。修改形状会改变实际kernel PCM，
恢复旧无形状state会清除未来编辑；定向2项及UI6项/tsc通过。

## 实测：标准ARA委托在当前宿主未启用

Release `bidirectional-fades-01`正常构建，源提交`904dfcad`；实际使用独立profile、
临时1秒恒定0.5 WAV、manual in/out长度0.25s、7.81新c/S轴全0、首位HFS插件。
隔离REAPER PID45016从冷启动加载，没有向已有实例送脚本，未修改用户工程。
Factory广告TIMESTRETCH|CONTENT_FADES并接实时head/tail=0查询，但宿主给两次实际
createPlaybackRegion的flags都是`0x1`，没有head=8/tail=4。当前不能把厂商能力广告
解释为REAPER已委托。

44100Hz真实宿主导出44100帧，0.125s/0.5s/0.875s处分别约
0.37497735/0.5/0.37497735；半渐变处相对中段为0.74995470，不是HFS快起族的
0.73204285。单HFS包络oracle判定失败。源码正确保留了未委托的宿主路径，没有叠加
HFS音频包络假装完成；本机的自定义HFS形状暂时会明确提示未获委托。

## 历史替代路线（未实现，按用户新许可停止）

同一导出中，整个in区间的宿主默认包络与`2t-t²`的最大差约1.49e-7（24-bit输出），
这是实测，不是推断HFS已经支持。可进一步验证不同长度、auto交叉、两采样率、重叠、
冷恢复以及精确7.81版本，再评估“只允许已校准默认宿主包络的输出补偿层”；若采取它，
必须逐region在混合前补偿且验证零点/边界/有界增益，不能对未知shape反算、不能修改
REAPER长度为0，也不能把整轨重叠混音后除一个包络。最终仍须真实GUI形状变化后的
单次HFS音频oracle通过。SDK正式扩展/禁用host普通fade接口也可继续核查，不猜opcode。

## 证据与剩余工作

原始资料仅在ignored `.build-tmp/owned-fade-oracle/`：run.lua、plugin.log、oracle.RPP、
actual-fades.wav、analyse.py。隔离实例已正常退出。
用户随后明确允许走REAPER渐变，要求HFS调整渐变区宽度必须同步REAPER。因此采用
宿主单次包络，停止补偿研究；形状编辑关闭而弯曲示意保持。长度回写/撤销/保存恢复
仍需验证，完整双向目标保持活跃。不是技术已证明不可实现或整批已通过。
