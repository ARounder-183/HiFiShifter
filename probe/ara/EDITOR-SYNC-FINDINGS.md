# 原GUI自动应用与播放头同步批次

## 最新用户渐变裁定与实现（覆盖下方历史未校准门）

2026-10-05用户允许渐变“走hifishifter自己的”，不要求完整还原宿主曲线。新轴保留
原始c/S数值，`visualFadeGain`混合本应用已有幂曲率与S族，定义为HFS示意曲线，不
把它冒称REAPER公式。零轴显示线性；未知版本用明确示意线。独立App/legacy轴仍用
原`fadeGainSigned`逐值路径。音频责任不变，普通fade仅由REAPER应用一次。

Canvas不再只画长度边界；元数据也经active/inactive take、时间线/参数编辑器波形
投影进入同一个共享求值器，避免“线画一种、波形画另一种”。手动/auto有效长度原语义
保留，真实宿主改动仍走已有独立UI revision，不丢pending编辑/不重复神经渲染。

实测7文件27项通过/exit0，覆盖双轴有界单调/端点、两轴更新、App原值不变、实际
Canvas不被旧linear快路径短路、普通/auto长度/take投影与波形几何响应。类型检查
修正新夹具漏channelMode后exit0；原始输出`.build-tmp/hfs-fade-ui-01.log`。
不是原生GUI验收；Release/完整最终Verify/宿主四BUG仍待最终门。

用户改范围前启动过无插件精确校准的隔离REAPER45948；没有得到cases/WAV/RPP。
进程仍在，MainWindowHandle=0、CloseMainWindow返回false，未强杀/未发送后续脚本。
其新profile自动附加了公共VST3路径，当前未知停在扫描还是模态；不猜运行状态。
已经停止校准并将两个未提交的一次性脚本移到ignored `.build-tmp/fade-axis-oracle-01`
保留，不带进产品/下一轮执行。此隔离进程须正常关闭后才可开最终用户测试实例。
用户原RPP摘要未变；没有改主develop、SDK或用户现有profile。

中文记录，2026-10-05。一次性诊断记录，不替代最终REAPER验收。用户要求全部源码完成后
统一启动隔离实例交其集中测试，本批未启动REAPER、未操作Computer Use。

## 已有证据与修改

- actor在循环顶部执行到期应用，不再要求只读请求FIFO出现150ms空闲；仅在已成功入队
  的写入全部处理后取最新票据。切换素材后缺失的项目原线key由actor组装完整clip分析缓存，
  不依赖GUI读取原线或新的ClipPitchReady。重叠选择保持优先，用户目标曲线不覆盖。
- render_error与Conflict/授权error分离；暂时渲染失败后1秒重试，Conflict仍保留未提交曲线。
  气声/张力/共振峰与pitch共用原处理器需求判定；需要合成且缺原线时pending，None/纯混音
  不被此门禁阻塞。HNSEP仍整段处理，没有新增分块。
- get_playback_state经同文档route/view准入后直接读取原子宿主时钟，不排到神经合成后面，
  不读编辑timeline锁、不调用host API、不IO。实际时钟来自所属工程GetPlayPositionEx；
  前端宿主分支不再把整段请求RTT（含排队）加到实际位置。插件视觉插值最多100ms，
  真正后退seek/loop仍接受，独立App原外推路径不变。
- fade单独UI版本刷新，真实唯一region绑定后的普通/auto长度和轴数值自动显示，不force
  reload、不改变音频generation。提交参数时清普通fade，音频由REAPER应用一次。

## 本批回归（实测）

- 后端actor/fade/版本批次：29项，首轮28通过；新增张力门禁夹具漏开breath_enabled，
  与原内核的实际分离开关约定冲突。修正夹具后该项及typed host合同共14项通过，exit0。
- 前端6文件28项通过，exit0；tsc -b exit0。涵盖长RTT不能前漂、真实后退seek、
  独立App原行为、插件插值上限、新轴与旧曲线分离及原canvas回归。
- 新快查询的回归持住actor timeline锁仍在200ms内返回宿主位置；关闭文档后明确拒绝。
- 暂时不支持的content-fade令自动合成失败，恢复后无需任何reload自动发布。

## 不声称完成的部分

REAPER7.81官方头明确C_FADE*SHAPE仅适用于7.80及以前，新轴为D_FADE*DIR_NEW及
D_FADE*DIR2_NEW。已核对GetAppVersion typed ABI，未知版本明确标记unknown。
新轴两项为零显示直线，其它新轴只画准确范围边界/压暗区和原始c/S数值，**没有使用旧
七预设公式冒充精确新曲线**。任意新曲率的精确曲线尚缺宿主oracle校准，完整fade形状门
仍open；官方头没有公开fade求值函数。UI数值同步不是完整形状校准证据。

延迟project绑定、四BUG原GUI操作与实际HiFiGAN输出尚待最后集中native验收；本批无
实机成功结论。正常长宿主源的旧限制/资源策略、同路径换源及旧v2几何迁移门仍继续。
