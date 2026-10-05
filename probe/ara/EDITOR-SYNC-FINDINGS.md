# 原GUI自动应用与播放头同步批次

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
