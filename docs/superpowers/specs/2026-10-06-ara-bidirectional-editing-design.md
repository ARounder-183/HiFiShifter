# REAPER 双向片段编辑与 HiFiShifter 渐变权威

中文设计，2026-10-06。用户要求设为目标并开始开发；本文覆盖旧计划的“宿主几何只读”
与“普通渐变声音由宿主处理、HFS仅示意显示”限制。保留独立App、同文档多轨GUI、
逐轨输出、已有参数源坐标锚点、线性拉伸、保存恢复与实时安全。短辅音、倒放、
非线性stretch markers仍不在本批范围。

## 目标与权威

1. 在原HiFiShifter GUI导入音频clip、移动/裁切/编辑clip、正向线性拉伸与调整渐变，
   修改写入该插件所属REAPER工程，不依赖活动工程或最后实例。
2. REAPER item静音同步到GUI和最终播放/离线输出，解除静音恢复；静音不删除参数。
3. REAPER保留渐变长度和默认方式；HFS自定义shape/curvature由插件保存与执行，
   不仅改变图形。长度双向同步；不能叠加REAPER默认包络形成双重淡化。
4. 几何修改由REAPER接收后通过真实ARA模型回流；GUI不把私有timeline当第二份几何权威。
   自定义渐变和修音仍属于HFS编辑状态。撤销/重做跨宿主几何与插件参数正确归属。
5. 保存工程、关闭所有GUI、冷重开、多轨与切工程后仍正确。原独立App不走REAPER接口。

## 架构

- 当前typed `IReaperHostApplication`已有直接parent project/take与唯一assigned region绑定。
  在此基础上增加主线程写命令，冻结region/GUID与请求代次，执行前重新查验所属工程、
  item/take以及租约。禁止从ARA hostRef强转、按位置/名称猜item或用当前工程API兜底。
- 原WebView命令口识别宿主几何操作；actor规划/文件准备不触碰REAPER API，native UI
  任务执行宿主写操作。宿主调用不持document/timeline/views锁，允许ARA同步重入。
- 命令限定已接入轨道/真实region与显式导入路径；新增item通过已核对的官方API创建，
  再等宿主建立ARA source/modification/region。导入文件不替代现有ARA音频授权。
- 复用原frontend交互与payload，不另建简化GUI。连续拖动合并，成功后的模型回流带来源
  票据避免反馈循环；失败恢复宿主真值，明确部分失败，不能丢失本地参数。
- 几何操作使用所属project的Undo块；参数历史和自定义渐变保存与已有组件state同源。
  撤销顺序需明确协调，不能用旧几何timeline快照覆盖宿主。

## 静音

官方锁定SDK `c0eafe87863b2bf69c5c822760f1b32a753b211b` 区分
`B_MUTE`（item solo覆盖后的mute）与`B_MUTE_ACTUAL`（忽略solo的原始mute）。
GUI与实际输出采用有效`B_MUTE`；原始标记不得误当有效solo行为。UI/model线程采集，
每region renderer读取原子静音门，不在process查API/加锁/做推理。模型变化与初次绑定、
关闭GUI/离线准备也要刷新，不将“打开GUI时才正确”当完成。

## 渐变音频责任门

锁定ARAInterface.h说明content-based fade flags可让宿主将边界淡化委托给插件。
这是候选正确路径，**尚未实测**；必须先验证REAPER在保留item长度/默认形状元数据时
是否停止应用默认fade，以及head/tail查询与flags边界。然后接原kernel fade处理及
自定义shape持久化。不能仅广告能力、直接解除unsupported或把普通fade烘焙一次就
宣称已完成。不能用未经校准的“除宿主包络”近似，尤其零点/重叠/自动交叉渐变。
如候选不可用，保留完整目标并评估另一条官方接口路线；不改成只显示或清零用户的
REAPER fade长度作为假成功。

## 验收与交付

定向源码合同在各批末跑，最终集中原GUI/播放导出/保存验收；不反复全套测试/review。
验证导入、移动、裁切、线性拉伸、两方向长度同步、HFS独立渐变形状、item mute/solo、
Ctrl+Z/redo、快速连续编辑、多轨、关闭GUI/冷重开与工程隔离。音频oracle验证单次包络
而非只比较图形。新Release安装到D:\VST\HiFiShifter.vst3，先退出REAPER并备份旧包；
只明确路径git add、仅本地提交，不自动push。工作区仅.worktrees/ara-plugin，主develop不动。
