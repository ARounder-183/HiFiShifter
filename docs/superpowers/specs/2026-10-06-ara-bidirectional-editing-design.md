# REAPER 双向片段编辑与渐变宽度同步

中文设计，2026-10-06。用户要求设为目标并开始开发；本文覆盖旧计划的“宿主几何只读”
与“普通渐变声音由宿主处理、HFS仅示意显示”限制。保留独立App、同文档多轨GUI、
逐轨输出、已有参数源坐标锚点、线性拉伸、保存恢复与实时安全。短辅音、倒放、
非线性stretch markers仍不在本批范围。

## 目标与权威

1. 在原HiFiShifter GUI导入音频clip、移动/裁切/编辑clip、正向线性拉伸与调整渐变，
   修改写入该插件所属REAPER工程，不依赖活动工程或最后实例。
2. REAPER item静音同步到GUI和最终播放/离线输出，解除静音恢复；静音不删除参数。
3. 按用户随后明确许可，当前REAPER不委托content fades时由REAPER执行渐变声音。
   HFS拖动手动/自动渐变宽度必须写回REAPER，两方向长度自动同步；HFS保留弯曲示意
   显示，不承诺精确还原新c/S轴，不开放当前无法兑现的插件形状/曲率编辑。
   已有自有渐变状态与委托端代码保留兼容，但不烘焙未委托端、不叠加双重淡化。
4. 几何修改由REAPER接收后通过真实ARA模型回流；GUI不把私有timeline当第二份几何权威。
   修音仍属于HFS编辑状态。撤销/重做跨宿主几何与插件参数正确归属。
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

## 渐变音频责任与当前实测

锁定ARAInterface.h说明content-based fade flags可让宿主将边界淡化委托给插件。
这是候选正确路径；隔离REAPER7.81实测flags仅0x1，未启用委托，恒定PCM导出为宿主
默认包络而非HFS形状（见probe/ara/OWNED-FADE-FINDINGS.md）。原目标曾要求验证
REAPER在保留item长度/默认形状元数据时
是否停止应用默认fade，以及head/tail查询与flags边界。然后接原kernel fade处理及
自定义shape持久化。不能仅广告能力、直接解除unsupported或把普通fade烘焙一次就
宣称已完成。不能用未经校准的“除宿主包络”近似，尤其零点/重叠/自动交叉渐变。
用户最新许可采用REAPER渐变，覆盖此前“实际声音必须归HFS”的门，不再推进输出
补偿/未知包络反算。这不降低宽度双向写入、撤销与保存恢复要求；不得清零用户长度。

## 移动回流时序

自动刷新不强制覆盖插件拖动预览或在途几何写入；每次交互/写入的开始和结束推进
读取代次，丢弃跨代响应。丢弃不消费宿主版本，下次轮询补同步。原生写入回执等本次
确切clip的位置/长度/源起点/倍率/轨道及指定渐变长度真正回流，不因旧timeline仍可读
就报成功，也不重复setter。失败释放同步保护，宿主仍是唯一几何权威。

## 验收与交付

定向源码合同在各批末跑，最终集中原GUI/播放导出/保存验收；不反复全套测试/review。
验证导入、移动不闪回、裁切、线性拉伸、两方向渐变长度同步、item mute/solo、
Ctrl+Z/redo、快速连续编辑、多轨、关闭GUI/冷重开与工程隔离。音频oracle验证单次包络
而非只比较图形（当前采用REAPER单次包络）。新Release安装到D:\VST\HiFiShifter.vst3，先退出REAPER并备份旧包；
只明确路径git add、仅本地提交，不自动push。工作区仅.worktrees/ara-plugin，主develop不动。
