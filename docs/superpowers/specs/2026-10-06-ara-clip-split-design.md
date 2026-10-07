# 插件 clip 分割与短片段排查

## 范围

最新目标：排查共享App/插件的短片段无声；插件GUI支持clip分割。
此批不添加倒放、非线性拉伸、胶合/删除，不修改独立App的分割过渡行为。

## 分割合同

1. 独立能力`clipSplitting`：需要官方`SplitMediaItem`和`GetActiveTake`，不因可拖动就
   放开所有clip快捷键。插件提供S及精简右键菜单的播放头处分割。
2. 单/多选均使用已有前端秒域吸附和选区逻辑；原生整批核对当前item/take GUID、
   所属project、位置、长度、源起点与有效倍率。仅支持线性片段。
3. 宿主Undo块内直接分割所属item；原item保留为左段，新item为右段。
   读取新active take与两个真实GUID，不能只创建私有TimelineState片段。
4. 保存调用前父参数basis。宿主可能在API返回前同步缩短左段；确切新GUID登记谱系，
   右段继承正确源域参数，不从同源重叠clip中猜父。绑定完成清理临时副本。
5. 两段均真实回流且几何吻合后回复`created_clip_ids`、选中右段。
   静音item用GUI清单确认存在；失败/超时不假成功，宿主Undo保留。
6. 写入开始/结束废弃旧GUI读取，防止新两段被旧单段快照覆盖。
7. 渐变、take属性由REAPER原生分割继承；插件不额外套用App分割过渡设置。

## 本批计划与完成门

- 完成：官方ABI核对、能力门、原生规划/写入/回执、参数谱系、GUI菜单/快捷键/竞态。
- 定向检查：分割边界/重复批次；真实raw函数fixture的两GUID/源窗口；同源重叠及同步
  重入下参数继承；前端能力/旧刷新/右段选中。只做这一批定向检查及最终统一构建。
- 发包：完整VST3在REAPER退出时安装D:\VST，保留旧包备份。
- 仍需实际GUI闭环：分割后两段显示与声音、Ctrl+Z/Redo、保存重开。
  编译和fixture不等同实际REAPER验收，不把manifest nativeAcceptance改成true。
- 短片段排查结论与限制见probe/ara/SHORT-CLIP-FINDINGS.md，未复现实际无声前不宣告修复。

## 后续新增范围：元音丢音、clip音量、默认算法

- 用户实测：同素材独立App有声；旁路HFS FX有声，启用无声；不依赖合成开关。
  定位到ARA完整渲染入口的源尾严格检查：8073帧ka源、8084帧n源的窗口各多约1源帧。
  Source EOF按源网格round后最多允许一帧有界零尾，源起点在EOF或越界更多仍拒绝。
  不改宿主时间坐标/倍率/clip长度，不绕回源头，不靠关合成或换算法掩盖。
- Clip音量沿用原GUI增益徽章的拖拽、双击数值与多选，写宿主item D_VOL，0..4与原
  GUI范围一致。读取真实item/take D_VOL，take的负号极性保持；不改take音量。
  GUI扁平gain与active-take显示字段保持一致；向内核提交参数时还原源域gain，
  由宿主负责item增益，实际只施加一次的声音仍需REAPER验收。
- Snapshot新增可选只读diagnostics：当前renderer准备错误/版本、已发布PCM范围及前八
  clip的峰值/RMS。不复制输出音频，不在RT回调扫描，不调用宿主或写Commit。
- 插件新宿主轨道默认NSF-HiFiGAN；共享App默认本来就是HiFiGAN，不改已有明确算法覆盖。
- 本批不把当前旧DLL的源码推断当修后实测，最后统一包后验收ka/n、分割Undo和音量。
