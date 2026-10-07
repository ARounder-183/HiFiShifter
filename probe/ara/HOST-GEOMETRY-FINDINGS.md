# 宿主完整几何：源码事实与待验证绑定

中文调查记录，2026-10-05，Task38前置。不是REAPER加载/音频通过报告。

## 已核对的权威接口

REAPER官方 `justinfrankel/reaper-sdk` commit
`c0eafe87863b2bf69c5c822760f1b32a753b211b`：`sdk/reaper_vst3_interfaces.h`、
`sdk/reaper_plugin_functions.h`。本批直接读取锁定commit的官方raw文件，无clone或SDK修改。

- `IReaperHostApplication`在FUnknown后三槽为getReaperApi/getReaperParent/reaperExtended。
  parent selectors：1 track，2 take，3 project，4 fxdsp，5 trackchan。
  reaperExtended本头未提供ARA↔item转换opcode，不能猜。
- `GetMediaItemTake_Item(MediaItem_Take*) -> MediaItem*`给真实take的parent。
  `ValidatePtr2(ReaProject*,void*,const char*) -> bool`支持项目/轨道/item/take等类型。
  不透明ARA hostRef不能作为这些函数的参数。
- `GetMediaItemInfo_Value(MediaItem*,const char*) -> double`读取D_POSITION/D_LENGTH、
  D_FADEINLEN/D_FADEOUTLEN、D_FADEINLEN_AUTO/D_FADEOUTLEN_AUTO、timebase等。
- **7.81曲率变化**：D_FADEINDIR/D_FADEOUTDIR及C_FADE*SHAPE标注7.80及以前；
  7.81以后D_FADE*DIR_NEW与D_FADE*DIR2_NEW决定形状。只读旧C_FADEINSHAPE再投影
  原曲线不能证明7.81正确，须核对新的连续/S参数口径。
- `GetMediaItemTakeInfo_Value(MediaItem_Take*,const char*) -> double`读取D_STARTOFFS、
  D_PLAYRATE、B_PPITCH、I_CHANMODE、D_PITCH。保调开关与时间比率不能混为同一字段。
- marker数量函数真实名字是`GetTakeNumStretchMarkers`，不是`GetNumTakeStretchMarkers`。
  `GetTakeStretchMarker(take,int,double* pos,double* srcpos) -> int`：pos是item内位置、
  srcpos是源媒体位置；失败-1。`GetTakeStretchMarkerSlope(take,int) -> double`存在，
  当前头只交叉引用setter，没有数学公式/单位说明，不能根据名字编造。
- `GetSetMediaItemInfo_String`及`GetSetMediaItemTakeInfo_String`能读取GUID（setNewValue=false），
  属辅助元数据而不是音频权威。使用前还需读取完整字段与buffer契约。

## 已确认的产品缺口

- ARAPlaybackRegionProperties只给两个时间范围及flags，没有普通fade或REAPER marker数组。
  TimestretchReflectingTempo明确指context与modification的tempo关系，不等于总时长比。
- runtime现在广告TIMESTRETCH|REFLECT_TEMPO|CONTENT_FADES；后两项尚未实现。
  CONTENT_FADES协商意味着宿主可不再做普通淡化。当前snapshot拒绝content fade，不能
  将广告视为功能证据；需要先撤回虚假能力或完成真正契约，普通fade仍由宿主处理一次。
- 原kernel的ClipTake.stretch_markers字段只被定义/持久化保留，mixdown没有消费它。
  线性SoundTouch保调测试通过不能外推marker支持。
- 原App REAPER importer有`stretch_segments_full_cover`及分段导入实现，可参考原语义，
  但不能让插件读RPP替代ARA，也不能直接信旧SM字段=原生API单位。
  该helper的window裁断源位置使用线性插值，而velocity_start/end又表达坡度，完整坡度
  几何是否一致需要独立oracle；不能未经核对照抄。
- 当前EditState持久化轨级项目时间曲线，无源/区域局部锚点。移动/裁切/拉伸后参数重投影
  仍未实现；不要只改变clip长度就声称“保留编辑”。

## 绑定门与实施顺序

1. 在初始化/model/UI线程查询拥有引用的host extension，采集parent(2)是否真有take。
   查询必须核对当前document/owner仍活、ValidatePtr2所属project，不在process或worker调用。
2. 只在**直接take绑定**与该owner唯一真实assigned region同时成立时关联；多分配/无take
   不能按名字、路径、时间位置匹配。采集真实隐藏playback owner和editor-only owner行为。
   本批没有恢复此前Esc停止的Computer Use，因此实际parent返回尚未取得。
3. frozen HostClipGeometry只含Rust值/GUID，宿主API调用前释放内部锁以容许重入，调用后
   再核对doc/scope/model；端点必须与ARA权限及source窗口相容，否则明确拒绝/留缺口。
4. GUI ordinary fade元数据与音频处理责任分离；kernel渲染的ordinary fade归零，宿主
   最终应用一次。若content fade协商，需单独实现head/tail范围和计算，不能用普通字段代替。
5. source-coordinate曲线锚定/迁移，线性→分段→坡度→tempo/timebase按独立音频oracle
   验证；旧v2状态保持兼容。未获真实宿主绑定，不把typed fake测试当实际成功。

当前结论：线性正向保调已接通并有库回归；完整marker、tempo、fade和坐标迁移仍open。
