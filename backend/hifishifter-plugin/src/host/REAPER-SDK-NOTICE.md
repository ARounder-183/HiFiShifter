# REAPER SDK 接口适配来源

中文说明。`reaper.rs`是HiFiShifter手写适配，不是原SDK文件；接口与API签名依据
REAPER官方SDK仓库 `justinfrankel/reaper-sdk`，commit
`c0eafe87863b2bf69c5c822760f1b32a753b211b` 的
`sdk/reaper_vst3_interfaces.h`、`sdk/reaper_plugin_functions.h`。
官方入口：https://www.reaper.fm/sdk/plugin/plugin.php

核对事项：IReaperHostApplication三个FUnknown槽后依次getReaperApi/getReaperParent/
reaperExtended；parent=1 track、2 take、3 project、4 fxdsp、5 trackchan。
GetPlayPositionEx是latency-compensated actual-what-you-hear；GetPlayPosition2Ex是下一个
处理块位置，不能把它用作原GUI实际游标。只在原UI/model线程查询所属project。

Task38a只读typed元数据另核对ValidatePtr2、GetMediaItemTake_Item、
GetMediaItemInfo_Value、GetMediaItemTakeInfo_Value、GetSetMediaItemInfo_String、
GetSetMediaItemTakeInfo_String（GUID读取setNewValue=false）、GetTakeNumStretchMarkers、
GetTakeStretchMarker、GetTakeStretchMarkerSlope、GetProjectStateChangeCount。
普通fade legacy与7.81 DIR_NEW/DIR2_NEW分别保留；marker输出/坡度保持raw值，
未转换为kernel秒坐标。host接口引用不保活project/take，逐getter重检文档/owner/scope，
对project change integer只比较相等，不假设单调或正数。实际REAPER返回尚待native验收。

SDK许可原文（来源sdk/LICENSE）：

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
