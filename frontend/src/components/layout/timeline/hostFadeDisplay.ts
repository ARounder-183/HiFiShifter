// 宿主淡化只读显示：7.81新轴不能套用旧七预设公式；未校准曲线明确标记，不伪造轨迹。
import type {HostFadeMetadata} from "../../../types/api";

/** 新轴两项均为零才可确定为直线；其它新轴忠实显示数值，等待宿主oracle校准。 */
export function hostFadeDisplay(metadata:HostFadeMetadata|undefined,isOut:boolean):"legacy"|"linear"|"host_defined" {
    if (!metadata||metadata.curve_mode==="legacy") return "legacy";
    if (metadata.curve_mode!=="reaper_new") return "host_defined";
    const curvature=isOut?metadata.out_curvature:metadata.in_curvature;
    const s=isOut?metadata.out_s:metadata.in_s;
    return curvature===0&&s===0?"linear":"host_defined";
}

/** 直接报告宿主两个原始轴；问号表示版本语义不可用，不沿用过时shape名称。 */
export function hostFadeLabel(metadata:HostFadeMetadata,isOut:boolean):string {
    if (metadata.curve_mode==="unknown") return "REAPER ?";
    const curvature=isOut?metadata.out_curvature:metadata.in_curvature;
    const s=isOut?metadata.out_s:metadata.in_s;
    return `REAPER c=${curvature.toFixed(2)} S=${s.toFixed(2)}`;
}
