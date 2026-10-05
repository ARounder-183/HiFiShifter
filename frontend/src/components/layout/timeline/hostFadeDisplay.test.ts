// 宿主淡化显示边界：新轴与旧公式分离，未知版本不能冒充线性曲线。
import {expect,test} from "vitest";
import {hostFadeDisplay,hostFadeLabel} from "./hostFadeDisplay";
import type {HostFadeMetadata} from "../../../types/api";

const metadata:HostFadeMetadata={curve_mode:"reaper_new",in_curvature:-0.2,out_curvature:0,in_s:0.65,out_s:0};
test("new axes preserve both values and cannot use a legacy curve",()=>{
    expect(hostFadeDisplay(metadata,false)).toBe("host_defined");
    expect(hostFadeLabel(metadata,false)).toBe("REAPER c=-0.20 S=0.65");
    expect(hostFadeDisplay(metadata,true)).toBe("linear");
    expect(hostFadeDisplay({...metadata,curve_mode:"unknown"},true)).toBe("host_defined");
});
test("standalone and known legacy hosts retain the existing fade renderer",()=>{
    expect(hostFadeDisplay(undefined,false)).toBe("legacy");
    expect(hostFadeDisplay({...metadata,curve_mode:"legacy"},false)).toBe("legacy");
});
