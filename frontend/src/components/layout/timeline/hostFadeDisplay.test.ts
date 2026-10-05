// 宿主淡化示意显示：本应用包络有界/端点正确，保留原App路径，不伪称REAPER公式。
import {expect,test} from "vitest";
import {hostFadeDisplay,hostFadeLabel,visualFadeGain} from "./hostFadeDisplay";
import {fadeGainSigned} from "./reaperFade";
import type {HostFadeMetadata} from "../../../types/api";

const metadata:HostFadeMetadata={curve_mode:"reaper_new",in_curvature:-0.2,out_curvature:0,in_s:0.65,out_s:0};
test("new axes preserve both values and use an explicit HFS visual style",()=>{
    expect(hostFadeDisplay(metadata,false)).toBe("hifishifter");
    expect(hostFadeLabel(metadata,false)).toBe("REAPER c=-0.20 S=0.65");
    expect(hostFadeDisplay(metadata,true)).toBe("linear");
    expect(hostFadeDisplay({...metadata,curve_mode:"unknown"},true)).toBe("hifishifter");
});

test("HFS visuals are bounded monotone with exact in/out endpoints for both axes",()=>{
    for (const c of [-1,-0.35,0,0.7,1]) for (const s of [-1,-0.25,0,0.5,1]) for (const mode of ["in","out"] as const) {
        const axes={...metadata,in_curvature:c,out_curvature:c,in_s:s,out_s:s};
        const values=Array.from({length:201},(_,i)=>visualFadeGain(axes,6,0.8,mode,i/200));
        expect(values[0]).toBe(mode==="in"?0:1);expect(values[200]).toBe(mode==="in"?1:0);
        values.forEach((v,i)=>{expect(v).toBeGreaterThanOrEqual(0);expect(v).toBeLessThanOrEqual(1);
            if (i) expect(mode==="in"?v-values[i-1]:values[i-1]-v).toBeGreaterThanOrEqual(-1e-12);});
    }
});
test("changing either axis changes visual curves and App/legacy remain exactly unchanged",()=>{
    const a=visualFadeGain(metadata,0,0,"in",0.3);
    expect(visualFadeGain({...metadata,in_curvature:0.7},0,0,"in",0.3)).not.toBe(a);
    expect(visualFadeGain({...metadata,in_s:-0.65},0,0,"in",0.3)).not.toBe(a);
    for (const mode of ["in","out"] as const) for (const t of [0,0.025,0.3,0.85,1]) {
        expect(visualFadeGain(undefined,5,-0.3,mode,t)).toBe(fadeGainSigned(5,-0.3,mode,t));
        expect(visualFadeGain({...metadata,curve_mode:"legacy"},5,-0.3,mode,t)).toBe(fadeGainSigned(5,-0.3,mode,t));
    }
    expect(visualFadeGain({...metadata,in_curvature:0,in_s:0},0,0,"in",0.3)).toBe(0.3);
    expect(Number.isFinite(visualFadeGain({...metadata,in_curvature:NaN,in_s:Infinity},0,0,"in",NaN))).toBe(true);
});
test("standalone and known legacy hosts retain the existing fade renderer",()=>{
    expect(hostFadeDisplay(undefined,false)).toBe("legacy");
    expect(hostFadeDisplay({...metadata,curve_mode:"legacy"},false)).toBe("legacy");
});
