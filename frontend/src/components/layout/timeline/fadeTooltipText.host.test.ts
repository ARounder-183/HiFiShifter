// 宿主淡化浮标回归：原始新轴优先，不用旧形状图标冒充校准完成。
import {expect,test} from "vitest";
import {renderToStaticMarkup} from "react-dom/server";
import {createElement,Fragment} from "react";
import {buildSingleFadeInfoText,buildSingleFadeInfoContent} from "./fadeTooltipText";

test("host tooltip reports both raw axes and its calibration boundary",()=>{
    const args={isOut:false,shape:6,dir:0.8,lengthSec:0.75,t:(key:string)=>key,
        formatCtx:{primaryTimeUnit:"seconds" as const,secondaryTimeUnit:"none" as const,bpm:120,beatsPerBar:4,grid:"1/4"},
        hostFades:{curve_mode:"reaper_new" as const,in_curvature:-0.2,out_curvature:0,in_s:0.65,out_s:0}};
    const text=buildSingleFadeInfoText(args);expect(text).toContain("REAPER c=-0.20 S=0.65");
    expect(text).toContain("未校准");expect(text).not.toContain("fade_shape");
    const rich=renderToStaticMarkup(createElement(Fragment,null,buildSingleFadeInfoContent(args)));
    expect(rich).toContain("S=0.65");expect(rich).not.toContain("<svg");
});
