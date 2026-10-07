// 插件宿主时钟回归：停播seek由DAW权威驱动，独立app行为不变。
import { expect, test, vi } from "vitest";
import reducer from "./sessionSlice";
import { syncPlaybackState } from "./thunks/transportThunks";
import {playOriginal} from "./thunks/transportThunks";

test("host stopped seek overwrites local cursor without starting playback", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const next = reducer({ ...base, playheadSec: 7 }, syncPlaybackState.fulfilled({
        ok: true, is_playing: false, base_sec: 0, position_sec: 2, duration_sec: 20,
        host_authoritative: true,
    } as never, "host-seek", undefined as never));
    expect(next.playheadSec).toBe(2);
    expect(next.runtime.isPlaying).toBe(false);
});

test("standalone stopped polling still preserves local editor cursor", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const next = reducer({ ...base, playheadSec: 7 }, syncPlaybackState.fulfilled({
        ok: true, is_playing: false, base_sec: 0, position_sec: 2, duration_sec: 20,
    } as never, "app-idle", undefined as never));
    expect(next.playheadSec).toBe(7);
});
test("ARA request acknowledgement does not claim the DAW has started",()=>{
    const base=reducer(undefined,{type:"@@INIT"});
    const next=reducer(base,playOriginal.fulfilled({ok:true,clipId:null,anchorSec:2,host_request:true} as never,"ara-request",undefined));
    expect(next.runtime.isPlaying).toBe(false);
});

test("host position never includes queue or round-trip delay and backward seeks remain valid",()=>{
    const timer=vi.spyOn(performance,"now").mockReturnValue(10000);
    try {
        const base=reducer(undefined,{type:"@@INIT"});
        const received=(state:typeof base,position:number,dispatched:number)=>reducer(state,syncPlaybackState.fulfilled({
            ok:true,is_playing:true,base_sec:0,position_sec:position,duration_sec:20,host_authoritative:true,
        } as never,"host-poll",{epoch:state._transportEpoch,dispatchedAtMs:dispatched}));
        const first=received(base,4.25,1000);expect(first.playheadSec).toBe(4.25);
        const next=received(first,4.3,9950);expect(next.playheadSec).toBe(4.3);
        expect(received(next,1.5,9000).playheadSec).toBe(1.5);
        const standalone=reducer(base,syncPlaybackState.fulfilled({ok:true,is_playing:true,position_sec:4,base_sec:0} as never,
            "app-poll",{epoch:base._transportEpoch,dispatchedAtMs:9900}));
        expect(standalone.playheadSec).toBeCloseTo(4.1);
    } finally {timer.mockRestore();}
});
