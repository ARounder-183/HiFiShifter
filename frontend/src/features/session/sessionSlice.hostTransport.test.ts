// 插件宿主时钟回归：停播seek由DAW权威驱动，独立app行为不变。
import { expect, test } from "vitest";
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
