/*
 * 脏标记（「未保存更改」的数据源）回归测试。
 *
 * 【为什么必须有】`project.dirty` 是**逐 reducer / thunk 手工挂**的标记，
 * 而它决定"切换工程 / 退出时是否询问"。漏挂的后果是静默数据丢失 ——
 * 实测过两条：改 BPM（Tempo Map 提交）与增删轨道都不标脏，「新建工程」
 * 不询问、直接丢弃。这类缺口不会有异常、不会有日志，只有用户丢掉工作。
 *
 * 【审计边界（本轮已核对并修复）】
 *   - Tempo Map 提交（BPM / 拍号）：`setTempoMapRemote.fulfilled` → 标脏；
 *   - 增删轨道：`addTrackRemote/removeTrackRemote.fulfilled` → 标脏；
 *   - 轨道音量与轨道名：`setTrackVolume` / `setTrackName` → 标脏（此前漏挂）；
 *   - 已有的：clip 增删、自动化点增删移、工程笔记、checkpointHistory，以及后端
 *     回报 `project.dirty` 的四条（基础/自定义音阶、时间线设置、拉伸设置）。
 *
 * 【仍未覆盖（如实记录，供后续跟进，不要当作已修）】轨道颜色 / 算法等其余轨道
 * 设置的提交路径。它们走的是后端快照通路（`applyTimelineState`），在那里统一
 * 标脏会把加载 / 撤销重做也误标 —— 需要一次专门的梳理，不适合塞进本轮。
 */
import { describe, expect, test } from "vitest";

import { markProjectDirty } from "./sessionDirtyState.js";
import sessionReducer, {
    addClip,
    setTempoMap,
    setMetronomeConfig,
    setTempoMapVisible,
    setTrackVolume,
} from "./sessionSlice";
import { setTempoMapRemote } from "./thunks/tempoMapThunks";
import { addTrackRemote } from "./thunks/timelineThunks";

const initial = sessionReducer(undefined, { type: "@@init" });
const trackId = initial.tracks[0]?.id ?? "track-1";

/** 用给定状态跑一条 action（本测试只关心 `project.dirty` 的翻转）。 */
function run(state: typeof initial, action: { type: string }) {
    return sessionReducer(state, action as never);
}

test("markProjectDirty 只做一件事：把标记置为 true", () => {
    const project = { dirty: false };
    markProjectDirty(project);
    expect(project.dirty).toBe(true);
});

describe("工程级编辑必须标脏", () => {
    test("新增片段与变更轨道音量会标脏", () => {
        expect(run(initial, addClip({ trackId })).project.dirty).toBe(true);
        expect(run(initial, setTrackVolume({ trackId, volume: 0.5 })).project.dirty).toBe(true);
    });

    test("Tempo Map 提交标脏 —— 即使后端回声与本地乐观值完全一致", () => {
        /*
         * 乐观路径：提交前本地已 `setTempoMap(nextMap)`，因此"回声与本地不同"
         * 这个判据在此恒为 false。脏标记不能建立在它之上 —— 这正是改 BPM 不
         * 提示的根因（改完 BPM 直接新建工程会静默丢弃）。
         */
        const optimistic = run(initial, setTempoMap(null));
        const normalized = optimistic.tempoMap;
        const fulfilled = {
            type: setTempoMapRemote.fulfilled.type,
            payload: {
                ok: true,
                tracks: initial.tracks,
                clips: initial.clips,
                tempoMap: normalized,
            },
            meta: { arg: normalized },
        };
        expect(run(optimistic, fulfilled).project.dirty).toBe(true);
    });

    test("新增轨道标脏（结构变更同样要询问）", () => {
        const fulfilled = {
            type: addTrackRemote.fulfilled.type,
            payload: { ok: true, tracks: initial.tracks, clips: initial.clips },
            meta: { arg: { name: "Track", parentId: null } },
        };
        expect(run(initial, fulfilled).project.dirty).toBe(true);
    });
});

describe("纯 UI 状态不标脏", () => {
    test("节拍器开关与速度图显示开关不会把工程标成已修改", () => {
        // 这两个开关只影响回放与显示，写进工程文件是噪音 —— 一旦标脏，
        // 用户每次切工程都会被问"要不要保存"。
        expect(
            run(initial, setMetronomeConfig({ metronomeEnabled: !initial.metronomeEnabled }))
                .project.dirty,
        ).toBe(false);
        expect(run(initial, setTempoMapVisible(true)).project.dirty).toBe(false);
    });
});
