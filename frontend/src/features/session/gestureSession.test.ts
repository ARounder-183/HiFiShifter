// 手势会话纯逻辑：三条规则（深度、推迟、落地）各自钉死。
import { describe, expect, it } from "vitest";
import {
    beginGestureSession,
    deferSnapshot,
    endGestureSession,
    isGestureActive,
    isStaleSnapshot,
    readPayloadRevision,
    type GestureSessionState,
} from "./gestureSession";
import type { TimelineState } from "../../types/api";

function slot(): GestureSessionState {
    return {
        _gestureDepth: 0,
        _gestureKind: null,
        _gestureBaseRevision: 7,
        _deferredSnapshot: null,
    };
}

const payload = (tag: string) => ({ tag }) as unknown as TimelineState;

describe("readPayloadRevision", () => {
    it("数字直接采纳，字符串（u64 超 2^53）解析成数字", () => {
        expect(readPayloadRevision({ revision: 42 })).toBe(42);
        expect(readPayloadRevision({ revision: "9007199254740993" })).toBe(9007199254740992);
    });

    it("缺失 / 非法 / null 一律视为「旧后端」", () => {
        expect(readPayloadRevision({})).toBeNull();
        expect(readPayloadRevision(null)).toBeNull();
        expect(readPayloadRevision({ revision: null })).toBeNull();
        expect(readPayloadRevision({ revision: Number.NaN })).toBeNull();
        expect(readPayloadRevision({ revision: "" })).toBeNull();
    });
});

describe("beginGestureSession / endGestureSession", () => {
    it("深度 0→1 记录基线种类与版本；嵌套不再覆盖种类", () => {
        const state = slot();
        beginGestureSession(state, { kind: "clip-move" }, 11);
        expect(state._gestureDepth).toBe(1);
        expect(state._gestureKind).toBe("clip-move");
        expect(state._gestureBaseRevision).toBe(11);

        beginGestureSession(state, { kind: "pan" }, 99);
        expect(state._gestureDepth).toBe(2);
        // 最早的那次才是这次连续交互的语义归属。
        expect(state._gestureKind).toBe("clip-move");
        expect(state._gestureBaseRevision).toBe(11);
    });

    it("嵌套时只有最后一个结束才交回推迟项", () => {
        const state = slot();
        beginGestureSession(state, { kind: "param-paint" }, 0);
        beginGestureSession(state, undefined, 0);
        deferSnapshot(state, { revision: 5, payload: payload("p") });
        expect(endGestureSession(state)).toBeNull();
        expect(state._deferredSnapshot).not.toBeNull();
        expect(endGestureSession(state)).toEqual({ revision: 5, payload: expect.anything() });
        expect(state._deferredSnapshot).toBeNull();
        expect(state._gestureDepth).toBe(0);
        expect(state._gestureKind).toBeNull();
    });

    it("没开会就收会：无害，但必须清掉残留（否则那份快照永远落不了地）", () => {
        const state = slot();
        state._deferredSnapshot = { revision: 3, payload: payload("stale") };
        expect(endGestureSession(state)).toBeNull();
        expect(state._deferredSnapshot).toBeNull();
        expect(state._gestureDepth).toBe(0);
    });
});

describe("deferSnapshot", () => {
    it("只保留版本号更大的那一份", () => {
        const state = slot();
        expect(deferSnapshot(state, { revision: 10, payload: payload("a") })).toBe(true);
        expect(deferSnapshot(state, { revision: 9, payload: payload("b") })).toBe(false);
        expect(state._deferredSnapshot?.revision).toBe(10);
        expect(deferSnapshot(state, { revision: 11, payload: payload("c") })).toBe(true);
        expect(state._deferredSnapshot?.revision).toBe(11);
    });

    it("两边都没有版本号（旧后端）时保留后到的那份 —— 不比旧行为更差", () => {
        const state = slot();
        deferSnapshot(state, { revision: null, payload: payload("first") });
        deferSnapshot(state, { revision: null, payload: payload("second") });
        expect((state._deferredSnapshot?.payload as unknown as { tag: string }).tag).toBe("second");
    });

    it("一份有版本一份没有时，以有版本的那份为准（缺版本 ≠ 0）", () => {
        const state = slot();
        deferSnapshot(state, { revision: null, payload: payload("no-rev") });
        deferSnapshot(state, { revision: 1, payload: payload("rev") });
        expect(state._deferredSnapshot?.revision).toBe(1);
        deferSnapshot(state, { revision: null, payload: payload("no-rev-2") });
        expect(state._deferredSnapshot?.revision).toBe(1);
    });
});

describe("isGestureActive / isStaleSnapshot", () => {
    it("深度 > 0 才算在会话中", () => {
        expect(isGestureActive({ _gestureDepth: 0 })).toBe(false);
        expect(isGestureActive({ _gestureDepth: 1 })).toBe(true);
        expect(isGestureActive({ _gestureDepth: Number.NaN })).toBe(false);
    });

    it("相等不算旧（同版本号的载荷可以覆盖 —— 内容相同）", () => {
        expect(isStaleSnapshot(5, 5)).toBe(false);
        expect(isStaleSnapshot(4, 5)).toBe(true);
        expect(isStaleSnapshot(6, 5)).toBe(false);
        expect(isStaleSnapshot(null, 5)).toBe(false);
    });
});
