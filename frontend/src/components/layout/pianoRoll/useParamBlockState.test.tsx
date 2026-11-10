// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";
import { useParamBlockState } from "./useParamBlockState";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
const probe: {
    state: ParamBlockState;
    setState: ((next: ParamBlockState | ((prev: ParamBlockState) => ParamBlockState)) => void) | null;
    stateRef: { current: ParamBlockState } | null;
} = {
    state: EMPTY_PARAM_BLOCK_STATE,
    setState: null,
    stateRef: null,
};

function Harness() {
    const { state, stateRef, setState } = useParamBlockState();
    useEffect(() => {
        probe.state = state;
        probe.setState = setState;
        probe.stateRef = stateRef;
    }, [state, stateRef, setState]);
    return null;
}

async function render(): Promise<void> {
    await act(async () => {
        root.render(<Harness />);
    });
}

beforeEach(() => {
    probe.state = EMPTY_PARAM_BLOCK_STATE;
    probe.setState = null;
    probe.stateRef = null;
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

describe("useParamBlockState", () => {
    it("starts from the empty state", async () => {
        await render();
        expect(probe.state).toBe(EMPTY_PARAM_BLOCK_STATE);
        expect(probe.stateRef?.current).toBe(EMPTY_PARAM_BLOCK_STATE);
    });

    it("updates the ref synchronously and the React state on commit", async () => {
        await render();
        const next: ParamBlockState = { ...EMPTY_PARAM_BLOCK_STATE, framePeriodMs: 10 };
        act(() => probe.setState?.(next));
        expect(probe.stateRef?.current).toBe(next);
        await render();
        expect(probe.state).toBe(next);
    });

    it("applies functional updaters against the latest ref, not a render closure", async () => {
        await render();
        act(() => {
            probe.setState?.((state) => ({ ...state, framePeriodMs: state.framePeriodMs + 1 }));
            probe.setState?.((state) => ({ ...state, framePeriodMs: state.framePeriodMs + 1 }));
        });
        expect(probe.stateRef?.current.framePeriodMs).toBe(7);
        await render();
        expect(probe.state.framePeriodMs).toBe(7);
    });

    it("does not notify for the same object", async () => {
        await render();
        const before = probe.stateRef?.current;
        act(() => probe.setState?.(before as ParamBlockState));
        expect(probe.stateRef?.current).toBe(before);
        expect(probe.state).toBe(before);
    });
});
