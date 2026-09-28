// @vitest-environment jsdom
/*
 * 时间轴编辑事件消费端的回归测试。
 *
 * 【为什么必须有】`hifi:timelineEditOp` 不只有键盘一个来源：菜单里的「粘贴」、
 * 记事本暂存块的「插入到时间轴」都是**合成派发**。而粘贴曾经在消费端无条件
 * 布防长按重复（holdRepeat），而 holdRepeat 只靠 `keyup` / `blur` 终止 ——
 * 合成事件没有键可以松，于是布下一个永远停不下来的 50ms 定时器：
 * **点一次「插入到时间轴」就无限往时间轴里插**。
 *
 * 因此这里锁定的契约是：消费端收到粘贴事件只粘一次，布防长按是**派发方**
 * （App 的键盘路径，那里确实有键按着）的责任。
 */

import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, expect, test, vi } from "vitest";

import {
    isHoldRepeatActive,
    keybindingsReducer,
    stopHoldRepeat,
} from "../../../../features/keybindings";
import {
    useTimelineEventHandlers,
    type UseTimelineEventHandlersArgs,
} from "./useTimelineEventHandlers";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** 消费端只读这几个 session 字段（粘贴分支甚至不读），给最小的可用形状即可。 */
const SESSION_STUB = {
    multiSelectedClipIds: [] as string[],
    selectedClipId: null,
    clips: [],
    ignoreGrouping: false,
    disabledGroupIds: [] as string[],
};

function ref<T>(value: T): { current: T } {
    return { current: value };
}

function buildArgs(): UseTimelineEventHandlersArgs {
    return {
        dispatch: vi.fn(),
        sessionRef: ref(SESSION_STUB),
        scrollRef: ref(null),
        viewport: {
            getScrollLeft: () => 0,
            setScrollLeft: () => 0,
            getViewportWidth: () => 1000,
            getViewportHeight: () => 500,
            getScrollTop: () => 0,
            setScrollTop: () => {},
        },
        trackListScrollRef: ref(null),
        pxPerSecRef: ref(100),
        viewportWidthRef: ref(1000),
        keyboardZoomPendingRef: ref(null),
        pxPerSec: 100,
        setPxPerSec: vi.fn(),
        commitScrollLeftState: vi.fn(),
        rowHeight: 40,
        setMultiSelectedClipIds: vi.fn(),
        copyClips: vi.fn(async () => true),
        cutClips: vi.fn(),
        pasteClipsAtPlayhead: vi.fn(),
        splitSelectedAtPlayhead: vi.fn(),
        normalizeClips: vi.fn(),
        groupClips: vi.fn(),
        ungroupClips: vi.fn(),
        contextMenu: null,
        trackAreaMenu: null,
        setContextMenu: vi.fn(),
        setTrackAreaMenu: vi.fn(),
        syncScrollLeft: vi.fn(),
    } as unknown as UseTimelineEventHandlersArgs;
}

function Harness({ args }: { args: UseTimelineEventHandlersArgs }) {
    useTimelineEventHandlers(args);
    return null;
}

/** 挂载消费端，返回卸载函数。 */
async function mount(args: UseTimelineEventHandlersArgs): Promise<() => Promise<void>> {
    // keybindings 用真实 reducer：消费端的粘贴分支会读 `clip.paste` 的绑定，
    // 用傀儡 store 会让那段代码抛错并被事件派发吞掉 —— 测试就会对着一个
    // "看起来通过"的假象放行。
    const store = configureStore({
        reducer: {
            session: (state: typeof SESSION_STUB = SESSION_STUB) => state,
            keybindings: keybindingsReducer,
        },
    });
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            // 运行期只需要一个能回答 `store.getState().session` 的 store；
            // 类型上它与应用的 RootState 无关，故在此收敛掉。
            <Provider store={store as never}>
                <Harness args={args} />
            </Provider>,
        );
    });
    return async () => {
        await act(async () => root.unmount());
        host.remove();
    };
}

afterEach(() => {
    stopHoldRepeat();
    vi.useRealTimers();
});

test("合成派发的粘贴只粘一次，且不布防长按重复", async () => {
    vi.useFakeTimers();
    const args = buildArgs();
    const unmount = await mount(args);
    try {
        await act(async () => {
            window.dispatchEvent(
                new CustomEvent("hifi:timelineEditOp", { detail: { op: "paste" } }),
            );
        });

        expect(args.pasteClipsAtPlayhead).toHaveBeenCalledTimes(1);
        // 这一条就是本测试存在的理由：合成事件没有键可松，一旦布防长按，
        // 计时器就再也停不下来。
        expect(isHoldRepeatActive(), "合成粘贴不得布防长按重复").toBe(false);

        // 时间推过 holdRepeat 的初始延时（400ms）与数个重复间隔：不能出现第二拍。
        await act(async () => {
            vi.advanceTimersByTime(2000);
        });
        expect(args.pasteClipsAtPlayhead).toHaveBeenCalledTimes(1);
    } finally {
        await unmount();
    }
});

test("粘贴之外的编辑操作不受影响（删除只发一次请求）", async () => {
    const args = buildArgs();
    const unmount = await mount(args);
    try {
        await act(async () => {
            window.dispatchEvent(
                new CustomEvent("hifi:timelineEditOp", { detail: { op: "delete" } }),
            );
        });
        // 选区为空 → 删除分支直接 return，不发请求（与既有语义一致）。
        expect(args.dispatch).not.toHaveBeenCalled();
        expect(isHoldRepeatActive()).toBe(false);
    } finally {
        await unmount();
    }
});
