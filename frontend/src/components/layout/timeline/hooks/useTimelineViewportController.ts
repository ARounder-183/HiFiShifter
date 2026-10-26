// 时间轴视口控制：滚动/缩放落地、标尺滚轮、竖直滚动条与粘贴聚焦。
import React from "react";
import { createTimelineViewportAccess } from "./timelineViewportAccess";
import { useCaptureReveal } from "./useCaptureReveal";
import { createTimelineAxis, playheadLineLeftPx } from "../../renderKernel/timelineAxis";
import { computeFocusCursorScrollLeft } from "../../../../utils/autoFollowScroll";
import { setPendingPlayheadReveal } from "../../../../features/session/sessionSlice";
import type { TimelineKernelHost } from "../kernel/host/timelineKernelHost";
import type { TimelineSessionSlice, useTimelineState } from "./useTimelineState";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    s: TimelineSessionSlice;
    dispatch: AppDispatch;
    kernelHostRef: React.MutableRefObject<TimelineKernelHost | null>;
    setHorizontalScrollbarGutterPx: React.Dispatch<React.SetStateAction<number>>;
    visualPlayheadSecRef: React.MutableRefObject<number>;
};

export function useTimelineViewportController({
    state,
    s,
    dispatch,
    kernelHostRef,
    setHorizontalScrollbarGutterPx,
    visualPlayheadSecRef,
}: Deps) {
    const {
        scrollRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        pxPerSecRef,
        pxPerSec,
        setPxPerSec,
        dynamicProjectSec,
        syncScrollLeft,
        setScrollLeftState,
        trackIdFromClientY,
        rowHeight,
        sessionRef,
    } = state;
    const [viewportAccess] = React.useState(() =>
        createTimelineViewportAccess({ scrollRef, kernelHostRef }),
    );
    // useCaptureReveal 需要 MutableRefObject 形状；包装一次即可，实例本身仍是单例。
    const viewportAccessRef = React.useMemo(() => ({ current: viewportAccess }), [viewportAccess]);

    // 录完之后把视口移到刚录好的区间（见 `useCaptureReveal` 的文件头）。
    useCaptureReveal(viewportAccessRef);

    /**
     * 标尺播放头（竖线 + 倒三角）的**挂载定位**。
     *
     * 两者都由内核逐帧写（与轨道区播放头同源同帧、同一视口左缘），但内核宿主是在
     * 被动 effect 里创建的、首帧要等一个 rAF；在那之前元素若没有 `left`，会停在
     * 静态位置（左缘）闪一帧。这里用绘制前执行的 layout effect 补上初值 —— 之后由
     * 内核接管（含倒三角，见 `TimelineKernelDomSync.rulerPlayheadHead`）。
     */
    React.useLayoutEffect(() => {
        const left = playheadLineLeftPx(
            createTimelineAxis({
                pxPerSec: pxPerSecRef.current,
                scrollLeftPx: viewportAccess.getScrollLeft(),
                viewportWidthPx: viewportAccess.getViewportWidth(),
                dpr: window.devicePixelRatio || 1,
            }),
            visualPlayheadSecRef.current,
        );
        if (rulerPlayheadLineRef.current) {
            rulerPlayheadLineRef.current.style.left = `${left}px`;
        }
        if (rulerPlayheadHeadRef.current) {
            rulerPlayheadHeadRef.current.style.left = `${left}px`;
        }
        // 只在挂载时定位：之后的每一次写入都由内核负责（同一次帧提交）。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /**
     * clientY → 轨道 id（模式无关）。
     *
     * 旧实现读原生 scroller 的 `scrollTop`；内核模式下容器不再滚动（真值在宿主里），
     * 必须改用「容器矩形 + 宿主缓存的 scrollTop」。
     *
     * 特殊说明：宿主存在时走内核分支；宿主为 null（挂载前 / 测试环境）时委托给
     * `trackIdFromClientY` 兜底。旧的原生 scroller 实现已随"渲染内核唯一路径"改造删除，
     * `scrollRef` 已无 JSX 挂载点，因此这里**不是**在两种渲染模式之间做选择。
     */
    const resolveTrackIdAtClientY = React.useCallback(
        (clientY: number): string | null => {
            const host = kernelHostRef.current;
            if (host === null) return trackIdFromClientY(clientY);
            const rect = host.getContainerRect();
            if (rect === null) return null;
            const y = clientY - rect.top + host.getViewport().scrollTop;
            const idx = Math.floor(y / rowHeight);
            const tracks = sessionRef.current.tracks;
            if (idx < 0 || idx >= tracks.length) return null;
            return tracks[idx]?.id ?? null;
        },
        [rowHeight, sessionRef, trackIdFromClientY, kernelHostRef],
    );

    /**
     * 内核水平滚动提交（每跨过 256px 一次）。
     *
     * 走旧实现的同一条通知链（`syncScrollLeft`）：更新 React state（标尺刻度
     * 窗口）**并**把视口写入 `timelineViewportSync`（参数编辑器同步的写侧）。
     * 若直接传 `setScrollLeftState`，同步写侧会被漏掉——表现为「内核滚动时
     * 参数编辑器不跟随」。
     */
    const handleKernelScrollLeftCommit = React.useCallback(
        (next: number, nextPxPerSec: number) => {
            syncScrollLeft(next);
            if (Math.abs(nextPxPerSec - pxPerSec) > 1e-9) {
                // 【缩放变化时位置必须**同批**进 React】标尺刻度窗口是 React 侧
                // `(pxPerSec, scrollLeft)` 的函数（见 useTimelineState 的 `tickAxis`），
                // 只改其中一项就会有一帧按**新缩放**换算**旧像素位置**——锚点对应的
                // 时间错位，标尺某段既没有刻度线也没有文本。纯滚动仍走
                // `syncScrollLeft` 的量化路径，保住"滚动帧不进 React"。
                setPxPerSec(nextPxPerSec);
                setScrollLeftState(next);
            }
        },
        [syncScrollLeft, pxPerSec, setPxPerSec, setScrollLeftState, kernelHostRef],
    );

    /**
     * 滚轮缩放的**待落地目标**（内核请求、React 落地）。
     *
     * 标尺是 DOM、由 React 用 `pxPerSec` 布局，轨道区归内核。内核若自己先切缩放，
     * 缩放过程中标尺会整整落后一个滚轮步（约 10% 缩放 → 视口宽 1000px 时末端差
     * 上百像素），就是「标尺抽动 / 网格与标尺没对齐」。旧实现没有这个问题，因为
     * 缩放只有 React 一个真值源、画布也在 React 提交后才画。
     *
     * 因此这里照旧：内核只"请求"，React 用 `setPxPerSec` 触发重排，并在**同一次
     * 提交的 layout effect** 里把缩放应用到内核——DOM 重排与内核缩放落在同一帧的
     * 绘制中，两层不可能分叉。
     */
    const pendingKernelZoomRef = React.useRef<{ pxPerSec: number; scrollLeft: number } | null>(
        null,
    );
    const handleKernelZoomRequest = React.useCallback(
        (next: { pxPerSec: number; scrollLeft: number }) => {
            if (next.pxPerSec === pxPerSec) {
                // 缩放已在上/下限（或该步被钳制回原值）：请求退化为"位置无变化"，
                // 直接应用一次即可。**不能**留成待落地——`setPxPerSec` 传同值不会
                // 触发重渲染，下面的 layout effect 也就永远不会跑，请求会被搁置。
                pendingKernelZoomRef.current = null;
                viewportAccess.setZoomAndScroll(next.pxPerSec, next.scrollLeft);
                // 同上：位置变了，标尺刻度的窗口就必须跟着变（这一支不改 `pxPerSec`，
                // 不会触发下面的 layout effect，所以位置要在这里自己提交）。
                setScrollLeftState(next.scrollLeft);
                return;
            }
            pendingKernelZoomRef.current = next;
            // 只改缩放：位置由下面的 layout effect 与缩放**原子**应用（分两次写会让
            // 中间那一帧出现"新缩放 + 旧位置"的错位）。
            setPxPerSec(next.pxPerSec);
            // 【位置也必须与缩放**同批**提交给 React】标尺刻度是按 React 的
            // `scrollLeft` 状态取窗口的（见 `useTimelineState` 的 `tickAxis`），而缩放
            // 会大幅改变位置：以光标为锚放大 10 倍时，位置可以从 1000px 直接跳到
            // 14500px。若这一批只提交 `pxPerSec`，刻度的窗口仍按**旧位置**生成 ——
            // 可见视口完全落在窗口之外，标尺文字整片消失（用户报告的"水平缩放时标尺
            // 文本闪烁 / 消失"）；小步缩放则表现为窗口错位一帧的闪烁。量化提交
            // （`onScrollLeftCommit`）虽然最终会纠正它，但那要等到下一次 rAF。
            // 同批提交后，标尺刻度、内核视口（layout effect 里原子应用）与 GL 网格
            // 落在同一帧的同一投影上。
            setScrollLeftState(next.scrollLeft);
        },
        [pxPerSec, setPxPerSec, setScrollLeftState, viewportAccess],
    );
    /**
     * 标尺上的滚轮：**与画布内的滚轮完全同义**。
     *
     * 标尺是内核容器**之外**的 DOM 条，过去它的 `onWheel` 只 `preventDefault` 把滚轮
     * 吞掉（而那行在 React 的 passive 监听下还是空操作），用户在标尺上滚轮毫无反应。
     * 现在只做"转交"：把事件原样交给内核的滚轮入口，keybinding 判定（自由滚动 /
     * 横纵滚动 / 横纵缩放）、锚点换算、上下限全部复用画布那一套，标尺不定义第二种语义。
     */
    const handleRulerWheel = React.useCallback(
        (event: WheelEvent) => {
            kernelHostRef.current?.dispatchWheel(event);
        },
        [kernelHostRef],
    );

    React.useLayoutEffect(() => {
        const pending = pendingKernelZoomRef.current;
        if (pending === null) return;
        pendingKernelZoomRef.current = null;
        viewportAccess.setZoomAndScroll(pending.pxPerSec, pending.scrollLeft);
        // 【同任务把内核画出来】本提交已经把**标尺文本**按新缩放排好了（React 布局），
        // 而网格 / clip / 波形（GL）与标尺内容层的 translate 由内核绘制。若等下一次
        // rAF，被绘制出来的那一帧就是"新文本 + 旧网格"，缩放过程中标尺文本看起来在
        // 闪。这里同步提交一帧，文本与网格同帧切换（与参数编辑器的 `commitViewportNow`
        // 同一手法）。
        kernelHostRef.current?.paintNow();
    }, [pxPerSec, viewportAccess, kernelHostRef]);

    // ── 轨道头与时间轴区域的竖直滚动对齐 ─────────────────
    // 右侧时间轴 scroller 常驻水平滚动条（占高 h），其竖直滚动范围因此比
    // 轨道头少 h 像素：内容同为「轨道数 × rowHeight」时，轨道头滚到底会
    // 比时间轴多滚 h 像素，行无法对齐。这里实测 h（offsetHeight 减去
    // clientHeight），平台/样式自适应（overlay 滚动条占位为 0）。
    React.useLayoutEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;
        const measure = () => {
            setHorizontalScrollbarGutterPx(scroller.offsetHeight - scroller.clientHeight);
        };
        measure();
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(measure);
            observer.observe(scroller);
            return () => observer.disconnect();
        }
        window.addEventListener("resize", measure);
        return () => window.removeEventListener("resize", measure);
    }, [scrollRef, setHorizontalScrollbarGutterPx]);

    // ── 粘贴后“聚焦播放光标”（提交后执行）────────────────────
    // 粘贴可能大幅扩充工程全长（dynamicProjectSec / 水平可滚动范围随之
    // 扩大）。滚动必须在本状态与对应 DOM（paddedContentWidth）都提交后
    // 执行，否则会被旧的滚动上限钳制，导致光标无法进入画面。
    // pendingPlayheadRevealSec 由粘贴 fulfilled reducer 记录，这里在
    // useLayoutEffect 中消费并立即清除。
    const pendingPlayheadRevealSec = s.pendingPlayheadRevealSec;
    React.useLayoutEffect(() => {
        if (pendingPlayheadRevealSec == null) return;
        // 仅当新光标位置不在可视范围内时才滚动（需求语义：画面内不扰动视图）。
        // 视口来源必须模式无关：内核模式下旧滚动容器不挂载，若在这里早退会
        // **静默丢弃**聚焦请求（表现为「粘贴后视图不跟随」）。
        const x = Math.max(0, pendingPlayheadRevealSec) * pxPerSec;
        const left = viewportAccess.getScrollLeft();
        const right = left + viewportAccess.getViewportWidth();
        if (x >= left && x <= right) {
            dispatch(setPendingPlayheadReveal(null));
            return;
        }
        const next = computeFocusCursorScrollLeft({
            playheadSec: pendingPlayheadRevealSec,
            pxPerSec,
            contentWidth: dynamicProjectSec * pxPerSec,
        });
        if (Math.abs(viewportAccess.getScrollLeft() - next) > 0.5) {
            syncScrollLeft(viewportAccess.setScrollLeft(next));
        }
        dispatch(setPendingPlayheadReveal(null));
    }, [
        pendingPlayheadRevealSec,
        pxPerSec,
        dynamicProjectSec,
        viewportAccess,
        syncScrollLeft,
        dispatch,
    ]);

    return {
        viewportAccess,
        viewportAccessRef,
        resolveTrackIdAtClientY,
        handleKernelScrollLeftCommit,
        handleKernelZoomRequest,
        handleRulerWheel,
    };
}
