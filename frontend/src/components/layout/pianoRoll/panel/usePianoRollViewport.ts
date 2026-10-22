/**
 * 参数编辑器的**视口**：滚动位置 / 缩放 / 与轨道时间轴的同步。
 *
 * 【为什么是独立的一个 hook】这一组状态原本散落在面板模型的 825-1372 行，被
 * 十来个彼此无关的功能（选区、绘制、剪贴板、颤音…）夹在中间，但它们其实是
 * 一台机器：`scrollLeft` / `pxPerSec` 两个 state、四个镜像 ref、一条共享视口
 * 订阅、一条"测量两面板水平偏移"的布局 effect、以及缩放的 rAF 原子提交。
 * 它们之间互相写 ref、互相读"内核真值"，任何一处改动都要同时照顾另外几处；
 * 拆出来之后，这条链路第一次可以整块读。
 *
 * 【所有权不变】状态仍然属于**面板**（面板重挂载即归零，与拆分前一致），
 * 这里只是把声明搬进了子 hook：调用点、hook 顺序、依赖数组全部保持原位，
 * 因此渲染行为与拆分前逐字节相同。
 *
 * 【对 lint 的已知影响】ref 从"本函数里的 `useRef` 结果"变成"另一个 hook 的返回值"
 * 之后，`react-hooks/exhaustive-deps` 无法再证明它们的引用稳定性，于是面板模型里
 * 用到这些 ref 的 effect 会各多一条 missing-dependency 警告。那是**误报**：ref 对象
 * 的身份与拆分前完全相同（同一次渲染内恒等），不必为了消警告去改依赖数组。
 *
 * 【坐标约定（读这个文件前必须先知道）】**绘制坐标**是参数编辑器画布里的 x
 * 对应的位置，**原生坐标**是轨道时间轴的滚动位置；两者相差一个 `timelineOffset`
 * （见下方测量 effect）。所有跨面板的写入都必须先换算回原生坐标，否则两个面板
 * 会永久错位。
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import type { RootState } from "../../../../app/store";
import { useAppSelector } from "../../../../app/hooks";
import { resolveScrollableProjectSec } from "../../../../features/session/projectBoundary";
import { subscribeCapture } from "../../../../features/capture/captureRange";
import { resolveSyncOffsetForms } from "../../../../features/dock/dockSchema";
import { getActiveSurface } from "../../../../features/uiFocus/focusSurface";
import { PANEL_PARAM_EDITOR, PANEL_TIMELINE } from "../../../dock/registerBuiltinPanels";
import { readUiValue, writeUiValue } from "../../../../services/uiStorage";
import {
    measureTabbedViewportOffsetPx,
    measureTimelineViewportOffsetPx,
    timelineViewportNativeToState,
    timelineViewportStateToNative,
    timelineViewportSync,
} from "../../../../utils/timelineViewportSync";
import { DEFAULT_PX_PER_SEC, MAX_PX_PER_SEC, MIN_PX_PER_SEC } from "../../timeline";
import {
    resolveHorizontalWheelZoom,
    resolveTimelineScrollRange,
} from "../../timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../../timeline/runtime/timelineZoomBounds";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import { PIANO_ROLL_SYNC_ORIGIN } from "./panelHelpers";

/** 本 hook 读取的会话切片（只声明真正用到的字段，避免与 store 类型整体耦合）。 */
type ViewportSession = Pick<
    RootState["session"],
    "bpm" | "clips" | "paramEditorSyncTimeline" | "playheadSec" | "projectSec"
>;

type PianoRollViewportInputs = {
    /** 会话切片（同上）。 */
    s: ViewportSession;
    /** 内核宿主的稳定引用；宿主尚未挂载时为 null。 */
    hostRef: { current: PianoRollKernelHost | null };
    /** 本窗体在停靠布局里的 id（判断是否与时间轴同屏）。 */
    dockFormId: string | undefined;
    /** 视觉插值播放头（缩放锚点必须用它，不能用 33Hz 轮询的滞后值）。 */
    getVisualPlayheadSec: () => number;
    /** 内核 + 各图层的一次原子提交（定义在面板模型里，只在事件期被调用）。 */
    commitViewportNow: (drawingScrollLeft: number, pxPerSecAtCommit?: number) => void;
};

/**
 * 视口状态 + 共享视口同步 + 水平缩放。
 *
 * 调用点必须固定在面板模型里原来的位置：hook 顺序、`useLayoutEffect` 的提交
 * 时机都是行为的一部分（例如"缩放先提交 state，再由 layout effect 落地"）。
 */
export function usePianoRollViewport({
    s,
    hostRef,
    dockFormId,
    getVisualPlayheadSec,
    commitViewportNow,
}: PianoRollViewportInputs) {
    // 视口专属的三只 ref：滚动容器、上一次已提交的绘制坐标、可视区尺寸。
    // 它们的消费者几乎都在本 hook 内（测量、缩放钳制、同步落地），因此随视口
    // 一起搬进来；面板仍按原名拿到它们（返回值里原样透出）。
    const scrollerRef = useRef<HTMLDivElement | null>(null);
    const lastScrollLeftRef = useRef<number | null>(null);
    const viewSizeRef = useRef({ w: 1, h: 1 });

    // 可滚域时长必须与时间轴内核**同源**：两处各取一个来源时，内容宽 / 可滚上限
    // 会分叉，共享 scrollLeft 的同步会被浏览器钳制、两边永久错位（缺陷 7）。
    // 详见 `resolveScrollableProjectSec` 的说明。
    const dynamicProjectSec = useMemo(
        () => resolveScrollableProjectSec(s.projectSec, s.clips),
        [s.projectSec, s.clips],
    );
    const [scrollLeft, setScrollLeft] = useState(0);
    const [pxPerSec, setPxPerSec] = useState(() => {
        const stored = Number(readUiValue("hifishifter.paramPxPerSec"));
        return Number.isFinite(stored) && stored > 0
            ? Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, stored))
            : DEFAULT_PX_PER_SEC;
    });
    // 渲染时根 ?BPM 换算 pxPerBeat：pxPerBeat = pxPerSec × (60 / bpm)
    const pxPerBeat = pxPerSec * (60 / Math.max(1e-6, s.bpm));
    const scrollLeftRef = useRef(scrollLeft);
    /** 标尺交互用的实时滚动位置；与内核绘制同源，stable 引用避免 TimeRuler 重渲。 */
    const getScrollLeftLive = useCallback(
        () => hostRef.current?.getViewport().scrollLeft ?? scrollLeftRef.current,
        [],
    );
    const pxPerBeatRef = useRef(pxPerBeat);
    const pxPerSecRef = useRef(pxPerSec);
    // 同步开关的 ref 镜像：rAF 原子提交与宿主帧回调读取最新值，避免陈旧闭包。
    const paramEditorSyncTimelineRef = useRef(s.paramEditorSyncTimeline);
    // 渲染期把 state 同步进 ref。**缩放**照旧无条件同步（它是原子提交的，state 就是
    // 真值）；**横向位置**只在宿主尚未挂载时才同步。
    //
    // 【为什么位置必须有条件（这是一个真实缺陷的根因）】横向位置的真值在**内核**，
    // 而 React state 只是 256px 步长的**量化提交**（见宿主 `SCROLL_COMMIT_STEP_PX`）：
    // 滚轮 / 拖 thumb / 触摸 / 自动滚屏只改内核与原生镜像，state 会滞后最多 255px
    // （最后一次不足一步的位移永不提交）。内核挂载后 `scrollLeftRef` 由 `onFrame`
    // 每帧对齐内核真值；若无条件用 state 回写，每一次 React 重渲都会把滞后值**灌回**
    // ref，于是"渲染按内核、交互按 refs"两套视口并存——框选出的选区因此与鼠标划过
    // 的区域相差同一距离（用户报告"实际产生的选区与光标划定的区域不一致"，且先做
    // 一次缩放后滚动才出现：缩放经 flushSync 原子对齐 state，滚动又让它重新滞后；
    // 再缩放一次又对齐，于是"缩放后恢复正常"）。
    //
    // 宿主未挂载时（挂载期）没有内核真值，ref 必须由 state 供给，此时回写是唯一
    // 正确行为。缩放 / pxPerBeat 与滚动无关（由 zomm 的原子提交直接写入 ref），
    // 保持无条件同步以免影响既有缩放路径。
    if (hostRef.current === null) {
        if (scrollLeftRef.current !== scrollLeft) scrollLeftRef.current = scrollLeft;
    }
    if (pxPerBeatRef.current !== pxPerBeat) pxPerBeatRef.current = pxPerBeat;
    if (pxPerSecRef.current !== pxPerSec) pxPerSecRef.current = pxPerSec;
    paramEditorSyncTimelineRef.current = s.paramEditorSyncTimeline;
    const timelineSyncApplyingRef = useRef(false);
    const timelineOffsetRef = useRef(0);
    const [timelineOffsetPx, setTimelineOffsetPx] = useState(0);

    /**
     * 同步偏移是否适用 —— 判据是**两个面板都可见**。
     *
     * 【为什么不是"必须上下堆叠"】偏移 = 轨道区左缘 − 参数编辑器绘制区左缘，把参数
     * 编辑器的内容按它平移后，同一时刻会落在**同一个屏幕 x** 上。只要两个面板同时
     * 可见，这个对齐就有意义（上下相邻是主场景，并排或一个浮在另一个之上同样成立：
     * 偏移可正可负，负值由 `minScrollLeft = -offset` 兜住）。
     *
     * 早期实现把第一个参数写成了 `dockFormId`（本面板自己的窗体 id），两个参数于是
     * 是同一个窗体 —— 判定必然为假、偏移被强制为 0，"同步时间轴视图"的像素对齐因此
     * **整个失效**。这里改为分别取时间轴与本面板的窗体 id，并只要求"都可见"。
     */
    const dockLayout = useAppSelector((state) => state.dock.layout);
    /**
     * 同步所需的两个窗体。`sameTabset` 决定偏移怎么算：
     * 同一标签组时用"内容内边距之差"（两个宿主同槽位），否则用屏幕左缘之差。
     */
    const syncPair = useMemo(
        () => resolveSyncOffsetForms(dockLayout, PANEL_TIMELINE, PANEL_PARAM_EDITOR, dockFormId),
        [dockFormId, dockLayout],
    );
    const syncOffsetApplicable = syncPair !== null;
    const syncOffsetTabbed = syncPair?.sameTabset === true;

    // 待落地的同步视口：**只记缩放**。位置在落地时直接取共享视口的当前值（权威且最新），
    // 不再捕获快照——见下方落地 effect 的说明（捕获值 + 比对 React state 的老做法会在
    // state 被同期写入点覆盖时静默取消落地，造成随机错位）。
    const pendingParamSyncViewportRef = useRef<{ pxPerSec: number } | null>(null);
    const horizontalZoomPendingRef = useRef<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>(null);
    const horizontalZoomChainRef = useRef<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>(null);
    // 水平缩放提交的 rAF 合并：一帧内多次滚轮/快捷键缩放只做一次原子提交。
    const zoomRafRef = useRef<number | null>(null);

    // 测量轨道时间线区与参数编辑器画布区之间的全局水平偏移，
    // 用于同步时把参数编辑器的绘制坐标与轨道视图按同一屏幕位置对齐。
    //
    // 【为什么必须容错 + 重试（这是一个真实缺陷的根因）】
    // 偏移是两个视口元素左缘之差，而**两个面板的挂载顺序不固定**：时间轴的内核容器
    // 可能晚于参数编辑器出现（面板按需挂载、WebGL 初始化、加载顺序）。元素缺失时
    // `measureTimelineViewportOffsetPx()` 返回 `null`（旧实现返回 0，与"真的对齐"不可
    // 区分）——若把 0 当成测量值，参数编辑器就会**丢掉整段同步位移**，错位量恰好等于
    // 偏移本身（= 轨道头宽度 − 键盘列宽度 ≈ 轨道头区域宽度）；更糟的是那一刻观察器也
    // 没绑上时间轴元素，之后再没有重测时机，只有恰好一次布局尺寸变化才会恢复——表现
    // 为**随机**错位（实测复现：偏移恒为 0，两面板相差整整 200px，并持续存在）。
    //
    // 因此这里：测不到就保持上一次的有效值，并在后续帧重试（时间轴元素可能刚出现）；
    // 每次重测都重新绑定观察器，元素后出现时也能补上。
    useLayoutEffect(() => {
        // 时间轴不可见（或本面板不可见）：偏移没有意义，固定为 0 并停止重试/观察。
        if (!syncOffsetApplicable) {
            timelineOffsetRef.current = 0;
            setTimelineOffsetPx((prev) => (prev === 0 ? prev : 0));
            return;
        }

        /** 视口元素缺失时的重试上限（帧）；超过后退回低频轮询，避免长期每帧空转。 */
        const RETRY_FRAMES = 120;
        /** 低频轮询间隔（ms）。 */
        const RETRY_INTERVAL_MS = 500;

        let observer: ResizeObserver | null = null;
        let frame: number | null = null;
        let retryTimer: ReturnType<typeof setInterval> | null = null;
        let stopped = false;

        const stopRetry = () => {
            if (frame !== null) {
                cancelAnimationFrame(frame);
                frame = null;
            }
            if (retryTimer !== null) {
                clearInterval(retryTimer);
                retryTimer = null;
            }
        };

        // 当前已绑定的观察目标（元素身份比较，见 observeViewports 的卫兵说明）。
        let observedPiano: Element | null = null;
        let observedTrack: Element | null = null;

        /** 把两边的视口元素（重新）挂到观察器上：元素可能后于本面板出现。 */
        const observeViewports = () => {
            if (typeof ResizeObserver === "undefined") return;
            const piano = scrollerRef.current;
            const track = document.querySelector<HTMLElement>("[data-timeline-scroller]");
            // 【目标集合未变化时绝不重绑】disconnect + observe 会对同一元素再次
            // 投递初始回调，回调路径（measureAndApply → applyMeasured）又会走到
            // 这里 —— 每帧一次投递的永久循环，正是 "ResizeObserver loop
            // completed with undelivered notifications" 以 60Hz 刷屏的来源。
            // 只在目标元素真的出现 / 被替换时才重新绑定。
            if (observer !== null && piano === observedPiano && track === observedTrack) {
                return;
            }
            observer = observer ?? new ResizeObserver(() => measureAndApply());
            observer.disconnect();
            observedPiano = piano;
            observedTrack = track;
            if (piano) observer.observe(piano);
            if (track) observer.observe(track);
        };

        /**
         * 量一次偏移。
         *
         * 同一标签组（插件默认视图）里两个面板永不共屏，绝对坐标不可用（非活动标签的
         * 宿主停在视口外），改用"内容内边距之差"；其余布局仍是屏幕左缘之差。
         */
        const measureOffset = () =>
            syncOffsetTabbed ? measureTabbedViewportOffsetPx() : measureTimelineViewportOffsetPx();

        /** 采用一次有效测量（并停止重试）。 */
        const applyMeasured = (value: number) => {
            stopRetry();
            timelineOffsetRef.current = value;
            setTimelineOffsetPx((prev) => (Math.abs(prev - value) < 0.5 ? prev : value));
            observeViewports();
        };

        const scheduleRetry = () => {
            if (stopped || frame !== null || retryTimer !== null) return;
            let attempts = 0;
            const tick = () => {
                frame = null;
                if (stopped) return;
                const next = measureOffset();
                if (next !== null) {
                    applyMeasured(next);
                    return;
                }
                observeViewports();
                attempts += 1;
                if (attempts >= RETRY_FRAMES) {
                    retryTimer = setInterval(measureAndApply, RETRY_INTERVAL_MS);
                } else {
                    frame = requestAnimationFrame(tick);
                }
            };
            frame = requestAnimationFrame(tick);
        };

        function measureAndApply(): void {
            if (stopped) return;
            const next = measureOffset();
            if (next !== null) {
                applyMeasured(next);
                return;
            }
            // 尚不可测（视口元素未挂载）：保持上一次的有效偏移并重试——**绝不能**当成 0。
            observeViewports();
            scheduleRetry();
        }

        measureAndApply();
        if (typeof ResizeObserver === "undefined") {
            window.addEventListener("resize", measureAndApply);
        }
        return () => {
            stopped = true;
            stopRetry();
            observer?.disconnect();
            window.removeEventListener("resize", measureAndApply);
        };
        // 【为什么要依赖布局】停靠重排会把面板的 DOM 宿主搬到别处，而**尺寸可能
        // 完全不变** —— 那样 ResizeObserver 不会触发，本 effect 若只挂载时跑一次，
        // 偏移就永远停在旧布局测出的值（实测：两面板相差数百像素，且只在恰好发生
        // 一次尺寸变化时才自愈）。因此布局一变就重测。
    }, [syncOffsetApplicable, syncOffsetTabbed, dockLayout]);

    useEffect(() => {
        // 捕获区间推进：只请内核补一帧**叠加层**（不重建曲线几何，也不触发 React
        // 渲染）。带子的几何由内核在帧提交里按最新视口写（见 `overlay.captureRange`）。
        return subscribeCapture(() => {
            hostRef.current?.invalidatePlayhead();
        });
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => {
            writeUiValue("hifishifter.paramPxPerSec", String(pxPerSec));
        }, 500);
        return () => clearTimeout(timer);
    }, [pxPerSec]);

    // 同步开关（双向交互）：订阅共享视口并应用到本面板。
    // 原生滚动位置 = 共享视口值（轨道坐标）；绘制坐标 = 原生 - 左右偏移。
    useLayoutEffect(() => {
        if (!s.paramEditorSyncTimeline) return;
        horizontalZoomPendingRef.current = null;
        horizontalZoomChainRef.current = null;
        const applyViewport = () => {
            // 未播种（时间轴尚未把当前位置写入共享视口）时拒绝应用：
            // 模块默认 {0,150} 只是占位，提前应用会把一帧错误缩放/位置
            // 画出来再被纠正（启动"一闪"）。时间轴在挂载/切换的 layout
            // effect（首帧绘制前）播种，播种后 emit 会驱动本订阅应用。
            if (!timelineViewportSync.isSeeded()) return;
            // 【跳过自己发布的广播】参数编辑器既发布又订阅，把刚发布的值再应用一遍
            // 没有意义，却会在缩放事务中间（pending 尚未落地）把位置/缩放拉回上一份
            // 快照——表现就是「水平缩放时波形/参数线抽搐一帧」。时间轴侧早就有同一条
            // 来源判定（`getOrigin() === TIMELINE_SYNC_ORIGIN`），这里补齐对称的一半。
            if (timelineViewportSync.getOrigin() === PIANO_ROLL_SYNC_ORIGIN) return;
            const store = timelineViewportSync.get();
            const offset = timelineOffsetRef.current;
            const drawingScrollLeft = timelineViewportNativeToState(store.scrollLeft, offset);
            // 纯滚动（pxPerSec 未变）：在同一个事件帧内同步落地——原生
            // scroller、标尺/网格层（applyScrollLayers）与轨道视图同帧提交，
            // 两个面板严丝合缝。state 仅作事后对齐（React 在绘制前提交）。
            //
            // 【判据必须取内核真值，不能取渲染期 ref】`pxPerSecRef` 在渲染期就被同步成
            // state 的新值，而并发渲染可能"渲染了但被丢弃"：用 ref 判定会把一次**缩放**
            // 广播误判成纯滚动（于是只应用位置、不应用缩放）；反过来也会把纯滚动误判成
            // 缩放而走延迟落地。内核才是这台面板的视口真值（与 `resolvePanelRenderViewport`
            // 同一约定）。
            const kernelPxPerSec = hostRef.current?.getViewport().pxPerSec ?? pxPerSecRef.current;
            const scroller = scrollerRef.current;
            if (scroller && Math.abs(store.pxPerSec - kernelPxPerSec) <= 1e-9) {
                timelineSyncApplyingRef.current = true;
                pxPerSecRef.current = store.pxPerSec;
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                // 同步落地走统一载体（宿主；它会换算原生坐标并镜像回写），
                // 并在**同一任务**里提交各图层（DOM / Canvas2D / GL）——否则 GL
                // 侧的曲线 / 播放头会比标尺、网格慢一帧到几帧（见 `paintNow`）。
                commitViewportNow(drawingScrollLeft);
                setScrollLeft(drawingScrollLeft);
                timelineSyncApplyingRef.current = false;
                return;
            }
            // 缩放（pxPerSec 变化）：内容宽度必须先按新 pxPerSec 重排，维持
            // “先提交 state，再由 layout effect 落地”的既有路径。
            timelineSyncApplyingRef.current = true;
            pendingParamSyncViewportRef.current = { pxPerSec: store.pxPerSec };
            setScrollLeft(drawingScrollLeft);
            setPxPerSec(store.pxPerSec);
            timelineSyncApplyingRef.current = false;
        };
        const unsubscribe = timelineViewportSync.subscribe(applyViewport);
        // 启用瞬间以轨道视图当前值为基准：原生位置对齐共享视口。
        applyViewport();
        const scroller = scrollerRef.current;
        return () => {
            unsubscribe();
            pendingParamSyncViewportRef.current = null;
            horizontalZoomPendingRef.current = null;
            horizontalZoomChainRef.current = null;
            // 禁用时移除偏移补偿：把位置还原为绘制坐标。
            if (scroller) {
                const next = Math.max(0, scrollLeftRef.current);
                scrollLeftRef.current = next;
                lastScrollLeftRef.current = next;
                commitViewportNow(next);
                setScrollLeft(next);
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- commitViewportNow 随渲染重建，加入会让共享视口订阅随每次渲染反复退订/重订（既有热路径口径）
    }, [s.paramEditorSyncTimeline]);

    // 布局偏移变化时，同一共享视口对应的绘制坐标也会变化。
    // 重新按当前偏移计算状态，让同步对齐始终使用最新几何位置。
    useEffect(() => {
        if (!s.paramEditorSyncTimeline) return;
        const store = timelineViewportSync.get();
        const drawingScrollLeft = timelineViewportNativeToState(
            store.scrollLeft,
            timelineOffsetRef.current,
        );
        timelineSyncApplyingRef.current = true;
        pendingParamSyncViewportRef.current = { pxPerSec: store.pxPerSec };
        setScrollLeft(drawingScrollLeft);
        timelineSyncApplyingRef.current = false;
    }, [timelineOffsetPx, s.paramEditorSyncTimeline]);

    // 同步视口必须等内容宽度按新 pxPerSec 更新后再落到 DOM。
    // 否则设置 scroller.scrollLeft 时会被浏览器钳回旧的最大滚动位置，
    // 形成“缩放已变、滚动没变”的水平漂移。
    //
    // 【为什么位置取"共享视口的当前值"而不是捕获时的快照，且不再比对 React state】
    // 这条落地路径原先要求 `React scrollLeft state` 恰好等于 pending 的目标值，否则
    // **直接 return 把 pending 搁置**。而 state 是异步提交的：同期可能被其它写入点
    // （量化提交的 rAF、纯滚动广播、以及参数编辑器自己发布的回声）覆盖成更旧的值，
    // 于是这次落地被静默取消——面板停在旧位置、缩放却已生效，两个面板从此错开一段
    // 距离（用户报告：「参数编辑器的偏移量有误」，且**随机**出现）。位置本身在共享
    // 视口里就是权威且最新的，直接取它即可；只要该视口的**缩放**与本次 pending 一致
    // （说明这是"同一个缩放代"的落地），落地就是对的。缩放若已被更新的广播取代，
    // 那份广播会写下新的 pending，由它落地。
    useLayoutEffect(() => {
        const pending = pendingParamSyncViewportRef.current;
        if (!pending || !s.paramEditorSyncTimeline) return;
        if (Math.abs(pxPerSec - pending.pxPerSec) > 1e-9) return;
        if (Math.abs(timelineOffsetPx - timelineOffsetRef.current) > 0.5) return;

        const store = timelineViewportSync.get();
        // 更新的缩放已在路上：等它自己的 pending（否则会用旧缩放的位置落地）。
        if (Math.abs(store.pxPerSec - pending.pxPerSec) > 1e-9) return;

        pendingParamSyncViewportRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const offset = timelineOffsetRef.current;
        const drawingScrollLeft = timelineViewportNativeToState(store.scrollLeft, offset);
        timelineSyncApplyingRef.current = true;
        pxPerSecRef.current = pending.pxPerSec;
        pxPerBeatRef.current = pending.pxPerSec * (60 / Math.max(1e-6, s.bpm));
        scrollLeftRef.current = drawingScrollLeft;
        // `store.scrollLeft` 是共享视口的原生值；换算成绘制坐标后走统一载体
        // （宿主会再加回偏移，落到与共享视口相同的位置），并在**同一任务**里把
        // DOM / Canvas2D / GL 一起提交（见 `paintNow`）。
        //
        // 【这里不能再调 syncScrollLeft】它是「读原生 scroller → 采纳为真值」的
        // 路径，而此时原生镜像还停留在**上一帧的旧值**（本函数的写入要等内核下一帧才
        // 回写），于是会把刚提交的目标位置又覆盖回旧值——表现为「同步从 1200 拨回 0
        // 时参数编辑器不动」。
        // 缩放与位置一起落进内核再绘制（见 `commitViewportNow`）：否则同帧绘制会用
        // 内核的旧缩放画新位置，而 DOM 侧标尺/网格已按新缩放排好——一帧内两套缩放。
        commitViewportNow(drawingScrollLeft, pending.pxPerSec);
        timelineSyncApplyingRef.current = false;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pxPerSec, scrollLeft, s.paramEditorSyncTimeline, timelineOffsetPx]);

    useLayoutEffect(() => {
        const pending = horizontalZoomPendingRef.current;
        if (!pending) return;
        // 【先消费再判定】请求一旦"过期"就必须丢弃，不能留在 ref 里等未来某次渲染
        // 的 state 凑巧等于它的缩放——那会用一次**旧缩放的位置**落地（旧快照的
        // `nextScrollLeft`），并在同步模式下把它推回共享视口：两个面板一起跳到旧位置。
        // 因此这里无条件清空，只有"当前 state 正是这次请求的缩放"才真正落地。
        horizontalZoomPendingRef.current = null;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        horizontalZoomChainRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const syncEnabled = s.paramEditorSyncTimeline;
        const offset = syncEnabled ? timelineOffsetRef.current : 0;
        const native = pending.nextScrollLeft;
        const next = timelineViewportNativeToState(native, offset);
        // 缩放落地：缩放 + 位置一次原子写入内核，再同任务提交各图层（见 `commitViewportNow`）。
        commitViewportNow(next, pending.nextScale);
        if (lastScrollLeftRef.current !== next) {
            lastScrollLeftRef.current = next;
            scrollLeftRef.current = next;
        }
        // 同步模式下手动缩放后必须把新的 pxPerSec 写回共享视口；即使滚动位置
        // 没有变化（例如光标位于左侧同步空白区时锚定在工程起点，next 仍为 -offset），
        // 也要广播缩放，否则轨道视图不会跟着缩放。
        //
        // 【广播的是"落地值"而不是"请求值"】`applyHorizontalScrollPosition` 已经
        // 把目标交给内核（内核按当前边界钳制），所以这里回读一次：共享视口里只能是
        // 两个面板都到得了的位置。广播请求值（可能越界）会让轨道视图先按自己的上限
        // 钳回来，两个面板在工程结尾处错开一个差量。
        if (syncEnabled && !timelineSyncApplyingRef.current) {
            const appliedDrawing = hostRef.current?.getViewport().scrollLeft ?? next;
            timelineViewportSync.setViewport(
                {
                    scrollLeft: timelineViewportStateToNative(appliedDrawing, offset),
                    pxPerSec,
                },
                PIANO_ROLL_SYNC_ORIGIN,
            );
        }
        // 原生滚动位置的钳制校正由宿主的镜像回写负责（它每帧都会把原生位置对齐真值）
        // ——不要再写原生 scroller，否则会与内核真值打架。
        setScrollLeft(next);
        // eslint-disable-next-line react-hooks/exhaustive-deps -- commitViewportNow 随渲染重建，加入会让缩放落地 effect 随每次渲染重跑（既有热路径口径）
    }, [pxPerSec, s.paramEditorSyncTimeline]);

    const zoomTimelineStateRef = useRef({
        playheadSec: s.playheadSec,
        projectSec: dynamicProjectSec,
    });
    useLayoutEffect(() => {
        zoomTimelineStateRef.current = {
            playheadSec: s.playheadSec,
            projectSec: dynamicProjectSec,
        };
    });

    // 指定缩放下的内容宽（口径与轨道视图一致：工程宽向上取整）。缩放提交时要用
    // **目标**缩放的内容宽来钳位置——用当前缩放算出的上限会在放大时把位置压低。
    // 定义在缩放入口之前：`useCallback` 的依赖数组在 render 期求值，写在后面会命中 TDZ。
    const contentWidthAt = useCallback(
        (scale: number) => Math.max(1, Math.ceil(dynamicProjectSec * scale)),
        [dynamicProjectSec],
    );

    const queueHorizontalZoom = useCallback(
        (nextPxPerSec: number, nextNativeScrollLeft: number) => {
            // 事件内只记录缩放意图，**绝不**在提交前改写 refs/state：任何
            // “ref 先行”都会让夹缝中的 rAF 绘制读到“新缩放 + 旧滚动”的
            // 混合投影——参数线/原始音高线/参考线等所有线条整体抽搐一帧。
            horizontalZoomPendingRef.current = {
                nextScale: nextPxPerSec,
                nextScrollLeft: nextNativeScrollLeft,
            };
            // 「一帧一次原子提交」：rAF 合并同帧内的连续缩放事件；
            // flushSync 把两个 state 在同一批提交中落地，
            // DOM 内容宽度按新缩放重排后，layout effect 在同一提交内写原生
            // scrollLeft 并同帧重绘标尺/网格/画布/波形（applyScrollLayers）。
            // refs 只随 render 写入、与 state 同帧移动——绘制前所有图层拿到
            // 同一对投影值，不存在任何混合帧窗口。
            if (zoomRafRef.current == null) {
                zoomRafRef.current = requestAnimationFrame(() => {
                    zoomRafRef.current = null;
                    const pending = horizontalZoomPendingRef.current;
                    if (!pending) return;
                    const offset = paramEditorSyncTimelineRef.current
                        ? timelineOffsetRef.current
                        : 0;
                    const drawingScrollLeft = timelineViewportNativeToState(
                        pending.nextScrollLeft,
                        offset,
                    );
                    flushSync(() => {
                        setPxPerSec(pending.nextScale);
                        setScrollLeft(drawingScrollLeft);
                    });
                });
            }
        },
        [setPxPerSec, setScrollLeft],
    );

    const handleHorizontalZoom = useCallback(
        (nextPxPerSec: number, nextScrollLeft: number) => {
            // 计算结果为绘制坐标；同步时需换算回原生（轨道）坐标再交给 layout effect。
            const nativeNextScrollLeft = timelineViewportStateToNative(
                nextScrollLeft,
                s.paramEditorSyncTimeline ? timelineOffsetRef.current : 0,
            );
            // 【为什么必须在这里钳一次】共享位置（原生坐标）必须落在两个面板**公共**
            // 的可滚范围内（= `resolveTimelineScrollRange` 的上限，按**目标**缩放的
            // 工程宽算，与轨道视图同一份口径）。不钳的话，缩放锚点可以把位置算到范围
            // 之外：参数编辑器会广播一个自己都到不了的值，轨道视图收到后按自己的上限
            // 钳回来 —— 两个面板在工程结尾处错开，且位置互相追赶（抖动）。
            const range = resolveTimelineScrollRange({
                contentWidth: contentWidthAt(nextPxPerSec),
                viewportWidth: viewSizeRef.current.w,
            });
            const clampedNativeScrollLeft = Math.min(
                range.maxScrollLeft,
                Math.max(range.minScrollLeft, nativeNextScrollLeft),
            );
            queueHorizontalZoom(nextPxPerSec, clampedNativeScrollLeft);
        },
        [s.paramEditorSyncTimeline, contentWidthAt, queueHorizontalZoom],
    );

    // 工具栏/快捷键聚焦缩放（hifi:zoomTimelineFocus）。放在 handleHorizontalZoom
    // 定义之后，避免 render 期求值 deps 时命中 const 的 TDZ。
    useEffect(() => {
        function onZoomFocused(e: Event) {
            const { projectSec } = zoomTimelineStateRef.current;
            const inPianoRoll = getActiveSurface() === "pianoRoll";
            if (!inPianoRoll) return;

            const factor = Number((e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1);
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollerRef.current;
            if (!scroller) return;

            const syncEnabled = s.paramEditorSyncTimeline;
            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pxPerSecRef.current,
                // 与滚轮路径同一契约：base 一律用绘制坐标（scrollLeftRef），
                // 并显式传入同步模式的负向最小滚动与锚点偏移；结果同样交给
                // handleHorizontalZoom 换算回原生坐标——两个缩放入口的坐标
                // 空间保持一致，避免同步模式下基准坐标空间错位。
                baseScrollLeft: scrollLeftRef.current,
                totalSec: projectSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: true,
                playheadSec: getVisualPlayheadSec(),
                anchorScreenX: 0,
                minPxPerSec: resolveTimelineMinPxPerSec({
                    baseMinPxPerSec: MIN_PX_PER_SEC,
                    projectSec,
                    viewportWidthPx: scroller.clientWidth,
                }),
                maxPxPerSec: MAX_PX_PER_SEC,
                minScrollLeft: syncEnabled ? -timelineOffsetRef.current : 0,
                anchorOffsetPx: syncEnabled ? timelineOffsetRef.current : 0,
            });
            if (!zoom) return;

            handleHorizontalZoom(zoom.nextPxPerSec, zoom.nextScrollLeft);
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
    }, [getVisualPlayheadSec, s.paramEditorSyncTimeline, handleHorizontalZoom]);

    return {
        // 状态与镜像
        scrollLeft,
        setScrollLeft,
        pxPerSec,
        pxPerBeat,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        zoomRafRef,
        paramEditorSyncTimelineRef,
        timelineOffsetRef,
        timelineOffsetPx,
        timelineSyncApplyingRef,
        horizontalZoomChainRef,
        lastScrollLeftRef,
        scrollerRef,
        viewSizeRef,
        // 读取器与缩放入口
        getScrollLeftLive,
        contentWidthAt,
        handleHorizontalZoom,
    };
}
