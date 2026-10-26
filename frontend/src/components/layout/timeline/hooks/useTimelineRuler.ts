// 时间轴标尺：时间格式上下文、主/副单位、Tempo Map、播放头复制与拖拽 seek。
import { type TimeValueFormatContext } from "../timeValueText";
import { planSeekGesture } from "../kernel/interaction/seekGesture";
import {
    persistUiSettings,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setTempoMap,
} from "../../../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../../../features/session/thunks/tempoMapThunks";
import { registerDragAbort } from "../../../../utils/gestureFocusGuard";
import { formatCursorTime } from "../../timeline";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "../../timeline";
import { SNAP_HIGHLIGHT_GROUP, clearSnapHighlights } from "../../../../utils/snapHighlight";
import type { TempoMap } from "../../../../utils/tempoMap";
import type { ScaleLike } from "../../../../utils/musicalScales";
import { useTimelineState } from "../hooks/useTimelineState";
import { store } from "../../../../app/store";
import { isLegacyMouseEventFromStylus } from "../../../../utils/penInput";
import { timelineViewportBus } from "../../../../utils/timelineViewportBus";
import type { TimelineKernelHost } from "../kernel/host/timelineKernelHost";
import type { AppDispatch } from "../../../../app/store";
import React from "react";
import type { TimelineSessionSlice } from "./useTimelineState";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    s: TimelineSessionSlice;
    dispatch: AppDispatch;
    kernelHostRef: React.MutableRefObject<TimelineKernelHost | null>;
    setTimeDisplaySettingsOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

export function useTimelineRuler({
    state,
    s,
    dispatch,
    kernelHostRef,
    setTimeDisplaySettingsOpen,
}: Deps) {
    const { setPlayheadFromClientX } = state;
    const timeContext = React.useMemo<TimeFormatContext>(
        () => ({
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.bpm, s.beats, s.grid, s.tempoMap],
    );

    // 时间量 ToolTips 的格式化上下文：主/副时间单位 + 工程计时参数 + Tempo Map。
    //
    // 一个对象同时服务两种口径（见 `timeValueText` 文件头）：
    // - **时长**（淡化长度、吸附偏移）走 `formatDurationText`，按定义忽略 `tempoMap`；
    // - **时刻**（吸附偏移的绝对位置）走 `formatPositionText`，必须感知 Tempo Map。
    const timeValueFormatCtx = React.useMemo<TimeValueFormatContext>(
        () => ({
            primaryTimeUnit: s.primaryTimeUnit,
            secondaryTimeUnit: s.secondaryTimeUnit,
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.primaryTimeUnit, s.secondaryTimeUnit, s.bpm, s.beats, s.grid, s.tempoMap],
    );

    const projectScale = React.useMemo<ScaleLike | null>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project.baseScale, s.project.customScale, s.project.useCustomScale],
    );

    const handleTempoMapChange = React.useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
        },
        [dispatch],
    );
    const handleTempoMapCommit = React.useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
            void dispatch(setTempoMapRemote(next));
        },
        [dispatch],
    );
    const handlePrimaryUnitChange = React.useCallback(
        (unit: TimeUnit) => {
            dispatch(setPrimaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleSecondaryUnitChange = React.useCallback(
        (unit: TimeUnitChoice) => {
            dispatch(setSecondaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleCopyPlayheadTime = React.useCallback(async () => {
        const text = formatCursorTime(
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            Number(s.playheadSec ?? 0) || 0,
            timeContext,
        ).combined;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            try {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            } catch {
                // 忽略复制失败
            }
        }
    }, [s.primaryTimeUnit, s.secondaryTimeUnit, s.playheadSec, timeContext]);

    // TimeRuler 是 React.memo 组件：传给它的 handler 必须是稳定引用，内联箭头
    // 函数会让 memo 每次渲染都失效（拖拽 / 滚动期间标尺子树整棵重渲）。
    const handleRulerOpenSettings = React.useCallback(() => {
        setTimeDisplaySettingsOpen(true);
    }, [setTimeDisplaySettingsOpen]);
    const handleRulerCopyPlayheadTime = React.useCallback(() => {
        void handleCopyPlayheadTime();
    }, [handleCopyPlayheadTime]);
    const handleRulerMouseDown = React.useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            // 数位笔 / 触摸不触发标尺 seek：pen 的兼容 mouse 事件（无
            // pointerType）会让悬停画线起笔误定位播放头。
            if (isLegacyMouseEventFromStylus()) return;
            if (e.button !== 0) return;
            // 水平滚动位置取自内核视口。取值的「实时性」是硬要求——拖拽期间滚动
            // 位置可能被自动滚动改变，因此每次换算都重新读，不缓存。
            // 读滚动真值：内核宿主是唯一来源（旧的原生 scroller 分支已删除：
            // `scrollRef` 无 JSX 挂载点，恒为 null）。
            const readScrollLeft = (): number | null => {
                const host = kernelHostRef.current;
                return host != null ? host.getViewport().scrollLeft : null;
            };
            if (readScrollLeft() === null) return;
            const ruler = e.currentTarget as HTMLDivElement;

            const updateAt = (clientX: number, commit: boolean): number =>
                setPlayheadFromClientX(
                    clientX,
                    ruler.getBoundingClientRect(),
                    readScrollLeft() ?? 0,
                    commit,
                );

            // 播放态实时判定（store，而非滞后的镜像）：与轨道空白区共用同一
            // 套阶段策略（`planSeekGesture`）—— 播放中按下与拖拽都不写播放头，
            // 只有松手才提交落点并延续播放。标尺若不共用，从标尺开始拖拽仍会
            // 出现完全相同的闪回（按下把光标拉到指针处、下一帧被播放轮询拽回）。
            const isPlayingNow = (): boolean => Boolean(store.getState().session.runtime.isPlaying);

            // 标尺没有其他编辑操作需要区分：未播放时按下立即提交一次 seek
            // （既有行为）；播放中按下不写播放头，落点延后到松手那一拍。
            const pressCommitted = planSeekGesture("press", isPlayingNow()).playhead === "commit";
            if (pressCommitted) {
                updateAt(e.clientX, true);
            }
            let moved = false;
            let lastClientX = e.clientX;

            const onMove = (ev: MouseEvent) => {
                moved = true;
                lastClientX = ev.clientX;
                // 播放中的拖拽帧不写播放头（闪回防护）。
                if (planSeekGesture("move", isPlayingNow()).playhead === "preview") {
                    updateAt(ev.clientX, false);
                }
            };

            // 拖拽期间**视口变化**（滚轮滚动/缩放、键盘滚动、跨面板同步）后按最后
            // 指针位置重放一次落点。
            //
            // 【为什么必须重放】落点 = 指针客户端 X 换算出的时间，而换算依赖当前
            // 视口（`readScrollLeft` 是实时的）。滚轮**不产生任何 mousemove**，视口
            // 变了而指针没动时若不重放，播放头就停在旧落点上、与光标脱开，直到用户
            // 再动一下鼠标才归位（用户报告的"拖拽中滚动/缩放导致偏移"）。订阅总线
            // 而非 React 属性：总线在每次视口提交时**同步**广播，重放与绘制同帧。
            const unsubscribeViewport = timelineViewportBus.subscribe(() => {
                if (planSeekGesture("move", isPlayingNow()).playhead === "preview") {
                    updateAt(lastClientX, false);
                }
            });

            // 失焦取消：切屏期间 mouseup 不送达本窗口，blur 时以
            // 最后一次已知位置收尾（提交 seek + 清吸附高亮），
            // 防止监听器泄漏：否则下次点击会被旧的 onEnd 消费。
            const finish = () => {
                unregisterAbort();
                unsubscribeViewport();
                window.removeEventListener("mousemove", onMove, true);
                window.removeEventListener("mouseup", onEnd, true);
                window.removeEventListener("mouseleave", onEnd, true);
                // 拖拽松手提交最终落点：播放中这是**唯一**写播放头的一拍
                // （后端 seek 不改 is_playing —— 即"延续播放状态"）。
                // 未拖动的单击在未播放时已由按下那一拍提交，此处不重复提交
                // （重复提交是多余的一次后端 seek）。
                const releaseCommits =
                    planSeekGesture("release", isPlayingNow()).playhead === "commit";
                if (releaseCommits && (moved || !pressCommitted)) {
                    updateAt(lastClientX, true);
                }
                // 拖拽中间帧可能发布过吸附高亮，必须在此清除：否则拖拽标尺后
                // 网格吸附的竖线会冻结在画面上，且任何单击跳转都不再清理它。
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            };
            const onEnd = (ev: MouseEvent) => {
                lastClientX = ev.clientX;
                finish();
            };
            const unregisterAbort = registerDragAbort(finish);

            window.addEventListener("mousemove", onMove, true);
            window.addEventListener("mouseup", onEnd, true);
            window.addEventListener("mouseleave", onEnd, true);
        },
        [kernelHostRef, setPlayheadFromClientX],
    );

    // ── 记录最近点击的 clientX，用于 Shift 范围选择的锚点位置

    return {
        timeContext,
        timeValueFormatCtx,
        projectScale,
        handleTempoMapChange,
        handleTempoMapCommit,
        handlePrimaryUnitChange,
        handleSecondaryUnitChange,
        handleCopyPlayheadTime,
        handleRulerOpenSettings,
        handleRulerCopyPlayheadTime,
        handleRulerMouseDown,
    };
}
