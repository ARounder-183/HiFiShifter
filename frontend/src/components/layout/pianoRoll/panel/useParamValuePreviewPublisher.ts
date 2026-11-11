/**
 * 参数值浮窗的**唯一写入入口**（合帧后 setState）。
 *
 * 【为什么单独成 hook】它同时喂给 `usePianoRollInteractions`（参数线 / 颤动 / 四点
 * 编辑）与块手势（拖块 / 新建）。内联在 interactions 的入参对象里，块手势就只能另写
 * 一份 setState —— 那会绕过合帧，并且出现两个写入者（上一版拖块时读数被悬停读数
 * 按旧值盖回去，正是这么来的）。
 *
 * 【为什么要合帧】pointermove 在高刷设备上可达数百 Hz，逐事件 setState 会让整个面板
 * 以事件频率重渲。这里只记最新载荷、每渲染帧至多提交一次；浮窗是纯读数 UI，滞后
 * 一帧不可感知。
 */
import { useCallback, useEffect, useRef, type RefObject } from "react";

/** 浮窗载荷（不含容器 rect —— 那个在 rAF 内随读数同帧采集）。 */
export type ParamValuePreviewPayload = {
    readonly clientX: number;
    readonly clientY: number;
    readonly value: number;
    readonly displayText?: string;
};

/** 提交给面板 state 的完整读数（含容器 rect 快照）。 */
export type ParamValuePreviewEntry = ParamValuePreviewPayload & {
    readonly rectLeft: number;
    readonly rectTop: number;
};

export function useParamValuePreviewPublisher(args: {
    readonly canvasRef: RefObject<HTMLCanvasElement | null>;
    readonly setPreview: (next: ParamValuePreviewEntry | null) => void;
}): (next: ParamValuePreviewPayload | null) => void {
    const { canvasRef, setPreview } = args;
    const rafRef = useRef<number | null>(null);
    const pendingRef = useRef<{ payload: ParamValuePreviewPayload | null } | null>(null);
    useEffect(() => {
        return () => {
            if (rafRef.current != null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
            pendingRef.current = null;
        };
    }, []);
    return useCallback(
        (next: ParamValuePreviewPayload | null) => {
            pendingRef.current = { payload: next };
            if (rafRef.current != null) return;
            rafRef.current = requestAnimationFrame(() => {
                rafRef.current = null;
                const pending = pendingRef.current;
                pendingRef.current = null;
                if (!pending) return;
                /*
                 * 容器 rect 在这里（rAF 内、事件后）随读数同帧快照：EditorArea 渲染期
                 * 只消费数值，不再 getBoundingClientRect（渲染期强制同步布局，hover
                 * 时每帧都付一次）。
                 */
                const rect = canvasRef.current?.getBoundingClientRect();
                setPreview(
                    pending.payload && {
                        ...pending.payload,
                        rectLeft: rect?.left ?? 0,
                        rectTop: rect?.top ?? 0,
                    },
                );
            });
        },
        [canvasRef, setPreview],
    );
}
