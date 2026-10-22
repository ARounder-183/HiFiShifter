/**
 * 标尺按下（定位播放头 / 拖动）。
 *
 * 【为什么单独成文件】它原本是 `usePianoRollInteractions` 里的一个 `useCallback`
 * （71 行）。这些回调只做一件事，却和另外几十个回调挤在同一个 hook 里。
 *
 * 【依赖怎么拿】上下文由调用方打包传入（`Pick<PointerDeps, …>`，字段清单由 AST
 * 扫描得出），函数体原样搬过来，开头解构回局部变量。
 */
import { seekPlayhead, setplayheadSec } from "../../../../features/session/sessionSlice";
import { isLegacyMouseEventFromStylus } from "../../../../utils/penInput";
import { clamp } from "../../timeline";
import { secFromViewportClientX } from "../seekPlayheadMapping";
import type { MouseEvent as ReactMouseEvent } from "react";

import type { PointerDeps } from "./pointerDeps";

/** 本回调需要的上下文。 */
type Deps = Pick<PointerDeps, "axisFromRefs" | "dispatch">;

/** 标尺按下（定位播放头 / 拖动）。 */
export function handleRulerPointerDown(deps: Deps, e: ReactMouseEvent<HTMLDivElement>): void {
    const { axisFromRefs, dispatch } = deps;

    // 数位笔 / 触摸不触发标尺 seek：pen 的兼容 mouse 事件（无
    // pointerType）会让"悬停画线起笔"误定位播放头；悬停时间气泡
    // （只读）不受影响。按"最近一次真实 pointer 事件"的设备类型判定。
    if (isLegacyMouseEventFromStylus()) return;
    if (e.button !== 0) return;
    const ruler = e.currentTarget as HTMLDivElement;
    let moved = false;

    const updateAt = (clientX: number, commit: boolean): number => {
        const bounds = ruler.getBoundingClientRect();
        const sec = clamp(
            secFromViewportClientX({
                clientX,
                viewportLeft: bounds.left,
                // 同上：seek 的投影也必须取内核真值，否则滚动后点标尺会落到
                // 与画面播放头不同的位置（量化残差）。
                axis: axisFromRefs(),
            }),
            0,
            1e12,
        );

        if (commit) {
            dispatch(setplayheadSec(sec));
            void dispatch(seekPlayhead(sec));
        }
        return sec;
    };

    // 标尺没有其他编辑操作需要区分，按下时立即提交一次 seek。
    let lastSec = updateAt(e.clientX, true);

    // 拖动中的 setplayheadSec 逐事件派发会让所有订阅 session 的组件
    // 随 mousemove 频率重渲；rAF 合帧后与画布重绘节奏一致。
    let rafId: number | null = null;
    let queuedSec: number | null = null;

    const onMove = (ev: MouseEvent) => {
        moved = true;
        queuedSec = updateAt(ev.clientX, false);
        if (rafId == null) {
            rafId = requestAnimationFrame(() => {
                rafId = null;
                if (queuedSec == null) return;
                dispatch(setplayheadSec(queuedSec));
                queuedSec = null;
            });
        }
    };

    const onEnd = (ev: MouseEvent) => {
        window.removeEventListener("mousemove", onMove, true);
        window.removeEventListener("mouseup", onEnd, true);
        window.removeEventListener("mouseleave", onEnd, true);
        if (rafId != null) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }
        if (!moved) return;
        lastSec = updateAt(ev.clientX, true);
        void dispatch(seekPlayhead(lastSec));
    };

    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onEnd, true);
    window.addEventListener("mouseleave", onEnd, true);
}
