// 参数编辑器的视觉播放头 / 播放跟随驱动。
//
// 【为什么单独一个组件】面板模型的 store 选择器包含整个 session；若 `playheadSec`
// 也由模型订阅，33Hz 的轮询会让整块参数编辑器（数百个 hook 与子组件）每帧重渲染。
// 把这条高频通路收进一个只返回 null 的小组件后，面板只在真正的编辑状态变化时渲染，
// 而光标仍由共享视觉时钟逐帧驱动。
import React from "react";
import { shallowEqual } from "react-redux";
import { useAppSelector } from "../../../../app/hooks";
import { publishVisualPlayheadSec } from "../../../../features/capture/visualPlayheadBus";
import { isTransportAdvancing } from "../../../../features/session/selectors/transport";
import { useVisualPlayhead } from "../../../../hooks/useVisualPlayhead";
import { computeAutoFollowScrollLeft } from "../../../../utils/autoFollowScroll";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";

interface PianoRollPlayheadBridgeProps {
    visualPlayheadSecRef: React.MutableRefObject<number>;
    hostRef: React.MutableRefObject<PianoRollKernelHost | null>;
    scrollerRef: React.MutableRefObject<HTMLDivElement | null>;
    pxPerSecRef: React.MutableRefObject<number>;
    /** 当前内容宽度（px）。只在缩放 / 工程长度变化时更新。 */
    contentWidth: number;
    /** 无内核时的整帧标脏入口。 */
    invalidate: () => void;
}

/**
 * 逐帧写视觉播放头，并在未启用同步时负责参数编辑器自己的自动滚屏。
 *
 * 同步到时间轴时，滚动位置由时间轴那边统一发布；本组件只负责把共享视觉时钟
 * 推给参数编辑器内核，避免两个面板各自跑一条时钟造成光标相对滚动抽动。
 */
export const PianoRollPlayheadBridge = React.memo(function PianoRollPlayheadBridge({
    visualPlayheadSecRef,
    hostRef,
    scrollerRef,
    pxPerSecRef,
    contentWidth,
    invalidate,
}: PianoRollPlayheadBridgeProps) {
    const transport = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            playheadSampledAtMs: state.session.playheadSampledAtMs,
            isPlaying: state.session.runtime.isPlaying,
            positionAdvancing: state.session.runtime.positionAdvancing,
            positionAgeMs: state.session.runtime.positionAgeMs,
            transportSource: state.session.runtime.transportSource,
            playbackWaitingForRender: state.session.runtime.playbackWaitingForRender,
            autoScrollEnabled: state.session.autoScrollEnabled,
            paramEditorSyncTimeline: state.session.paramEditorSyncTimeline,
        }),
        shallowEqual,
    );
    const transportAdvancing = isTransportAdvancing(transport);

    useVisualPlayhead({
        visualPlayheadSecRef,
        syncedPlayheadSec: transport.playheadSec,
        syncedAtMs: transport.playheadSampledAtMs,
        isTransportAdvancing: transportAdvancing,
        onFrame: React.useCallback(
            (visualPlayheadSec: number) => {
                // `visualPlayheadSecRef` 已由 useVisualPlayhead 的两阶段通知在第一阶段写好；
                // 这里不能再依赖 onFrame 才写——时间轴自动滚屏会在本回调之前触发一次同步绘制。
                // 捕获带右缘读这一份：参数编辑器单独可见时时间轴桥可能没有挂载。
                publishVisualPlayheadSec(visualPlayheadSec);

                let followed = false;
                if (
                    !transport.paramEditorSyncTimeline &&
                    transport.autoScrollEnabled &&
                    transport.isPlaying
                ) {
                    const host = hostRef.current;
                    const scroller = scrollerRef.current;
                    const viewportWidth =
                        host?.getViewport().viewportWidth ?? scroller?.clientWidth ?? 0;
                    if (viewportWidth > 0) {
                        const currentScroll =
                            host?.getViewport().scrollLeft ?? scroller?.scrollLeft ?? 0;
                        const next = computeAutoFollowScrollLeft({
                            playheadSec: visualPlayheadSec,
                            pxPerSec: pxPerSecRef.current,
                            viewportWidth,
                            contentWidth,
                        });
                        if (Math.abs(currentScroll - next) > 0.5) {
                            if (host !== null) {
                                // 内核写入会经滚动订阅标脏；不在这里同步整帧重绘，
                                // 让本帧的绘制预算留给时间轴与浏览器合成。
                                host.setScrollLeft(next);
                            } else if (scroller !== null) {
                                scroller.scrollLeft = next;
                                invalidate();
                            }
                            followed = true;
                        }
                    }
                }

                if (!followed) {
                    const host = hostRef.current;
                    if (host !== null) host.invalidatePlayhead();
                    else invalidate();
                }
            },
            [
                contentWidth,
                hostRef,
                invalidate,
                pxPerSecRef,
                scrollerRef,
                transport.autoScrollEnabled,
                transport.isPlaying,
                transport.paramEditorSyncTimeline,
            ],
        ),
    });

    return null;
});
