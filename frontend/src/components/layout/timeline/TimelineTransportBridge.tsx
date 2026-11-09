// 播放头桥接：逐帧写视觉播放头并请求内核重绘。
import React from "react";
import { useAppSelector } from "../../../app/hooks";
import { shallowEqual } from "react-redux";
import { type TimelineViewportAccess } from "./hooks/timelineViewportAccess";
import { useVisualPlayhead } from "../../../hooks/useVisualPlayhead";
import { isTransportAdvancing } from "../../../features/session/selectors/transport";
import { publishVisualPlayheadSec } from "../../../features/capture/visualPlayheadBus";
import { computeAutoFollowScrollLeft } from "../../../utils/autoFollowScroll";

export const TimelineTransportBridge = React.memo(function TimelineTransportBridge(props: {
    pxPerSecRef: React.MutableRefObject<number>;
    /**
     * 模式无关的视口访问器。
     *
     * 自动滚屏与播放头定位都需要「视口宽度 + 当前横向滚动量」，内核模式下这些
     * 量由 `ScrollKernel` 持有（DOM 容器的 scrollLeft 恒为 0），因此不能再直接读
     * 原生 scroller。
     */
    viewport: TimelineViewportAccess;
    /** 接收每帧视觉插值播放头（秒），供缩放锚点等命令式读取（与绘制同源）。 */
    visualPlayheadRef: React.MutableRefObject<number>;
    syncScrollLeft: (next: number) => void;
    autoScrollEnabled: boolean;
    projectSec: number;
    /** 请求时间轴内核重绘播放头（见 kernelHost.invalidatePlayhead 的说明）。 */
    requestPlayheadRepaint: () => void;
}) {
    const {
        pxPerSecRef,
        viewport,
        visualPlayheadRef,
        syncScrollLeft,
        autoScrollEnabled,
        projectSec,
        requestPlayheadRepaint,
    } = props;
    const transport = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            playheadSampledAtMs: state.session.playheadSampledAtMs,
            isPlaying: state.session.runtime.isPlaying,
            // 【为什么要带上它】`isPlaying` 是镜像，起播第一条采样到达之前是假的；
            // 而"位置真的在前进"可以由两条采样直接算出来（见 applyTransportSample）。
            positionAdvancing: state.session.runtime.positionAdvancing,
            // 读数的年龄："宿主说在播"必须与"这个说法有多旧"一起读（见 selector）。
            positionAgeMs: state.session.runtime.positionAgeMs,
            playbackWaitingForRender: state.session.runtime.playbackWaitingForRender,
        }),
        // 无 shallowEqual 时每次 dispatch 都产生新对象引用，
        // 该桥接组件会在任意 store 更新（含 33Hz 播放轮询）时重渲染。
        shallowEqual,
    );

    /*
     * 【光标必须满足的唯一不变量】只要宿主说在播放，光标就按 1x 前进；宿主的位置
     * 读数只用来**纠偏**，不用来决定"要不要前进"。
     *
     * 旧判据有三个开关：`isPlaying`、`!playbackWaitingForRender`、
     * `playbackPositionSec > 1e-4`。而这三个都可能在**起播那一瞬**短暂为假：
     *   * 第一次采样带回位置时 `is_playing` 仍是 false（或在 0 秒）；
     *   * 预渲染守卫把 `playbackWaitingForRender` 置真几百毫秒。
     * 于是光标"跳到位 → 冻住 → 再一格一格地走"—— 正是用户报的那条。
     *
     * 【为什么去掉"等待渲染"这一条也不违反原来的约束】不变量说"宿主在播就前进"，
     * 而"我听到的是旧音频"是另一件事（状态栏的 `stale`）。把光标停住并不能让音频
     * 变新，只会让用户以为播放停了 —— 而音频其实在响。
     */
    // 【唯一判据在 selector 里】见 `isTransportAdvancing`：三处三个公式，恰好
    // 在"位置冻结"这个场景互相分歧 —— 那正是"停止后光标还在走"。
    const transportAdvancing = isTransportAdvancing(transport);

    useVisualPlayhead({
        visualPlayheadSecRef: visualPlayheadRef,
        syncedPlayheadSec: transport.playheadSec,
        syncedAtMs: transport.playheadSampledAtMs,
        isTransportAdvancing: transportAdvancing,
        onFrame: React.useCallback(
            (visualPlayheadSec: number) => {
                // 缩放锚点（以播放头为锚时的锚点解析）读取的必须是与绘制同源的插值
                // 播放头。`visualPlayheadRef` 已由 useVisualPlayhead 在第一阶段写好；
                // 不能等到 onFrame 才写——自动滚屏的同步绘制发生在本回调之前。
                // 捕获带的右缘读这一份：它必须与光标同源，否则两者永远差一个轮询周期。
                publishVisualPlayheadSec(visualPlayheadSec);

                // 自动滚动先行：syncScrollLeft 内部会用 Redux 同步播放头（滞后于
                // 视觉插值）重写播放头位置 —— 若先定位播放头再滚动，播放头每帧
                // 会在"视觉位置"与"同步位置"之间跳动（自动滚屏抽搐的根因）。
                // 滚动先行、播放头定位收尾，最终写入获胜。
                if (autoScrollEnabled && transport.isPlaying) {
                    const next = computeAutoFollowScrollLeft({
                        playheadSec: visualPlayheadSec,
                        pxPerSec: pxPerSecRef.current,
                        viewportWidth: viewport.getViewportWidth(),
                        contentWidth: projectSec * pxPerSecRef.current,
                    });
                    if (Math.abs(viewport.getScrollLeft() - next) > 0.5) {
                        // 写后回读实际生效值再广播：跟随滚动接近工程右端时请求值
                        // 可能被载体钳制，跟随视口的图层必须与真实视口同源。
                        syncScrollLeft(viewport.setScrollLeft(next));
                    }
                }

                // 标尺播放头线**不在这里写**：它由内核在 draw() 内与轨道区播放头
                // 一起写（同一次帧提交、同一份内核视口、同一个 `playheadLineLeftPx`）。
                //
                // 【为什么必须收走】这里曾用 `visualPlayheadSec × pxPerSec`（**内容
                // 坐标**）写同一条线，而线现在位于内容平移层之外（视口坐标），两者
                // 差一个 scrollLeft；再加上 React 渲染路径与内核路径，同一条线一度有
                // 三个写者、两套坐标。内核的去重逻辑还会因此**跳过**自己的正确写入，
                // 让错误的坐标一直留到下一次滚动 —— 这正是"标尺线与主体线不像同一条
                // 线"的根因之一。
                // 请求内核重绘轨道区播放头。
                //
                // 【为什么必须显式请求】内核的渲染循环是纯脏标记驱动的
                // （`renderKernel/renderLoop`：`start()` 不绘制、无常驻 rAF），而
                // 轨道区播放头元素**只在 `draw()` 内被写**。播放头位置经数据镜像
                // 流入内核，镜像变化本身不会标脏——不请求就永远不重绘：点标尺
                // seek 后标尺播放头（React 声明式）动了，网格上的播放头却冻结在
                // 旧位置（缺陷 #2 的根因）。自动滚屏开着时之所以看不到，是因为它
                // 每帧写 scrollLeft 触发滚动订阅顺带标脏；而它默认关闭。
                requestPlayheadRepaint();
            },
            [
                autoScrollEnabled,
                pxPerSecRef,
                viewport,
                syncScrollLeft,
                transport.isPlaying,
                projectSec,
                requestPlayheadRepaint,
            ],
        ),
    });

    return null;
});
