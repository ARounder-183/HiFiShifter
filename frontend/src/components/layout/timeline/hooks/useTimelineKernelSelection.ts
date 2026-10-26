// 时间轴内核框选、右键菜单与 Clip 轻量动作。拆自 TimelinePanel。
import {
    buildCrossfadeGripInfoContent,
    buildSingleFadeInfoContent,
    publishFadeRichTooltip,
    type FadeLabelLookup,
} from "../fadeTooltipText";
import { effectiveFadeSec } from "../kernel/interaction/fadeTargets";
import { requestOpenFadeContextMenu, type FadeContextMenuRequest } from "../fadeContextMenuBus";
import {
    openClipFormantToolWindow,
    selectTrackRemote,
    seekPlayhead,
    setPlayheadSec,
    selectClipRemote,
} from "../../../../features/session/sessionSlice";
import type { TimelineSessionSlice } from "./useTimelineState";
import type { useTimelineClipActions } from "./useTimelineClipActions";
import { computeTimelineRectSelection } from "../useTimelineSelectionRect";
import type { TimeValueFormatContext } from "../timeValueText";
import React from "react";
import type { useI18n } from "../../../../i18n/I18nProvider";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    sessionRef: React.MutableRefObject<TimelineSessionSlice>;
    multiSelectedClipIds: string[];
    setMultiSelectedClipIds: (ids: string[]) => void;
    dispatch: AppDispatch;
    clearContextMenu: () => void;
    setContextMenu: ReturnType<typeof useTimelineClipActions>["setContextMenu"];
    setTrackAreaMenu: ReturnType<typeof useTimelineClipActions>["setTrackAreaMenu"];
    fadeTooltipAnchor: () => HTMLElement | null;
    timeValueFormatCtx: TimeValueFormatContext;
    t: ReturnType<typeof useI18n>["t"];
    timelineRuntimeIsPlaying: boolean;
    activateTrackLaneTake: (clipId: string, takeId: string) => void;
    toggleTrackLaneClipMuted: (clipId: string, nextMuted: boolean) => void;
    setRateEditorClipId: (id: string | null) => void;
    setRateEditorPosition: (value: { x: number; y: number } | null) => void;
};

export function useTimelineKernelSelection({
    sessionRef,
    multiSelectedClipIds,
    setMultiSelectedClipIds,
    dispatch,
    clearContextMenu,
    setContextMenu,
    setTrackAreaMenu,
    fadeTooltipAnchor,
    timeValueFormatCtx,
    t,
    timelineRuntimeIsPlaying,
    activateTrackLaneTake,
    toggleTrackLaneClipMuted,
    setRateEditorClipId,
    setRateEditorPosition,
}: Deps) {
    const kernelBoxSelectOriginRef = React.useRef<string[] | null>(null);
    /** 框选期间「已同步为焦点 clip」的去重（避免每帧重复一次同样的后端请求）。 */
    const kernelBoxSelectSingleRef = React.useRef<string | null>(null);
    const multiSelectedIdsRef = React.useRef<string[]>([]);
    multiSelectedIdsRef.current = multiSelectedClipIds;

    /**
     * 内核框选预览：复用既有合并语义。
     *
     * 合并规则（是否保留原有选择、主修饰键切换）由
     * `computeTimelineRectSelection` 决定——两处各写一份会让「按修饰键框选」
     * 的行为在两种渲染模式下分叉。
     */
    const handleKernelBoxSelectPreview = React.useCallback(
        (args: { clipIds: readonly string[]; additive: boolean }) => {
            if (kernelBoxSelectOriginRef.current === null) {
                // 拖动前的基线 = 多选集合（非空时），否则退化为**单选**目标
                // ——旧实现 `useTimelineSelectionRect` 同一口径。只看多选集合
                // 会让"单选了 A，再框选一片空白"时 A 的取消/保留判定出错。
                const multi = multiSelectedIdsRef.current;
                const single = sessionRef.current.selectedClipId;
                kernelBoxSelectOriginRef.current =
                    multi.length > 0 ? [...multi] : single ? [single] : [];
            }
            const selected = computeTimelineRectSelection({
                selectionBeforeDrag: kernelBoxSelectOriginRef.current,
                selectedInRect: [...args.clipIds],
                primaryModifierPressedAtStart: args.additive,
            });
            setMultiSelectedClipIds(selected);
            // 框内只剩一个 clip 时，顺手把它同步为**焦点 clip**（后端 `selected_clip`
            // 落库）——旧实现 `onSingleSelect` 同源。焦点 clip 决定参数编辑器的编辑
            // 目标，缺了这一步框选出单个 clip 后参数编辑器还停在旧目标上。
            //
            // 去重（同一目标不重复请求）是纯优化：旧实现每帧都发，行为不可见。
            if (selected.length === 1) {
                if (kernelBoxSelectSingleRef.current !== selected[0]) {
                    kernelBoxSelectSingleRef.current = selected[0];
                    void dispatch(selectClipRemote(selected[0]));
                }
            } else {
                kernelBoxSelectSingleRef.current = null;
            }
        },
        [dispatch, sessionRef, setMultiSelectedClipIds],
    );

    /** 内核框选收尾：取消时恢复拖动前的选择。 */
    const handleKernelBoxSelectCommit = React.useCallback(
        (args: { clipIds: readonly string[]; additive: boolean; cancelled: boolean }) => {
            const origin = kernelBoxSelectOriginRef.current;
            kernelBoxSelectOriginRef.current = null;
            if (args.cancelled) {
                if (origin !== null) setMultiSelectedClipIds(origin);
                return;
            }
            kernelBoxSelectSingleRef.current = null;
            // 预览已写入最终选择；这里只补「框内为空且非叠加」应清空选择的语义。
            if (args.clipIds.length === 0 && !args.additive) {
                setMultiSelectedClipIds([]);
            }
        },
        [setMultiSelectedClipIds],
    );

    /**
     * 内核右键框选（按住 `modifier.clipRangeToParamSelection`）→ 被框选的 clip
     * 范围**并入**参数编辑器选区。
     *
     * 这是**额外**效果，不替代 clip 多选：同一次手势的框选预览/收尾走的是普通
     * 框选那条路径（`handleKernelBoxSelectPreview` / `handleKernelBoxSelectCommit`），
     * 框内 clip 因此也照常纳入 clip 选择 —— 与未按住该修饰键的普通框选一致。
     *
     * 与单个块的右键单击（`handleKernelDoubleClickClip` 的 `toggle`）同一修饰键
     * 家族：单击 = 该块并入 / 挖掉；拖框 = 框内**全部**并入（叠加在既有参数选区
     * 上，重叠/相接自动合并）。复用 `addClipsToParamSelection` 编辑操作——
     * `PianoRollPanel` 的接收端已实现"只取当前根轨道组内的块 + 归一化"语义，
     * 这里不再重写一份。
     */
    const handleKernelBoxSelectToParamSelection = React.useCallback(
        (args: { clipIds: readonly string[]; cancelled: boolean }) => {
            if (args.cancelled || args.clipIds.length === 0) return;
            clearContextMenu();
            window.dispatchEvent(
                new CustomEvent("hifi:editOp", {
                    detail: { op: "addClipsToParamSelection", clipIds: [...args.clipIds] },
                }),
            );
        },
        [clearContextMenu],
    );

    /**
     * 内核右键菜单：复用既有分支（clip 菜单 / 轨道区菜单）。
     *
     * 与旧实现的差别只在**命中来源**：旧实现用 `trackIdFromClientY` +
     * `beatFromClientX`（依赖原生 scroller 的 scrollLeft），内核直接给命中的
     * 轨道与 clip 列表，菜单分支本身不变。
     */
    const handleKernelContextMenu = React.useCallback(
        (args: {
            clientX: number;
            clientY: number;
            clipIds: readonly string[];
            trackId: string | null;
            sec: number;
        }) => {
            setContextMenu(null);
            setTrackAreaMenu(null);
            if (args.trackId === null) return;
            if (args.clipIds.length > 0) {
                // 最上面那个（startSec 最大）作为主目标，其余作为"重叠选择"入口——
                // 与旧实现取 `clipsHere[clipsHere.length - 1]` 一致。
                setContextMenu({
                    x: args.clientX,
                    y: args.clientY,
                    clipId: args.clipIds[args.clipIds.length - 1],
                    overlappingClipIds: args.clipIds.length > 1 ? [...args.clipIds] : undefined,
                });
                return;
            }
            // 插件仍保留空白区菜单；各动作由菜单按真实宿主能力分别准入。
            if (sessionRef.current.selectedTrackId !== args.trackId) {
                // 轨道区空白右键 = "把这条轨道设为当前轨道"（随后弹粘贴/建轨菜单）：
                // 只切焦点，不得让后端把全局记住的 `selected_clip_id` 恢复回来。
                // 后端的选中记忆是**全工程唯一**的（不是每轨一份，见
                // `state.rs::select_track`），恢复出来的 clip 完全可能属于另一条
                // 轨道，从而在"本地取消选中"之后被异步复活。契约与
                // `handleKernelSeek` 的空白点击同源（019e93ed）。
                void dispatch(
                    selectTrackRemote({ trackId: args.trackId, applySelectedClip: false }),
                );
            }
            setTrackAreaMenu({
                x: args.clientX,
                y: args.clientY,
                trackId: args.trackId,
                timeSec: args.sec,
            });
        },
        [dispatch, sessionRef, setContextMenu, setTrackAreaMenu],
    );

    /**
     * 内核淡变专属右键菜单：转发到全局总线（菜单宿主挂在面板上）。
     *
     * 复用旧实现的 `requestOpenFadeContextMenu`——载荷类型与语义完全一致，因此
     * 曲率滑块 / 形状 / 长度 / 重置等菜单项在内核模式下与旧实现同源。
     * 通用 clip 菜单要显式关掉（旧实现里两者互斥：右键落点只属于其中一方）。
     */
    const handleKernelFadeContextMenu = React.useCallback(
        (request: FadeContextMenuRequest) => {
            setContextMenu(null);
            setTrackAreaMenu(null);
            requestOpenFadeContextMenu(request);
        },
        [setContextMenu, setTrackAreaMenu],
    );

    /**
     * 内核淡变悬停浮标：发布到 AppTooltip 的富内容注册表。
     *
     * 【为什么在面板拼装内容】提示文案（形状名 i18n、长度按时间轴显示设置格式化、
     * 内联曲线图标）是渲染无关的领域知识，且与旧实现 `FadeHitLayer` /
     * `OverlapEditLayer` 共用同一套 `buildSingleFadeInfoContent` /
     * `buildCrossfadeGripInfoContent`——内核侧重写会产生第二份文案规则。
     *
     * 【锚点元素】内核的淡变控件是自绘的、没有 DOM，宿主因此把**内核容器本身**
     * 标记为浮标载体（`data-hs-fade-tooltip-anchor`）：容器本就是指针事件的目标，
     * 复用它就不需要插入任何会吞事件的浮层。面板把内容发布到它上面即可整体复用
     * AppTooltip 的显示 / 钉住 / 随指针跟随 / 菜单抑制语义。
     *
     * 长度取**生效值**（自动交叉淡化优先于手动），与绘制端和右键菜单同一规则。
     */
    const handleKernelFadeHover = React.useCallback(
        (
            args: {
                clipId: string;
                side: "in" | "out";
                isLine: boolean;
                partnerClipId?: string;
            } | null,
        ) => {
            const anchor = fadeTooltipAnchor();
            if (anchor === null) return;
            if (args === null) {
                // 收起：content 传 null → Provider 移除该元素的内容（浮标消失）。
                publishFadeRichTooltip(anchor, null);
                return;
            }
            const clips = sessionRef.current.clips;
            const sideOf = (clipId: string, isOut: boolean) => {
                const clip = clips.find((item) => item.id === clipId);
                if (clip === undefined) return null;
                return {
                    shape: (isOut ? clip.fadeOutShape : clip.fadeInShape) ?? 0,
                    hostFades: clip.hostFades,
                    dir: (isOut ? clip.fadeOutDir : clip.fadeInDir) ?? 0,
                    lengthSec: effectiveFadeSec(
                        isOut ? clip.fadeOutSec : clip.fadeInSec,
                        isOut ? clip.autoFadeOutSec : clip.autoFadeInSec,
                    ),
                };
            };
            // 交叉点抓手 = 双列（前块淡出在前、后块淡入在后，与右键菜单列序一致）。
            if (args.partnerClipId !== undefined) {
                const earlier = sideOf(args.partnerClipId, true);
                const later = sideOf(args.clipId, false);
                if (earlier === null || later === null) {
                    publishFadeRichTooltip(anchor, null);
                    return;
                }
                publishFadeRichTooltip(
                    anchor,
                    buildCrossfadeGripInfoContent({
                        earlier,
                        later,
                        formatCtx: timeValueFormatCtx,
                        t: t as unknown as FadeLabelLookup,
                    }),
                );
                return;
            }
            const isOut = args.side === "out";
            const side = sideOf(args.clipId, isOut);
            if (side === null) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            publishFadeRichTooltip(
                anchor,
                buildSingleFadeInfoContent({
                    isOut,
                    ...side,
                    formatCtx: timeValueFormatCtx,
                    t: t as unknown as FadeLabelLookup,
                }),
            );
        },
        [timeValueFormatCtx, sessionRef, t, fadeTooltipAnchor],
    );

    /**
     * 内核单击 inactive take lane：切换活跃 Take（复用旧实现的提交入口）。
     *
     * 与旧实现同源：暂停 / 停止时还会把播放光标带到点击位置（播放中不打断当前
     * 播放位置），切换本身走 `setClipActiveTakeRemote`（含后端落库与撤销步）。
     *
     * 特殊说明：本回调**确实要移动播放头**，因此同样遵循「乐观写 + seek 成对派发」
     * 契约（`setPlayheadSec` 必须先于 `seekPlayhead`）——只派发 seek 时后端原样
     * 回显请求值，`seekPlayhead.fulfilled` 的采纳分支不命中，store 的
     * `playheadSec` 不会变（同 `handleKernelSeek` 的说明）。
     */
    const handleKernelActivateTake = React.useCallback(
        (clipId: string, takeId: string, sec: number) => {
            if (!timelineRuntimeIsPlaying) {
                dispatch(setPlayheadSec(sec));
                void dispatch(seekPlayhead(sec));
            }
            activateTrackLaneTake(clipId, takeId);
        },
        [activateTrackLaneTake, dispatch, timelineRuntimeIsPlaying],
    );

    /**
     * 内核右键手势（按住 `modifier.clipRangeToParamSelection`，默认 Alt，右键单击
     * clip）：请求参数编辑器按 clip 起止范围创建选区。
     *
     * 左键双击 clip 主体已不再走此入口（易误触且会抢走 clip 的复制/剪切焦点），
     * 「在参数编辑器内为 Clip 创建选区」统一由上述右键手势承担。关闭右键菜单 →
     * 派发 `hifi:editOp/selectClipParamRange`，交互焦点随之切到参数编辑器侧。
     * 用 window 事件而不是直接调 store：参数编辑器在另一棵子树里监听它，
     * 这条契约与渲染模式无关（内核 / 旧实现共用同一入口）。
     */
    const handleKernelClipParamSelectionGesture = React.useCallback(
        (clipId: string, mode: "replace" | "toggle" = "replace") => {
            clearContextMenu();
            window.dispatchEvent(
                new CustomEvent("hifi:editOp", {
                    // mode 缺省 replace；按住 `modifier.clipRangeToParamSelection`
                    // （默认 Alt）右键单击时内核传 "toggle"，由参数编辑器并入 / 挖掉
                    // 该块范围。
                    detail: { op: "selectClipParamRange", clipId, mode },
                }),
            );
        },
        [clearContextMenu],
    );

    /**
     * 内核静音切换：复用旧实现的处理器（内部含乐观更新、分组联动与远端提交）。
     */
    const handleKernelToggleClipMute = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            toggleTrackLaneClipMuted(clipId, nextMuted);
        },
        [toggleTrackLaneClipMuted],
    );

    /**
     * 内核打开共振峰工具窗口：锚点用内核给的指针屏幕坐标。
     *
     * 旧实现取按钮右缘 +12 / 上缘（见 `ClipFormantButton`）；内核模式下指针就在
     * 徽标上，直接用指针位置等价且更简单（浮窗自身会钳制到视口内）。
     */
    const handleKernelOpenClipFormant = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            dispatch(
                openClipFormantToolWindow({
                    clipId,
                    anchor: { x: Math.round(screenX + 12), y: Math.round(screenY) },
                }),
            );
        },
        [dispatch],
    );

    /**
     * 内核速率高级编辑（右键速率标签）：复用既有 `ClipRateEditorDialog`。
     *
     * 该对话框渲染在内核开关之外（见文件末尾），两种渲染模式共用，此前只是缺入口。
     */
    const handleKernelRateBadgeMenu = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            setRateEditorClipId(clipId);
            setRateEditorPosition({ x: screenX, y: screenY });
        },
        [],
    );

    /**
     * 内核音量旋钮拖拽：按下时的基准（参与集合 + 各自增益 + 精细调整轴向状态）。
     *
     * 特殊说明：参与集合是**多选集合**，不展开编组——旧实现 `useEditDrag` 的
     * `supportsGroupExpansion` 明确排除 gain（增益拖拽只作用于选中的 clip）。
     */
    return {
        handleKernelBoxSelectPreview,
        handleKernelBoxSelectCommit,
        handleKernelBoxSelectToParamSelection,
        handleKernelContextMenu,
        handleKernelFadeContextMenu,
        handleKernelFadeHover,
        handleKernelActivateTake,
        handleKernelClipParamSelectionGesture,
        handleKernelToggleClipMute,
        handleKernelOpenClipFormant,
        handleKernelRateBadgeMenu,
    };
}
