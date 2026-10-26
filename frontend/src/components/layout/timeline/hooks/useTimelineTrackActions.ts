// 时间轴轨道头动作与分组派生。拆自 TimelinePanel。
import { useMemo } from "react";
import {
    isPluginMode,
    canCreateHostTracks,
    canGroupPluginTracks,
} from "../../../../services/hostCapabilities";
import {
    addTrackRemote,
    createHostTrackRemote,
    duplicateTrackRemote,
    removeTrackRemote,
    selectTrackRemote,
    setTrackStateRemote,
    moveTrackRemote,
    setTrackName,
    setTrackVolume,
} from "../../../../features/session/sessionSlice";
import { getInsertBelowTargetIndex } from "../trackContextMenuPlacement";
import { isMirrorEcho } from "../scrollEcho";
import type { TimelineSessionSlice } from "./useTimelineState";
import type { TimelineKernelHost } from "../kernel/host/timelineKernelHost";
import type { AppDispatch } from "../../../../app/store";
import React from "react";

type Deps = {
    dispatch: AppDispatch;
    sessionRef: React.MutableRefObject<TimelineSessionSlice>;
    setTrackVolumeUi: React.Dispatch<React.SetStateAction<Record<string, number>>>;
    kernelHostRef: React.MutableRefObject<TimelineKernelHost | null>;
    s: TimelineSessionSlice;
    multiSelectedClipIds: string[];
    disabledGroupIds: string[];
};

export function useTimelineTrackActions({
    dispatch,
    sessionRef,
    setTrackVolumeUi,
    kernelHostRef,
    s,
    multiSelectedClipIds,
    disabledGroupIds,
}: Deps) {
    const clipById = useMemo(
        () => new Map(s.clips.map((clip) => [clip.id, clip] as const)),
        [s.clips],
    );

    /**
     * 轨道头左键按下：把该轨道设为当前轨道（面板 → `TrackList` 的 `onSelectTrack`）。
     *
     * 特殊说明：`applySelectedClip: false` —— 轨道头点击的**全部**意图就是换当前
     * 轨道（手册：「点击左侧轨道头切换当前轨道」），不包含"恢复某条 clip 的选中"。
     * 后端的选中记忆是**全工程唯一**的（`state.rs::select_track` 只改
     * `selected_track_id`），恢复出来的 clip 可能属于另一条轨道，会把用户刚做完的
     * "点空白取消选中"异步复活。实测（忠实 mock）：纯字符串形式下点轨道头会让
     * `selectedClipId` 从 `null` 变回 `track-1-clip-1`。
     */
    const handleSelectTrack = React.useCallback(
        (trackId: string) => {
            if (sessionRef.current.selectedTrackId === trackId) {
                return;
            }
            void dispatch(selectTrackRemote({ trackId, applySelectedClip: false }));
        },
        [dispatch, sessionRef],
    );
    const handleRemoveTrack = React.useCallback(
        (trackId: string) => {
            if (isPluginMode()) return;
            dispatch(removeTrackRemote(trackId));
        },
        [dispatch],
    );
    const handleMoveTrack = React.useCallback(
        (payload: { trackId: string; targetIndex: number; parentTrackId: string | null }) => {
            if (isPluginMode() && !canGroupPluginTracks()) return;
            dispatch(
                moveTrackRemote({
                    trackId: payload.trackId,
                    targetIndex: payload.targetIndex,
                    parentTrackId: payload.parentTrackId,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackMute = React.useCallback(
        (trackId: string, nextMuted: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    muted: nextMuted,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackSolo = React.useCallback(
        (trackId: string, nextSolo: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    solo: nextSolo,
                }),
            );
        },
        [dispatch],
    );
    const handleToggleTrackCompose = React.useCallback(
        (trackId: string, nextComposeEnabled: boolean) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    composeEnabled: nextComposeEnabled,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackVolumeUiChange = React.useCallback(
        (trackId: string, nextVolume: number) => {
            setTrackVolumeUi((prev) => ({
                ...prev,
                [trackId]: nextVolume,
            }));
        },
        [setTrackVolumeUi],
    );
    const handleTrackVolumeCommit = React.useCallback(
        (trackId: string, nextVolume: number) => {
            dispatch(setTrackVolume({ trackId, volume: nextVolume }));
            setTrackVolumeUi((prev) => {
                const copy = { ...prev };
                delete copy[trackId];
                return copy;
            });
            dispatch(
                setTrackStateRemote({
                    trackId,
                    volume: nextVolume,
                }),
            );
        },
        [dispatch, setTrackVolumeUi],
    );
    const handleAddTrack = React.useCallback(() => {
        // 插件里"添加轨道"建的是**宿主**轨道（`create_host_track`），不是插件自己的
        // 轨道；后者的命令被 `pluginAllowsAction` 的 `track.*` 规则挡掉。新建的宿主
        // 轨道没有音频，所以这个 FX 实例暂时还是空的 —— 菜单文案已经写明这一点。
        if (isPluginMode()) {
            if (!canCreateHostTracks()) return;
            void dispatch(createHostTrackRemote());
            return;
        }
        dispatch(addTrackRemote({}));
    }, [dispatch]);
    const handleTrackColorChange = React.useCallback(
        (trackId: string, color: string) => {
            if (isPluginMode()) return;
            dispatch(
                setTrackStateRemote({
                    trackId,
                    color,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackAlgoChange = React.useCallback(
        (trackId: string, algo: string) => {
            dispatch(
                setTrackStateRemote({
                    trackId,
                    pitchAnalysisAlgo: algo,
                }),
            );
        },
        [dispatch],
    );
    const handleTrackNameChange = React.useCallback(
        (trackId: string, name: string) => {
            if (isPluginMode()) return;
            dispatch(setTrackName({ trackId, name }));
            dispatch(
                setTrackStateRemote({
                    trackId,
                    name,
                }),
            );
        },
        [dispatch],
    );
    const handleDuplicateTrack = React.useCallback(
        (trackId: string) => {
            dispatch(duplicateTrackRemote(trackId));
        },
        [dispatch],
    );
    // “复制拖动”修饰键 + 轨道头拖拽：在拖放位置克隆轨道（含子树）。
    const handleDuplicateTrackTo = React.useCallback(
        (payload: { trackId: string; targetIndex: number; parentTrackId: string | null }) => {
            dispatch(
                duplicateTrackRemote({
                    trackId: payload.trackId,
                    parentTrackId: payload.parentTrackId,
                    targetIndex: payload.targetIndex,
                }),
            );
        },
        [dispatch],
    );
    const handleCreateTrackBelow = React.useCallback(
        (trackId: string) => {
            void (async () => {
                const existingTracks = [...sessionRef.current.tracks];
                const beforeIds = new Set(existingTracks.map((track) => track.id));
                const added = (await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap()) as {
                    tracks?: Array<{ id?: string }>;
                    selected_track_id?: string | null;
                };
                const nextTracks = Array.isArray(added.tracks) ? added.tracks : [];
                const createdTrackId =
                    nextTracks.find((track) => !beforeIds.has(String(track?.id)))?.id ??
                    added.selected_track_id ??
                    null;
                if (!createdTrackId) return;
                await dispatch(
                    moveTrackRemote({
                        trackId: String(createdTrackId),
                        targetIndex: getInsertBelowTargetIndex(existingTracks, trackId),
                        parentTrackId: null,
                    }),
                );
            })();
        },
        [dispatch, sessionRef],
    );
    const handleTrackListScrollTopChange = React.useCallback(
        (scrollTop: number) => {
            // 内核模式：轨道头只是**被动镜像**，它报来的 `scrollTop` 有两种来源，
            // 且数值上无法区分，必须靠"宿主刚写过的值"来判（见 `timeline/scrollEcho`）：
            //
            // 1. **镜像回声**（绝大多数）：宿主每帧写 `trackList.scrollTop`，这次写入
            //    触发的原生 `scroll` 事件会把**滞后一帧**的值报回来。若照单全收，内核
            //    每帧被拉回一次——实测 40 步拖拽里 19 次调用**全部**是回声、每次
            //    `delta ≈ −9px`，即用户报告的"纵向拖起来卡卡的、像被吸附"。
            // 2. **真实输入**：焦点在轨道头控件上时，浏览器原生 scroll-into-view
            //    （Tab / PageDown / End）会真正改变容器位置（实测 0 → 308 / 132 / 361）。
            //    这类必须继续回灌，否则焦点导航时轨道头会与时间轴脱节。
            //
            // 【为什么旧判据失效】旧实现拿事件值与**内核当前值**比（< 0.5px 即忽略）。
            // 拖拽时内核每帧前进，而事件报的是上一帧镜像值，两者相差约 9px，因此回声
            // 被误当成用户输入收下，形成「内核 → DOM → 内核」的回退循环。
            const host = kernelHostRef.current;
            if (host != null) {
                // 画布竖直缩放"内核先行"的在途窗口：窗口内的 scroll 事件是内容高变化的
                // 钳制回弹 / 滚动锚定补偿，与"上次写入值"必然不符，绝不能当用户输入回灌
                // ——否则会把视口从锚点位置拽回（竖直抽动）。位置由宿主在行高落地后补写，
                // 见 `verticalZoomInFlight`。
                //
                // 特殊说明：主路径（宿主接上了 `onRowHeightChange` 落地通道）**不会**进入
                // 这个状态——它走"请求 → React 落地 → 同一次提交原子应用"，内核不领先
                // React，轨道头 DOM 与内核在同一次提交里一起变。本守卫只为兜底路径而存在。
                if (host.isVerticalZoomInFlight()) {
                    return;
                }
                if (
                    isMirrorEcho({
                        mirroredPx: host.getMirroredTrackListScrollTop(),
                        nativePx: scrollTop,
                    })
                ) {
                    return;
                }
                host.setScrollTop(scrollTop);
                return;
            }
            // 宿主不可用（挂载前）时不回灌：旧的原生 scroller 分支已删除——
            // `scrollRef` 无 JSX 挂载点，`scrollRef.current` 恒为 null。
        },
        [kernelHostRef],
    );

    const activeGroupIds = useMemo(() => {
        const ids = new Set<string>();
        for (const cid of multiSelectedClipIds) {
            const gid = clipById.get(cid)?.groupId;
            if (gid && !disabledGroupIds.includes(gid)) ids.add(gid);
        }
        if (s.selectedClipId) {
            const gid = clipById.get(s.selectedClipId)?.groupId;
            if (gid && !disabledGroupIds.includes(gid)) ids.add(gid);
        }
        return ids.size > 0 ? ids : undefined;
    }, [multiSelectedClipIds, clipById, s.selectedClipId, disabledGroupIds]);
    /**
     * 内核用分组数组。
     *
     * `activeGroupIds` 是派生的 `Set`，内核侧要数组；这里转一次并保持引用稳定，
     * 避免每次渲染都产生新数组（数据镜像引用抖动会让宿主每帧判定「内容已变」）。
     */
    const kernelActiveGroupIds = React.useMemo(
        () => (activeGroupIds === undefined ? [] : Array.from(activeGroupIds)),
        [activeGroupIds],
    );
    return {
        clipById,
        handleSelectTrack,
        handleRemoveTrack,
        handleMoveTrack,
        handleToggleTrackMute,
        handleToggleTrackSolo,
        handleToggleTrackCompose,
        handleTrackVolumeUiChange,
        handleTrackVolumeCommit,
        handleAddTrack,
        handleTrackColorChange,
        handleTrackAlgoChange,
        handleTrackNameChange,
        handleDuplicateTrack,
        handleDuplicateTrackTo,
        handleCreateTrackBelow,
        handleTrackListScrollTopChange,
        kernelActiveGroupIds,
    };
}
