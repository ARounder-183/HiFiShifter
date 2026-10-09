/**
 * FadeContextMenuHost — 淡变上下文菜单的宿主。
 *
 * 挂在 TimelinePanel 内；监听全局打开事件，渲染 FadeContextMenu，并把
 * 形状/曲率修改提交到 Redux（乐观）与后端。
 *
 * ## 快照过期问题（重要）
 * 打开事件载荷只携带 `{clipId, isOut}` 标识——**不能**携带 shape/dir 的
 * 静态快照。否则曲率滑块每次提交虽更新了 Redux，菜单渲染仍引用打开
 * 瞬间的冻结对象，受控 `value` 永远回弹 = "滑块无法调整"。因此每帧
 * 渲染时按标识从 Redux 实时解析形状/曲率/有效长度。
 *
 * ## 多选（问题 4）
 * 右键命中的 clip 属于多选集合时，形状/曲率/重置应用到**全部选中 clip**
 * （与 gain/静音等批量编辑同一判定：getBulkEditableClipIds）。
 *
 * ## 撤销管理（问题 3）
 * 后端为撤销权威。菜单会话采用**惰性 undo group**：首次提交时才
 * beginUndoGroup（压入"会话前"快照），会话内全部写入 checkpoint:false，
 * 菜单关闭（任何路径）时 flush 节流尾值并 endUndoGroup —— 整个会话 =
 * 单个撤销步，且不会在"只打开未修改"时产生空撤销步。
 * 双击重置曲率可能发生在菜单外（直接双击包络线）：无活动会话时自行
 * 开/关一个 undo group，保证重置也可单步撤销。
 */
import React from "react";
import { batch } from "react-redux";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { setClipFades } from "../../../features/session/sessionSlice";
import { webApi } from "../../../services/webviewApi";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { defaultFadeDirFor } from "./reaperFade";
import { hostFadeDisplayShape } from "./hostFadeDisplay";
import { FadeContextMenu, type FadeContextSide } from "./FadeContextMenu";
import { onFadeContextMenuRequest, onFadeCurvatureReset } from "./fadeContextMenuBus";
import { getBulkEditableClipIds } from "./hooks/bulkClipEdit";

const remoteDebounceMs = 120;

type SideRef = { clipId: string; isOut: boolean };
type MenuState = {
    clientX: number;
    clientY: number;
    primary: SideRef;
    secondary: SideRef | null;
};

/** 节流积压条目：等待收尾 flush 的最新补丁。 */
type PendingEntry = {
    clipId: string;
    isOut: boolean;
    patch: { shape?: number; dir?: number; s?: number };
};

function extractSide(clip: ClipInfo | undefined, ref: SideRef): FadeContextSide | null {
    if (!clip) return null;
    const shape = Number.isFinite(ref.isOut ? clip.fadeOutShape : clip.fadeInShape)
        ? ref.isOut
            ? clip.fadeOutShape
            : clip.fadeInShape
        : 0;
    const dir = (ref.isOut ? clip.fadeOutDir : clip.fadeInDir) ?? 0;
    // S 轴只对 continuous 宿主存在；legacy/独立 App 恒 0。
    const continuous = clip.hostFades?.curve_mode === "reaper_new";
    const s = continuous ? Number((ref.isOut ? clip.fadeOutS : clip.fadeInS) ?? 0) || 0 : 0;
    const autoSec = ref.isOut ? (clip.autoFadeOutSec ?? 0) : (clip.autoFadeInSec ?? 0);
    const manualSec = Math.max(0, ref.isOut ? (clip.fadeOutSec ?? 0) : (clip.fadeInSec ?? 0));
    const lengthSec = autoSec > 0 ? autoSec : manualSec;
    return {
        clipId: ref.clipId,
        isOut: ref.isOut,
        shape,
        dir,
        s,
        lengthSec,
        // 宿主读数从 Redux 实时解析（不是打开时的快照）——与 shape/dir 同一条纪律。
        hostFades: clip.hostFades,
    };
}

export const FadeContextMenuHost: React.FC = () => {
    const dispatch = useAppDispatch();
    const [menu, setMenu] = React.useState<MenuState | null>(null);
    const lastRemoteRef = React.useRef<Record<string, number>>({});
    const clips = useAppSelector((state) => state.session.clips);

    // ── 多选展开（Redux 选区；ref 镜像避免事件回调读到过期选区）──────
    const multiSelectedClipIds = useAppSelector((state) => state.session.multiSelectedClipIds);
    const multiSelectedRef = React.useRef(multiSelectedClipIds);
    multiSelectedRef.current = multiSelectedClipIds;
    const multiSelectedSet = React.useMemo(
        () => new Set(multiSelectedClipIds),
        [multiSelectedClipIds],
    );
    const multiSelectedSetRef = React.useRef(multiSelectedSet);
    multiSelectedSetRef.current = multiSelectedSet;

    /** 重点 clip 在多选集合内 → 应用到全部选中；否则仅该 clip。 */
    const expandTargets = React.useCallback((activeClipId: string): string[] => {
        return getBulkEditableClipIds({
            activeClipId,
            multiSelectedClipIds: multiSelectedRef.current,
            multiSelectedSet: multiSelectedSetRef.current,
        });
    }, []);

    // ── 会话级 undo group（惰性开、关闭路径统一收尾）────────────────
    /** 已打开的 undo group 链（打开后所有远端写入排在其后）。 */
    const sessionGroupRef = React.useRef<Promise<unknown> | null>(null);
    /** 远端写入链（防 flush 与节流写入乱序）。 */
    const remoteChainRef = React.useRef<Promise<unknown>>(Promise.resolve());
    /** 节流未发的最新补丁积压（菜单关闭时 flush 兜底，防尾值丢失）。 */
    const pendingPatchesRef = React.useRef<Map<string, PendingEntry>>(new Map());
    /** 会话收尾已完成（幂等）。 */
    const endedRef = React.useRef(false);

    /** 首次实际提交时打开会话 undo group；已开则复用。 */
    const ensureSessionGroup = React.useCallback((): Promise<unknown> => {
        if (!sessionGroupRef.current) {
            sessionGroupRef.current = webApi.beginUndoGroup("edit_clip");
        }
        return sessionGroupRef.current;
    }, []);

    /** 立即发送某 key 的最新积压补丁（checkpoint:false，并入活动组）。 */
    const sendPendingKey = React.useCallback((key: string) => {
        const entry = pendingPatchesRef.current.get(key);
        if (!entry) return;
        lastRemoteRef.current[key] = Date.now();
        // S 只在 continuous 宿主上出现（`handleSChange` 才会填它）。legacy 宿主上
        // 传 S 会被后端 `validate_fade_axes` 拒绝，所以这里按"有没有值"条件展开。
        const sPatch =
            entry.patch.s === undefined
                ? {}
                : entry.isOut
                  ? { fadeOutS: entry.patch.s }
                  : { fadeInS: entry.patch.s };
        remoteChainRef.current = remoteChainRef.current
            .then(() =>
                webApi.setClipState({
                    clipId: entry.clipId,
                    ...(entry.isOut
                        ? { fadeOutShape: entry.patch.shape, fadeOutDir: entry.patch.dir }
                        : { fadeInShape: entry.patch.shape, fadeInDir: entry.patch.dir }),
                    ...sPatch,
                    checkpoint: false,
                }),
            )
            .catch(() => undefined);
    }, []);

    /** 会话收尾：flush 节流尾值 → 关闭 undo group（幂等，多路径安全）。 */
    const endSession = React.useCallback(() => {
        if (endedRef.current) return;
        endedRef.current = true;
        for (const key of [...pendingPatchesRef.current.keys()]) {
            sendPendingKey(key);
        }
        pendingPatchesRef.current.clear();
        const group = sessionGroupRef.current;
        if (group) {
            sessionGroupRef.current = null;
            void remoteChainRef.current
                .catch(() => undefined)
                .then(() => group)
                .then(() => webApi.endUndoGroup())
                .catch(() => undefined);
        }
    }, [sendPendingKey]);

    // 菜单关闭（含 primary clip 被删除、组件卸载）→ 统一收尾。
    const closeMenu = React.useCallback(() => {
        setMenu(null); // 卸载时 FadeContextMenu 的清理函数复位抑制标志。
    }, []);

    React.useEffect(() => () => endSession(), [endSession]);

    React.useEffect(() => {
        return onFadeContextMenuRequest((request) => {
            endedRef.current = false; // 新会话重新武装收尾。
            setMenu({
                clientX: request.clientX,
                clientY: request.clientY,
                primary: { clipId: request.primary.clipId, isOut: request.primary.isOut },
                secondary: request.secondary
                    ? { clipId: request.secondary.clipId, isOut: request.secondary.isOut }
                    : null,
            }); // 打开时 FadeContextMenu 自身会置位全局抑制标志。
        });
    }, []);

    // ── 双击重置曲率（离散动作，立即提交不节流）─────────────────────
    // clips 以 ref 镜像：事件回调触发时读取的必须是最新 Redux 形态，
    // 才能把曲率重置到"当前形状"的默认值。
    const clipsRef = React.useRef(clips);
    clipsRef.current = clips;
    React.useEffect(() => {
        return onFadeCurvatureReset(({ sides }) => {
            void (async () => {
                // 菜单会话进行中 → 并入会话组；否则自开自关一个组，
                // 保证"菜单外双击重置"也可单步撤销。
                const standalone = !sessionGroupRef.current;
                if (standalone) {
                    sessionGroupRef.current = webApi.beginUndoGroup("edit_clip");
                }
                try {
                    await sessionGroupRef.current;
                    for (const side of sides) {
                        const targets = expandTargets(side.clipId);
                        for (const targetId of targets) {
                            const clip = clipsRef.current.find((c) => c.id === targetId);
                            if (!clip) continue;
                            // 【为什么形状也要一起送】"曲率"落在哪个宿主字段取决于版本，
                            // 而两套轴**不是同一套参数化**（实测 HFS dir 0.75 ↔ 宿主
                            // curvature 0.6878）。只送 HFS 的 dir，连续轴宿主会把它当
                            // curvature 直接写进 `D_FADEINDIR_NEW` —— 那是个别族的数值。
                            // 连形状一起送，后端就会按实测表写预设自己的
                            // `(curvature, S)`；旧轴宿主则照旧写 `C_FADE*SHAPE` + dir。
                            // 这与形状循环走的是同一条规则。
                            const shapeRaw = side.isOut ? clip.fadeOutShape : clip.fadeInShape;
                            const shape = hostFadeDisplayShape(
                                clip.hostFades,
                                side.isOut,
                                Number.isFinite(shapeRaw) ? shapeRaw : 0,
                            );
                            const dir = defaultFadeDirFor(shape, side.isOut);
                            const patch = side.isOut
                                ? { fadeOutShape: shape, fadeOutDir: dir }
                                : { fadeInShape: shape, fadeInDir: dir };
                            dispatch(setClipFades({ clipId: targetId, ...patch }));
                            remoteChainRef.current = remoteChainRef.current
                                .then(() =>
                                    webApi.setClipState({
                                        clipId: targetId,
                                        ...patch,
                                        checkpoint: false,
                                    }),
                                )
                                .catch(() => undefined);
                        }
                    }
                } finally {
                    if (standalone) {
                        sessionGroupRef.current = null;
                        void remoteChainRef.current
                            .catch(() => undefined)
                            .then(() => webApi.endUndoGroup())
                            .catch(() => undefined);
                    }
                }
            })();
        });
    }, [dispatch, expandTargets]);

    // 实时从 Redux 解析两侧的最新形态（菜单打开期间任何提交都会回流到这里）。
    const primary = menu
        ? extractSide(
              clips.find((c) => c.id === menu.primary.clipId),
              menu.primary,
          )
        : null;
    const secondaryRef = menu?.secondary ?? null;
    const secondary = secondaryRef
        ? extractSide(
              clips.find((c) => c.id === secondaryRef.clipId),
              secondaryRef,
          )
        : null;
    // 菜单会话是否仍存活：primary 解析失败（clip 被删）等同关闭 → 收尾。
    const menuActive = menu != null && primary != null;
    const prevMenuActiveRef = React.useRef(menuActive);
    React.useEffect(() => {
        if (prevMenuActiveRef.current && !menuActive) {
            endSession();
        }
        prevMenuActiveRef.current = menuActive;
    }, [menuActive, endSession]);

    /**
     * 单次提交：应用到（展开后的）全部目标 clip。
     *
     * - Redux 乐观更新走 batch（多 clip 一次渲染）；
     * - 远端写入带前导节流（防滑块连发刷爆后端），最新补丁积压，
     *   菜单关闭时统一 flush —— 杜绝"关菜单丢尾值"。
     * - 全部写入 checkpoint:false，并入会话 undo group（首次提交时惰性开启）。
     */
    const commit = React.useCallback(
        (clipId: string, isOut: boolean, patch: { shape?: number; dir?: number; s?: number }) => {
            void ensureSessionGroup();
            const targets = expandTargets(clipId);
            batch(() => {
                for (const targetId of targets) {
                    const sPatch =
                        patch.s === undefined
                            ? {}
                            : isOut
                              ? { fadeOutS: patch.s }
                              : { fadeInS: patch.s };
                    dispatch(
                        setClipFades({
                            clipId: targetId,
                            ...(isOut
                                ? { fadeOutShape: patch.shape, fadeOutDir: patch.dir }
                                : { fadeInShape: patch.shape, fadeInDir: patch.dir }),
                            ...sPatch,
                        }),
                    );
                }
            });
            for (const targetId of targets) {
                const key = `${targetId}:ctx-${isOut ? "out" : "in"}`;
                pendingPatchesRef.current.set(key, { clipId: targetId, isOut, patch });
                const now = Date.now();
                const last = lastRemoteRef.current[key] || 0;
                if (now - last > remoteDebounceMs) {
                    sendPendingKey(key);
                }
            }
        },
        [dispatch, ensureSessionGroup, expandTargets, sendPendingKey],
    );

    const handleShapeChange = React.useCallback(
        (clipId: string, isOut: boolean, shape: number) => {
            // 切换形状重置默认曲率（REAPER 语义）。离散、低频：随 commit
            // 的节流窗口立即直连提交 —— 同为会话单撤销步的一部分。
            const dir = defaultFadeDirFor(shape, isOut);
            commit(clipId, isOut, { shape, dir });
        },
        [commit],
    );

    const handleDirChange = React.useCallback(
        (clipId: string, isOut: boolean, dir: number) => {
            commit(clipId, isOut, { dir });
        },
        [commit],
    );

    // S 轴滑杆（只有 REAPER ≥7.81 有）。曲率与 S 是同一族曲线的两个分量，
    // 单独动 S 时不带 shape —— 后端只写 `D_FADE*DIR2_NEW`，曲率保持不变。
    const handleSChange = React.useCallback(
        (clipId: string, isOut: boolean, s: number) => {
            commit(clipId, isOut, { s });
        },
        [commit],
    );

    if (!menu || !primary) return null; // 目标 clip 已被删除等情形：直接关闭。
    return (
        <FadeContextMenu
            x={menu.clientX}
            y={menu.clientY}
            primary={primary}
            secondary={secondary}
            onClose={closeMenu}
            onShapeChange={handleShapeChange}
            onDirChange={handleDirChange}
            onSChange={handleSChange}
        />
    );
};

/** 七预设形状 id 列表便捷引用（供循环点击使用）。 */
