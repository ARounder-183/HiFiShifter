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
 */
import React from "react";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { setClipFades } from "../../../features/session/sessionSlice";
import { webApi } from "../../../services/webviewApi";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { defaultFadeDirFor, FADE_PRESETS } from "./reaperFade";
import { FadeContextMenu, type FadeContextSide } from "./FadeContextMenu";
import { onFadeContextMenuRequest, onFadeCurvatureReset } from "./fadeContextMenuBus";

const remoteDebounceMs = 120;

type SideRef = { clipId: string; isOut: boolean };
type MenuState = {
    clientX: number;
    clientY: number;
    primary: SideRef;
    secondary: SideRef | null;
};

function extractSide(clip: ClipInfo | undefined, ref: SideRef): FadeContextSide | null {
    if (!clip) return null;
    const shape = Number.isFinite(ref.isOut ? clip.fadeOutShape : clip.fadeInShape)
        ? ref.isOut
            ? clip.fadeOutShape
            : clip.fadeInShape
        : 0;
    const dir = (ref.isOut ? clip.fadeOutDir : clip.fadeInDir) ?? 0;
    const autoSec = ref.isOut ? (clip.autoFadeOutSec ?? 0) : (clip.autoFadeInSec ?? 0);
    const manualSec = Math.max(0, ref.isOut ? (clip.fadeOutSec ?? 0) : (clip.fadeInSec ?? 0));
    const lengthSec = autoSec > 0 ? autoSec : manualSec;
    return { clipId: ref.clipId, isOut: ref.isOut, shape, dir, lengthSec };
}

export const FadeContextMenuHost: React.FC = () => {
    const dispatch = useAppDispatch();
    const [menu, setMenu] = React.useState<MenuState | null>(null);
    const lastRemoteRef = React.useRef<Record<string, number>>({});
    const clips = useAppSelector((state) => state.session.clips);

    React.useEffect(() => {
        return onFadeContextMenuRequest((request) => {
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
            for (const side of sides) {
                const clip = clipsRef.current.find((c) => c.id === side.clipId);
                if (!clip) continue;
                const shapeRaw = side.isOut ? clip.fadeOutShape : clip.fadeInShape;
                const shape = Number.isFinite(shapeRaw) ? shapeRaw : 0;
                const dir = defaultFadeDirFor(shape, side.isOut);
                dispatch(
                    setClipFades({
                        clipId: side.clipId,
                        ...(side.isOut ? { fadeOutDir: dir } : { fadeInDir: dir }),
                    }),
                );
                try {
                    void webApi.setClipState({
                        clipId: side.clipId,
                        ...(side.isOut ? { fadeOutDir: dir } : { fadeInDir: dir }),
                        checkpoint: false,
                    });
                } catch {
                    // Best-effort remote update.
                }
            }
        });
    }, [dispatch]);

    const closeMenu = React.useCallback(() => {
        setMenu(null); // 卸载时 FadeContextMenu 的清理函数复位抑制标志。
    }, []);

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

    const commit = React.useCallback(
        (clipId: string, isOut: boolean, patch: { shape?: number; dir?: number }) => {
            dispatch(
                setClipFades({
                    clipId,
                    ...(isOut
                        ? { fadeOutShape: patch.shape, fadeOutDir: patch.dir }
                        : { fadeInShape: patch.shape, fadeInDir: patch.dir }),
                }),
            );
            try {
                const now = Date.now();
                const key = `${clipId}:ctx-${isOut ? "out" : "in"}`;
                const last = lastRemoteRef.current[key] || 0;
                if (now - last > remoteDebounceMs) {
                    lastRemoteRef.current[key] = now;
                    void webApi.setClipState({
                        clipId,
                        ...(isOut
                            ? { fadeOutShape: patch.shape, fadeOutDir: patch.dir }
                            : { fadeInShape: patch.shape, fadeInDir: patch.dir }),
                        checkpoint: false,
                    });
                }
            } catch {
                // Best-effort remote preview update.
            }
        },
        [dispatch],
    );

    const handleShapeChange = React.useCallback(
        (clipId: string, isOut: boolean, shape: number) => {
            // 切换形状重置默认曲率（REAPER 语义）。离散、低频：立即直连提交，
            // 不节流 —— 保证撤销点外数值不残留旧曲率。
            const dir = defaultFadeDirFor(shape, isOut);
            dispatch(
                setClipFades({
                    clipId,
                    ...(isOut
                        ? { fadeOutShape: shape, fadeOutDir: dir }
                        : { fadeInShape: shape, fadeInDir: dir }),
                }),
            );
            try {
                void webApi.setClipState({
                    clipId,
                    ...(isOut
                        ? { fadeOutShape: shape, fadeOutDir: dir }
                        : { fadeInShape: shape, fadeInDir: dir }),
                    checkpoint: false,
                });
            } catch {
                // Best-effort.
            }
        },
        [dispatch],
    );

    const handleDirChange = React.useCallback(
        (clipId: string, isOut: boolean, dir: number) => {
            commit(clipId, isOut, { dir });
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
        />
    );
};

/** 七预设形状 id 列表便捷引用（供循环点击使用）。 */
export const CYCLE_SHAPES = FADE_PRESETS.map((preset) => preset.shape);
