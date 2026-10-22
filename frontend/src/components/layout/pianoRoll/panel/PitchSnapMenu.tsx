/**
 * 音高吸附按钮与它的右键菜单（关 / 半音 / 音阶，以及"设置…"）。
 *
 * 按钮显示当前吸附模式（关闭 / 半音 / 音阶），图标随模式切换；右键弹出菜单
 * 在三种模式之间切换，底部一项打开 `PitchSnapSettingsDialog`（由工具栏渲染，
 * 这里只负责把 open 置位）。
 *
 * 【与绘制工具菜单同构】两者都是"锚点外壳 + AppAnchoredMenu"：菜单本体挂到
 * `document.body`，外壳只提供矩形，因此外壳必须始终挂载。区别只在菜单内容
 * 与激活态来源（这里读 `effectivePitchSnapVisual`，它已经把"吸附开关"与
 * "当前量化单位"合并成一个视觉真值）。
 */
import { Box } from "@radix-ui/themes";
import { CheckIcon } from "@radix-ui/react-icons";
import {
    persistUiSettings,
    setPitchSnapUnit,
    togglePitchSnap,
} from "../../../../features/session/sessionSlice";
import { AppAnchoredMenu, AppIconButton } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "dispatch"
        | "effectivePitchSnapVisual"
        | "pitchSnapMenuAnchorRef"
        | "pitchSnapMenuOpen"
        | "pitchSnapMenuRef"
        | "pitchSnapScaleIcon"
        | "pitchSnapSemitoneIcon"
        | "s"
        | "setPitchSnapMenuOpen"
        | "setPitchSnapOpen"
        | "t"
        | "tf"
    >;
};

export function PitchSnapMenu({ m }: Props) {
    const {
        dispatch,
        effectivePitchSnapVisual,
        pitchSnapMenuAnchorRef,
        pitchSnapMenuOpen,
        pitchSnapMenuRef,
        pitchSnapScaleIcon,
        pitchSnapSemitoneIcon,
        s,
        setPitchSnapMenuOpen,
        setPitchSnapOpen,
        t,
        tf,
    } = m;

    return (
        <>
            {/* 锚点外壳：菜单本身挂到 `document.body`（见 AppAnchoredMenu）。 */}
            <Box ref={pitchSnapMenuAnchorRef} data-hs-context-menu>
                <AppIconButton
                    active={effectivePitchSnapVisual}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={`${t("pitch_snap")}: ${
                        effectivePitchSnapVisual
                            ? s.pitchSnapUnit === "semitone"
                                ? tf("quantize_semitone")
                                : tf("quantize_scale")
                            : tf("pitch_snap_off")
                    }`}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(togglePitchSnap());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setPitchSnapMenuOpen(true);
                    }}
                    icon={
                        <Box
                            style={{
                                position: "relative",
                                width: 15,
                                height: 15,
                            }}
                        >
                            <Box
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                }}
                            >
                                {!effectivePitchSnapVisual ? (
                                    <Box
                                        style={{
                                            position: "relative",
                                            width: 15,
                                            height: 15,
                                            opacity: 0.45,
                                        }}
                                    >
                                        {pitchSnapSemitoneIcon}
                                        <svg
                                            className="absolute inset-0"
                                            width="15"
                                            height="15"
                                            viewBox="0 0 15 15"
                                            fill="none"
                                            xmlns="http://www.w3.org/2000/svg"
                                        >
                                            <path
                                                d="M3 3L12 12"
                                                stroke="currentColor"
                                                strokeWidth="1.2"
                                                strokeLinecap="round"
                                            />
                                        </svg>
                                    </Box>
                                ) : s.pitchSnapUnit === "semitone" ? (
                                    pitchSnapSemitoneIcon
                                ) : (
                                    pitchSnapScaleIcon
                                )}
                            </Box>
                            <Box
                                style={{
                                    position: "absolute",
                                    right: -1,
                                    bottom: -1,
                                    width: 6,
                                    height: 6,
                                    opacity: 0.7,
                                }}
                            >
                                <svg
                                    width="6"
                                    height="6"
                                    viewBox="0 0 6 6"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path d="M0 6L6 0V6Z" fill="currentColor" />
                                </svg>
                            </Box>
                        </Box>
                    }
                />

                <AppAnchoredMenu
                    open={pitchSnapMenuOpen}
                    anchorRef={pitchSnapMenuAnchorRef}
                    menuRef={pitchSnapMenuRef}
                >
                    <button
                        type="button"
                        className="hs-menu__item"
                        onClick={() => {
                            dispatch(setPitchSnapUnit("semitone"));
                            if (!s.pitchSnapEnabled) {
                                dispatch(togglePitchSnap());
                            }
                            void dispatch(persistUiSettings());
                            setPitchSnapMenuOpen(false);
                        }}
                        onPointerDown={(e) => e.stopPropagation()}
                    >
                        <span className="flex min-w-0 items-center gap-2">
                            <Box className="hs-menu__icon">{pitchSnapSemitoneIcon}</Box>
                            <span className="hs-menu__label-text">
                                {tf("pitch_snap_menu_semitone")}
                            </span>
                        </span>
                        <span className="hs-menu__trail">
                            {s.pitchSnapUnit === "semitone" ? (
                                <span className="hs-menu__check">
                                    <CheckIcon />
                                </span>
                            ) : null}
                        </span>
                    </button>
                    <button
                        type="button"
                        className="hs-menu__item"
                        onClick={() => {
                            dispatch(setPitchSnapUnit("scale"));
                            if (!s.pitchSnapEnabled) {
                                dispatch(togglePitchSnap());
                            }
                            void dispatch(persistUiSettings());
                            setPitchSnapMenuOpen(false);
                        }}
                        onPointerDown={(e) => e.stopPropagation()}
                    >
                        <span className="flex min-w-0 items-center gap-2">
                            <Box className="hs-menu__icon">{pitchSnapScaleIcon}</Box>
                            <span className="hs-menu__label-text">
                                {tf("pitch_snap_menu_scale")}
                            </span>
                        </span>
                        <span className="hs-menu__trail">
                            {s.pitchSnapUnit === "scale" ? (
                                <span className="hs-menu__check">
                                    <CheckIcon />
                                </span>
                            ) : null}
                        </span>
                    </button>
                    <div className="hs-menu__separator" role="separator" />
                    <button
                        type="button"
                        className="hs-menu__item"
                        onClick={() => {
                            setPitchSnapMenuOpen(false);
                            setPitchSnapOpen(true);
                        }}
                        onPointerDown={(e) => e.stopPropagation()}
                    >
                        <span className="hs-menu__label-text">
                            {tf("pitch_snap_settings_action")}
                        </span>
                    </button>
                </AppAnchoredMenu>
            </Box>
        </>
    );
}
