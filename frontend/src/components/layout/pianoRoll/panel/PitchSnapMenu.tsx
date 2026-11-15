/**
 * 音高吸附按钮与它的右键菜单（半音 / 音阶，以及"设置…"）。
 *
 * 【交互与"自动滚屏""波纹编辑"同构】这是本应用里**所有"开关 × 模式"按钮**的统一形状：
 * - 左键 = 开关（`togglePitchSnap`）；
 * - 右键 = 在按钮下方展开模式菜单（半音 / 音阶）；
 * - **图标画模式，不画开关**：开关由按钮的实心/强调态（`active`）+ `aria-pressed`
 *   表达。旧版把"关闭"也画成一个图标（半音图标压暗 + 一道斜杠），于是同一枚按钮
 *   有三种图形，而其中两种只差一个开关量 —— 用户得先分辨"这是关着还是半音模式"。
 * - **tooltip 固定是「功能名：模式名」**（`音高吸附：半音`）。旧版在关闭时换成
 *   "音高吸附：禁用"，那是把开关塞进了一个只该说模式的位置。
 * - 右下角小三角由 `AppIconButton` 的 `menuCaret` 画（与工具栏其它按钮同源）。
 *
 * 【与绘制工具菜单同构】两者都是"锚点外壳 + AppAnchoredMenu"：菜单本体挂到主题作用域，
 * 外壳只提供矩形，因此外壳必须始终挂载。区别只在菜单内容与激活态来源。
 *
 * 【激活态为什么读 `effectivePitchSnapVisual`】拖动中按住"临时反转吸附"修饰键时，
 * 按钮要显示反转后的状态（与时间轴吸附按钮同一手法）。那是**开关**的临时反转，
 * 与"当前是半音还是音阶"无关，因此只影响 `active`，不影响图标与 tooltip。
 */
import { Box } from "@radix-ui/themes";
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

    /** 当前量化单位的本地化名字（tooltip 用它；图标与菜单勾选直接看 `pitchSnapUnit`）。 */
    const unitLabel =
        s.pitchSnapUnit === "semitone" ? tf("quantize_semitone") : tf("quantize_scale");

    return (
        // 锚点外壳：菜单贴在这个盒子下方展开（与工具栏上的其它按钮同一种弹出面）。
        <Box ref={pitchSnapMenuAnchorRef} data-hs-context-menu>
            <AppIconButton
                active={effectivePitchSnapVisual}
                // 激活时用主题强调色（与相邻开关同款；不带 color 时 Radix 回落强调色）
                emphasis="accent"
                // 固定「功能名：模式名」—— 它同时是 aria-label，所以必须自带功能名。
                tooltip={`${t("pitch_snap")}: ${unitLabel}`}
                tabIndex={-1}
                // 右下角小三角：提示"右键还有模式可选"（与工具栏其它按钮同一个记号）。
                menuCaret
                onClick={() => {
                    dispatch(togglePitchSnap());
                    void dispatch(persistUiSettings());
                }}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setPitchSnapMenuOpen(true);
                }}
                icon={s.pitchSnapUnit === "semitone" ? pitchSnapSemitoneIcon : pitchSnapScaleIcon}
            />

            <AppAnchoredMenu
                open={pitchSnapMenuOpen}
                anchorRef={pitchSnapMenuAnchorRef}
                menuRef={pitchSnapMenuRef}
                ariaLabel={t("pitch_snap")}
                onClose={() => setPitchSnapMenuOpen(false)}
                items={[
                    {
                        key: "pitch-snap-semitone",
                        // 菜单项只说**模式名**：功能名已经在按钮的 tooltip 里，
                        // 在菜单里再念一遍"音高吸附："只会让两个选项读起来更慢。
                        label: tf("quantize_semitone"),
                        icon: pitchSnapSemitoneIcon,
                        checked: s.pitchSnapUnit === "semitone",
                        onSelect: () => {
                            dispatch(setPitchSnapUnit("semitone"));
                            // 选模式 = 立即生效（与自动滚屏 / 波纹编辑同一条策略）。
                            if (!s.pitchSnapEnabled) dispatch(togglePitchSnap());
                            void dispatch(persistUiSettings());
                        },
                    },
                    {
                        key: "pitch-snap-scale",
                        label: tf("quantize_scale"),
                        icon: pitchSnapScaleIcon,
                        checked: s.pitchSnapUnit === "scale",
                        onSelect: () => {
                            dispatch(setPitchSnapUnit("scale"));
                            if (!s.pitchSnapEnabled) dispatch(togglePitchSnap());
                            void dispatch(persistUiSettings());
                        },
                    },
                    {
                        key: "pitch-snap-settings",
                        label: tf("pitch_snap_settings_action"),
                        separatorBefore: true,
                        onSelect: () => setPitchSnapOpen(true),
                    },
                ]}
            />
        </Box>
    );
}
