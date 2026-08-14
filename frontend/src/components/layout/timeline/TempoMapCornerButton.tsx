/**
 * 轨道头 / 参数编辑器左上角矩形区域右下角的“速度映射”小按钮。
 *
 * - 无 Tempo Map 或速度映射被隐藏：点击后自动显示速度映射，
 *   并在工程没有 Tempo Map 时为其创建一个 Tempo Map
 *   （仅 0 位置的初始点 = 工程基准记录，不添加其他变化点）；
 * - 存在 Tempo Map 且正在显示：图标转为红色提醒，点击弹出确认对话框：
 *   可以“清空”整个 Tempo Map，也可以“仅隐藏”（与 视图 → 速度映射 为同一设置）。
 *
 * 按钮的模式/图标完全由 Redux（tempoMap + tempoMapVisible）驱动，
 * 因此切换 视图 → 速度映射 会自动更新本按钮。
 */
import { useCallback, useMemo, useState } from "react";
import { Button, Dialog, Flex, Text } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { useI18n } from "../../../i18n/I18nProvider";
import {
    persistUiSettings,
    setTempoMap,
    setTempoMapVisible,
} from "../../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../../features/session/thunks/tempoMapThunks";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    clampBpm,
    clampDenominator,
    clampNumerator,
    createTempoPointId,
    scaleLikeToScaleData,
} from "../../../utils/tempoMap";

/** “显示/创建”模式图标：空心四分音符 + 右上角加号（表示“显示并创建”）。 */
function TempoShowIcon() {
    return (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            {/* 四分音符：符头 + 符干 + 符尾 */}
            <ellipse
                cx="4.6"
                cy="11.4"
                rx="2.6"
                ry="1.9"
                transform="rotate(-18 4.6 11.4)"
                stroke="currentColor"
                strokeWidth="1.1"
                fill="none"
            />
            <path
                d="M6.7 10.8 V3.2"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
            />
            <path
                d="M6.7 3.4 C 8.6 3.2 10.1 4.2 10.9 6.4"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
                fill="none"
            />
            {/* 右上角加号 */}
            <path d="M12.4 1.2 v3.4 M10.7 2.9 h3.4" stroke="currentColor" strokeWidth="1.3" />
        </svg>
    );
}

/** “已启用”模式图标：红色实心节拍器（提醒用户点击可清空/隐藏）。 */
function TempoActiveIcon() {
    return (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            {/* 节拍器机身（实心） */}
            <path
                d="M6 2.6 H10 L10 9.9 A3.1 3.1 0 1 1 6 9.9 Z"
                fill="currentColor"
                stroke="currentColor"
                strokeWidth="0.8"
                strokeLinejoin="round"
            />
            {/* 摆锤槽（机身上方的小缺口） */}
            <path d="M8 2.6 V1.4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            <circle cx="8" cy="7" r="0.9" fill="var(--qt-window)" />
        </svg>
    );
}

export const TempoMapCornerButton: React.FC = () => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const s = useAppSelector((state) => state.session);
    const [dialogOpen, setDialogOpen] = useState(false);

    const hasMap = s.tempoMap != null && s.tempoMap.points.length > 0;
    /** 存在 Tempo Map 且正在显示（红色提醒模式）。 */
    const active = hasMap && s.tempoMapVisible;

    const projectScaleLike = useMemo<ScaleLike | null>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project],
    );

    /** 确保工程存在 Tempo Map（仅 0 位置初始点 = 工程基准记录），并显示速度映射。 */
    const ensureShown = useCallback(() => {
        if (!hasMap) {
            const initialMap = {
                points: [
                    {
                        id: createTempoPointId(),
                        positionSec: 0,
                        bpm: clampBpm(s.bpm || 120),
                        timeSignature: {
                            numerator: clampNumerator(s.beats || 4),
                            denominator: clampDenominator(
                                s.project.timeSignatureDenominator ?? 4,
                            ),
                        },
                        scale: scaleLikeToScaleData(
                            projectScaleLike ?? undefined,
                            s.project.useCustomScale
                                ? (s.project.customScale?.name ?? undefined)
                                : undefined,
                        ),
                    },
                ],
            };
            dispatch(setTempoMap(initialMap));
            void dispatch(setTempoMapRemote(initialMap));
        }
        dispatch(setTempoMapVisible(true));
        void dispatch(persistUiSettings());
    }, [dispatch, hasMap, s.bpm, s.beats, s.project, projectScaleLike]);

    const clearMap = useCallback(() => {
        dispatch(setTempoMap(null));
        void dispatch(setTempoMapRemote(null));
        setDialogOpen(false);
    }, [dispatch]);

    const hideOnly = useCallback(() => {
        dispatch(setTempoMapVisible(false));
        void dispatch(persistUiSettings());
        setDialogOpen(false);
    }, [dispatch]);

    return (
        <>
            <button
                type="button"
                className="absolute rounded-[3px] flex items-center justify-center outline-none focus-visible:outline-none transition-colors"
                style={{
                    right: 3,
                    bottom: 3,
                    width: 16,
                    height: 16,
                    color: active ? "var(--qt-danger, #e5484d)" : "var(--qt-text-muted)",
                    backgroundColor: "transparent",
                    border: "none",
                    padding: 0,
                    cursor: "pointer",
                }}
                data-tooltip={
                    active
                        ? tAny("tempo_map_active_tooltip")
                        : tAny("tempo_map_show_tooltip")
                }
                onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.backgroundColor =
                        "color-mix(in srgb, var(--qt-hover) 80%, transparent)";
                }}
                onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.backgroundColor =
                        "transparent";
                }}
                onClick={(e) => {
                    e.stopPropagation();
                    if (active) {
                        setDialogOpen(true);
                        return;
                    }
                    ensureShown();
                }}
            >
                {active ? <TempoActiveIcon /> : <TempoShowIcon />}
            </button>

            <Dialog.Root open={dialogOpen} onOpenChange={setDialogOpen}>
                <Dialog.Content maxWidth="360px">
                    <Dialog.Title>{tAny("tempo_map_clear_dialog_title")}</Dialog.Title>
                    <Dialog.Description>
                        <Text size="2" className="text-qt-text-muted">
                            {tAny("tempo_map_clear_dialog_message")}
                        </Text>
                    </Dialog.Description>
                    <Flex justify="end" align="center" mt="4" gap="2">
                        <Button
                            variant="soft"
                            color="gray"
                            size="1"
                            onClick={() => setDialogOpen(false)}
                        >
                            {tAny("cancel")}
                        </Button>
                        <Button variant="soft" color="gray" size="1" onClick={hideOnly}>
                            {tAny("tempo_map_hide_only")}
                        </Button>
                        <Button variant="solid" color="red" size="1" onClick={clearMap}>
                            {tAny("tempo_map_clear_confirm")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
        </>
    );
};
