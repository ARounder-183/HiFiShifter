/*
 * 吸附 / 网格设置。
 *
 * 【为什么重写结构】上一版是 25 行挤在一条长滚动里（内容 1175px、视口 492px，
 * 约 2.4 屏），而且：
 *   - 节标题 12px/700 比它统领的 14px 复选框行**还小**，滚动时没有路标；
 *   - 同一张表单里两种标签：`AppField` 是 11px muted、裸复选框行是 14px；
 *   - 3 个下拉、4 个数字框、1 个滑块**全部不能滚轮调值**；
 *   - 5 条 Radix `Separator`（全仓第四种分隔线做法）；
 *   - 矩阵区用内联 130/90 像素宽度，与标签列错位。
 *
 * 现在：五个 `AppFormSection` 分组（节标题 13px/600，靠字重与留白分层），
 * 布尔行统一 `AppSwitchRow`，取值控件统一走能力层原语（滚轮与精细调整内建），
 * 矩阵列与 `AppField` 的标签列同宽。
 *
 * 【为什么不做页签】设置项之间有关联（"独立吸附间距"开关决定下面的间距下拉
 * 是否有意义），页签会把上下文藏起来。用分组 + 留白解决扫读，而不是用导航。
 */
import { Checkbox } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    checkpointHistory,
    moveClipStart,
    moveClipsRemote,
    persistUiSettings,
    setGrid,
    setProjectTimelineSettingsRemote,
    setTimelineSnapSettings,
} from "../../features/session/sessionSlice";
import type { GridSize, TimelineSnapSettings } from "../../features/session/sessionTypes";
import { alignClipsToSwingGrid } from "../../utils/timelineSnapping";
import { useDebouncedCallback } from "../../utils/useDebouncedCallback";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppFormSection, AppSwitchRow } from "../../ui/Field";
import { AppNumberField } from "../../ui/NumberField";
import { AppSelect } from "../../ui/Select";
import { AppSlider, AppSliderReadout } from "../../ui/Slider";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const GRID_SIZES: readonly GridSize[] = [
    "1/1",
    "1/2",
    "1/4",
    "1/8",
    "1/16",
    "1/32",
    "1/64",
    "1/1d",
    "1/2d",
    "1/4d",
    "1/8d",
    "1/16d",
    "1/32d",
    "1/64d",
    "1/1t",
    "1/2t",
    "1/4t",
    "1/8t",
    "1/16t",
    "1/32t",
    "1/64t",
];

/** 网格档位的下拉选项（值即标签）。 */
const GRID_OPTIONS = GRID_SIZES.map((grid) => ({ value: grid as string, label: grid }));

/** 矩阵标签列宽：与 `AppField` 的标签列同宽，保证矩阵与上方各行左缘对齐。 */
const MATRIX_LABEL_WIDTH = 112;
/** 矩阵「吸附至网格」列宽：三行共用，列头与单元格居中于同一宽度。 */
const MATRIX_GRID_COLUMN_WIDTH = 96;

export function SnapGridSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const session = useAppSelector((state) => state.session);
    const snap = session.timelineSnap;

    const patch = (next: Partial<TimelineSnapSettings>) => {
        dispatch(setTimelineSnapSettings(next));
    };

    const persist = () => {
        void dispatch(persistUiSettings());
    };

    /*
     * 工程基准网格的后端同步**去抖**。
     *
     * 【为什么必须去抖】`setProjectTimelineSettingsRemote` 最终打到一个**同步** Tauri
     * 命令（`set_project_timeline_settings`）。Tauri 2.10 里只有 `async fn` 命令会进
     * 线程池，普通 `fn` 的命令体在 **UI 线程上内联执行**；而该命令会全量重建节拍器
     * 响点表（最多 200 万项 + 稳定排序），带 Tempo Map 时还会整份克隆时间轴并压一条
     * 撤销记录。滚轮逐格调用它，累计约 5s 消息泵饥饿，Windows 即判定"未响应"
     * （已实测复现）。
     *
     * 因此本地 `setGrid` 立即生效（廉价、界面即时响应），后端同步在操作停止后下发一次。
     */
    const syncProjectGrid = useDebouncedCallback((gridSize: string) => {
        void dispatch(
            setProjectTimelineSettingsRemote({
                beatsPerBar: session.beats,
                timeSignatureDenominator: session.project.timeSignatureDenominator,
                gridSize,
            }),
        );
    }, 250);

    /** 拖动中只更新设置值（轻量 Redux 写）。 */
    const applySwingPreview = (percent: number) => {
        dispatch(
            setTimelineSnapSettings({
                swingPercent: percent,
                swingEnabled: percent > 0 || snap.swingEnabled,
            }),
        );
    };

    /**
     * 提交摇摆强度：对齐剪辑 + checkpoint + IPC 一律推迟到这里。
     *
     * 拖动会以高频率触发 onValueChange，逐 tick 全量重排会刷满撤销栈并打爆
     * move_clips / save_ui_settings IPC（与 TimelineDisplaySettingsDialog 的
     * onValueCommit 模式一致）。
     */
    const commitSwing = (percent: number, forceAlign = false) => {
        const nextSettings = {
            ...snap,
            swingPercent: percent,
            swingEnabled: percent > 0 || snap.swingEnabled,
        };
        dispatch(
            setTimelineSnapSettings({
                swingPercent: nextSettings.swingPercent,
                swingEnabled: nextSettings.swingEnabled,
            }),
        );
        if (nextSettings.adjustClipsOnSwingChange && (snap.swingEnabled || forceAlign)) {
            const updates = alignClipsToSwingGrid({
                clips: session.clips,
                settings: nextSettings,
                grid: session.grid,
                tempoMap: session.tempoMap,
                bpm: session.bpm,
            });
            const moves = Object.entries(updates).map(([clipId, startSec]) => ({
                clipId,
                startSec,
            }));
            if (moves.length > 0) {
                dispatch(checkpointHistory());
                for (const move of moves) {
                    dispatch(moveClipStart(move));
                }
                void dispatch(moveClipsRemote({ moves, moveLinkedParams: false }));
            }
        }
        void dispatch(persistUiSettings());
    };

    /** 矩阵三行：行标签 + 两个目标列的勾选状态与写回。 */
    const matrixRows = [
        {
            key: "clips",
            label: t("snap_clips"),
            toMarkersCursor: snap.snapClipsToSelectionMarkersCursor,
            setToMarkersCursor: (v: boolean) => patch({ snapClipsToSelectionMarkersCursor: v }),
            toGrid: snap.snapClipsToGrid,
            setToGrid: (v: boolean) => patch({ snapClipsToGrid: v }),
        },
        {
            key: "selection",
            label: t("snap_selection"),
            toMarkersCursor: snap.snapSelectionToSelectionMarkersCursor,
            setToMarkersCursor: (v: boolean) =>
                patch({ snapSelectionToSelectionMarkersCursor: v }),
            toGrid: snap.snapSelectionToGrid,
            setToGrid: (v: boolean) => patch({ snapSelectionToGrid: v }),
        },
        {
            key: "cursor",
            label: t("snap_cursor"),
            toMarkersCursor: snap.snapCursorToSelectionMarkersCursor,
            setToMarkersCursor: (v: boolean) => patch({ snapCursorToSelectionMarkersCursor: v }),
            toGrid: snap.snapCursorToGrid,
            setToGrid: (v: boolean) => patch({ snapCursorToGrid: v }),
        },
    ];

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("snap_grid_settings_title")}
            description={t("snap_grid_settings_desc")}
            size="md"
            // 动作区与长内容之间需要一条分割线（本窗口内容远高于视口）
            footerDivider
            actions={[
                {
                    id: "ok",
                    label: t("ok"),
                    intent: "primary",
                    onClick: () => onOpenChange(false),
                },
            ]}
        >
            {/*
             * 纯布尔列表：控件贴左、标签紧随（与同表单的字段行不同 —— 这里
             * 大多数行都是开关，标签列会把控件推到 112px 之后，读起来很别扭）。
             */}
            <AppForm booleanRow="leading">
                <AppFormSection title={t("snap_section_grid")}>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_grid_show_lines")}
                        checked={snap.gridVisible}
                        onCheckedChange={(v) => {
                            patch({ gridVisible: v });
                            persist();
                        }}
                    />
                    <AppField label={t("snap_grid_spacing")}>
                        <AppSelect
                            value={session.grid}
                            ariaLabel={t("snap_grid_spacing")}
                            options={GRID_OPTIONS}
                            onValueChange={(v) => {
                                // 本地立即生效（廉价），后端同步去抖 —— 见 syncProjectGrid
                                dispatch(setGrid(v as GridSize));
                                syncProjectGrid.call(v);
                            }}
                        />
                    </AppField>
                    <AppField label={t("snap_grid_min_spacing_px")}>
                        <AppNumberField
                            value={snap.gridMinSpacingPx}
                            unit="pixels"
                            min={2}
                            max={200}
                            suffix="px"
                            ariaLabel={t("snap_grid_min_spacing_px")}
                            onCommit={(v) => {
                                patch({ gridMinSpacingPx: v });
                                persist();
                            }}
                        />
                    </AppField>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_grid_swing")}
                        checked={snap.swingEnabled}
                        onCheckedChange={(enabled) => {
                            patch({ swingEnabled: enabled });
                            if (enabled && session.clips.length > 0) {
                                // 离散开关动作：直接走提交（对齐 + checkpoint + 持久化）
                                commitSwing(snap.swingPercent, true);
                            } else {
                                persist();
                            }
                        }}
                    />
                    <AppField label={t("snap_grid_swing_strength")}>
                        <div className="flex items-center gap-2">
                            <AppSlider
                                value={snap.swingPercent}
                                unit="percent"
                                min={0}
                                max={100}
                                ariaLabel={t("snap_grid_swing_strength")}
                                onChange={applySwingPreview}
                                onCommit={(v) => commitSwing(v, !snap.swingEnabled && v > 0)}
                            />
                            <AppSliderReadout>{Math.round(snap.swingPercent)}%</AppSliderReadout>
                        </div>
                    </AppField>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_grid_adjust_clips_on_swing")}
                        checked={snap.adjustClipsOnSwingChange}
                        onCheckedChange={(v) => {
                            patch({ adjustClipsOnSwingChange: v });
                            persist();
                        }}
                    />
                </AppFormSection>

                <AppFormSection title={t("snap_section_master")}>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_enable_snapping")}
                        checked={snap.enabled}
                        onCheckedChange={(v) => {
                            patch({ enabled: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_show_highlight")}
                        checked={snap.snapHighlightEnabled}
                        onCheckedChange={(v) => {
                            patch({ snapHighlightEnabled: v });
                            persist();
                        }}
                    />
                    <AppField label={t("snap_distance_px")}>
                        <AppNumberField
                            value={snap.snapDistancePx}
                            unit="pixels"
                            min={0}
                            max={200}
                            suffix="px"
                            ariaLabel={t("snap_distance_px")}
                            onCommit={(v) => {
                                patch({ snapDistancePx: v });
                                persist();
                            }}
                        />
                    </AppField>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_relative_to_grid")}
                        checked={snap.snapRelativeToGrid}
                        onCheckedChange={(v) => {
                            patch({ snapRelativeToGrid: v });
                            persist();
                        }}
                    />
                </AppFormSection>

                {/*
                 * 吸附对象 × 目标矩阵：三行两列。
                 *
                 * 列头与行标签同宽（`MATRIX_LABEL_WIDTH` 与 AppField 的标签列一致），
                 * 因此矩阵与上方所有 `AppField` 行左缘对齐 —— 原实现用内联
                 * 130/90，与 112px 的标签列错位。
                 */}
                <AppFormSection title={t("snap_section_targets")}>
                    <div className="flex items-center gap-2">
                        <span
                            className="hs-type-caption shrink-0"
                            style={{ width: MATRIX_LABEL_WIDTH }}
                        />
                        <span className="hs-type-caption flex-1">
                            {t("snap_to_selection_markers_cursor")}
                        </span>
                        <span
                            className="hs-type-caption shrink-0 text-center"
                            style={{ width: MATRIX_GRID_COLUMN_WIDTH }}
                        >
                            {t("snap_to_grid")}
                        </span>
                    </div>
                    {matrixRows.map((row) => (
                        <div key={row.key} className="flex items-center gap-2">
                            <span
                                className="hs-type-label shrink-0"
                                style={{ width: MATRIX_LABEL_WIDTH }}
                            >
                                {row.label}
                            </span>
                            <div className="flex flex-1 items-center">
                                <Checkbox
                                    checked={row.toMarkersCursor}
                                    aria-label={`${row.label} — ${t("snap_to_selection_markers_cursor")}`}
                                    onCheckedChange={(v) => {
                                        row.setToMarkersCursor(Boolean(v));
                                        persist();
                                    }}
                                />
                            </div>
                            <div
                                className="flex shrink-0 items-center justify-center"
                                style={{ width: MATRIX_GRID_COLUMN_WIDTH }}
                            >
                                <Checkbox
                                    checked={row.toGrid}
                                    aria-label={`${row.label} — ${t("snap_to_grid")}`}
                                    onCheckedChange={(v) => {
                                        row.setToGrid(Boolean(v));
                                        persist();
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </AppFormSection>

                <AppFormSection title={t("snap_section_grid_behavior")}>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_follow_grid_visibility")}
                        checked={snap.snapFollowsGridVisibility}
                        onCheckedChange={(v) => {
                            patch({ snapFollowsGridVisibility: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_any_distance")}
                        checked={snap.snapToGridAnyDistance}
                        onCheckedChange={(v) => {
                            patch({ snapToGridAnyDistance: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_independent_spacing")}
                        checked={snap.useIndependentSnapSpacing}
                        onCheckedChange={(v) => {
                            patch({ useIndependentSnapSpacing: v });
                            persist();
                        }}
                    />
                    <AppField label={t("snap_grid_spacing")}>
                        <AppSelect
                            value={snap.snapSpacing}
                            ariaLabel={t("snap_grid_spacing")}
                            options={GRID_OPTIONS}
                            onValueChange={(v) => {
                                patch({ snapSpacing: v as GridSize });
                                persist();
                            }}
                        />
                    </AppField>
                    <AppField label={t("snap_spacing_min_px")}>
                        <AppNumberField
                            value={snap.snapSpacingMinPx}
                            unit="pixels"
                            min={2}
                            max={200}
                            suffix="px"
                            ariaLabel={t("snap_spacing_min_px")}
                            onCommit={(v) => {
                                patch({ snapSpacingMinPx: v });
                                persist();
                            }}
                        />
                    </AppField>
                </AppFormSection>

                <AppFormSection title={t("snap_section_interactions")}>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_clip_edges")}
                        checked={snap.snapClipEdges}
                        onCheckedChange={(v) => {
                            patch({ snapClipEdges: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_clip_snap_offset")}
                        checked={snap.snapClipSnapOffset}
                        onCheckedChange={(v) => {
                            patch({ snapClipSnapOffset: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_across_tracks")}
                        checked={snap.snapAcrossTracks}
                        onCheckedChange={(v) => {
                            patch({ snapAcrossTracks: v });
                            persist();
                        }}
                    />
                    <AppField label={t("snap_track_distance")}>
                        <AppNumberField
                            value={snap.snapTrackDistance}
                            unit="integer"
                            min={0}
                            max={32}
                            ariaLabel={t("snap_track_distance")}
                            onCommit={(v) => {
                                patch({ snapTrackDistance: v });
                                persist();
                            }}
                        />
                    </AppField>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_razor_edits")}
                        checked={snap.snapRazorEdits}
                        onCheckedChange={(v) => {
                            patch({ snapRazorEdits: v });
                            persist();
                        }}
                    />
                </AppFormSection>

                <AppFormSection title={t("snap_section_advanced")}>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_project_sample_rate")}
                        checked={snap.snapToProjectSampleRate}
                        onCheckedChange={(v) => {
                            patch({ snapToProjectSampleRate: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_source_edges")}
                        checked={snap.snapClipsToSourceMedia}
                        onCheckedChange={(v) => {
                            patch({ snapClipsToSourceMedia: v });
                            persist();
                        }}
                    />
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_force_selection_multiples")}
                        checked={snap.forceSelectionsToMultiples}
                        onCheckedChange={(v) => {
                            patch({ forceSelectionsToMultiples: v });
                            persist();
                        }}
                    />
                    <AppField label={t("snap_selection_multiple")}>
                        <AppSelect
                            value={snap.selectionMultiple}
                            ariaLabel={t("snap_selection_multiple")}
                            options={GRID_OPTIONS}
                            onValueChange={(v) => {
                                patch({ selectionMultiple: v as GridSize });
                                persist();
                            }}
                        />
                    </AppField>
                    <AppSwitchRow
                        control="checkbox"
                        label={t("snap_sync_grid_views")}
                        checked={snap.syncArrangeAndMidiGrid}
                        onCheckedChange={(v) => {
                            patch({ syncArrangeAndMidiGrid: v });
                            persist();
                        }}
                    />
                </AppFormSection>
            </AppForm>
        </AppDialog>
    );
}
