import {
    Button,
    Checkbox,
    Dialog,
    Flex,
    ScrollArea,
    Select,
    Separator,
    Slider,
    Text,
    TextField,
} from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    checkpointHistory,
    moveClipStart,
    moveClipsRemote,
    persistUiSettings,
    setProjectTimelineSettingsRemote,
    setTimelineSnapSettings,
} from "../../features/session/sessionSlice";
import type { GridSize, TimelineSnapSettings } from "../../features/session/sessionTypes";
import { alignClipsToSwingGrid } from "../../utils/timelineSnapping";

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

function NumberField({
    value,
    onCommit,
    min,
    max,
    step = 1,
    className,
}: {
    value: number;
    onCommit: (next: number) => void;
    min: number;
    max: number;
    step?: number;
    className?: string;
}) {
    const [text, setText] = useState(String(value));
    useEffect(() => {
        setText(String(value));
    }, [value]);
    const commit = () => {
        const parsed = Number(text);
        if (!Number.isFinite(parsed)) {
            setText(String(value));
            return;
        }
        onCommit(Math.min(max, Math.max(min, parsed)));
    };
    return (
        <TextField.Root
            size="1"
            type="number"
            value={text}
            step={step}
            onChange={(e) => setText(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
                if (e.key === "Enter") commit();
            }}
            className={className}
            style={{ width: 72 }}
        />
    );
}

export function SnapGridSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const session = useAppSelector((state) => state.session);
    const snap = session.timelineSnap;

    const patch = (next: Partial<TimelineSnapSettings>) => {
        dispatch(setTimelineSnapSettings(next));
    };

    const handleSwingChange = (nextPercent: number, forceAlign = false) => {
        const prev = session.timelineSnap;
        const nextSettings = {
            ...prev,
            swingPercent: nextPercent,
            swingEnabled: nextPercent > 0 || prev.swingEnabled,
        };
        dispatch(
            setTimelineSnapSettings({
                swingPercent: nextPercent,
                swingEnabled: nextSettings.swingEnabled,
            }),
        );
        if (nextSettings.adjustItemsOnSwingChange && (prev.swingEnabled || forceAlign)) {
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

    const persist = () => {
        void dispatch(persistUiSettings());
    };

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 560 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("snap_grid_settings_title")}</Dialog.Title>
                <ScrollArea style={{ maxHeight: "70vh" }}>
                    <Flex direction="column" gap="3" mt="3">
                        {/* ── Grid ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_grid")}
                        </Text>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.gridVisible}
                                onCheckedChange={(v) => {
                                    patch({ gridVisible: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_grid_show_lines")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_grid_spacing")}
                            </Text>
                            <Select.Root
                                value={session.grid}
                                size="1"
                                onValueChange={(v) => {
                                    void dispatch(
                                        setProjectTimelineSettingsRemote({
                                            beatsPerBar: session.beats,
                                            timeSignatureDenominator:
                                                session.project.timeSignatureDenominator,
                                            gridSize: v,
                                        }),
                                    );
                                }}
                            >
                                <Select.Trigger />
                                <Select.Content>
                                    {GRID_SIZES.map((grid) => (
                                        <Select.Item key={grid} value={grid}>
                                            {grid}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_grid_min_spacing_px")}
                            </Text>
                            <NumberField
                                value={snap.gridMinSpacingPx}
                                min={2}
                                max={200}
                                onCommit={(v) => {
                                    patch({ gridMinSpacingPx: v });
                                    persist();
                                }}
                            />
                            <Text size="1" className="text-qt-text-muted">
                                px
                            </Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.swingEnabled}
                                onCheckedChange={(v) => {
                                    const enabled = Boolean(v);
                                    patch({ swingEnabled: enabled });
                                    if (enabled && session.clips.length > 0) {
                                        handleSwingChange(snap.swingPercent, true);
                                    } else {
                                        persist();
                                    }
                                }}
                            />
                            <Text size="2">{tAny("snap_grid_swing")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_grid_swing_strength")}
                            </Text>
                            <Slider
                                value={[snap.swingPercent]}
                                min={0}
                                max={100}
                                step={1}
                                onValueChange={(values) =>
                                    handleSwingChange(
                                        values[0] ?? 0,
                                        !snap.swingEnabled && (values[0] ?? 0) > 0,
                                    )
                                }
                                style={{ flex: 1 }}
                            />
                            <Text size="1" style={{ width: 36, textAlign: "right" }}>
                                {Math.round(snap.swingPercent)}%
                            </Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.adjustItemsOnSwingChange}
                                onCheckedChange={(v) => {
                                    patch({ adjustItemsOnSwingChange: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_grid_adjust_items_on_swing")}</Text>
                        </Flex>

                        <Separator size="4" />

                        {/* ── Snap master ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_master")}
                        </Text>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.enabled}
                                onCheckedChange={(v) => {
                                    patch({ enabled: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_enable_snapping")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_distance_px")}
                            </Text>
                            <NumberField
                                value={snap.snapDistancePx}
                                min={0}
                                max={200}
                                onCommit={(v) => {
                                    patch({ snapDistancePx: v });
                                    persist();
                                }}
                            />
                            <Text size="1" className="text-qt-text-muted">
                                px
                            </Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapRelativeToGrid}
                                onCheckedChange={(v) => {
                                    patch({ snapRelativeToGrid: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_relative_to_grid")}</Text>
                        </Flex>

                        <Separator size="4" />

                        {/* ── Snap targets matrix ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_targets")}
                        </Text>
                        <Flex gap="2">
                            <Text size="1" style={{ width: 130 }} />
                            <Text size="1" className="text-qt-text-muted" style={{ flex: 1 }}>
                                {tAny("snap_to_selection_markers_cursor")}
                            </Text>
                            <Text size="1" className="text-qt-text-muted" style={{ width: 90 }}>
                                {tAny("snap_to_grid")}
                            </Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ width: 130 }}>
                                {tAny("snap_media_items")}
                            </Text>
                            <Checkbox
                                style={{ flex: 1 }}
                                checked={snap.snapMediaItemsToSelectionMarkersCursor}
                                onCheckedChange={(v) => {
                                    patch({ snapMediaItemsToSelectionMarkersCursor: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Checkbox
                                style={{ width: 90 }}
                                checked={snap.snapMediaItemsToGrid}
                                onCheckedChange={(v) => {
                                    patch({ snapMediaItemsToGrid: Boolean(v) });
                                    persist();
                                }}
                            />
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ width: 130 }}>
                                {tAny("snap_selection")}
                            </Text>
                            <Checkbox
                                style={{ flex: 1 }}
                                checked={snap.snapSelectionToSelectionMarkersCursor}
                                onCheckedChange={(v) => {
                                    patch({ snapSelectionToSelectionMarkersCursor: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Checkbox
                                style={{ width: 90 }}
                                checked={snap.snapSelectionToGrid}
                                onCheckedChange={(v) => {
                                    patch({ snapSelectionToGrid: Boolean(v) });
                                    persist();
                                }}
                            />
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ width: 130 }}>
                                {tAny("snap_cursor")}
                            </Text>
                            <Checkbox
                                style={{ flex: 1 }}
                                checked={snap.snapCursorToSelectionMarkersCursor}
                                onCheckedChange={(v) => {
                                    patch({ snapCursorToSelectionMarkersCursor: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Checkbox
                                style={{ width: 90 }}
                                checked={snap.snapCursorToGrid}
                                onCheckedChange={(v) => {
                                    patch({ snapCursorToGrid: Boolean(v) });
                                    persist();
                                }}
                            />
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapToTakeMarkers}
                                onCheckedChange={(v) => {
                                    patch({ snapToTakeMarkers: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_take_markers")}</Text>
                        </Flex>

                        <Separator size="4" />

                        {/* ── Grid snap behavior ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_grid_behavior")}
                        </Text>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.gridSnapFollowsGridVisibility}
                                onCheckedChange={(v) => {
                                    patch({ gridSnapFollowsGridVisibility: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_follow_grid_visibility")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapToGridAnyDistance}
                                onCheckedChange={(v) => {
                                    patch({ snapToGridAnyDistance: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_any_distance")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.useIndependentSnapSpacing}
                                onCheckedChange={(v) => {
                                    patch({ useIndependentSnapSpacing: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_independent_spacing")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_grid_spacing")}
                            </Text>
                            <Select.Root
                                value={snap.snapSpacing}
                                size="1"
                                onValueChange={(v) => {
                                    patch({ snapSpacing: v as GridSize });
                                    persist();
                                }}
                            >
                                <Select.Trigger />
                                <Select.Content>
                                    {GRID_SIZES.map((grid) => (
                                        <Select.Item key={grid} value={grid}>
                                            {grid}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_spacing_min_px")}
                            </Text>
                            <NumberField
                                value={snap.snapSpacingMinPx}
                                min={2}
                                max={200}
                                onCommit={(v) => {
                                    patch({ snapSpacingMinPx: v });
                                    persist();
                                }}
                            />
                            <Text size="1" className="text-qt-text-muted">
                                px
                            </Text>
                        </Flex>

                        <Separator size="4" />

                        {/* ── Item & special interactions ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_interactions")}
                        </Text>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapItemStart}
                                onCheckedChange={(v) => {
                                    patch({ snapItemStart: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_item_start")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapItemSnapOffset}
                                onCheckedChange={(v) => {
                                    patch({ snapItemSnapOffset: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_item_snap_offset")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapAcrossTracks}
                                onCheckedChange={(v) => {
                                    patch({ snapAcrossTracks: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_across_tracks")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_track_distance")}
                            </Text>
                            <NumberField
                                value={snap.snapTrackDistance}
                                min={0}
                                max={32}
                                onCommit={(v) => {
                                    patch({ snapTrackDistance: v });
                                    persist();
                                }}
                            />
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapFixedLaneCompAreas}
                                onCheckedChange={(v) => {
                                    patch({ snapFixedLaneCompAreas: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_fixed_lane_comp_areas")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapAutomationItems}
                                onCheckedChange={(v) => {
                                    patch({ snapAutomationItems: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_automation_items")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapRazorEdits}
                                onCheckedChange={(v) => {
                                    patch({ snapRazorEdits: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_razor_edits")}</Text>
                        </Flex>

                        <Separator size="4" />

                        {/* ── Advanced ── */}
                        <Text size="1" weight="bold" className="text-qt-text-muted">
                            {tAny("snap_section_advanced")}
                        </Text>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapToProjectSampleRate}
                                onCheckedChange={(v) => {
                                    patch({ snapToProjectSampleRate: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_project_sample_rate")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.snapMediaEdgesToSource}
                                onCheckedChange={(v) => {
                                    patch({ snapMediaEdgesToSource: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_source_edges")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.forceSelectionsToMultiples}
                                onCheckedChange={(v) => {
                                    patch({ forceSelectionsToMultiples: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_force_selection_multiples")}</Text>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 110 }}>
                                {tAny("snap_selection_multiple")}
                            </Text>
                            <Select.Root
                                value={snap.selectionMultiple}
                                size="1"
                                onValueChange={(v) => {
                                    patch({ selectionMultiple: v as GridSize });
                                    persist();
                                }}
                            >
                                <Select.Trigger />
                                <Select.Content>
                                    {GRID_SIZES.map((grid) => (
                                        <Select.Item key={grid} value={grid}>
                                            {grid}
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                        <Flex align="center" gap="2">
                            <Checkbox
                                checked={snap.syncArrangeAndMidiGrid}
                                onCheckedChange={(v) => {
                                    patch({ syncArrangeAndMidiGrid: Boolean(v) });
                                    persist();
                                }}
                            />
                            <Text size="2">{tAny("snap_sync_grid_views")}</Text>
                        </Flex>
                    </Flex>
                </ScrollArea>
                <Flex justify="end" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("ok")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
