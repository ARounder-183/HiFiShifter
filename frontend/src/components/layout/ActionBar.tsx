import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
    Flex,
    Select,
    TextField,
    Button,
    IconButton,
    Separator,
    Text,
} from "@radix-ui/themes";
import {
    PauseIcon,
    PlayIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import { PitchSnapSettingsDialog } from "./PitchSnapSettingsDialog";
import { CustomScaleDialog } from "./CustomScaleDialog";

import {
    playOriginal,
    stopAudioPlayback,
    setBpm,
    updateTransportBpm,
    setProjectTimelineSettingsRemote,
    toggleAutoCrossfade,
    toggleGridSnap,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleParamEditorSeekPlayhead,
    persistUiSettings,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
} from "../../features/session/sessionSlice";
import { SCALE_KEYS, SCALE_LABELS } from "../../utils/musicalScales";
import { toggleVisible } from "../../features/fileBrowser/fileBrowserSlice";

export function ActionBar() {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const fileBrowserVisible = useAppSelector(
        (state: RootState) => state.fileBrowser.visible,
    );
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);
    const [customScaleOpen, setCustomScaleOpen] = useState(false);
    const [gridSnapMenuPos, setGridSnapMenuPos] = useState<{ x: number; y: number } | null>(null);

    const [bpmText, setBpmText] = useState(() =>
        String(Math.round(s.bpm || 120)),
    );
    const bpmDirtyRef = useRef(false);

    useEffect(() => {
        if (!bpmDirtyRef.current) {
            setBpmText(String(Math.round(s.bpm || 120)));
        }
    }, [s.bpm]);

    function commitBpm(nextText?: string) {
        const raw = (nextText ?? bpmText).trim();
        const next = Number(raw);
        bpmDirtyRef.current = false;
        if (!Number.isFinite(next)) {
            setBpmText(String(Math.round(s.bpm || 120)));
            return;
        }
        dispatch(setBpm(next));
        void dispatch(updateTransportBpm(next));
        setBpmText(String(Math.round(next)));
    }

    // Custom styles for Radix components to match Qt look
    // Note: Radix Themes handles a lot, but we might need overrides for exact pixel matching if needed.
    // For now, we use standard Radix "gray" theme which fits well.

    return (
        <Flex
            align="center"
            gap="3"
            className="h-8 bg-qt-window border-b border-qt-border px-1 text-qt-text flex-nowrap overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/* BPM & Time */}
            <Flex align="center" gap="2" className="shrink-0">
                <Text size="1" className="text-qt-text-muted">
                    {t("bpm")}:
                </Text>
                <TextField.Root
                    size="1"
                    value={bpmText}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        setBpmText(e.target.value);
                    }}
                    onBlur={() => commitBpm()}
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === "Enter") {
                            e.preventDefault();
                            commitBpm();
                            (e.currentTarget as HTMLInputElement).blur();
                        } else if (e.key === "Escape") {
                            e.preventDefault();
                            bpmDirtyRef.current = false;
                            setBpmText(String(Math.round(s.bpm || 120)));
                            (e.currentTarget as HTMLInputElement).blur();
                        }
                    }}
                    style={{
                        width: 60,
                        textAlign: "center",
                        backgroundColor: "var(--qt-base)",
                    }}
                />
                <Text size="1" className="text-qt-text-muted">
                    {t("beats_per_bar")}:
                </Text>
                <Flex align="center" gap="1">
                    <TextField.Root
                        size="1"
                        type="number"
                        value={
                            Number.isFinite(s.beats)
                                ? Math.round(s.beats).toString()
                                : "4"
                        }
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            const raw = e.target.value.trim();
                            const parsed = Number(raw);
                            if (!Number.isFinite(parsed)) return;
                            // Clamp locally to avoid sending huge values to backend
                            const clamped = Math.min(32, Math.max(1, Math.round(parsed)));
                            if (clamped === Math.round(s.beats || 0)) return;
                            void dispatch(
                                setProjectTimelineSettingsRemote({
                                    beatsPerBar: clamped,
                                    gridSize: s.grid,
                                }),
                            );
                        }}
                        style={{
                            width: 42,
                            textAlign: "center",
                            backgroundColor: "var(--qt-base)",
                        }}
                    />
                    <Text size="1" className="text-qt-text-muted">
                        / 4
                    </Text>
                </Flex>

                <Text size="1" className="text-qt-text-muted">
                    {t("grid")}:
                </Text>
                <Select.Root
                    value={s.grid}
                    size="1"
                    onValueChange={(v) => {
                        void dispatch(
                            setProjectTimelineSettingsRemote({
                                beatsPerBar: s.beats,
                                gridSize: v,
                            }),
                        );
                    }}
                >
                    <Select.Trigger
                        style={{ backgroundColor: "var(--qt-base)" }}
                    />
                    <Select.Content>
                        <Select.Group>
                            <Select.Label>{tAny("grid_note_normal")}</Select.Label>
                            <Select.Item value="1/1">1/1</Select.Item>
                            <Select.Item value="1/2">1/2</Select.Item>
                            <Select.Item value="1/4">1/4</Select.Item>
                            <Select.Item value="1/8">1/8</Select.Item>
                            <Select.Item value="1/16">1/16</Select.Item>
                            <Select.Item value="1/32">1/32</Select.Item>
                            <Select.Item value="1/64">1/64</Select.Item>
                        </Select.Group>
                        <Select.Separator />
                        <Select.Group>
                            <Select.Label>{tAny("grid_note_dotted")}</Select.Label>
                            <Select.Item value="1/2d">1/2.</Select.Item>
                            <Select.Item value="1/4d">1/4.</Select.Item>
                            <Select.Item value="1/8d">1/8.</Select.Item>
                            <Select.Item value="1/16d">1/16.</Select.Item>
                            <Select.Item value="1/32d">1/32.</Select.Item>
                            <Select.Item value="1/64d">1/64.</Select.Item>
                        </Select.Group>
                        <Select.Separator />
                        <Select.Group>
                            <Select.Label>{tAny("grid_note_triplet")}</Select.Label>
                            <Select.Item value="1/2t">1/2t</Select.Item>
                            <Select.Item value="1/4t">1/4t</Select.Item>
                            <Select.Item value="1/8t">1/8t</Select.Item>
                            <Select.Item value="1/16t">1/16t</Select.Item>
                            <Select.Item value="1/32t">1/32t</Select.Item>
                            <Select.Item value="1/64t">1/64t</Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
                <Text size="1" className="text-qt-text-muted">
                    {t("base_scale")}:
                </Text>
                <Select.Root
                    value={
                        s.project?.useCustomScale && s.project?.customScale
                            ? "__custom__"
                            : (s.project?.baseScale ?? "C")
                    }
                    size="1"
                    onValueChange={(v) => {
                        if (v === "__custom_dialog__") {
                            setCustomScaleOpen(true);
                            return;
                        }
                        if (v === "__custom__" && s.project?.customScale) {
                            dispatch(setProjectCustomScaleRemote(s.project.customScale));
                            return;
                        }
                        if ((SCALE_KEYS as readonly string[]).includes(v)) {
                            dispatch(setProjectBaseScaleRemote(v));
                        }
                    }}
                >
                    <Select.Trigger style={{ backgroundColor: "var(--qt-base)" }} />
                    <Select.Content>
                        <Select.Group>
                            {SCALE_KEYS.map((k) => (
                                <Select.Item key={k} value={k}>
                                    {SCALE_LABELS[k]}
                                </Select.Item>
                            ))}
                        </Select.Group>
                        {s.project?.customScale ? (
                            <>
                                <Select.Separator />
                                <Select.Group>
                                    <Select.Item value="__custom__">
                                        {`${tAny("custom_scale_label")}: ${s.project.customScale.name}`}
                                    </Select.Item>
                                </Select.Group>
                            </>
                        ) : null}
                        <Select.Separator />
                        <Select.Group>
                            <Select.Item value="__custom_dialog__">
                                {tAny("custom_scale_action")}
                            </Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* Transport */}
            <Flex gap="1" className="shrink-0">
                <Button
                    variant="soft"
                    color="gray"
                    size="1"
                    onClick={() => {
                        dispatch(stopAudioPlayback({ restoreAnchor: true }));
                    }}
                    title={t("action_stop")}
                >
                    <StopIcon />
                </Button>
                <IconButton
                    variant="solid"
                    size="1"
                    onClick={() => {
                        if (s.runtime.isPlaying) {
                            dispatch(stopAudioPlayback());
                            return;
                        }
                        dispatch(playOriginal());
                    }}
                    title={s.runtime.isPlaying ? tAny("action_pause") : t("action_play_out")}
                >
                    {s.runtime.isPlaying ? <PauseIcon /> : <PlayIcon />}
                </IconButton>
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* File Browser Toggle */}
            <Flex gap="1" className="shrink-0">
                <IconButton
                    size="1"
                    variant={fileBrowserVisible ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("fb_title")}
                    onClick={() => dispatch(toggleVisible())}
                >
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 3.5C2 3.22386 2.22386 3 2.5 3H5.29289L6.64645 4.35355C6.74021 4.44732 6.86739 4.5 7 4.5H12.5C12.7761 4.5 13 4.72386 13 5V11.5C13 11.7761 12.7761 12 12.5 12H2.5C2.22386 12 2 11.7761 2 11.5V3.5Z" fill="currentColor"/>
                    </svg>
                </IconButton>
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* Toolbar Toggles */}
            <Flex align="center" gap="1" className="shrink-0">
                {/* Auto Crossfade */}
                <IconButton
                    size="1"
                    variant={s.autoCrossfadeEnabled ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("auto_crossfade")}
                    tabIndex={-1}
                    onClick={() => { dispatch(toggleAutoCrossfade()); void dispatch(persistUiSettings()); }}
                >
                    {/* X icon for crossfade */}
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 12L7.5 3L13 12" stroke="currentColor" strokeWidth="1.2" fill="none"/>
                        <path d="M2 3L7.5 12L13 3" stroke="currentColor" strokeWidth="1.2" fill="none" opacity="0.5"/>
                    </svg>
                </IconButton>

                {/* Grid Snap */}
                <IconButton
                    size="1"
                    variant={s.gridSnapEnabled ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("grid_snap")}
                    tabIndex={-1}
                    onClick={() => { dispatch(toggleGridSnap()); void dispatch(persistUiSettings()); }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setGridSnapMenuPos({ x: e.clientX, y: e.clientY });
                    }}
                >
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 2V13M5.5 2V13M9 2V13M12.5 2V13" stroke="currentColor" strokeWidth="0.8" opacity="0.5"/>
                        <path d="M7.5 4L7.5 11" stroke="currentColor" strokeWidth="1.5"/>
                        <path d="M5.5 6L7.5 4L9.5 6" stroke="currentColor" strokeWidth="1"/>
                        <path d="M5.5 9L7.5 11L9.5 9" stroke="currentColor" strokeWidth="1"/>
                    </svg>
                </IconButton>


                {/* Playhead Zoom */}
                <IconButton
                    size="1"
                    variant={s.playheadZoomEnabled ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("playhead_zoom")}
                    tabIndex={-1}
                    onClick={() => { dispatch(togglePlayheadZoom()); void dispatch(persistUiSettings()); }}
                >
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7.5 2V13" stroke="currentColor" strokeWidth="1.2"/>
                        <path d="M6 3L7.5 1.5L9 3" stroke="currentColor" strokeWidth="1"/>
                        <path d="M4 7.5H2M13 7.5H11" stroke="currentColor" strokeWidth="1"/>
                        <path d="M3.5 5L2 7.5L3.5 10" stroke="currentColor" strokeWidth="0.8"/>
                        <path d="M11.5 5L13 7.5L11.5 10" stroke="currentColor" strokeWidth="0.8"/>
                    </svg>
                </IconButton>

                {/* Auto Scroll */}
                <IconButton
                    size="1"
                    variant={s.paramEditorSeekPlayheadEnabled ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("param_editor_seek_playhead")}
                    tabIndex={-1}
                    onClick={() => { dispatch(toggleParamEditorSeekPlayhead()); void dispatch(persistUiSettings()); }}
                >
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 2.5H13" stroke="currentColor" strokeWidth="0.8" opacity="0.5"/>
                        <path d="M2 12.5H13" stroke="currentColor" strokeWidth="0.8" opacity="0.5"/>
                        <path d="M7.5 3.5V11.5" stroke="currentColor" strokeWidth="1.2"/>
                        <path d="M6 4.5L7.5 3L9 4.5" stroke="currentColor" strokeWidth="1"/>
                        <path d="M7.8 8.2C8.9 8.2 9.8 9.1 9.8 10.2C9.8 11.3 8.9 12.2 7.8 12.2C6.9 12.2 6.2 11.6 6 10.8H7.8V8.2Z" fill="currentColor"/>
                    </svg>
                </IconButton>

                {/* Auto Scroll (horizontal arrows) */}
                <IconButton
                    size="1"
                    variant={s.autoScrollEnabled ? "solid" : "ghost"}
                    color="gray"
                    title={tAny("auto_scroll")}
                    tabIndex={-1}
                    onClick={() => { dispatch(toggleAutoScroll()); void dispatch(persistUiSettings()); }}
                >
                    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7.5 2V13" stroke="currentColor" strokeWidth="1.2"/>
                        <path d="M3 6L1.5 7.5L3 9" stroke="currentColor" strokeWidth="1"/>
                        <path d="M12 6L13.5 7.5L12 9" stroke="currentColor" strokeWidth="1"/>
                        <path d="M2 7.5H13" stroke="currentColor" strokeWidth="0.8" opacity="0.5"/>
                    </svg>
                </IconButton>

            </Flex>

            {/* Pitch Snap Settings Dialog */}
            {pitchSnapOpen && (
                <PitchSnapSettingsDialog
                    open={pitchSnapOpen}
                    onOpenChange={setPitchSnapOpen}
                />
            )}

            {customScaleOpen && (
                <CustomScaleDialog
                    open={customScaleOpen}
                    onOpenChange={setCustomScaleOpen}
                />
            )}

            {/* Grid Snap Context Menu */}
            {gridSnapMenuPos && (
                <GridSnapContextMenu
                    x={gridSnapMenuPos.x}
                    y={gridSnapMenuPos.y}
                    currentGrid={s.grid}
                    onSelect={(grid) => {
                        void dispatch(
                            setProjectTimelineSettingsRemote({
                                beatsPerBar: s.beats,
                                gridSize: grid,
                            }),
                        );
                        setGridSnapMenuPos(null);
                    }}
                    onClose={() => setGridSnapMenuPos(null)}
                    t={tAny}
                />
            )}
        </Flex>
    );
}

/** Grid snap note type definitions for the context menu */
const GRID_SNAP_ITEMS: Array<{ value: string; labelKey: string } | "separator"> = [
    { value: "1/1", labelKey: "grid_snap_whole" },
    { value: "1/2", labelKey: "grid_snap_half" },
    { value: "1/4", labelKey: "grid_snap_quarter" },
    { value: "1/8", labelKey: "grid_snap_8th" },
    { value: "1/16", labelKey: "grid_snap_16th" },
    { value: "1/32", labelKey: "grid_snap_32nd" },
    { value: "1/64", labelKey: "grid_snap_64th" },
    "separator",
    { value: "1/2d", labelKey: "grid_snap_dotted_half" },
    { value: "1/4d", labelKey: "grid_snap_dotted_quarter" },
    { value: "1/8d", labelKey: "grid_snap_dotted_8th" },
    { value: "1/16d", labelKey: "grid_snap_dotted_16th" },
    { value: "1/32d", labelKey: "grid_snap_dotted_32nd" },
    { value: "1/64d", labelKey: "grid_snap_dotted_64th" },
    "separator",
    { value: "1/2t", labelKey: "grid_snap_triplet_half" },
    { value: "1/4t", labelKey: "grid_snap_triplet_quarter" },
    { value: "1/8t", labelKey: "grid_snap_triplet_8th" },
    { value: "1/16t", labelKey: "grid_snap_triplet_16th" },
    { value: "1/32t", labelKey: "grid_snap_triplet_32nd" },
    { value: "1/64t", labelKey: "grid_snap_triplet_64th" },
];

function GridSnapContextMenu({
    x, y, currentGrid, onSelect, onClose, t,
}: {
    x: number; y: number; currentGrid: string;
    onSelect: (grid: string) => void; onClose: () => void;
    t: (key: string) => string;
}) {
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClick = (e: globalThis.MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose();
            }
        };
        const handleKey = (e: globalThis.KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("mousedown", handleClick, true);
        window.addEventListener("keydown", handleKey, true);
        return () => {
            window.removeEventListener("mousedown", handleClick, true);
            window.removeEventListener("keydown", handleKey, true);
        };
    }, [onClose]);

    // Clamp menu position to viewport edges
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) el.style.left = `${Math.max(0, vw - rect.width)}px`;
        if (rect.bottom > vh) el.style.top = `${Math.max(0, vh - rect.height)}px`;
    }, [x, y]);

    // Adjust position to fit within viewport
    const style: React.CSSProperties = {
        position: "fixed",
        left: x,
        top: y,
        zIndex: 10000,
        minWidth: 180,
        maxHeight: "70vh",
        overflowY: "auto",
        background: "var(--color-panel-solid)",
        border: "1px solid var(--gray-6)",
        borderRadius: 6,
        padding: "4px 0",
        boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    };

    return (
        <div ref={menuRef} style={style}>
            {GRID_SNAP_ITEMS.map((item, i) => {
                if (item === "separator") {
                    return <div key={`sep-${i}`} style={{ height: 1, background: "var(--gray-6)", margin: "4px 0" }} />;
                }
                const isActive = item.value === currentGrid;
                return (
                    <div
                        key={item.value}
                        onClick={() => onSelect(item.value)}
                        style={{
                            padding: "5px 12px",
                            cursor: "pointer",
                            fontSize: 13,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            background: isActive ? "var(--accent-3)" : "transparent",
                            color: isActive ? "var(--accent-11)" : "inherit",
                        }}
                        onMouseEnter={(e) => {
                            if (!isActive) (e.currentTarget as HTMLDivElement).style.background = "var(--gray-3)";
                        }}
                        onMouseLeave={(e) => {
                            (e.currentTarget as HTMLDivElement).style.background = isActive ? "var(--accent-3)" : "transparent";
                        }}
                    >
                        <span>{t(item.labelKey)}</span>
                        {isActive && <span style={{ marginLeft: 8 }}>✓</span>}
                    </div>
                );
            })}
        </div>
    );
}
