import React from "react";
import { Button, Flex, Slider, Switch, Text } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import type {
    ClipFormantAnalysisState,
    ClipInfo,
    ClipFormantMorph,
} from "../../../../features/session/sessionTypes";
import { setClipFormantAnalysis } from "../../../../features/session/sessionSlice";
import { timelineApi } from "../../../../services/api/timeline";
import { useI18n } from "../../../../i18n/I18nProvider";
import { VowelChart } from "./VowelChart";
import { useClipFormantEditor } from "./useClipFormantEditor";
import { registerDragAbort } from "../gestureFocusGuard";
import {
    CLIP_FORMANT_ACTIVE_ATTR,
    shouldSuppressFormantToolSpaceDefault,
} from "./clipFormantInteractionGuards";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** 浊音占比低于该阈值时提示"素材不适合做共振峰调整"。 */
const VOICED_RATIO_HINT_THRESHOLD = 0.15;

export const ClipFormantToolWindow: React.FC<{
    clip: ClipInfo;
    status: "ready" | "rebuilding" | "failed";
    x: number;
    y: number;
    onCommit: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
    onMove: (x: number, y: number) => void;
    onClose: () => void;
}> = ({ clip, status, x, y, onCommit, onMove, onClose }) => {
    const { t } = useI18n();
    const dispatch = useAppDispatch();
    const analysis = useAppSelector(
        (state) => state.session.clipFormantAnalysis[clip.id] as ClipFormantAnalysisState | undefined,
    );
    const { draft, updateDraft, flush } = useClipFormantEditor({
        clipId: clip.id,
        value: clip.formantMorph,
        onCommit,
    });
    // Bypass：仅试听旁通，直接走 onCommit(checkpoint:false)（不产生撤销步，
    // 不触碰编辑器 dirtyRef 语义），松开时恢复草稿值。
    const bypassRef = React.useRef(false);
    const [position, setPosition] = React.useState({ x, y });
    const positionRef = React.useRef(position);
    const dragOffsetRef = React.useRef<{ dx: number; dy: number } | null>(null);
    const draggingRef = React.useRef(false);

    React.useEffect(() => {
        if (!draggingRef.current) {
            setPosition({ x, y });
        }
    }, [x, y]);

    React.useEffect(() => {
        positionRef.current = position;
    }, [position]);

    React.useEffect(() => {
        const onMovePointer = (event: PointerEvent) => {
            if (!draggingRef.current || !dragOffsetRef.current) return;
            const nextX = clamp(
                event.clientX - dragOffsetRef.current.dx,
                8,
                window.innerWidth - 72,
            );
            const nextY = clamp(
                event.clientY - dragOffsetRef.current.dy,
                8,
                window.innerHeight - 48,
            );
            setPosition({ x: nextX, y: nextY });
        };

        const onEndPointer = () => {
            if (!draggingRef.current) return;
            draggingRef.current = false;
            dragOffsetRef.current = null;
            onMove(positionRef.current.x, positionRef.current.y);
        };

        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur 时
        // 以最后一次位置收尾（与 pointerup 语义一致）。
        const unregisterAbort = registerDragAbort(onEndPointer);

        window.addEventListener("pointermove", onMovePointer, true);
        window.addEventListener("pointerup", onEndPointer, true);
        window.addEventListener("pointercancel", onEndPointer, true);
        return () => {
            unregisterAbort();
            window.removeEventListener("pointermove", onMovePointer, true);
            window.removeEventListener("pointerup", onEndPointer, true);
            window.removeEventListener("pointercancel", onEndPointer, true);
        };
    }, [onMove]);

    React.useEffect(() => {
        return () => {
            flush();
        };
    }, [flush]);

    // 窗口打开 / 切换 clip 时拉取一次源共振峰分析（后端有缓存，重复请求廉价）。
    // 与 clip 消费窗口相关的键由后端管理，前端无需感知窗口变化。
    React.useEffect(() => {
        let cancelled = false;
        dispatch(
            setClipFormantAnalysis({
                clipId: clip.id,
                analysis: {
                    status: "loading",
                    sourceF1Hz: 0,
                    sourceF2Hz: 0,
                    track: [],
                    voicedRatio: 0,
                    message: null,
                },
            }),
        );
        timelineApi
            .analyzeClipFormants(clip.id)
            .then((result) => {
                if (cancelled) return;
                dispatch(
                    setClipFormantAnalysis({
                        clipId: clip.id,
                        analysis: {
                            status: "ready",
                            sourceF1Hz: result.sourceF1Hz,
                            sourceF2Hz: result.sourceF2Hz,
                            track: result.track,
                            voicedRatio: result.voicedRatio,
                            message: result.message,
                        },
                    }),
                );
            })
            .catch(() => {
                if (cancelled) return;
                dispatch(
                    setClipFormantAnalysis({
                        clipId: clip.id,
                        analysis: {
                            status: "failed",
                            sourceF1Hz: 0,
                            sourceF2Hz: 0,
                            track: [],
                            voicedRatio: 0,
                            message: null,
                        },
                    }),
                );
            });
        return () => {
            cancelled = true;
        };
    }, [clip.id, dispatch]);

    /** 试听旁通：按下临时禁用（checkpoint:false 不产生撤销步），松开恢复。 */
    const bypassHandlers = {
        onPointerDown: (event: React.PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            if (bypassRef.current) return;
            bypassRef.current = true;
            onCommit(clip.id, { ...draft, enabled: false }, false);
        },
        onPointerUp: (event: React.PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            if (!bypassRef.current) return;
            bypassRef.current = false;
            onCommit(clip.id, draft, false);
        },
        onPointerCancel: () => {
            if (!bypassRef.current) return;
            bypassRef.current = false;
            onCommit(clip.id, draft, false);
        },
    };

    React.useEffect(() => {
        document.body.setAttribute(CLIP_FORMANT_ACTIVE_ATTR, "true");
        return () => {
            document.body.removeAttribute(CLIP_FORMANT_ACTIVE_ATTR);
        };
    }, []);

    const strengthPercent = Math.round(draft.strength * 100);
    const statusText = !draft.enabled
        ? t("clip_formant_status_disabled")
        : status === "rebuilding"
          ? t("clip_formant_status_rebuilding")
          : status === "failed"
            ? t("clip_formant_status_failed")
            : t("clip_formant_status_ready");
    const statusClassName =
        status === "failed"
            ? "text-qt-danger-text"
            : status === "rebuilding"
              ? "text-qt-warning-text"
              : "text-qt-text-muted";

    return (
        <div
            className="fixed z-[260] rounded-xl border border-qt-border bg-qt-window text-qt-text shadow-2xl"
            style={{
                left: position.x,
                top: position.y,
                width: 468,
                userSelect: "none",
                WebkitUserSelect: "none",
            }}
            tabIndex={0}
            onPointerDown={(event) => {
                event.stopPropagation();
            }}
            onMouseDown={(event) => {
                event.stopPropagation();
            }}
            onClick={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.stopPropagation()}
            onKeyDownCapture={(event) => {
                if (
                    shouldSuppressFormantToolSpaceDefault({
                        code: event.code,
                        key: event.key,
                    })
                ) {
                    event.preventDefault();
                }
            }}
        >
            <Flex
                align="center"
                justify="between"
                className="cursor-grab border-b border-qt-border bg-qt-panel px-3 py-2 active:cursor-grabbing"
                onPointerDown={(event) => {
                    if ((event.target as HTMLElement | null)?.closest("button")) return;
                    event.preventDefault();
                    event.stopPropagation();
                    draggingRef.current = true;
                    dragOffsetRef.current = {
                        dx: event.clientX - position.x,
                        dy: event.clientY - position.y,
                    };
                }}
            >
                <Flex align="center" gap="2" className="min-w-0">
                    <div
                        className={`h-2.5 w-2.5 rounded-full ${status === "failed" ? "bg-qt-danger-border" : status === "rebuilding" ? "bg-qt-warning-border" : draft.enabled ? "bg-qt-highlight" : "bg-qt-border"}`}
                    />
                    <Text size="2" weight="medium">
                        {t("clip_formant_title")}
                    </Text>
                    <Text size="1" color="gray" className="truncate">
                        {clip.name}
                    </Text>
                </Flex>
                <Button size="1" variant="ghost" color="gray" onClick={onClose}>
                    {t("close")}
                </Button>
            </Flex>

            <Flex direction="column" gap="3" className="bg-qt-base px-3 py-3">
                <Flex align="center" justify="between">
                    <Flex align="center" gap="2">
                        <Switch
                            checked={draft.enabled}
                            disabled={status === "failed"}
                            onCheckedChange={(checked) => updateDraft({ enabled: checked })}
                        />
                        <Text size="2">{t("clip_formant_enabled")}</Text>
                    </Flex>
                    <Button
                        size="1"
                        variant="soft"
                        color="gray"
                        disabled={!draft.enabled}
                        {...bypassHandlers}
                    >
                        {t("clip_formant_bypass")}
                    </Button>
                </Flex>

                <div className="rounded-lg border border-qt-border bg-qt-panel p-2">
                    <VowelChart
                        targetF1Hz={draft.targetF1Hz}
                        targetF2Hz={draft.targetF2Hz}
                        disabled={!draft.enabled}
                        onChange={updateDraft}
                        sourceF1Hz={analysis?.status === "ready" ? analysis.sourceF1Hz : undefined}
                        sourceF2Hz={analysis?.status === "ready" ? analysis.sourceF2Hz : undefined}
                        track={analysis?.status === "ready" ? analysis.track : undefined}
                        onPickVowel={(f1, f2) =>
                            updateDraft({
                                targetF1Hz: Math.round(f1),
                                targetF2Hz: Math.round(f2),
                            })
                        }
                    />
                    <Flex justify="between" mt="2">
                        <Text size="1" color="gray">
                            {t("clip_formant_source")}: F1{" "}
                            {analysis?.status === "ready" && analysis.sourceF1Hz > 0
                                ? Math.round(analysis.sourceF1Hz)
                                : "—"}{" "}
                            / F2{" "}
                            {analysis?.status === "ready" && analysis.sourceF2Hz > 0
                                ? Math.round(analysis.sourceF2Hz)
                                : "—"}{" "}
                            Hz
                        </Text>
                        <Text size="1" color="gray">
                            {t("clip_formant_target")}: F1 {Math.round(draft.targetF1Hz)} / F2{" "}
                            {Math.round(draft.targetF2Hz)} Hz
                        </Text>
                    </Flex>
                </div>

                <div className="rounded-lg border border-qt-border bg-qt-panel px-3 py-2">
                    <Flex align="center" justify="between" mb="2">
                        <Text size="2">{t("clip_formant_strength")}</Text>
                        <input
                            type="number"
                            min={0}
                            max={100}
                            value={strengthPercent}
                            disabled={!draft.enabled}
                            onChange={(event) => {
                                const next = Number(event.target.value);
                                if (!Number.isFinite(next)) return;
                                updateDraft({ strength: clamp(next / 100, 0, 1) });
                            }}
                            className="w-14 rounded border border-qt-border bg-qt-window px-1 py-0.5 text-right text-xs text-qt-text"
                        />
                    </Flex>
                    <Slider
                        value={[strengthPercent]}
                        min={0}
                        max={100}
                        disabled={!draft.enabled}
                        onValueChange={(nextValue) =>
                            updateDraft({
                                strength: Math.max(
                                    0,
                                    Math.min(1, Number(nextValue[0] ?? strengthPercent) / 100),
                                ),
                            })
                        }
                    />
                </div>

                <Text size="1" className={statusClassName}>
                    {statusText}
                </Text>
                {analysis?.status === "ready" &&
                (analysis.message === "no_voiced_frames" ||
                    analysis.voicedRatio < VOICED_RATIO_HINT_THRESHOLD) ? (
                    <Text size="1" className="text-qt-warning-text">
                        {t("clip_formant_no_voiced")}
                    </Text>
                ) : null}
            </Flex>
        </div>
    );
};
