import React from "react";
import { Button, Flex, Slider, Text } from "@radix-ui/themes";
import type { ClipInfo, ClipFormantMorph } from "../../../../features/session/sessionTypes";
import { useI18n } from "../../../../i18n/I18nProvider";
import { VowelChart } from "./VowelChart";
import { useClipFormantEditor } from "./useClipFormantEditor";
import {
    CLIP_FORMANT_ACTIVE_ATTR,
    CLIP_FORMANT_FOCUS_WINDOW,
    shouldSuppressFormantToolSpaceDefault,
} from "./clipFormantInteractionGuards";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

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
    const { draft, updateDraft, flush } = useClipFormantEditor({
        clipId: clip.id,
        value: clip.formantMorph,
        onCommit,
    });
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

        window.addEventListener("pointermove", onMovePointer, true);
        window.addEventListener("pointerup", onEndPointer, true);
        window.addEventListener("pointercancel", onEndPointer, true);
        return () => {
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

    React.useEffect(() => {
        document.body.setAttribute(CLIP_FORMANT_ACTIVE_ATTR, "true");
        document.body.setAttribute("data-hs-focus-window", CLIP_FORMANT_FOCUS_WINDOW);
        return () => {
            document.body.removeAttribute(CLIP_FORMANT_ACTIVE_ATTR);
            if (document.body.getAttribute("data-hs-focus-window") === CLIP_FORMANT_FOCUS_WINDOW) {
                document.body.removeAttribute("data-hs-focus-window");
            }
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
            onFocus={() => {
                document.body.setAttribute("data-hs-focus-window", CLIP_FORMANT_FOCUS_WINDOW);
            }}
            onPointerDown={(event) => {
                document.body.setAttribute("data-hs-focus-window", CLIP_FORMANT_FOCUS_WINDOW);
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
                <label className="flex items-center gap-2 text-sm text-qt-text">
                    <input
                        type="checkbox"
                        checked={draft.enabled}
                        onChange={(event) => updateDraft({ enabled: event.target.checked })}
                    />
                    <span>{t("clip_formant_enabled")}</span>
                </label>

                <div className="rounded-lg border border-qt-border bg-qt-panel p-2">
                    <VowelChart
                        targetF1Hz={draft.targetF1Hz}
                        targetF2Hz={draft.targetF2Hz}
                        disabled={!draft.enabled}
                        onChange={updateDraft}
                    />
                    <Flex justify="between" mt="2">
                        <Text size="1" color="gray">
                            F1 {Math.round(draft.targetF1Hz)} Hz
                        </Text>
                        <Text size="1" color="gray">
                            F2 {Math.round(draft.targetF2Hz)} Hz
                        </Text>
                    </Flex>
                </div>

                <div className="rounded-lg border border-qt-border bg-qt-panel px-3 py-2">
                    <Flex align="center" justify="between" mb="2">
                        <Text size="2">{t("clip_formant_strength")}</Text>
                        <Text size="1" color="gray">
                            {strengthPercent}%
                        </Text>
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
            </Flex>
        </div>
    );
};
