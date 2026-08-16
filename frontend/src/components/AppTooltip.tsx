import { useEffect, useState, type PropsWithChildren } from "react";
import { createPortal } from "react-dom";

export type AppTooltipPosition = {
    x: number;
    y: number;
};

function clampTooltipPosition(position: AppTooltipPosition): AppTooltipPosition {
    return {
        x: Math.min(position.x + 14, Math.max(8, window.innerWidth - 220)),
        y: Math.min(position.y + 18, Math.max(8, window.innerHeight - 56)),
    };
}

export function AppTooltipBubble({
    text,
    position,
}: {
    text: string;
    position: AppTooltipPosition | null;
}) {
    if (!position || !text) return null;

    const clamped = clampTooltipPosition(position);
    return createPortal(
        <div className="app-tooltip" role="tooltip" style={{ left: clamped.x, top: clamped.y }}>
            {text}
        </div>,
        document.body,
    );
}

export function AppTooltipProvider({ children }: PropsWithChildren) {
    const [tooltip, setTooltip] = useState<{
        text: string;
        position: AppTooltipPosition;
    } | null>(null);

    useEffect(() => {
        let currentElement: Element | null = null;

        const updateFromEvent = (event: PointerEvent) => {
            const target = event.target instanceof Element ? event.target : null;
            const element = target?.closest?.("[data-tooltip]") ?? null;

            if (element !== currentElement) {
                currentElement = element;
                if (!element) {
                    setTooltip(null);
                    return;
                }
            }
            if (!element) return;

            const text = element.getAttribute("data-tooltip") || "";
            if (!text) {
                setTooltip(null);
                return;
            }

            setTooltip({
                text,
                position: {
                    x: event.clientX,
                    y: event.clientY,
                },
            });
        };

        const onPointerOver = (event: PointerEvent) => updateFromEvent(event);

        const onPointerMove = (event: PointerEvent) => {
            if (!currentElement) return;
            const text = currentElement.getAttribute("data-tooltip") || "";
            if (!text) {
                setTooltip(null);
                return;
            }
            setTooltip({
                text,
                position: {
                    x: event.clientX,
                    y: event.clientY,
                },
            });
        };

        const onPointerOut = (event: PointerEvent) => {
            const target = event.target instanceof Element ? event.target : null;
            const relatedTarget =
                event.relatedTarget instanceof Element ? event.relatedTarget : null;
            if (
                target?.closest?.("[data-tooltip]") &&
                !relatedTarget?.closest?.("[data-tooltip]")
            ) {
                currentElement = null;
                setTooltip(null);
            }
        };

        document.addEventListener("pointerover", onPointerOver);
        document.addEventListener("pointermove", onPointerMove);
        document.addEventListener("pointerout", onPointerOut);
        return () => {
            document.removeEventListener("pointerover", onPointerOver);
            document.removeEventListener("pointermove", onPointerMove);
            document.removeEventListener("pointerout", onPointerOut);
        };
    }, []);

    return (
        <>
            {children}
            <AppTooltipBubble text={tooltip?.text ?? ""} position={tooltip?.position ?? null} />
        </>
    );
}
