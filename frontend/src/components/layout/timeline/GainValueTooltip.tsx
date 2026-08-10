import { createPortal } from "react-dom";

export type GainTooltipPosition = {
    x: number;
    y: number;
};

export function GainValueTooltip({
    text,
    position,
}: {
    text: string;
    position: GainTooltipPosition | null;
}) {
    if (!position) return null;

    return createPortal(
        <div
            role="tooltip"
            style={{
                position: "fixed",
                left: Math.min(position.x + 14, Math.max(8, window.innerWidth - 220)),
                top: Math.min(position.y + 18, Math.max(8, window.innerHeight - 56)),
                zIndex: 2147483647,
                pointerEvents: "none",
                backgroundColor: "var(--qt-panel)",
                color: "var(--qt-text)",
                padding: "4px 8px",
                borderRadius: 6,
                fontSize: 12,
                lineHeight: "16px",
                border: "1px solid var(--qt-border)",
                boxShadow: "0 2px 10px var(--qt-overlay)",
                whiteSpace: "nowrap",
            }}
        >
            {text}
        </div>,
        document.body,
    );
}
