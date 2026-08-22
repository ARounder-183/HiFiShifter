import React from "react";
import { useAppDispatch, useAppSelector } from "../../../../app/hooks";
import type { ClipInfo } from "../../../../features/session/sessionTypes";
import { useI18n } from "../../../../i18n/I18nProvider";
import { openClipFormantToolWindow } from "../../../../features/session/sessionSlice";
import { getClipFormantButtonStyle } from "./clipFormantButtonStyle";

export const ClipFormantButton: React.FC<{
    clip: ClipInfo;
    hidden?: boolean;
    opacity?: number;
    width: number;
    height: number;
    baseBackgroundColor: string;
    baseBorderColor: string;
    baseTextColor: string;
}> = ({
    clip,
    hidden = false,
    opacity = 1,
    width,
    height,
    baseBackgroundColor,
    baseBorderColor,
    baseTextColor,
}) => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const status = useAppSelector((state) => state.session.clipFormantStatus[clip.id] ?? "ready");
    const buttonRef = React.useRef<HTMLButtonElement | null>(null);

    const accentStyle = React.useMemo(
        () =>
            getClipFormantButtonStyle({
                baseBackgroundColor,
                baseBorderColor,
                baseTextColor,
                enabled: Boolean(clip.formantMorph?.enabled),
                status,
            }),
        [baseBackgroundColor, baseBorderColor, baseTextColor, clip.formantMorph?.enabled, status],
    );

    if (hidden) return null;

    return (
        <button
            ref={buttonRef}
            className="rounded flex items-center justify-center border transition-all text-[10px] font-bold"
            data-tooltip={t("clip_formant_title")}
            style={{
                opacity,
                width,
                height,
                ...accentStyle,
            }}
            onPointerDown={(event) => {
                event.preventDefault();
                event.stopPropagation();
            }}
            onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                const rect = buttonRef.current?.getBoundingClientRect();
                dispatch(
                    openClipFormantToolWindow({
                        clipId: clip.id,
                        anchor: {
                            x: Math.round((rect?.right ?? event.clientX) + 12),
                            y: Math.round(rect?.top ?? event.clientY),
                        },
                    }),
                );
            }}
        >
            F
        </button>
    );
};
