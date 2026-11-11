import { useEffect, useState } from "react";
import { pianoRollViewportBus } from "../pianoRollViewportBus";

export interface PianoRollSnapGuideViewport {
    readonly scrollLeftPx: number;
    readonly pxPerSec: number;
}

/**
 * Live projection for the snap-guide layer. React's quantized scrollLeft is
 * deliberately not the source: the guide must move in the same paint frame as
 * the canvas, including edge auto-scroll.
 */
export function usePianoRollSnapGuideViewport(): PianoRollSnapGuideViewport {
    const [viewport, setViewport] = useState<PianoRollSnapGuideViewport>(() => {
        const axis = pianoRollViewportBus.getAxis();
        return { scrollLeftPx: axis.scrollLeftPx, pxPerSec: axis.pxPerSec };
    });
    useEffect(
        () =>
            pianoRollViewportBus.subscribe((scrollLeftPx, pxPerSec) => {
                setViewport((previous) =>
                    previous.scrollLeftPx === scrollLeftPx && previous.pxPerSec === pxPerSec
                        ? previous
                        : { scrollLeftPx, pxPerSec },
                );
            }),
        [],
    );
    return viewport;
}
