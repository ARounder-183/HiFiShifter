export type FineAxisDragState = {
    raw: number;
    adjusted: number;
    fineActive: boolean;
};

const FINE_AXIS_DRAG_SCALE = 0.2;
const FINE_AXIS_DRAG_PULLBACK_RATIO = 0.35;

export function advanceFineAxisDrag(
    state: FineAxisDragState,
    nextRaw: number,
    fineActive: boolean,
): number {
    const delta = nextRaw - state.raw;
    if (fineActive && !state.fineActive) {
        state.adjusted += delta * (1 - FINE_AXIS_DRAG_PULLBACK_RATIO);
    } else {
        const scale = fineActive ? FINE_AXIS_DRAG_SCALE : 1;
        state.adjusted += delta * scale;
    }
    state.raw = nextRaw;
    state.fineActive = fineActive;
    return state.adjusted;
}
