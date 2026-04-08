import { useCallback, useEffect, useRef } from "react";

import { paramsApi } from "../../../services/api";

import type { ParamName, ParamViewSegment, StrokeMode, StrokePoint } from "./types";
export function useLiveParamEditing(args: {
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;

    paramView: ParamViewSegment | null;
    setParamView: (next: ParamViewSegment | null) => void;

    bumpRefreshToken: () => void;
    invalidate: () => void;
}) {
    const {
        rootTrackId,
        editParam,
        pitchEnabled,
        paramView,
        setParamView,
        bumpRefreshToken,
        invalidate,
    } = args;

    const liveEditOverrideRef = useRef<{ key: string; edit: number[] } | null>(null);

    useEffect(() => {
        if (!paramView) {
            liveEditOverrideRef.current = null;
            return;
        }
        if (liveEditOverrideRef.current?.key !== paramView.key) {
            liveEditOverrideRef.current = null;
        }
    }, [paramView?.key]);

    const ensureLiveEditBase = useCallback((pv: ParamViewSegment) => {
        const cur = liveEditOverrideRef.current;
        if (cur && cur.key === pv.key) return;
        liveEditOverrideRef.current = { key: pv.key, edit: pv.edit.slice() };
    }, []);

    const applyDenseToLiveEdit = useCallback(
        (
            pv: ParamViewSegment,
            denseStartFrame: number,
            dense: number[] | null,
            minF: number,
            maxF: number,
            mode: StrokeMode,
        ) => {
            ensureLiveEditBase(pv);
            const cur = liveEditOverrideRef.current;
            if (!cur || cur.key !== pv.key) return;
            const nextEdit = cur.edit.slice();

            const start = pv.startFrame;
            const step = pv.stride;
            for (let i = 0; i < nextEdit.length; i += 1) {
                const f = start + i * step;
                if (f < minF || f > maxF) continue;
                if (mode === "restore") {
                    nextEdit[i] = pv.orig[i] ?? nextEdit[i];
                } else if (dense) {
                    const j = f - denseStartFrame;
                    if (j >= 0 && j < dense.length) nextEdit[i] = dense[j] ?? nextEdit[i];
                }
            }

            liveEditOverrideRef.current = { key: pv.key, edit: nextEdit };
        },
        [ensureLiveEditBase],
    );

    const commitStroke = useCallback(
        async (points: StrokePoint[], mode: StrokeMode) => {
            const trackId = rootTrackId;
            if (!trackId) return;
            if (points.length < 1) return;
            if (editParam === "pitch" && !pitchEnabled) return;

            // IMPORTANT:
            // - During pointer-move, we update the live preview by applying each segment
            //   in the *time order* of the stroke (later segments overwrite earlier ones).
            // - If we sort points by frame here, that overwrite order can change when the
            //   user slightly backtracks in X, causing the committed curve to differ from
            //   what was previewed (often perceived as "spikes" / "glitches").
            // So: keep stroke order, only de-dupe consecutive same-frame samples.
            const ordered = points.filter(
                (p) => Number.isFinite(p.frame) && Number.isFinite(p.value),
            );
            const uniq: StrokePoint[] = [];
            for (const p of ordered) {
                const f = Math.max(0, Math.floor(p.frame));
                const v = p.value;
                const last = uniq[uniq.length - 1];
                if (last && last.frame === f) {
                    last.value = v;
                } else {
                    uniq.push({ frame: f, value: v });
                }
            }
            if (uniq.length < 1) return;

            let minF = Number.POSITIVE_INFINITY;
            let maxF = 0;
            for (const p of uniq) {
                minF = Math.min(minF, p.frame);
                maxF = Math.max(maxF, p.frame);
            }
            minF = Math.max(0, Math.floor(minF));
            maxF = Math.max(minF, Math.floor(maxF));

            const pv = paramView;

            function applyToParamViewDense(denseStartFrame: number, dense: number[] | null) {
                if (!pv) return;
                if (pv.stride <= 0) return;
                const start = pv.startFrame;
                const step = pv.stride;
                const nextEdit = pv.edit.slice();
                for (let i = 0; i < nextEdit.length; i += 1) {
                    const f = start + i * step;
                    if (f < minF || f > maxF) continue;
                    if (mode === "restore") {
                        nextEdit[i] = pv.orig[i] ?? nextEdit[i];
                    } else if (dense) {
                        const j = f - denseStartFrame;
                        if (j >= 0 && j < dense.length) nextEdit[i] = dense[j] ?? nextEdit[i];
                    }
                }
                setParamView({ ...pv, edit: nextEdit });
            }

            if (mode === "restore") {
                applyToParamViewDense(minF, null);
                liveEditOverrideRef.current = null;
                invalidate();
                await paramsApi.restoreParamFrames(trackId, editParam, minF, maxF - minF + 1, true);
                bumpRefreshToken();
                return;
            }

            const len = maxF - minF + 1;
            const out = new Array<number>(len);

            // Prefer committing the exact dense values that were shown in the live preview.
            // This keeps the committed curve identical to what the user saw.
            const pvEdit = (() => {
                const pvNow = paramView;
                if (!pvNow) return null;
                const live = liveEditOverrideRef.current;
                if (live && live.key === pvNow.key) return live.edit;
                return pvNow.edit;
            })();
            const pvStart = paramView?.startFrame ?? 0;
            const pvStride = paramView?.stride ?? 1;

            const canSliceFromPv =
                Boolean(paramView) && pvStride === 1 && Array.isArray(pvEdit) && pvEdit.length > 0;

            if (canSliceFromPv) {
                for (let f = minF; f <= maxF; f += 1) {
                    const i = f - pvStart;
                    out[f - minF] =
                        i >= 0 && i < (pvEdit as number[]).length
                            ? ((pvEdit as number[])[i] ?? 0)
                            : 0;
                }
            } else {
                // Fallback: replay the stroke in time order onto a dense buffer.
                for (let i = 0; i < len; i += 1) out[i] = uniq[0].value;
                for (let sIdx = 0; sIdx < uniq.length - 1; sIdx += 1) {
                    const a = uniq[sIdx];
                    const b = uniq[sIdx + 1];
                    const minSeg = Math.min(a.frame, b.frame);
                    const maxSeg = Math.max(a.frame, b.frame);
                    const denom = b.frame - a.frame;
                    for (let f = minSeg; f <= maxSeg; f += 1) {
                        if (f < minF || f > maxF) continue;
                        const t = denom === 0 ? 1 : (f - a.frame) / denom;
                        out[f - minF] = a.value + (b.value - a.value) * t;
                    }
                }
            }

            await paramsApi.setParamFrames(trackId, editParam, minF, out, true);
            applyToParamViewDense(minF, out);
            liveEditOverrideRef.current = null;
            invalidate();
            bumpRefreshToken();
        },
        [
            rootTrackId,
            editParam,
            pitchEnabled,
            paramView,
            setParamView,
            invalidate,
            bumpRefreshToken,
        ],
    );

    return {
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        commitStroke,
    };
}
