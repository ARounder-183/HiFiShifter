import { useCallback, useRef, useState } from "react";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";

/**
 * Parameter-block state with one synchronous writer.
 *
 * Commands need to read the edit set produced by the previous command in the
 * same tick, while React state may still be a render behind. Writing the ref
 * from render (the old approach) can also put a discarded render's stale state
 * back into the ref. This hook owns both values: setState updates the ref
 * first, then schedules the React render.
 */
export function useParamBlockState(initial: ParamBlockState = EMPTY_PARAM_BLOCK_STATE) {
    const [state, setReactState] = useState(initial);
    const stateRef = useRef(initial);

    const setState = useCallback(
        (next: ParamBlockState | ((prev: ParamBlockState) => ParamBlockState)) => {
            const value =
                typeof next === "function"
                    ? (next as (prev: ParamBlockState) => ParamBlockState)(stateRef.current)
                    : next;
            if (value === stateRef.current) return;
            stateRef.current = value;
            setReactState(value);
        },
        [],
    );

    return { state, stateRef, setState };
}
