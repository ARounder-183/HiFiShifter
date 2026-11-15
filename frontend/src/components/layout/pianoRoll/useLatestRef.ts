import { useLayoutEffect, useRef, type MutableRefObject } from "react";

/**
 * Keep a ref in sync with the latest render value without writing it during
 * render. Event handlers and async callbacks run after layout effects, so they
 * always observe the committed value.
 */
export function useLatestRef<T>(value: T): MutableRefObject<T> {
    const ref = useRef(value);
    useLayoutEffect(() => {
        ref.current = value;
    }, [value]);
    return ref;
}

/** Same contract for a ref that already exists for declaration-order reasons. */
export function useLatestRefValue<T>(ref: MutableRefObject<T>, value: T): void {
    useLayoutEffect(() => {
        ref.current = value;
    }, [ref, value]);
}
