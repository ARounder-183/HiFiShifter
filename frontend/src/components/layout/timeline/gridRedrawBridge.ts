/**
 * Bridge used by PianoRollPanel to invoke BackgroundGrid's imperative draw
 * function without storing it as a custom property on the DOM node.
 *
 * The grid layer is a React-rendered element shared between the timeline and
 * the parameter editor. Keeping the redraw callback in a WeakMap avoids
 * mutating React-managed DOM nodes and keeps the callback isolated from the
 * component lifecycle.
 */

type GridRedrawHandler = (scrollLeft: number) => void;

const gridRedrawHandlers = new WeakMap<HTMLElement, GridRedrawHandler>();

export function setGridRedrawHandler(element: HTMLElement, handler: GridRedrawHandler): void {
    gridRedrawHandlers.set(element, handler);
}

export function clearGridRedrawHandler(element: HTMLElement): void {
    gridRedrawHandlers.delete(element);
}

export function invokeGridRedrawHandler(
    element: HTMLElement | null | undefined,
    scrollLeft: number,
): void {
    if (!element) return;
    gridRedrawHandlers.get(element)?.(scrollLeft);
}
