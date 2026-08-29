export type TimelineViewportSnapshot = {
    scrollLeft: number;
    pxPerSec: number;
    viewportWidth: number;
};

export function shouldDispatchTimelineViewport(args: {
    previous: TimelineViewportSnapshot | null;
    next: TimelineViewportSnapshot;
}): boolean {
    if (args.previous == null) {
        return true;
    }

    return (
        Math.abs(args.previous.scrollLeft - args.next.scrollLeft) > 1e-6 ||
        Math.abs(args.previous.pxPerSec - args.next.pxPerSec) > 1e-9 ||
        Math.abs(args.previous.viewportWidth - args.next.viewportWidth) > 1e-6
    );
}
