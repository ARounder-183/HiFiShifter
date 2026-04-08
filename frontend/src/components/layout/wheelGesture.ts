const WHEEL_AXIS_EPSILON = 0.5;

export type ParamEditorWheelAction =
    | "horizontal-scroll"
    | "vertical-pan"
    | "vertical-zoom"
    | "horizontal-zoom";

export type TimelineWheelAction =
    | "horizontal-scroll"
    | "vertical-scroll"
    | "vertical-zoom"
    | "horizontal-zoom"
    | "native";

export function getWheelGestureAxis(input: {
    deltaX: number;
    deltaY: number;
}): "horizontal" | "vertical" {
    const absX = Math.abs(input.deltaX);
    const absY = Math.abs(input.deltaY);

    if (absX > WHEEL_AXIS_EPSILON && absX > absY) {
        return "horizontal";
    }

    return "vertical";
}

export function getParamEditorWheelAction(input: {
    deltaX: number;
    deltaY: number;
    horizontalScrollRequested: boolean;
    verticalPanRequested: boolean;
    verticalZoomRequested: boolean;
    horizontalZoomRequested: boolean;
}): ParamEditorWheelAction {
    if (input.horizontalScrollRequested) {
        return "horizontal-scroll";
    }

    if (input.verticalPanRequested) {
        return "vertical-pan";
    }

    if (input.verticalZoomRequested) {
        return "vertical-zoom";
    }

    if (input.horizontalZoomRequested) {
        return "horizontal-zoom";
    }

    return getWheelGestureAxis(input) === "horizontal" ? "horizontal-scroll" : "horizontal-zoom";
}

export function getTimelineWheelAction(input: {
    deltaX: number;
    deltaY: number;
    horizontalScrollRequested: boolean;
    verticalScrollRequested: boolean;
    verticalZoomRequested: boolean;
    horizontalZoomRequested: boolean;
}): TimelineWheelAction {
    if (input.horizontalScrollRequested) {
        return "horizontal-scroll";
    }

    if (input.verticalScrollRequested) {
        return "vertical-scroll";
    }

    if (input.verticalZoomRequested) {
        return "vertical-zoom";
    }

    if (input.horizontalZoomRequested) {
        return "horizontal-zoom";
    }

    return getWheelGestureAxis(input) === "horizontal" ? "horizontal-scroll" : "native";
}
