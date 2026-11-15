import { useCallback } from "react";

import { valueToY as valueToYImpl, yToValue as yToValueImpl } from "./sceneBuilders";
import type { ModelDeps } from "./modelDeps";
import type { ParamName } from "../types";

type ProjectionRefs = Pick<
    ModelDeps,
    "paramAxisDirectionsRef" | "paramViewsRef" | "pitchViewRef" | "processorParamsRef"
>;

/** 参数值 ↔ 视口 y 的唯一投影入口；方向设置变化由 ref 直接生效。 */
export function usePianoRollProjection(refs: ProjectionRefs): {
    valueToY: (param: ParamName, v: number, h: number) => number;
    yToValue: (param: ParamName, y: number, h: number) => number;
} {
    const { paramAxisDirectionsRef, paramViewsRef, pitchViewRef, processorParamsRef } = refs;
    const valueToY = useCallback(
        (param: ParamName, v: number, h: number) =>
            valueToYImpl(
                { paramAxisDirectionsRef, paramViewsRef, pitchViewRef, processorParamsRef },
                param,
                v,
                h,
            ),
        [processorParamsRef, paramAxisDirectionsRef, paramViewsRef, pitchViewRef],
    );
    const yToValue = useCallback(
        (param: ParamName, y: number, h: number) =>
            yToValueImpl(
                { paramAxisDirectionsRef, paramViewsRef, pitchViewRef, processorParamsRef },
                param,
                y,
                h,
            ),
        [processorParamsRef, paramAxisDirectionsRef, paramViewsRef, pitchViewRef],
    );
    return { valueToY, yToValue };
}
