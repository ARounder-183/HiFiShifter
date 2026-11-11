import { useCallback } from "react";

import { formatParamValuePreviewValue } from "../paramValueFormatting";
import type { ParamAxisUnit } from "../paramAxisUnits";
import type { ParamName } from "../types";

/** 把当前参数/单位/语言绑定到共享读数格式化器。 */
export function useParamValueFormatter(args: {
    readonly editParam: ParamName;
    readonly axisUnit: ParamAxisUnit;
    readonly tf: (key: string) => string;
}): (value: number) => string {
    const { editParam, axisUnit, tf } = args;
    return useCallback(
        (value: number): string => formatParamValuePreviewValue({ value, editParam, axisUnit, tf }),
        [editParam, axisUnit, tf],
    );
}
