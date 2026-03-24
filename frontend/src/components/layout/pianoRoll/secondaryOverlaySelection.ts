import type { ParamName } from "./types.js";

export function toggleSecondaryParamVisibility(
    current: Partial<Record<ParamName, boolean>>,
    param: ParamName,
): Partial<Record<ParamName, boolean>> {
    if (current[param]) {
        const next = { ...current };
        delete next[param];
        return next;
    }
    return { ...current, [param]: true };
}

export function getVisibleSecondaryParamIds(args: {
    editParam: ParamName;
    processorParamIds: ParamName[];
    secondaryParamVisible: Partial<Record<ParamName, boolean>>;
}): ParamName[] {
    const { editParam, processorParamIds, secondaryParamVisible } = args;
    const candidateIds =
        editParam === "pitch"
            ? processorParamIds
            : ["pitch", ...processorParamIds];

    return candidateIds.filter(
        (paramId) =>
            paramId !== editParam && secondaryParamVisible[paramId] === true,
    );
}

export function resolveSecondaryOverlayValues(args: {
    orig: number[];
    edit: number[];
}): number[] {
    const { orig, edit } = args;
    const length = Math.max(orig.length, edit.length);
    const values = new Array<number>(length);

    for (let index = 0; index < length; index += 1) {
        const editValue = edit[index];
        const origValue = orig[index];
        const hasEditValue = Number.isFinite(editValue);
        const hasOrigValue = Number.isFinite(origValue);

        if (hasEditValue) {
            values[index] = editValue;
        } else if (hasOrigValue) {
            values[index] = origValue;
        } else {
            values[index] = 0;
        }
    }

    return values;
}
