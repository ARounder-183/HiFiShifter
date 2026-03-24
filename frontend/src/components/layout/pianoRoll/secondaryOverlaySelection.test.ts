import {
    getVisibleSecondaryParamIds,
    resolveSecondaryOverlayValues,
    toggleSecondaryParamVisibility,
} from "./secondaryOverlaySelection.js";

function assertDeepEqual<T>(actual: T, expected: T): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson !== expectedJson) {
        throw new Error(`Expected ${expectedJson}, received ${actualJson}`);
    }
}

const processorParamIds = ["tension", "breathiness", "energy"];

assertDeepEqual(toggleSecondaryParamVisibility({}, "tension"), {
    tension: true,
});

assertDeepEqual(
    toggleSecondaryParamVisibility({ tension: true }, "breathiness"),
    {
        tension: true,
        breathiness: true,
    },
);

assertDeepEqual(
    toggleSecondaryParamVisibility({ breathiness: true }, "breathiness"),
    {},
);

assertDeepEqual(
    getVisibleSecondaryParamIds({
        editParam: "pitch",
        processorParamIds,
        secondaryParamVisible: { energy: true },
    }),
    ["energy"],
);

assertDeepEqual(
    getVisibleSecondaryParamIds({
        editParam: "tension",
        processorParamIds,
        secondaryParamVisible: { breathiness: true },
    }),
    ["breathiness"],
);

assertDeepEqual(
    getVisibleSecondaryParamIds({
        editParam: "tension",
        processorParamIds,
        secondaryParamVisible: { pitch: true },
    }),
    ["pitch"],
);

assertDeepEqual(
    getVisibleSecondaryParamIds({
        editParam: "breathiness",
        processorParamIds,
        secondaryParamVisible: { breathiness: true },
    }),
    [],
);

assertDeepEqual(
    getVisibleSecondaryParamIds({
        editParam: "pitch",
        processorParamIds,
        secondaryParamVisible: { tension: true, energy: true },
    }),
    ["tension", "energy"],
);

assertDeepEqual(
    resolveSecondaryOverlayValues({
        orig: [1, 2, 3, 4],
        edit: [1, 9, Number.NaN, 4],
    }),
    [1, 9, 3, 4],
);

assertDeepEqual(
    resolveSecondaryOverlayValues({
        orig: [4, 5, 6],
        edit: [7],
    }),
    [7, 5, 6],
);

console.log("secondary overlay selection checks passed");
