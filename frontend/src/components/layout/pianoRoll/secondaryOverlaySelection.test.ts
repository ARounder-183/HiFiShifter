import { test } from "vitest";

import {
    getVisibleSecondaryParamIds,
    resolveSecondaryOverlayValues,
    toggleSecondaryParamVisibility,
} from "./secondaryOverlaySelection.js";

test("components/layout/pianoRoll/secondaryOverlaySelection.test.ts scripted checks", async () => {
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

    assertDeepEqual(toggleSecondaryParamVisibility({ tension: true }, "breathiness"), {
        tension: true,
        breathiness: true,
    });

    assertDeepEqual(toggleSecondaryParamVisibility({ breathiness: true }, "breathiness"), {});

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

    /**
     * 回归：结果**不得**是跨调用共享的数组。
     *
     * 【缺陷现象】此前本函数把结果写进模块级数组并原样返回引用。GL 曲线路径是
     * **延迟消费**的（面板先把图层描述符全部构建完，宿主随后才逐层投影），于是
     * 所有参考线 / 副参数图层读到的都是**最后一次写入**的内容——多条未被选中的
     * 参数线画出了同一条别的线的数据，表现为随图层顺序变化的渲染错乱。
     *
     * 判据：两次调用的返回值必须是不同对象，且第一次的返回值不被第二次调用改写。
     */
    const first = resolveSecondaryOverlayValues({
        orig: [1, 1, 1],
        edit: [1, 1, 1],
    });
    const second = resolveSecondaryOverlayValues({
        orig: [9, 9, 9],
        edit: [9, 9, 9],
    });
    if (first === second) {
        throw new Error("resolveSecondaryOverlayValues 返回了共享数组（跨调用别名）");
    }
    assertDeepEqual(first, [1, 1, 1]);
    assertDeepEqual(second, [9, 9, 9]);
});
