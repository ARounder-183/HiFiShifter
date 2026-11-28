// 修饰键视图的**结构契约**：所有「读一次修饰键就决定行为」的地方都必须经过
// `modifierView`。
//
// 【为什么是源码断言而不是行为测试】这条链的两端相隔十几个模块：芯片 →
// modifierState → modifierView → 各手势的修饰键快照 → keybinding 判定 → 行为。
// 行为测试只能覆盖到当前存在的每一条路径，而**新加一条路径**（例如将来新增
// 「按住 Ctrl 拖淡变曲率」）时它会静默地只支持桌面。源码断言反过来：只要
// 那几个快照构造点还在，它们就必须写着 `modifierView`。
//
// 与 `useKernelGestureInteractionLock` 的契约测试同一套路（读源码、断言结构）。
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// 与 `gestureInteractionLockContract.test.ts` 同一口径：vitest 的 cwd 是 `frontend/`。
const SRC = ".";

/** 每个条目：文件 + 它必须包含的片段（说明它为什么是必经之路）。 */
const CHOKEPOINTS: Array<{ file: string; must: string[]; why: string }> = [
    {
        file: "src/components/layout/timeline/kernel/host/timelineKernelHost.ts",
        must: ["modifierView(event)", "const view = modifierView(event);"],
        why: "内核所有拖拽（移动 / 修剪 / 淡变 / 增益 / 交叉淡化）的修饰键快照都在这里；还有重叠区按下与选择修饰键",
    },
    {
        file: "src/components/layout/timeline/fadeShapeCycle.ts",
        must: ["const view = modifierView(event);"],
        why: "「循环切换淡变形状」的修饰键判定",
    },
    {
        file: "src/components/layout/pianoRoll/gestures/freehandPointerDown.ts",
        must: ["...modifierView("],
        why: "自由绘制笔画里的精细调整 / 吸附切换快照",
    },
    {
        file: "src/components/layout/pianoRoll/gestures/vibratoPointerDown.ts",
        must: ["...modifierView("],
        why: "直线 / 颤音笔画的同一份快照",
    },
    {
        file: "src/features/keybindings/keybindingsSlice.ts",
        must: ['const view = "key" in (event as object) ? event : modifierView(event);'],
        why: "160 处 isModifierActive 读取点的唯一接缝（指针事件并入虚拟修饰键，键盘事件不并）",
    },
];

describe("修饰键视图的必经之路", () => {
    for (const point of CHOKEPOINTS) {
        it(`${point.file}：${point.why}`, () => {
            const text = readFileSync(join(SRC, point.file), "utf8");
            for (const fragment of point.must) {
                expect(text, `缺少 ${fragment}`).toContain(fragment);
            }
        });
    }
});
