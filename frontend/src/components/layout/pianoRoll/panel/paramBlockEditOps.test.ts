import { describe, expect, it, vi } from "vitest";
import {
    PARAM_BLOCK_MENU_OPS,
    isParamBlockMenuOp,
    runParamBlockMenuOp,
    type ParamBlockMenuOp,
    type ParamBlockOpHandlers,
} from "./paramBlockEditOps";

/** 每个 handler 都记下自己被调用的次数与入参。 */
function makeHandlers() {
    const calls: string[] = [];
    const split = vi.fn((context: { readonly snapToggleHeld: boolean }) => {
        calls.push(`split:${String(context.snapToggleHeld)}`);
    });
    const handlers: ParamBlockOpHandlers = {
        split,
        createFromSelection: () => calls.push("createFromSelection"),
        group: () => calls.push("group"),
        ungroup: () => calls.push("ungroup"),
        delete: () => calls.push("delete"),
        selectAll: () => calls.push("selectAll"),
        deselect: () => calls.push("deselect"),
        restoreGame: () => calls.push("restoreGame"),
        restoreDeleted: () => calls.push("restoreDeleted"),
        normalizePitch: () => calls.push("normalizePitch"),
    };
    return { handlers, calls, split };
}

describe("runParamBlockMenuOp", () => {
    it("runs exactly the handler named by the op, once", () => {
        for (const op of PARAM_BLOCK_MENU_OPS) {
            const { handlers, calls } = makeHandlers();
            runParamBlockMenuOp(op, handlers, { snapToggleHeld: false });
            const expected = op === "split" ? "split:false" : op;
            expect(calls, `${op} 只应触发自己的 handler`).toEqual([expected]);
        }
    });

    it("passes snapToggleHeld through to split", () => {
        const { handlers, calls, split } = makeHandlers();
        runParamBlockMenuOp("split", handlers, { snapToggleHeld: true });
        expect(split).toHaveBeenCalledTimes(1);
        expect(split).toHaveBeenCalledWith({ snapToggleHeld: true });
        expect(calls).toEqual(["split:true"]);
    });

    it("registers every handler exactly once in the table", () => {
        /*
         * 穷举交叉检查：表里的 op 集合必须与 handler 的字段集合一一对应。
         * 少一个 = 某条命令永远执行不到；多一个 = 表里出现了没有实现的 op。
         */
        const { handlers } = makeHandlers();
        expect([...PARAM_BLOCK_MENU_OPS].sort()).toEqual(Object.keys(handlers).sort());
        expect(PARAM_BLOCK_MENU_OPS.length).toBe(10);
    });
});

describe("isParamBlockMenuOp", () => {
    it("accepts every op in the table and rejects anything else", () => {
        for (const op of PARAM_BLOCK_MENU_OPS) expect(isParamBlockMenuOp(op)).toBe(true);
        for (const op of ["", "split ", "SPLIT", "transposeCents", "__proto__"]) {
            expect(isParamBlockMenuOp(op), op).toBe(false);
        }
    });

    it("narrows a menu string to the union", () => {
        const raw: string = "group";
        if (!isParamBlockMenuOp(raw)) throw new Error("guard rejected a table op");
        const op: ParamBlockMenuOp = raw;
        expect(op).toBe("group");
    });
});
