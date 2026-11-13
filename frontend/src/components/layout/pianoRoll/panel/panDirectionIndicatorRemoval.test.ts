import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

function walk(dir: string): string[] {
    const out: string[] = [];
    for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) out.push(...walk(path));
        else if (/\.(ts|tsx)$/.test(name)) out.push(path);
    }
    return out;
}

describe("声像轴方向指示删除契约", () => {
    it("frontend/src 不再引用 param_axis_pan_left_*", () => {
        const hits: string[] = [];
        for (const file of walk("src")) {
            if (file.endsWith("panDirectionIndicatorRemoval.test.ts")) continue;
            if (readFileSync(file, "utf8").includes("param_axis_pan_left_")) hits.push(file);
        }
        expect(hits).toEqual([]);
    });

    it("EditorArea 不再渲染 axisPanDirectionToggleAvailable", () => {
        const source = readFileSync(
            join("src", "components", "layout", "pianoRoll", "panel", "EditorArea.tsx"),
            "utf8",
        );
        expect(source).not.toContain("axisPanDirectionToggleAvailable");
    });
});
