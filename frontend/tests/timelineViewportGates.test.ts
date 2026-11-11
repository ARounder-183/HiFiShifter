/**
 * 渲染内核是时间轴视口的唯一真值源。
 *
 * 旧的原生滚动容器（scrollRef）保留为挂载期 / 测试环境兜底，但绝不能重新挂到
 * JSX 上：一旦浏览器又维护一份 scrollLeft，就会与内核宿主形成双真值源。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

function sources(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) sources(full, out);
        else if (/.(ts|tsx)$/.test(entry)) out.push(full);
    }
    return out;
}

test("scrollRef 没有 JSX 挂载点", () => {
    const dir = join(process.cwd(), "src/components/layout/timeline");
    const offenders: string[] = [];
    for (const file of sources(dir)) {
        const lines = readFileSync(file, "utf8").split("\n");
        lines.forEach((line, index) => {
            if (/ref={[^}]*scrollRef/.test(line)) {
                offenders.push(file + ":" + (index + 1));
            }
        });
    }
    expect(offenders).toEqual([]);
});
