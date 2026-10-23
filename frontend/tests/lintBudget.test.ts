/**
 * 门禁：ESLint 警告**只降不升**。
 *
 * 为什么需要：react-hooks 那一族警告此前是没人看的 —— 跑一次 lint 看到 140 条，
 * 于是没人再跑第二次。而这类警告里混着真问题（渲染期读 ref）与拆分带来的误报
 * （ref 从局部变量变成 hook 参数），不区分就只能一起放着。
 *
 * 门禁不要求清零（清零会逼着人加错误的依赖数组），只要求预算不涨：数字写在
 * package.json 的 lint:budget 里，新代码把它推高就会失败，于是每修一批就顺手把
 * 预算调低。与本仓其它棘轮门禁（Radix 表单原语计数、裸 select 白名单）同一手法。
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { delimiter, join } from "node:path";
import { expect, test } from "vitest";

/** 预算本身：改这个数字要同时在下面的失败信息里说明理由。 */
const WARNING_BUDGET = 140;

test("ESLint 警告不超过预算（npm run lint:budget）", () => {
    const scripts = JSON.parse(readFileSync(join(process.cwd(), "package.json"), "utf8")).scripts;
    const script: string | undefined = scripts["lint:budget"];
    expect(script, "package.json 缺少 lint:budget 脚本").toBeTruthy();
    // 预算必须写在脚本里，而不是靠人记得传参。
    expect(script).toContain(`--max-warnings ${WARNING_BUDGET}`);

    const binDir = join(process.cwd(), "node_modules", ".bin");
    const result = spawnSync(script as string, {
        cwd: process.cwd(),
        shell: true,
        encoding: "utf8",
        env: {
            ...process.env,
            PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}`,
        },
    });
    if (result.error) throw new Error(`无法启动 lint:budget：${result.error.message}`);
    const output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
    if (result.status === 0) return;
    throw new Error(
        [
            "ESLint 警告超过了预算。修掉新增的警告，或者在确认它们是误报之后",
            "把 WARNING_BUDGET 与 package.json 一起上调，并在这里写明理由。",
            "",
            "    npm run lint",
            "",
            output,
        ].join("\n"),
    );
}, 300_000);
