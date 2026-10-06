/*
 * 构建门禁：**测试通过 ⇒ 类型检查通过**。
 *
 * 【为什么需要】一次真实事故：3138 条测试全绿，构建却是坏的。
 *
 * 根因有两层，缺一层都不会出事：
 *
 *   1. `npx tsc --noEmit` 在本仓是**空操作** —— 根 `tsconfig.json` 是解决方案
 *      配置（`"files": []` + references），它本身没有可检查的文件，于是退出码 0。
 *      这个命令看起来正是"检查类型"的标准写法，谁都会这么敲一次，然后得到一个
 *      假的绿灯。
 *   2. Vitest 只**剥掉**类型（esbuild 的 transform），从不检查它们。因此测试
 *      全绿与类型错误可以并存，而且没有任何一条测试会因此变红。
 *
 * 【为什么门禁放在测试里，而不是 pre-commit 钩子或 `pretest` 脚本】
 * 因为**钩子与 `pretest` 都能被绕过**：直接跑 `npx vitest run` 就不经过 npm 的
 * `pretest`，`--no-verify` 也不经过 git 钩子。而"跑测试"是无论用什么姿势都绕不开
 * 的一步（CI 与本地都跑）。把检查放进测试套件，等于把它绑在**唯一**那个必然发生
 * 的动作上 —— 无论跑测试的人敲的是 `npm test` 还是 `npx vitest run`。
 *
 * 【为什么跑 npm 脚本而不是直接调 tsc】为了"类型检查"只有**一个定义**。
 * 命令写在 `package.json` 的 `typecheck` 里，`build` 也复用它 —— 否则 `build`
 * 与门禁各写一份 `tsc` 参数，改一处就会让两边对"什么叫类型检查"产生分歧，而
 * 那正是这类门禁最常见的失效方式。
 *
 * 【代价】约 15 秒（`tsc -b` 是增量构建，CI 冷启动更久）。因此超时给到 5 分钟。
 * 这点时间换"构建不可能在测试全绿时坏掉"，是划算的。
 *
 * 【为什么这个文件在 `tests/` 而不在 `src/`】它要起子进程，于是需要
 * `node:child_process` 与 `process` 的类型；而应用项目（`tsconfig.app.json`）只带
 * `types: ["vite/client"]`，看不到 @types/node（实测：`node:fs` 恰好能解析，
 * `node:child_process` 不能）。三条路里选这条：
 *   - 给应用项目加 `"node"` 类型 → **应用代码**从此也能无声地用上 process /
 *     Buffer，为一道门禁放松整棵源码树的类型约束，得不偿失；
 *   - 在测试里手写 node 接口类型 + 动态导入绕过 → 能用，但为省一个目录把代码写拧；
 *   - 挪出 `src/` → 门禁本来就是**仓库工具**而非应用代码，放在这里名副其实，
 *     应用项目的类型面一个字符都不用动。
 * `src/` 下那些门禁（`ui/designSystemGates` / `i18n/catalogIntegrity`）查的是
 * 应用自身的约定，留在原地；只有需要 node 能力的仓库级门禁才住在这里。
 */
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

/** `tsc -b` 冷启动在 CI 上可能要几十秒，给足余量。 */
const TYPECHECK_TIMEOUT_MS = 300_000;

/**
 * 读 `tsconfig.app.json` 的 compilerOptions。
 *
 * 【为什么用 TypeScript 自己的读取器】本仓的 tsconfig 带注释（JSONC），
 * `JSON.parse` 直接抛错；而"剥注释再解析"会连字符串里的 `//` 一起剥掉。
 * `ts.readConfigFile` 就是编译器自己用的那一套，注释、尾逗号都按 TS 的规则处理。
 */
async function readAppCompilerOptions(): Promise<Record<string, unknown>> {
    const ts = (await import("typescript")).default;
    const configPath = join(process.cwd(), "tsconfig.app.json");
    const parsed = ts.readConfigFile(configPath, ts.sys.readFile);
    if (parsed.error) {
        throw new Error(ts.flattenDiagnosticMessageText(parsed.error.messageText, "\n"));
    }
    return (parsed.config?.compilerOptions ?? {}) as Record<string, unknown>;
}

/**
 * TypeScript 6 起报错、7.0 移除的编译选项。
 *
 * 【为什么要盯着它们】弃用只在**装了新 TS 的环境**里报错：本仓的 `typescript`
 * 是 5.9，`npm run build` 一切正常，而编辑器里更新过的 TS 会直接标红 ——
 * 于是"构建通过、IDE 报错"。把清单钉在门禁里，升级时一次就看清全部要改的地方。
 * 处置方式见 `tsconfig.app.json` 里 `baseUrl` 那段注释：**迁移，不要 `ignoreDeprecations`**。
 */
const OPTIONS_REMOVED_IN_TS7 = [
    "baseUrl",
    "charset",
    "importsNotUsedAsValues",
    "keyofStringsOnly",
    "noImplicitUseStrict",
    "noStrictGenericChecks",
    "out",
    "preserveValueImports",
    "suppressExcessPropertyErrors",
    "suppressImplicitAnyIndexErrors",
];

test("tsconfig 不带 TypeScript 7.0 将移除的选项", async () => {
    const options = await readAppCompilerOptions();
    const present = OPTIONS_REMOVED_IN_TS7.filter((name) => name in options);
    expect(
        present.length === 0
            ? []
            : [
                  `以下选项已弃用（TS 6 报错、7.0 移除），请迁移而不是用 ignoreDeprecations：`,
                  ...present.map((name) => `  ${name}`),
              ].join("\n"),
    ).toEqual([]);
});

/*
 * 扩展入口别名：tsconfig 的 `paths` 与 vite 的 `resolve.alias` 必须指向同一批文件。
 *
 * 【为什么值得一道门禁】这两处的一致此前只写在 `tsconfig.app.json` 的注释里，
 * 没有任何东西守着；而**别名没有任何源码消费者**（应用内部代码不用它，只有第三方
 * 扩展作者会用），所以指错文件也不会有人报错 —— 直到某个扩展作者导入失败。
 *
 * 顺带守住这次修复的方向：`paths` 必须还在（删掉 `baseUrl` 是对的，删掉 `paths`
 * 不是），下面的双向比对会立刻发现少了一边。
 */
test("扩展别名在 tsconfig 与 vite 两侧指向同一批文件", async () => {
    const options = await readAppCompilerOptions();
    const paths = (options.paths ?? {}) as Record<string, string[]>;
    const fromTsconfig: Record<string, string> = {};
    for (const [key, targets] of Object.entries(paths)) {
        fromTsconfig[key] = targets[0].replace(/^\.\//, "");
    }

    // vite.config.ts 是 TS 源码，这里按文本取出 `"@hs/x": resolve(__dirname, "…")`。
    // 解析 TS 文件要引入编译 API，而这条门禁要守的只是"两侧列出同一批键与路径"。
    const viteSource = readFileSync(join(process.cwd(), "vite.config.ts"), "utf8");
    const fromVite: Record<string, string> = {};
    for (const match of viteSource.matchAll(
        /"(@hs\/[^"]+)":\s*resolve\(__dirname,\s*"([^"]+)"\)/g,
    )) {
        fromVite[match[1]] = match[2];
    }

    expect(fromVite).toEqual(fromTsconfig);
    expect(Object.keys(fromTsconfig).length, "别名不该为空").toBeGreaterThan(0);

    // 两侧指向的文件必须真的存在 —— 别名没有消费者，写错了没人会发现。
    for (const target of Object.values(fromTsconfig)) {
        expect(existsSync(join(process.cwd(), target)), `${target} 不存在`).toBe(true);
    }
});

test(
    "类型检查通过（npm run typecheck）",
    () => {
        // `shell: true` 是跨平台必需的：Windows 上 npm 是 npm.cmd，不经 shell 起不来。
        const result = spawnSync("npm", ["run", "--silent", "typecheck"], {
            // 与其它门禁一致：相对路径基于 vitest 的 cwd（frontend/）。
            cwd: process.cwd(),
            shell: true,
            encoding: "utf8",
        });

        if (result.error) {
            throw new Error(`无法启动 npm run typecheck：${result.error.message}`);
        }
        if (result.status === 0) return;

        // 把编译器的原始输出原样带出来：门禁失败时，用户要看到的是**哪一个文件
        // 哪一行**，而不是"类型检查失败"这一句话。
        const output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
        throw new Error(
            [
                "类型检查失败。这曾经在 3138 条测试全绿的情况下让构建挂掉过一次 ——",
                "所以它现在是测试套件的一部分。复现与修复：",
                "",
                "    npm run typecheck",
                "",
                "编译器输出：",
                output,
            ].join("\n"),
        );
    },
    TYPECHECK_TIMEOUT_MS,
);
