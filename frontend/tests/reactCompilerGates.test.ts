/**
 * 规则集与构建工具链必须一致。
 *
 * 本仓未启用 React Compiler，因此 ESLint 只启用经典 hooks 规则；若将来加入
 * babel-plugin-react-compiler，必须同步恢复 eslint-plugin-react-hooks 的
 * recommended 编译器规则并清零其告警，否则编译器与 lint 会对同一段渲染逻辑
 * 给出相互矛盾的判据。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const root = process.cwd();

test("未启用 React Compiler 时不加载编译器专属 lint 规则", () => {
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    const deps = { ...pkg.dependencies, ...pkg.devDependencies };
    const compilerEnabled = Object.keys(deps).some((name) => name.includes("react-compiler"));
    const config = readFileSync(join(root, "eslint.config.js"), "utf8");
    // 只看实际生效的代码，不把解释性注释里的规则名当成启用证据。
    const code = config
        .split("\n")
        .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
        .join("\n");
    if (!compilerEnabled) {
        expect(code).not.toContain("reactHooks.configs.flat.recommended");
        expect(code).toContain('"react-hooks/rules-of-hooks": "error"');
        expect(code).toContain('"react-hooks/exhaustive-deps": "error"');
    } else {
        expect(code, "启用 React Compiler 后必须恢复完整 recommended 规则").toContain(
            "reactHooks.configs.flat.recommended",
        );
    }
});
