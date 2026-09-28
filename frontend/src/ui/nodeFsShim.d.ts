/*
 * 测试专用：最小的文件读取声明。
 *
 * 【为什么需要】排版层级门禁要读 `src/index.css` 的**真实文本**（不能复述一遍
 * 数值，否则测试与被测对象同源、断言无意义）。而 `?raw` 对本项目的 `.css`
 * 无效 —— vitest 会把 `.css` 一律替换成空模块（已实测：`?raw` 与
 * `import.meta.glob(..., { query: "?raw" })` 都返回空串），`historyOpLabels.test.ts`
 * 那套 `?raw` 只对非 CSS 文件成立。
 *
 * 【为什么不用 `/// <reference types="node" />`】那会把整套 node 类型并入整个
 * 程序，可能让 `setTimeout` 之类在浏览器代码里被推断成 Node 的返回类型。
 * 这里只声明测试真正用到的那一个函数，作用面最小。
 *
 * 【为什么不用 vitest 的 `css: true`】那会改变全部 223 个测试文件的处理方式
 * （为一份排版断言付全量 CSS 处理成本），代价与收益不成比例。
 */
declare module "node:fs" {
    export function readFileSync(path: string | URL, encoding: "utf8"): string;
}
