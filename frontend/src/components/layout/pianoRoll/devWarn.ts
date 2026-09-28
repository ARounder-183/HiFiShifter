/**
 * 开发模式诊断出口（**唯一的 DEV 判定点**）。
 *
 * 【为什么收口】这类日志表达的是"某个不变量被破坏了"，只在开发构建里有意义：
 * 生产构建里它们既是噪声，也会把实现细节写进用户日志。判定收口在这里，各诊断点
 * 只表达"要说什么" —— 各自复制一份 `import.meta.env` 判定的做法迟早会漏掉守卫，
 * 让某条诊断在生产里开始刷屏。
 *
 * 【为什么包 try】`import.meta.env` 由 Vite / vitest 注入；纯 node 环境（单测可能
 * 直接 import 调用方模块）读取它会抛错。
 */

/** 开发构建下打印一条告警；生产构建静默。 */
export function warnDev(message: string): void {
    if (!isDevBuild()) return;
    console.warn(message);
}

/** 是否开发构建。 */
function isDevBuild(): boolean {
    try {
        return import.meta.env.DEV === true;
    } catch {
        return false;
    }
}
