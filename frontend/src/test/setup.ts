/**
 * Vitest 全局环境补丁。
 *
 * 【为什么集中在这里】此前只有部分 jsdom 测试文件各自设置
 * IS_REACT_ACT_ENVIRONMENT；漏掉的文件会打印上千条 act(...) 告警，真实错误
 * 被淹没。jsdom 的 canvas getContext 未实现也是同一类噪声：当前行为本来就是
 * 返回 null，只是每次调用都往 stderr 打一行 Not implemented。
 *
 * 这里只补齐测试环境缺口，不改变生产行为：
 * - act 标志是 React 官方要求的测试环境声明；
 * - canvas 返回 null 与 jsdom 现状一致，只是不再打印噪声。
 */
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

if (typeof HTMLCanvasElement !== "undefined") {
    HTMLCanvasElement.prototype.getContext = (() =>
        null) as typeof HTMLCanvasElement.prototype.getContext;
}
