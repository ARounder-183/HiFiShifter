/**
 * cpu-render-bench.mjs — CPU-only（软件光栅）渲染基准（仅本地验证用，不参与构建）。
 *
 * 【用途】
 * 以 ANGLE + SwiftShader 软件光栅启动 Chrome，程序化驱动时间轴横向滚动，输出帧间隔的
 * p50 / p95 / max 与慢帧计数，并打印探测到的 GL renderer 以证明确实走了软件栈。
 *
 * 【为什么需要它】时间轴渲染内核是**唯一**渲染路径（旧实现已删除，无回退）。因此
 * "没有 GPU 的机器能否正确渲染"必须可回归验证，而不能只留一次性口头结论。实测结论见
 * docs/superpowers/specs/2026-09-13-timeline-single-path-design.md §2.1：
 * 纯 CPU 环境（含无 GPU）可用，且与 GPU **像素级一致**（0.0046% 差异，几何真值相同）。
 *
 * 【为什么程序化驱动滚动，而不用 CDP 注入鼠标事件】
 * 实测：CDP 注入鼠标事件本身会产生 80~130ms 的**假慢帧**（idle 时 p50 16.7ms，一注入
 * 拖动就变成 89~131ms），会把注入开销算进渲染时间，使数据失去意义。这里改为在 rAF 循环
 * 里直接推进滚动位置（内核经 `setScrollLeft`，无内核时退化为写 `scrollLeft`），测到的
 * 才是渲染本身。同时提供 idle 对照：若 idle 也不满帧，说明瓶颈不在渲染。
 *
 * 用法（**必须在 frontend/ 下运行**，脚本 import `playwright-core`）：
 *     node scripts/cpu-render-bench.mjs
 *
 * 环境变量：
 * - `URL`   目标地址，默认 `http://127.0.0.1:5174/?mock=1`（需 dev server 已运行）
 * - `STEPS` 采样帧数，默认 120
 * - `DELTA` 每帧推进的 CSS px，默认 24（`120 × 24 = 2880px`，覆盖多个视口宽）
 * - `MODE`  仅接受 `kernel`（默认）。**已移除的 `legacy` 模式会被显式拒绝**，见下。
 *
 * 【为什么必须拒绝 MODE=legacy，而不是回落到内核基准】
 * 旧实现（原生滚动 + Canvas2D）已随"渲染内核唯一路径"改造**删除**，因此不存在可对比的
 * legacy 路径。若脚本对此静默无视、照常打印内核数据，调用者会把它当成"legacy 的数字"
 * 去和内核比较——得到**看似有效实则错误**的结论。这比直接报错危险得多，所以这里显式
 * 拒绝并以非零码退出（设计文档 §6 明确要求"不得静默给出错误数据"）。
 */
import { chromium } from "playwright-core";

const url = process.env.URL ?? "http://127.0.0.1:5174/?mock=1";
const steps = Number(process.env.STEPS ?? 120);
const delta = Number(process.env.DELTA ?? 24);
const mode = process.env.MODE ?? "kernel";

// 显式拒绝已移除的模式：静默回落到内核基准会产出"看似有效实则错误"的对比数据。
if (mode !== "kernel") {
    const detail =
        mode === "legacy"
            ? "旧实现（原生滚动 + Canvas2D）已随「渲染内核唯一路径」改造删除，不存在可对比的 legacy 路径。"
            : `未知模式 ${JSON.stringify(mode)}；本脚本只支持 MODE=kernel。`;
    console.error(
        [
            `✗ MODE=${mode} 不受支持：${detail}`,
            "",
            "  本脚本只测内核路径（MODE=kernel，默认）。若要对比历史数据，请查阅",
            "  docs/superpowers/specs/2026-09-13-timeline-single-path-design.md §2.1，",
            "  其中记录了改造前 legacy(Canvas2D) 与内核的实测对比。",
        ].join("\n"),
    );
    process.exit(2);
}

const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    // SwiftShader = ANGLE 的纯 CPU 后端；enable-unsafe-swiftshader 允许在无 GPU 时
    // 使用它（Chrome 曾默认禁用）；use-angle 显式选后端，避免悄悄走回硬件路径。
    args: [
        "--force-device-scale-factor=2",
        "--enable-unsafe-swiftshader",
        "--use-angle=swiftshader",
        "--disable-gpu-sandbox",
    ],
});
const page = await browser.newPage({
    viewport: { width: 1920, height: 1200 },
    deviceScaleFactor: 2,
});
await page.goto(url, { waitUntil: "load" });
// 等首帧与内核挂载完成（内核宿主在挂载时创建 GL 上下文，失败会走失败界面）。
await page.waitForTimeout(5000);

/**
 * 把帧间隔数组归约成分位数统计。
 *
 * 特殊说明：丢掉前 3 帧——首帧包含字体/图集预热，会污染 p50。
 * 该阈值与分位数口径在两个模式下保持一致，因此结果可直接横向比较。
 *
 * @param frames 帧间隔数组（毫秒）。
 * @returns 统计结果；样本不足时返回 null。
 */
function summarize(frames) {
    const f = frames.slice(3);
    if (f.length === 0) return null;
    const sorted = [...f].sort((a, b) => a - b);
    const q = (p) => +sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))].toFixed(1);
    return {
        frames: f.length,
        p50: q(0.5),
        p95: q(0.95),
        max: +Math.max(...f).toFixed(1),
        slowOver33ms: f.filter((x) => x > 33).length,
        slowOver100ms: f.filter((x) => x > 100).length,
    };
}

// 证明确实走了软件光栅：若这里报出的是硬件 renderer，整份数据都不具代表性。
const glInfo = await page.evaluate(() => {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return { webgl2: false, renderer: null };
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    return {
        webgl2: true,
        renderer: gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    };
});

/** 只跑 rAF、不推进滚动，作为 idle 对照。 */
const idle = summarize(
    await page.evaluate(async (n) => {
        const frames = [];
        let last = performance.now();
        await new Promise((resolve) => {
            let i = 0;
            const tick = () => {
                const now = performance.now();
                frames.push(now - last);
                last = now;
                if (++i < n) requestAnimationFrame(tick);
                else resolve();
            };
            requestAnimationFrame(tick);
        });
        return frames;
    }, steps),
);

/** 每帧推进一次横向滚动，测量渲染帧时间。 */
const scroll = summarize(
    await page.evaluate(
        async ({ n, step }) => {
            const el = document.querySelector("[data-timeline-scroller]");
            const host = window.__hfsKernel;
            const frames = [];
            let pos = 0;
            let last = performance.now();
            await new Promise((resolve) => {
                let i = 0;
                const tick = () => {
                    const now = performance.now();
                    frames.push(now - last);
                    last = now;
                    pos += step;
                    // 内核模式下必须经内核改视口（自绘滚动，原生 scroller 只是被动镜像）；
                    // 无内核时退化为写原生位置，使脚本在失败界面上也能给出对照数据。
                    if (host) host.setScrollLeft(pos);
                    else if (el) el.scrollLeft = pos;
                    if (++i < n) requestAnimationFrame(tick);
                    else resolve();
                };
                requestAnimationFrame(tick);
            });
            return frames;
        },
        { n: steps, step: delta },
    ),
);

const kernelMounted = await page.evaluate(() => !!window.__hfsKernel);

console.log(
    JSON.stringify(
        {
            url,
            mode,
            steps,
            deltaPx: delta,
            gl: glInfo,
            kernelMounted,
            idle,
            scroll,
            // 显式提示数据是否可用于比较：走的是硬件后端或内核未挂载时都不具代表性。
            note: glInfo.renderer?.includes("SwiftShader")
                ? undefined
                : "⚠️ 未走 SwiftShader 软件光栅，本结果不反映 CPU-only 表现",
        },
        null,
        1,
    ),
);
await browser.close();
