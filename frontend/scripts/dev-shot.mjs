/**
 * dev-shot.mjs — 开发期浏览器截图 / 交互验证脚本（仅本地调试用，不参与构建）。
 *
 * 用法：
 *   node scripts/dev-shot.mjs <url> <outPath> [waitMs] [actionsJson]
 *
 * actionsJson 支持的动作（数组，按序执行）：
 *   { "type": "wheel", "x": 800, "y": 400, "deltaY": -240, "ctrl": true }
 *   { "type": "drag", "button": "middle", "from": [800, 400], "to": [600, 400] }
 *   { "type": "click", "x": 800, "y": 400, "button": "left" }
 *   { "type": "move", "x": 800, "y": 400 }
 *   { "type": "key", "key": "PageDown" }
 *   { "type": "wait", "ms": 500 }
 *   { "type": "shot", "path": "/tmp/x.png" }
 *   { "type": "eval", "js": "return 1 + 1" }
 */
import { chromium } from "playwright-core";

const url = process.argv[2] ?? "http://localhost:5173/?mock=1";
const out = process.argv[3] ?? "/tmp/hfs-shot.png";
const waitMs = Number(process.argv[4] ?? 3500);
const actions = process.argv[5] ? JSON.parse(process.argv[5]) : [];

const vw = Number(process.env.VW ?? 1600);
const vh = Number(process.env.VH ?? 900);
const kernelFlag = process.env.KERNEL; // "0" | "1" | undefined（undefined = 不写入，用默认）

const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--force-device-scale-factor=2", "--use-gl=angle", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({
    viewport: { width: vw, height: vh },
    deviceScaleFactor: 2,
});
if (kernelFlag !== undefined) {
    await page.addInitScript((value) => {
        window.localStorage.setItem("hifishifter.timelineKernel", value);
    }, kernelFlag);
}

const logs = [];
page.on("console", (msg) => {
    const type = msg.type();
    if (type === "error" || type === "warning") logs.push(`[${type}] ${msg.text()}`);
    // 诊断日志（前缀过滤）：排查渲染问题时需要看应用内部的 log 输出。
    else if (type === "log" && /waveform-debug|kernel-debug/.test(msg.text())) {
        logs.push(`[log] ${msg.text()}`);
    }
});
page.on("pageerror", (err) => logs.push(`[pageerror] ${String(err)}`));

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
await page.waitForTimeout(waitMs);

for (const action of actions) {
    switch (action.type) {
        case "wheel": {
            await page.mouse.move(action.x, action.y);
            // 修饰键按 playwright 的键名传入（macOS 上主修饰键是 "Meta"）。
            const held = [];
            if (action.ctrl) held.push("Control");
            if (action.meta) held.push("Meta");
            if (action.alt) held.push("Alt");
            if (action.shift) held.push("Shift");
            for (const key of held) await page.keyboard.down(key);
            await page.mouse.wheel(action.deltaX ?? 0, action.deltaY ?? 0);
            for (const key of held) await page.keyboard.up(key);
            break;
        }
        case "drag": {
            await page.mouse.move(action.from[0], action.from[1]);
            await page.mouse.down({ button: action.button ?? "left" });
            const steps = action.steps ?? 12;
            for (let i = 1; i <= steps; i += 1) {
                const x = action.from[0] + ((action.to[0] - action.from[0]) * i) / steps;
                const y = action.from[1] + ((action.to[1] - action.from[1]) * i) / steps;
                await page.mouse.move(x, y);
                await page.waitForTimeout(12);
            }
            await page.mouse.up({ button: action.button ?? "left" });
            break;
        }
        case "click": {
            await page.mouse.click(action.x, action.y, { button: action.button ?? "left" });
            break;
        }
        case "down": {
            // 与 move / key / up 组合可表达「拖拽中途按键」这类手势（drag 动作
            // 是一次性完成 down→move→up，无法插入中间步骤）。
            await page.mouse.down({ button: action.button ?? "left" });
            break;
        }
        case "up": {
            await page.mouse.up({ button: action.button ?? "left" });
            break;
        }
        case "move": {
            await page.mouse.move(action.x, action.y);
            break;
        }
        case "key": {
            await page.keyboard.press(action.key);
            break;
        }
        case "keyDown": {
            // 与 click / down / up 组合可表达「按住修饰键再点击」这类手势
            // （wheel 动作自带修饰键参数，但鼠标点击没有）。
            await page.keyboard.down(action.key);
            break;
        }
        case "keyUp": {
            await page.keyboard.up(action.key);
            break;
        }
        case "type": {
            // 向当前聚焦元素输入文本（行内编辑 / 重命名这类验证需要真实键入，
            // 直接改 DOM 值不会触发 React 的受控更新）。
            await page.keyboard.type(action.text ?? "", { delay: action.delay ?? 10 });
            break;
        }
        case "wait": {
            await page.waitForTimeout(action.ms ?? 300);
            break;
        }
        case "shot": {
            await page.screenshot({ path: action.path });
            console.log("SHOT:", action.path);
            break;
        }
        case "shotElement": {
            // 截取单个元素（用于观察某个 canvas 自身的绘制内容）。
            const locator = page.locator(action.selector).first();
            await locator.screenshot({ path: action.path });
            console.log("SHOT-ELEMENT:", action.path);
            break;
        }
        case "eval": {
            // eslint-disable-next-line no-eval
            const result = await page.evaluate(new Function(action.js));
            console.log("EVAL:", JSON.stringify(result));
            break;
        }
        default:
            break;
    }
}

await page.screenshot({ path: out });
console.log("SHOT:", out);
console.log("LOGS:");
for (const line of logs.slice(0, 60)) console.log("  " + line);
if (logs.length === 0) console.log("  (no errors/warnings)");

await browser.close();
