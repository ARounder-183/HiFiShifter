/**
 * menu-scroll-probe.mjs — 菜单滚动模型的验收探针（开发期脚本，不参与构建）。
 *
 * 【它验的是什么】"菜单没有竖直滚动条，只有两条提示行"这条契约在**真实浏览器**里
 * 是否成立。jsdom 里 scrollHeight / clientHeight 恒为 0，所有与滚动有关的断言都会
 * 退化成"断言 undefined" —— 因此这些量只能在真浏览器里量。
 *
 * 用法：
 *   node scripts/menu-scroll-probe.mjs [case]        # case: all | notebook | menubar
 *   VH=420 node scripts/menu-scroll-probe.mjs menubar
 *
 * 退出码：0 = 全部断言成立；1 = 有断言失败。
 *
 * 【为什么自带 playwright 而不复用 dev-shot】一次验收要在同一张菜单上连续做
 * "量几何 → 悬停 600ms → 量 scrollTop → 点击 → 再量 → 长按 → 再量"，中间还要按
 * 量到的坐标移动鼠标。动作数组表达不了这种"用上一步结果做下一步输入"的链路。
 */
import { chromium } from "playwright-core";

const URL = process.env.URL ?? "http://localhost:5173/?mock=1";
const VW = Number(process.env.VW ?? 1600);
const VH = Number(process.env.VH ?? 420);
const WAIT_MS = Number(process.env.WAIT ?? 3500);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** 一次验收收集到的结果。 */
const checks = [];
function check(name, ok, detail) {
    checks.push({ name: name, ok: Boolean(ok), detail: detail });
}

/** 读取当前打开的那张菜单的几何与提示行状态。 */
async function readMenu(page) {
    return page.evaluate(() => {
        const menu = document.querySelector(".hs-menu");
        if (!menu) return null;
        const style = getComputedStyle(menu);
        const hints = [...menu.querySelectorAll(".hs-menu__scroll-hint")].map((hint) => {
            const rect = hint.getBoundingClientRect();
            const bar = getComputedStyle(hint, "::after");
            return {
                cls: hint.className,
                visible: hint.dataset.visible === "1",
                top: Math.round(rect.top),
                bottom: Math.round(rect.bottom),
                barHeight: bar.height,
            };
        });
        const content = menu.querySelector(".hs-menu__content");
        const rect = menu.getBoundingClientRect();
        return {
            scrollbarWidth: style.scrollbarWidth,
            overflowY: style.overflowY,
            padding: style.padding,
            maxHeight: style.maxHeight,
            scrollTop: menu.scrollTop,
            scrollHeight: menu.scrollHeight,
            clientHeight: menu.clientHeight,
            hasContentLayer: content !== null,
            contentPadding: content ? getComputedStyle(content).padding : null,
            hints: hints,
            items: menu.querySelectorAll(".hs-menu__item").length,
            rect: {
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                w: Math.round(rect.width),
                h: Math.round(rect.height),
            },
        };
    });
}

/** 提示行 ::after 命中区的中心点（宿主的盒高是 0，中心落在 ::after 上）。 */
async function hintCenter(page, which) {
    return page.evaluate((which) => {
        const hint = document.querySelector(
            which === "up" ? ".hs-menu__scroll-hint--up" : ".hs-menu__scroll-hint--down",
        );
        if (!hint) return null;
        const rect = hint.getBoundingClientRect();
        const bar = Number.parseFloat(getComputedStyle(hint, "::after").height) || 0;
        const x = Math.round(rect.left + rect.width / 2);
        const y =
            which === "up" ? Math.round(rect.top + bar / 2) : Math.round(rect.bottom - bar / 2);
        const hit = document.elementFromPoint(x, y);
        return {
            x: x,
            y: y,
            hitsHint: Boolean(
                hit && hit.classList && hit.classList.contains("hs-menu__scroll-hint"),
            ),
            hitClass: hit ? String(hit.className).slice(0, 60) : null,
        };
    }, which);
}

async function scrollTop(page) {
    return page.evaluate(() => document.querySelector(".hs-menu")?.scrollTop ?? -1);
}

async function setScrollTop(page, value) {
    await page.evaluate((value) => {
        const menu = document.querySelector(".hs-menu");
        if (menu) menu.scrollTop = value;
    }, value);
    await sleep(140);
}

/** 打开记事本里的右键菜单（手写壳一族）。 */
async function openNotebookMenu(page) {
    await page.evaluate(() => {
        window.__hfsStore.dispatch({ type: "dock/openPanel", payload: { panelId: "notebook" } });
    });
    await sleep(900);
    await page.evaluate(() => {
        window.__hfsStore.dispatch({
            type: "session/setProjectNotesMarkdown",
            payload: "# T\n\npara\n",
        });
    });
    await sleep(700);
    return page.evaluate(() => {
        const p = document.querySelector(".hs-notebook-rich .ProseMirror p");
        if (!p) return false;
        const rect = p.getBoundingClientRect();
        p.dispatchEvent(
            new MouseEvent("contextmenu", {
                bubbles: true,
                cancelable: true,
                clientX: Math.round(rect.left + 30),
                clientY: Math.round(rect.top + 8),
                button: 2,
            }),
        );
        return true;
    });
}

/** 打开菜单栏的第 index 个下拉（默认「视图」：条目最多）。 */
async function openMenuBarMenu(page, index) {
    const triggers = page.locator(".min-h-qt-bar-main [aria-haspopup='menu']");
    const count = await triggers.count();
    if (count <= index) return false;
    await triggers.nth(index).click();
    return true;
}

/** 关闭所有菜单。 */
async function closeMenus(page) {
    await page.keyboard.press("Escape");
    await sleep(120);
    await page.mouse.click(6, Math.round(VH / 2));
    await sleep(120);
}

async function probeNotebook(page) {
    if (!(await openNotebookMenu(page))) {
        check("记事本菜单能打开", false, "找不到编辑器段落");
        return;
    }
    await sleep(320);
    const menu = await readMenu(page);
    if (!menu) {
        check("记事本菜单能打开", false, "菜单未挂载");
        return;
    }
    check("记事本菜单能打开", true, menu.items + " 项");
    check("壳没有原生滚动条", menu.scrollbarWidth === "none", menu.scrollbarWidth);
    check("壳自己不留内边距", menu.padding === "0px", menu.padding);
    check(
        "内容层持有内边距",
        menu.hasContentLayer && menu.contentPadding !== "0px",
        menu.contentPadding,
    );
    const expectedMax = await page.evaluate(() => window.innerHeight - 16);
    check(
        "高度上限是视口预算",
        menu.maxHeight === expectedMax + "px",
        menu.maxHeight + " vs " + expectedMax + "px",
    );
    check("两条提示行都在 DOM 里", menu.hints.length === 2, menu.hints.length);
    if (menu.scrollHeight <= menu.clientHeight + 1) {
        check(
            "装得下时两条提示行都不显示",
            menu.hints.every((hint) => !hint.visible),
            JSON.stringify(menu.hints.map((hint) => hint.visible)),
        );
    }
    await closeMenus(page);
}

async function probeMenuBar(page) {
    if (!(await openMenuBarMenu(page, 3))) {
        check("菜单栏下拉能打开", false, "找不到触发器");
        return;
    }
    await sleep(380);
    const menu = await readMenu(page);
    if (!menu) {
        check("菜单栏下拉能打开", false, "菜单未挂载");
        return;
    }
    check("菜单栏下拉能打开", true, menu.items + " 项");
    check("壳没有原生滚动条", menu.scrollbarWidth === "none", menu.scrollbarWidth);
    check("两条提示行都在 DOM 里", menu.hints.length === 2, menu.hints.length);

    const overflow = menu.scrollHeight > menu.clientHeight + 1;
    check("矮视口下菜单溢出", overflow, menu.scrollHeight + "/" + menu.clientHeight);
    if (!overflow) {
        await closeMenus(page);
        return;
    }

    const down = menu.hints.find((hint) => hint.cls.includes("--down"));
    const up = menu.hints.find((hint) => hint.cls.includes("--up"));
    check(
        "在顶端时只显示下方提示",
        down.visible === true && up.visible === false,
        JSON.stringify({ up: up.visible, down: down.visible }),
    );
    check(
        "下方提示行贴着壳的下内缘",
        Math.abs(down.bottom - (menu.rect.y + menu.rect.h - 1)) <= 1,
        down.bottom + " vs " + (menu.rect.y + menu.rect.h - 1),
    );
    check("提示行视觉条高度取自令牌", down.barHeight === "18px", down.barHeight);

    const hitDown = await hintCenter(page, "down");
    check("提示行可命中（::after 是命中区）", hitDown.hitsHint === true, hitDown.hitClass);

    // 悬停慢滚：等过 hoverDelayMs（180ms）再多观察 700ms。
    await page.mouse.move(hitDown.x, hitDown.y);
    await sleep(760);
    const afterHover = await scrollTop(page);
    check("悬停提示行会缓慢滚动", afterHover > menu.scrollTop, menu.scrollTop + " → " + afterHover);

    // 指针离开即停。
    await page.mouse.move(menu.rect.x + Math.round(menu.rect.w / 2), menu.rect.y + 30);
    await sleep(420);
    const afterLeave = await scrollTop(page);
    await sleep(320);
    check("指针离开后停止滚动", (await scrollTop(page)) === afterLeave, String(afterLeave));

    // 点击翻一页。
    await setScrollTop(page, 0);
    const clickTarget = await hintCenter(page, "down");
    await page.mouse.click(clickTarget.x, clickTarget.y);
    await sleep(420);
    const afterClick = await scrollTop(page);
    const pageStep = Math.max(3 * 24, menu.clientHeight - 2 * 24);
    const maxScroll = menu.scrollHeight - menu.clientHeight;
    const expectedStep = Math.min(pageStep, maxScroll);
    check(
        "点击提示行翻一页",
        afterClick > 0 && Math.abs(afterClick - expectedStep) <= 10,
        afterClick + " vs " + expectedStep,
    );

    // 长按快滚。
    await setScrollTop(page, 0);
    const pressTarget = await hintCenter(page, "down");
    await page.mouse.move(pressTarget.x, pressTarget.y);
    await page.mouse.down();
    await sleep(900);
    const duringPress = await scrollTop(page);
    const panelsDuringPress = await page.evaluate(
        () => document.querySelectorAll('[role="menu"]').length,
    );
    await page.mouse.up();
    await sleep(220);
    check("长按提示行持续滚动", duringPress > afterClick, String(duringPress));
    check("长按期间页面里仍只有一张菜单", panelsDuringPress === 1, String(panelsDuringPress));

    // 滚到底后上方提示出现。
    await setScrollTop(page, 99999);
    await sleep(220);
    const atBottom = await readMenu(page);
    const bottomUp = atBottom.hints.find((hint) => hint.cls.includes("--up"));
    const bottomDown = atBottom.hints.find((hint) => hint.cls.includes("--down"));
    check(
        "滚到底时只显示上方提示",
        bottomUp.visible === true && bottomDown.visible === false,
        JSON.stringify({ up: bottomUp.visible, down: bottomDown.visible }),
    );
    check(
        "上方提示行贴着壳的上内缘",
        Math.abs(bottomUp.top - (menu.rect.y + 1)) <= 1,
        bottomUp.top + " vs " + (menu.rect.y + 1),
    );

    // 键盘 End：末项必须完整可见（不被下方提示行盖住）。
    await setScrollTop(page, 0);
    await page.keyboard.press("End");
    await sleep(280);
    const endCheck = await page.evaluate(() => {
        const menu = document.querySelector(".hs-menu");
        if (!menu) return null;
        const active = menu.querySelector('[data-active="1"]') ?? document.activeElement;
        if (!active || !menu.contains(active)) return null;
        const menuRect = menu.getBoundingClientRect();
        const itemRect = active.getBoundingClientRect();
        const hint = menu.querySelector(".hs-menu__scroll-hint--down");
        const visible = hint !== null && hint.dataset.visible === "1";
        const bar = Number.parseFloat(
            getComputedStyle(menu).getPropertyValue("--qt-menu-scroll-hint-h"),
        );
        const inset = visible && Number.isFinite(bar) ? bar : 0;
        return {
            itemBottom: Math.round(itemRect.bottom),
            limit: Math.round(menuRect.top + menu.clientTop + menu.clientHeight - inset),
        };
    });
    check(
        "键盘 End 后末项完整可见（不被提示行盖住）",
        endCheck !== null && endCheck.itemBottom <= endCheck.limit + 1,
        JSON.stringify(endCheck),
    );

    await closeMenus(page);
}

const which = process.argv[2] ?? "all";

const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--force-device-scale-factor=2", "--use-gl=angle", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: 2 });
const errors = [];
page.on("pageerror", (error) => {
    const text = String(error);
    // 浏览器里没有 Tauri 后端，这条是 mock 形态的既有噪声，与被测行为无关。
    if (text.includes("transformCallback")) return;
    errors.push(text);
});
await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
await page.waitForTimeout(WAIT_MS);

if (which === "all" || which === "notebook") await probeNotebook(page);
if (which === "all" || which === "menubar") await probeMenuBar(page);

await browser.close();

const failed = checks.filter((item) => !item.ok);
for (const item of checks) {
    process.stdout.write((item.ok ? "PASS  " : "FAIL  ") + item.name + "  (" + item.detail + ")\n");
}
if (errors.length > 0) {
    process.stdout.write("\n页面错误：\n  " + errors.slice(0, 5).join("\n  ") + "\n");
}
process.stdout.write(
    "\n" +
        (checks.length - failed.length) +
        "/" +
        checks.length +
        " 通过（视口 " +
        VW +
        "×" +
        VH +
        "）\n",
);
process.exit(failed.length === 0 && errors.length === 0 ? 0 : 1);
