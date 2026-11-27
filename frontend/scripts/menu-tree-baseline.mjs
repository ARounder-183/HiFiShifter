/**
 * menu-tree-baseline.mjs — 菜单**条目树**的录制与对比（开发期验证脚本，不参与构建）。
 *
 * 【它防的是什么】菜单条目是"手工转写一次、出错不报错"的代码：把
 * <DropdownMenu.Item onSelect={onOpenProject}> 改写成
 * { key, label, onSelect } 时，绑错回调、丢掉 disabled 条件、少写一条分隔线、
 * 把"最近工程"的 12 条截断成 10 条 —— 这些在类型检查、lint、单测里**全部静默
 * 通过**，只有用户点下去才会发现。
 *
 * 因此迁移前把每个菜单（含逐层子菜单）的条目树录成 JSON 基线，迁移后逐字段
 * 对比。基线的更新必须是显式动作（重新 record 并提交），不能被"顺手刷新"。
 *
 * 用法：
 *   node scripts/menu-tree-baseline.mjs record [outPath]
 *   node scripts/menu-tree-baseline.mjs compare <baselinePath>
 *   node scripts/menu-tree-baseline.mjs dump
 *
 * 退出码：0 = 录制成功 / 完全一致；1 = 有差异或录不到。
 *
 * 【为什么自带 playwright 而不复用 dev-shot】录制要按"路径"反复重开菜单
 * （见 collectTree），这是一段有状态的多步交互；把它塞进 dev-shot 的动作数组
 * 意味着每条路径一次页面往返，而路径数是条目数的量级。
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { chromium } from "playwright-core";

const URL = process.env.URL ?? "http://localhost:5173/?mock=1";
const VW = Number(process.env.VW ?? 1600);
const VH = Number(process.env.VH ?? 900);
const WAIT_MS = Number(process.env.WAIT ?? 3500);

/**
 * 要录制的菜单表面。
 *
 * id 进差异报告；selector 是"菜单栏上那些触发器"的选择器 —— 录的是
 * **每一个触发器打开后**的那张菜单。
 *
 * 【为什么 paramToolbar 允许缺席】它只在参数编辑器面板打开时才存在。缺席时
 * 基线里记 null，对比时也必须是 null —— 这样"面板没开"不会伪装成"菜单变空了"。
 */
const SURFACES = [
    { id: "menubar", selector: ".min-h-qt-bar-main [aria-haspopup='menu']" },
    { id: "paramToolbar", selector: ".param-pill__seg--chev[aria-haspopup='menu']" },
];

/** 条目选择器：三种菜单项角色 + 分隔线。标题行没有角色，靠文本行识别（见 readRows）。 */
const ROW_SELECTOR =
    '[role="menuitem"],[role="menuitemcheckbox"],[role="menuitemradio"],[role="separator"]';

/**
 * 页面内取行函数的**函数体**（作为字符串传给 page.evaluate，避免闭包丢失）。
 *
 * 【为什么是 body 而不是整段 function】把它当整段源码塞进 eval，函数声明会落到
 * 被求值函数的变量作用域上；同一个 evaluate 里再求值一次就是
 * "Identifier 'readRows' has already been declared"。用 new Function 拿到一个
 * 干净的函数对象，作用域不会互相污染。
 */
const READ_ROWS_BODY = [
    "return [...menu.querySelectorAll(selector)]",
    "    .filter(function (el) { return el.closest('[role=\"menu\"]') === menu; })",
    "    .map(function (el) {",
    "        return {",
    "            role: el.getAttribute('role'),",
    "            text: (el.textContent || '').replace(/\\s+/g, ' ').trim(),",
    "            disabled: el.getAttribute('aria-disabled') === 'true' || el.hasAttribute('disabled'),",
    "            checked: el.getAttribute('aria-checked'),",
    "            haspopup: el.getAttribute('aria-haspopup'),",
    "        };",
    "    });",
].join("\n");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * 从触发器打开菜单，沿 path（每层一个行下标）逐级展开子菜单，读回那一层的行。
 *
 * 【为什么必须按 closest('[role=menu]') 过滤】子菜单的面板在 DOM 上住在父菜单的
 * 子树里，querySelectorAll 会把两层的行一起捞出来 —— 于是"顶层有 17 项"会变成
 * "顶层有 40 项"，而这个数字看起来完全正常。
 *
 * 【为什么每条路径都重新打开一次菜单】子菜单的收起要精确模拟指针离开，而 Radix
 * 与我们的原语对"离开"的判据不同。与其猜它，不如每次从干净的顶层菜单重新走一遍
 * —— 代价是每层多一次点击，而这只发生在开发期的探针里。
 */
async function readAtPath(page, path) {
    /** 已确认打开的菜单链：每展开一级都必须真的多出一张菜单，否则就是没展开。 */
    const chain = [];
    // 硬上限：三层子菜单是本仓的既有深度（视图 → 布局 → 预设 → 删除预设）。
    // 超过它只可能是探针自己写错了，不该让它把浏览器跑满。
    if (path.length > 6) return null;
    for (const index of path) {
        const ok = await page.evaluate(
            function (args) {
                const menu = document.querySelector('[role="menu"]');
                if (!menu) return false;
                // 这里要的是**元素本身**（要往它身上派发指针事件），不是 readRows
                // 产出的纯数据，所以自己再取一次。
                const rows = [...menu.querySelectorAll(args.selector)].filter(function (el) {
                    return el.closest('[role="menu"]') === menu;
                });
                const target = rows[args.index];
                if (!target) return false;
                // Radix 的 SubTrigger 在 pointermove（鼠标）时展开；我们自己的
                // AppSubMenu 在 mouseenter 时展开。三种都派发，两族都覆盖。
                const fire = function (type, bubbles) {
                    target.dispatchEvent(
                        new PointerEvent(type, {
                            bubbles: bubbles,
                            pointerType: "mouse",
                            pointerId: 1,
                            isPrimary: true,
                        }),
                    );
                };
                fire("pointerover", true);
                fire("pointerenter", false);
                fire("pointermove", true);
                target.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
                target.dispatchEvent(new MouseEvent("mouseenter", { bubbles: false }));
                return { panels: document.querySelectorAll('[role="menu"]').length };
            },
            { index, readRowsBody: READ_ROWS_BODY, selector: ROW_SELECTOR },
        );
        if (!ok) return null;
        await sleep(200);
        /*
         * 【必须有这道守卫】子菜单没展开时，下面"取最后一张菜单"拿到的还是**父菜单**，
         * 于是同一批行会被当成子层的内容 —— collectTree 会为同一个 haspopup 行
         * 无限递归，探针永远跑不完（实测跑满十分钟）。
         */
        const panels = await page.evaluate(() => document.querySelectorAll('[role="menu"]').length);
        chain.push(panels);
        // 展开第 i 级之后，页面里应当有 i + 1 张菜单。少一张就是没展开。
        if (panels < chain.length + 1) return null;
    }
    return page.evaluate(
        function (args) {
            const readRows = new Function("menu", "selector", args.readRowsBody);
            const menus = [...document.querySelectorAll('[role="menu"]')];
            const menu = menus[menus.length - 1];
            if (!menu) return null;
            return readRows(menu, args.selector);
        },
        { readRowsBody: READ_ROWS_BODY, selector: ROW_SELECTOR },
    );
}

/** 关掉所有菜单。 */
async function closeAll(page) {
    await page.keyboard.press("Escape");
    await sleep(90);
    await page.mouse.click(4, Math.round(VH / 2));
    await sleep(90);
}

/** 递归地把某个触发器（+ 子路径）的条目树读出来。 */
async function collectTree(page, triggerHandle, path) {
    await triggerHandle.click();
    await sleep(220);
    let rows;
    try {
        rows = await readAtPath(page, path);
    } finally {
        await closeAll(page);
    }
    if (rows === null) return null;
    const tree = [];
    for (let i = 0; i < rows.length; i += 1) {
        const row = rows[i];
        if (row.haspopup === "menu") {
            row.items = await collectTree(page, triggerHandle, [...path, i]);
        }
        tree.push(row);
    }
    return tree;
}

async function record(page) {
    const out = {};
    for (const surface of SURFACES) {
        const triggers = await page.locator(surface.selector).all();
        if (triggers.length === 0) {
            out[surface.id] = null;
            continue;
        }
        const menus = [];
        for (const trigger of triggers) {
            const label = await trigger.evaluate((el) => (el.textContent || "").trim());
            menus.push({ label: label, tree: await collectTree(page, trigger, []) });
        }
        out[surface.id] = menus;
    }
    return out;
}

/** 逐字段比较，返回差异行（空数组 = 一致）。 */
function diffTrees(a, b, path, out) {
    if (a === b) return out;
    if (a === null || b === null || typeof a !== typeof b) {
        out.push(path + ": 基线 " + JSON.stringify(a) + " ≠ 现状 " + JSON.stringify(b));
        return out;
    }
    if (Array.isArray(a)) {
        if (!Array.isArray(b)) {
            out.push(path + ": 基线是数组，现状不是");
            return out;
        }
        if (a.length !== b.length) {
            out.push(path + ": 条目数 " + a.length + " → " + b.length);
        }
        const n = Math.min(a.length, b.length);
        for (let i = 0; i < n; i += 1) diffTrees(a[i], b[i], path + "[" + i + "]", out);
        return out;
    }
    if (typeof a === "object") {
        const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
        for (const key of keys) diffTrees(a[key], b[key], path + "." + key, out);
        return out;
    }
    out.push(path + ": 基线 " + JSON.stringify(a) + " ≠ 现状 " + JSON.stringify(b));
    return out;
}

const mode = process.argv[2] ?? "dump";
const arg = process.argv[3];

const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--force-device-scale-factor=2", "--use-gl=angle", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({
    viewport: { width: VW, height: VH },
    deviceScaleFactor: 2,
});
const errors = [];
page.on("pageerror", (error) => errors.push(String(error)));
await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });
await page.waitForTimeout(WAIT_MS);

const tree = await record(page);
await browser.close();

if (mode === "record") {
    const target = arg ?? "scripts/menu-tree.baseline.json";
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, JSON.stringify(tree, null, 2) + "\n", "utf8");
    process.stdout.write("recorded → " + target + "\n");
    process.exit(0);
}

if (mode === "dump") {
    process.stdout.write(JSON.stringify(tree, null, 2) + "\n");
    if (errors.length)
        process.stderr.write("页面错误：\n  " + errors.slice(0, 5).join("\n  ") + "\n");
    process.exit(0);
}

if (mode === "compare") {
    if (!arg) {
        process.stderr.write("compare 需要基线文件路径\n");
        process.exit(1);
    }
    const baseline = JSON.parse(readFileSync(arg, "utf8"));
    const diffs = diffTrees(baseline, tree, "$", []);
    if (diffs.length === 0) {
        process.stdout.write("菜单条目树与基线一致\n");
        process.exit(0);
    }
    process.stderr.write("菜单条目树与基线有 " + diffs.length + " 处差异：\n");
    for (const line of diffs.slice(0, 120)) process.stderr.write("  " + line + "\n");
    if (errors.length)
        process.stderr.write("页面错误：\n  " + errors.slice(0, 5).join("\n  ") + "\n");
    process.exit(1);
}

process.stderr.write("unknown mode: " + mode + "\n");
process.exit(1);
