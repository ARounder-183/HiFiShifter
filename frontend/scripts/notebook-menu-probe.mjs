/*
 * 生成 dev-shot 用的动作序列（开发期验证脚本，不参与构建）。
 *
 * 【为什么要单独一个文件】把这段 JSON 直接写在 shell 里，中文与 `\n` 要穿过
 * shell 引号 → heredoc → JSON → `new Function` 四层转义，任何一层没对齐就是
 * 一句没有上下文的 SyntaxError。这里用真正的 JS 源文件，转义只发生在
 * `JSON.stringify` 那一处。
 *
 * 【为什么在同一个 eval 里右键 + 读菜单】右键坐标要先从 DOM 量出来，
 * `dev-shot` 的动作之间没法回传变量。React 的合成事件对离散事件是同步 flush
 * 的，因此 `dispatchEvent(contextmenu)` 返回时菜单已经挂上去了。
 *
 * 用法：
 *   node scripts/notebook-menu-probe.mjs <case> > /tmp/actions.json
 *   VW=1600 VH=900 node scripts/dev-shot.mjs "http://localhost:5173/?mock=1" out.png 4000 "$(cat /tmp/actions.json)"
 */

const MD = [
    "# 标题",
    "",
    "一段普通文字，用来测试右键。",
    "",
    "- 列表甲",
    "- 列表乙",
    "",
    "[外站链接](https://example.com/x)",
    "",
    "| a | b |",
    "| --- | --- |",
    "| c | d |",
    "",
].join("\n");

/** 打开记事本面板并把正文换成固定样本（每次验证的起点都一样）。 */
const SETUP = [
    {
        type: "eval",
        js: 'window.__hfsStore.dispatch({type:"dock/openPanel",payload:{panelId:"notebook"}}); return "open";',
    },
    { type: "wait", ms: 900 },
    {
        type: "eval",
        js: `window.__hfsStore.dispatch({type:"session/setProjectNotesMarkdown",payload:${JSON.stringify(MD)}}); return "seeded";`,
    },
    { type: "wait", ms: 900 },
];

/**
 * 在编辑器内命中的元素上右键，并读回菜单。
 *
 * 【为什么拆成「派发 → 等一帧 → 读」两步】手工 `dispatchEvent` 之后 React 未必
 * 已经 flush（并发渲染下离散事件的同步 flush 是实现细节）。同一个 eval 里立刻
 * 读 DOM 会得到"菜单没出现"的假阴性 —— 第一版就是这样误判的。
 */
function probe(selector, dx, dy) {
    const target = selector === null ? "root" : `root.querySelector(${JSON.stringify(selector)})`;
    return [
        {
            type: "eval",
            js: `
const root = document.querySelector('.hs-notebook-rich .ProseMirror');
if (!root) return { found: false };
const el = ${target};
if (!el) return { found: false, selector: ${JSON.stringify(selector)} };
const r = el.getBoundingClientRect();
const x = Math.round(r.left + ${dx});
const y = Math.round(r.top + ${dy});
el.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 2 }));
return { found: true, at: [x, y] };
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const root = document.querySelector('.hs-notebook-rich .ProseMirror');
const menu = document.querySelector('[role="menu"]');
if (!menu) return { open: false };
const items = Array.from(menu.querySelectorAll('.hs-menu__item')).map((b) => ({
    text: (b.textContent || '').trim(),
    disabled: b.disabled,
    tip: b.getAttribute('data-tooltip') || null,
}));
const headings = Array.from(menu.querySelectorAll('.hs-menu__label')).map((h) => (h.textContent || '').trim());
const mr = menu.getBoundingClientRect();
return {
    open: true,
    portal: root ? !root.contains(menu) : null,
    items,
    headings,
    box: { x: Math.round(mr.x), y: Math.round(mr.y), w: Math.round(mr.width), h: Math.round(mr.height) },
    viewport: [window.innerWidth, window.innerHeight],
    visible: getComputedStyle(menu).visibility,
};
`,
        },
    ];
}

/** 关掉菜单（外部 pointerdown）。 */
const CLOSE = {
    type: "eval",
    js: 'document.body.dispatchEvent(new PointerEvent("pointerdown",{bubbles:true})); return !!document.querySelector(\'[role="menu"]\');',
};

/** 每个 case 都是「开面板 → 铺样本 → 某个右键 → 读菜单」。 */

/** 读菜单时一并带上勾选态（标题行 / 当前块类型靠它表达）。 */
const READ_WITH_CHECKED = {
    type: "eval",
    js: `
const menu = document.querySelector('[role="menu"]');
if (!menu) return { open: false };
return {
    open: true,
    checked: Array.from(menu.querySelectorAll('.hs-menu__item'))
        .filter((b) => b.getAttribute('aria-checked') === 'true')
        .map((b) => (b.textContent || '').trim()),
};
`,
};

/** 切视图模式。 */
function setMode(mode) {
    return {
        type: "eval",
        js: `window.__hfsStore.dispatch({type:"notebook/setNotebookMode",payload:${JSON.stringify(mode)}}); return ${JSON.stringify(mode)};`,
    };
}

/** 切右键菜单设置档位。 */
function setScope(scope) {
    return {
        type: "eval",
        js: `window.__hfsStore.dispatch({type:"notebook/patchNotebookSettings",payload:{contextMenu:${JSON.stringify(scope)}}}); return ${JSON.stringify(scope)};`,
    };
}

/** 在任意选择器上右键（用于源码视图 / 只读预览）。 */
function probeSelector(selector, dx, dy, extraRead = "") {
    return [
        {
            type: "eval",
            js: `
const el = document.querySelector(${JSON.stringify(selector)});
if (!el) return { found: false, selector: ${JSON.stringify(selector)} };
const r = el.getBoundingClientRect();
const x = Math.round(r.left + ${dx});
const y = Math.round(r.top + ${dy});
el.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 2 }));
return { found: true, at: [x, y] };
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
if (!menu) return { open: false };
return {
    open: true,
    items: Array.from(menu.querySelectorAll('.hs-menu__item')).map((b) => ({
        text: (b.textContent || '').trim(),
        disabled: b.disabled,
    })),
    headings: Array.from(menu.querySelectorAll('.hs-menu__label')).map((h) => (h.textContent || '').trim()),
    ${extraRead}
};
`,
        },
    ];
}

const CASES = {
    /** 诊断：设置值、视图模式、事件是否送达、菜单是否出现。 */
    diag: [
        ...SETUP,
        {
            type: "eval",
            js: `
const s = window.__hfsStore.getState();
const root = document.querySelector('.hs-notebook-rich .ProseMirror');
let seen = 0;
if (root) root.addEventListener('contextmenu', () => { seen += 1; }, true);
const p = root && root.querySelector('p');
let at = null;
if (p) {
    const r = p.getBoundingClientRect();
    at = [Math.round(r.left + 30), Math.round(r.top + 8)];
    p.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: at[0], clientY: at[1], button: 2 }));
}
return {
    contextMenu: s.notebook.settings.contextMenu,
    mode: s.notebook.mode,
    hasRich: !!root,
    seen,
    at,
    menu: !!document.querySelector('[role="menu"]'),
};
`,
        },
    ],
    /** 诊断：子菜单触发项是否带上了 data-menu-key（键盘展开要靠它定位）。 */
    attr: [
        ...SETUP,
        ...probe("td p", 6, 8),
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
return {
    keys: Array.from(menu.querySelectorAll('[data-menu-key]')).map((e) => e.dataset.menuKey),
    items: menu.querySelectorAll('.hs-menu__item').length,
};
`,
        },
    ],
    /** 正文普通段落（不在任何特殊节点上）。 */
    text: [...SETUP, ...probe("p", 30, 8)],
    /**
     * 纯键盘走通子菜单：Home 到「行」→ Enter 展开 → ArrowDown 在子面板里移动。
     *
     * 【为什么必须在真浏览器里验】jsdom **不实现**"按钮获得焦点时按 Enter 会派发
     * click"这条默认行为，所以"回车展开子菜单"在单测里根本测不出来 —— 而它依赖的
     * 恰恰就是那个原生 click（`AppContextMenu` 对触发项刻意不拦截 Enter，见那里的
     * 注释）。子面板内部的导航同理：它走 DOM 焦点，也要真浏览器才成立。
     */
    keyboardSubmenu: [
        ...SETUP,
        {
            type: "eval",
            js: `
window.__keyLog = [];
document.addEventListener('keydown', (e) => {
    window.__keyLog.push({ key: e.key, prevented: e.defaultPrevented });
});
return 'hooked';
`,
        },
        // 对照组：菜单还没打开时先敲一下，确认键盘事件确实能到达页面。
        { type: "key", key: "ArrowDown" },
        { type: "wait", ms: 150 },
        { type: "eval", js: "return { controlLog: window.__keyLog.slice() };" },
        ...probe("td p", 6, 8),
        { type: "key", key: "Home" },
        { type: "wait", ms: 150 },
        { type: "eval", js: "return { afterOpen: window.__keyLog.slice() };" },
        { type: "key", key: "Enter" },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const sub = document.querySelector('.hs-menu--submenu');
return { openedByEnter: !!sub, panels: document.querySelectorAll('[role="menu"]').length, log: window.__keyLog.slice() };
`,
        },
        // 子面板里的方向键：走 DOM 焦点，且高亮必须**看得见**（`:focus` 那条 CSS）。
        { type: "key", key: "ArrowDown" },
        { type: "wait", ms: 150 },
        {
            type: "eval",
            js: `
const sub = document.querySelector('.hs-menu--submenu');
const focused = document.activeElement;
return {
    focusedInSub: !!(sub && focused && sub.contains(focused)),
    focusedText: focused ? (focused.textContent || '').trim() : null,
    focusedBackground: focused ? getComputedStyle(focused).backgroundColor : null,
};
`,
        },
    ],
    /**
     * 表格落点 + 展开「行」子菜单。
     *
     * 读回三层信息：顶层项数（层级改造后应显著变短）、子面板是否存在且是**独立**
     * 的菜单表面、以及两者的几何（子面板必须落在外层右侧且不出视口）。
     */
    submenu: [
        ...SETUP,
        ...probe("td p", 6, 8),
        {
            type: "eval",
            js: `
const outer = document.querySelector('[role="menu"]');
// 属于本层的项：子菜单触发项被包在自己的 div 里，:scope > .hs-menu__item
// 数不到它们，所以按 closest([role=menu]) 判归属（与 useMenuKeyboard 同口径）。
const topLevel = (menu) =>
    Array.from(menu.querySelectorAll('.hs-menu__item')).filter(
        (b) => b.closest('[role="menu"]') === menu,
    ).length;
const trigger = Array.from(outer.querySelectorAll('.hs-menu__item'))
    .find((b) => (b.textContent || '').trim() === '行');
if (!trigger) return { found: false, topLevel: topLevel(outer) };
// 悬停展开（真实用户的主要路径）。
trigger.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
trigger.parentElement.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
return {
    found: true,
    topLevel: topLevel(outer),
    triggerRect: (() => { const r = trigger.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }; })(),
};
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const outer = document.querySelector('[role="menu"]');
const panels = Array.from(document.querySelectorAll('[role="menu"]'));
const sub = document.querySelector('.hs-menu--submenu');
if (!sub) return { opened: false, panels: panels.length };
const r = sub.getBoundingClientRect();
const outerRect = outer.getBoundingClientRect();
/*
 * 【必须用 elementFromPoint，不能只看矩形】子面板被父壳的 overflow 裁掉时，
 * getBoundingClientRect 照样报出完整矩形（那是**布局几何**，与裁切无关）——
 * 曾经因此放过了"子菜单只露出 5px、实际完全看不见"的缺陷。这里改判"子面板
 * 中心点上真的是不是子面板"。
 */
const midX = Math.round(r.left + r.width / 2);
const midY = Math.round(r.top + r.height / 2);
const hit = document.elementFromPoint(midX, midY);
return {
    opened: true,
    panels: panels.length,
    subItems: Array.from(sub.querySelectorAll('.hs-menu__item')).map((b) => (b.textContent || '').trim()),
    subRightOfOuter: r.left >= outerRect.left,
    insideViewport: r.right <= window.innerWidth && r.bottom <= window.innerHeight,
    // 真正可见：中心点上命中的元素必须属于子面板。
    actuallyVisible: !!(hit && hit.closest('.hs-menu--submenu')),
    hitText: hit ? (hit.textContent || '').trim().slice(0, 20) : null,
    outerOverflowY: getComputedStyle(outer).overflowY,
    box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
};
`,
        },
    ],
    /**
     * 菜单打开时按方向键：菜单高亮与编辑器光标是否**同时**动。
     *
     * 先真的点一下编辑器（`click` 动作），否则它没有焦点、也没有 DOM 选区 ——
     * 第一版探针就是这样得出"光标没动"的假结论。菜单原语在 document 冒泡阶段
     * 处理方向键，而 ProseMirror 的 keymap 挂在编辑器 DOM 上、**先于它**执行。
     */
    arrows: [
        ...SETUP,
        {
            type: "eval",
            js: `
const root = document.querySelector('.hs-notebook-rich .ProseMirror');
const p = root.querySelector('p');
const r = p.getBoundingClientRect();
const x = Math.round(r.left + 30);
const y = Math.round(r.top + 8);
// 真的让编辑器拿到焦点，并按坐标落一次光标（PM 在 mousedown 里做位置换算）。
root.focus();
p.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
p.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
p.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 2 }));
const sel = document.getSelection();
return { focused: document.activeElement === root, caretBefore: sel ? sel.anchorOffset : null };
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
const ae = document.activeElement;
const before = ae ? ae.className.toString().slice(0, 30) : null;
// 手动聚焦一次，看它是否"站得住"——站不住说明有别的东西在抢焦点。
menu.focus();
const immediately = document.activeElement === menu;
return { menuTabIndex: menu.getAttribute('tabindex'), before, immediately };
`,
        },
        { type: "wait", ms: 300 },
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
return { stillMenu: document.activeElement === menu, now: document.activeElement ? document.activeElement.tagName : null };
`,
        },
        { type: "key", key: "ArrowDown" },
        { type: "wait", ms: 200 },
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
const active = menu ? menu.querySelector('[data-active="1"]') : null;
const sel = document.getSelection();
return {
    menuActive: active ? (active.textContent || '').trim() : null,
    caretAfter: sel ? sel.anchorOffset : null,
    activeAfter: document.activeElement ? document.activeElement.tagName : null,
};
`,
        },
    ],
    /** 链接。 */
    link: [...SETUP, ...probe("a", 4, 8)],
    /** 列表项（第二项：可缩进）。 */
    list: [...SETUP, ...probe("ul li:nth-child(2)", 20, 8)],
    /** 表格正文单元格。 */
    table: [...SETUP, ...probe("td p", 6, 8)],
    /** 表格标题行单元格：验证「标题行」的勾选态。 */
    tableHeader: [...SETUP, ...probe("th p", 6, 8), READ_WITH_CHECKED],
    /** 标题落点：验证「一级标题」的勾选态。 */
    headingChecked: [...SETUP, ...probe("h1", 20, 10), READ_WITH_CHECKED],
    /** 关掉菜单：验证外部点击真的收起。 */
    close: [
        ...SETUP,
        // 先让编辑器真的拿到焦点（真实右键前必然如此）——否则"关闭后焦点还给
        // 触发者"根本无从验证：触发者本来就是 `<body>`。
        {
            type: "eval",
            js: `
const root = document.querySelector('.hs-notebook-rich .ProseMirror');
const p = root.querySelector('p');
const r = p.getBoundingClientRect();
const x = Math.round(r.left + 30);
const y = Math.round(r.top + 8);
root.focus();
p.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
p.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 }));
p.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 2 }));
return 'opened';
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
return { focusOnMenu: document.activeElement === menu };
`,
        },
        CLOSE,
        {
            type: "eval",
            js: 'return document.querySelector(\'[role="menu"]\') ? "still-open" : "closed";',
        },
        { type: "wait", ms: 150 },
        {
            type: "eval",
            js: `
const el = document.activeElement;
return {
    focusAfterClose: el ? el.className.toString().slice(0, 40) : null,
    focusBackInEditor: !!(el && el.closest && el.closest('.ProseMirror')),
};
`,
        },
    ],
    /** 键盘入口：`ContextMenu` 键在光标处开菜单。 */
    keyboard: [
        ...SETUP,
        {
            type: "eval",
            js: `
const el = document.querySelector('.hs-notebook-rich .ProseMirror p');
el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: 0, clientY: 0 }));
el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ContextMenu', bubbles: true, cancelable: true }));
return 'pressed';
`,
        },
        { type: "wait", ms: 250 },
        {
            type: "eval",
            js: `
const menus = Array.from(document.querySelectorAll('[role="menu"]'));
return {
    open: menus.length > 0,
    menus: menus.length,
    first: menus[0] ? {
        label: menus[0].getAttribute('aria-label'),
        count: menus[0].querySelectorAll('.hs-menu__item').length,
        head: Array.from(menus[0].querySelectorAll('.hs-menu__item')).slice(0, 6).map((b) => (b.textContent || '').trim()),
        parent: menus[0].parentElement ? menus[0].parentElement.tagName : null,
    } : null,
};
`,
        },
    ],
    /** 真的执行一条命令：列表项「增加缩进」。 */
    actionIndent: [
        ...SETUP,
        ...probe("ul li:nth-child(2)", 20, 8),
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
const btn = Array.from(menu.querySelectorAll('.hs-menu__item')).find((b) => (b.textContent || '').includes('增加缩进'));
if (!btn) return { clicked: false };
btn.click();
return { clicked: true };
`,
        },
        { type: "wait", ms: 350 },
        {
            type: "eval",
            js: `return { nested: !!document.querySelector('.hs-notebook-rich .ProseMirror ul ul'), menuGone: !document.querySelector('[role="menu"]') };`,
        },
    ],
    /** 真的执行一条命令：表格「在下方插入行」。 */
    actionTableRow: [
        ...SETUP,
        ...probe("td p", 6, 8),
        {
            type: "eval",
            js: `
const menu = document.querySelector('[role="menu"]');
const btn = Array.from(menu.querySelectorAll('.hs-menu__item')).find((b) => (b.textContent || '').includes('在下方插入行'));
if (!btn) return { clicked: false };
btn.click();
return { clicked: true };
`,
        },
        { type: "wait", ms: 350 },
        {
            type: "eval",
            js: `return { rows: document.querySelectorAll('.hs-notebook-rich .ProseMirror table tr').length };`,
        },
    ],
    /** 源码视图：textarea 的菜单。 */
    source: [
        ...SETUP,
        setMode("source"),
        { type: "wait", ms: 500 },
        ...probeSelector(".hs-notebook-source", 40, 20),
    ],
    /** 分栏只读预览：只有只读项。 */
    preview: [
        ...SETUP,
        setMode("split"),
        { type: "wait", ms: 700 },
        ...probeSelector(".hs-scroll-gutter-flush .ProseMirror p", 20, 6),
    ],
    /** compact 档：去掉格式 / 段落 / 插入三组。 */
    compact: [...SETUP, setScope("compact"), { type: "wait", ms: 300 }, ...probe("p", 30, 8)],
    /** off 档：右键不弹菜单。 */
    off: [...SETUP, setScope("off"), { type: "wait", ms: 300 }, ...probe("p", 30, 8)],
    /** 图片卡片菜单。 */
    image: [
        ...SETUP,
        {
            type: "eval",
            js: `window.__hfsStore.dispatch({type:"session/setProjectNotesMarkdown",payload:"![图](https://example.com/a.png)\\n\\n尾段\\n"}); return "seeded-image";`,
        },
        { type: "wait", ms: 900 },
        ...probeSelector(".hs-notebook-rich .ProseMirror img", 10, 10),
    ],
};

const which = process.argv[2] ?? "text";
const actions = CASES[which];
if (!actions) {
    process.stderr.write(`unknown case: ${which}\n`);
    process.exit(1);
}
process.stdout.write(JSON.stringify(actions));
