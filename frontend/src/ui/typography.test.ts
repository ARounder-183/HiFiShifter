/*
 * 排版层级门禁。
 *
 * 【为什么必须有】上一轮引入了字号令牌，但**没有规定层级**：标题挑到 13px，
 * 结果比 14px 的正文标签还小；节标题 12px 比它统领的行还小；字段标签 11px
 * 比它自己的提示（12px）还小。层级不是靠"提供了哪些字号"保证的，而是靠
 * **角色之间的关系**保证的 —— 所以这里把关系写成断言。
 *
 * 断言直接读 `src/index.css` 的角色定义（而不是渲染后取计算样式）：
 * 这样失败信息直接指向那行 CSS，且不依赖浏览器是否加载了样式表。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { describe, expect, test } from "vitest";

/*
 * 读 `src/index.css` 的真实文本。不用 `?raw`：vitest 把 `.css` 一律替换成
 * 空模块（已实测），因此 `?raw` 与 `import.meta.glob(query: "?raw")` 都拿不到内容。
 * 类型由同目录的 `nodeFsShim.d.ts` 提供（只声明用到的那一个函数）。
 */
const css = readFileSync(new URL("../index.css", import.meta.url), "utf8");

/** 取某个类名的规则体。 */
function ruleBody(selector: string): string {
    const re = new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`, "m");
    const match = css.match(re);
    if (!match) throw new Error(`找不到样式规则：${selector}`);
    return match[1];
}

/** 取 `--qt-xxx` 令牌的值（px 数字）。 */
function tokenPx(name: string): number {
    const re = new RegExp(`${name}:\\s*(\\d+)px`, "m");
    const match = css.match(re);
    if (!match) throw new Error(`找不到令牌：${name}`);
    return Number(match[1]);
}

/** 取规则体里的 `font-size`（解析成 px）。 */
function fontSizeOf(body: string): number {
    const varMatch = body.match(/font-size:\s*var\((--qt-fs-[\w-]+)\)/);
    if (varMatch) return tokenPx(varMatch[1]);
    const pxMatch = body.match(/font-size:\s*(\d+)px/);
    if (!pxMatch) throw new Error(`规则体里没有可解析的 font-size：${body}`);
    return Number(pxMatch[1]);
}

function fontWeightOf(body: string): number {
    const match = body.match(/font-weight:\s*(\d+)/);
    return match ? Number(match[1]) : 400;
}

function usesMuted(body: string): boolean {
    return /color:\s*var\(--qt-text-muted\)/.test(body);
}

const ROLE_FONT = {
    display: fontSizeOf(ruleBody(".hs-type-display")),
    section: fontSizeOf(ruleBody(".hs-type-section")),
    body: fontSizeOf(ruleBody(".hs-type-body")),
    label: fontSizeOf(ruleBody(".hs-type-label")),
    muted: fontSizeOf(ruleBody(".hs-type-muted")),
    caption: fontSizeOf(ruleBody(".hs-type-caption")),
    mono: fontSizeOf(ruleBody(".hs-type-mono")),
};

describe("排版角色层级", () => {
    test("七个角色都有定义", () => {
        for (const role of Object.keys(ROLE_FONT)) {
            expect(ruleBody(`.hs-type-${role}`), `.hs-type-${role} 缺失`).toBeTruthy();
        }
    });

    test("标题明显大于正文（这是上一轮的回归点）", () => {
        // 曾经 13px vs 14px —— 倒挂。现在要求至少拉开 5px。
        expect(ROLE_FONT.display - ROLE_FONT.body).toBeGreaterThanOrEqual(5);
        expect(ROLE_FONT.display).toBeGreaterThanOrEqual(18);
    });

    test("正文 > 标签 > 说明，逐级递减且不出现倒挂", () => {
        expect(ROLE_FONT.body).toBeGreaterThan(ROLE_FONT.label);
        expect(ROLE_FONT.label).toBeGreaterThan(ROLE_FONT.caption);
    });

    test("弱化正文与标签同字号，靠颜色分层", () => {
        // `.hs-type-muted` 承载"弱化的正文句"（面板说明、空态提示）——与标签同为
        // 12px，但用弱化色：它比 caption 大一档（11px 撑不起句子），比正文淡一档。
        expect(ROLE_FONT.muted).toBe(ROLE_FONT.label);
        expect(usesMuted(ruleBody(".hs-type-muted"))).toBe(true);
        expect(ROLE_FONT.body).toBeGreaterThan(ROLE_FONT.muted);
    });

    test("节标题不小于正文，靠字重分层而不是靠缩小字号", () => {
        expect(ROLE_FONT.section).toBeGreaterThanOrEqual(ROLE_FONT.body);
        const section = ruleBody(".hs-type-section");
        const body = ruleBody(".hs-type-body");
        expect(fontWeightOf(section)).toBeGreaterThan(fontWeightOf(body));
    });

    test("标签用正文色，不用弱化色（标签不该比自己的提示还淡）", () => {
        expect(usesMuted(ruleBody(".hs-type-label"))).toBe(false);
        expect(usesMuted(ruleBody(".hs-type-caption"))).toBe(true);
    });

    test("对话框副标题复用 .hs-type-muted，规则里不再写字号", () => {
        const dialogSource = readFileSync(new URL("./Dialog.tsx", import.meta.url), "utf8");
        expect(dialogSource, "副标题元素上没有挂 hs-type-muted").toContain("hs-type-muted");
        expect(css, "副标题规则又写回了自己的排版声明 —— 字号必须只有一个来源").not.toMatch(
            /\.app-dialog__description\s*\{[^}]*font-size/,
        );
    });

    /*
     * 标题**复用角色类**，而不是把角色的声明抄一遍。
     *
     * 【为什么要断言结构而不是取值】此前 `.app-dialog__title` 逐字复制了
     * `.hs-type-display` 的四条声明（只差 color），于是全应用最重要的标题反而成了
     * 角色层唯一没有消费者的地方 —— 改字号要改两处，忘一处就分叉。
     * 断言"规则里不再写 font-size"能钉住这一点：一旦有人把声明抄回来就会红。
     */
    test("对话框标题复用 .hs-type-display，不复制它的声明", () => {
        const dialogSource = readFileSync(new URL("./Dialog.tsx", import.meta.url), "utf8");
        expect(dialogSource, "标题元素上没有挂 hs-type-display").toContain("hs-type-display");
        expect(
            ruleBody(".app-dialog__title"),
            "标题规则又写回了自己的 font-size —— 字号必须只有一个来源",
        ).not.toMatch(/font-size:/);
    });

    /*
     * 对话框的**主消息**：这一档是本轮新增的，也是最容易被写错的一档。
     *
     * 【为什么单独断言颜色】此前的失败模式不是"字号小"一件事，而是
     * **主消息被放进了副标题槽位**：11px + 弱化色。所以这里不仅要求它等于正文字号，
     * 还要求它**不是弱化色** —— 主消息不许比标题淡两档。只断言字号会让
     * "13px 但仍然灰得看不清"通过。
     */
    test("对话框主消息复用 .hs-type-body，而不是复制它的声明", () => {
        /*
         * 【为什么断言结构而不是取值】主消息就是正文 —— 字号、行高、颜色都该来自
         * 角色层。这里再写一份，改正文字号就要改两处，忘一处就分叉（标题曾经就是
         * 这么分叉的，见上一段）。
         */
        const dialogSource = readFileSync(new URL("./Dialog.tsx", import.meta.url), "utf8");
        expect(dialogSource, "消息元素上没有挂 hs-type-body").toContain("hs-type-body");
        // 角色的颜色必须是正文色：主消息不许比标题淡两档
        expect(usesMuted(ruleBody(".hs-type-body"))).toBe(false);
        // 主消息必须比副标题大一档，否则两者在视觉上无法区分
        expect(ROLE_FONT.body).toBeGreaterThan(ROLE_FONT.label);
    });

    test("严重度只落在图标上，消息永远是平文本正文", () => {
        /*
         * 上一轮把 tone 画成"底色 + 左色条 + 彩色文字"的告警卡，实测在一枚
         * 400px 的确认框里占了三分之一，用户反馈"太怪"。桌面消息框的惯例是
         * 消息保持平文本，严重度住在图标与动作按钮上 —— 这里把"平文本"钉死：
         * 谁把填充告警卡写回来就红。
         */
        const tone = ruleBody(".app-dialog__message--tone");
        expect(tone, "消息块又画回了告警卡（background）").not.toMatch(/background:/);
        expect(tone, "消息块又画回了告警卡（border-left）").not.toMatch(/border-left:/);
        // 图标着色走语义令牌，不写死十六进制：两套主题各有一份取值。
        expect(css).toMatch(/\[data-tone="warning"\][^}]*var\(--qt-warning-text\)/);
        expect(css).toMatch(/\[data-tone="danger"\][^}]*var\(--qt-danger-text\)/);
    });

    test("字号阶梯覆盖到标题量级（否则作者只能拿最接近的值凑）", () => {
        expect(tokenPx("--qt-fs-2xl")).toBeGreaterThanOrEqual(18);
        // 阶梯必须单调递增，且每个角色都能在阶梯上找到落点
        const ladder = [
            "--qt-fs-3xs",
            "--qt-fs-micro",
            "--qt-fs-xs",
            "--qt-fs-sm",
            "--qt-fs-md",
            "--qt-fs-lg",
            "--qt-fs-xl",
            "--qt-fs-2xl",
        ].map(tokenPx);
        for (let i = 1; i < ladder.length; i += 1) {
            expect(ladder[i], `阶梯在 ${i} 处不递增`).toBeGreaterThan(ladder[i - 1]);
        }
    });

    test("行高与字号成对给出（不留继承导致的行距跳变）", () => {
        for (const role of Object.keys(ROLE_FONT)) {
            expect(ruleBody(`.hs-type-${role}`), `.hs-type-${role} 未给行高`).toMatch(
                /line-height:/,
            );
        }
    });

    /*
     * 同一个槽位只能有一种渲染。
     *
     * 【为什么要禁掉 `<Text>`】实测过：`description` 裸用时是 11px 弱化色，
     * 而套一层 `<Text size="2" color="gray">` 后是 14px —— **同一个字段两种字号**，
     * 取决于作者当时怎么写的。槽位的样式只能由槽位自己决定。
     */
    test("description / message 槽位里不得再套 Radix <Text>", () => {
        const offenders: string[] = [];
        const stack: string[] = ["."];
        while (stack.length > 0) {
            const dir = stack.pop()!;
            for (const entry of readdirSync(dir)) {
                if (entry === "node_modules") continue;
                const full = dir === "." ? entry : `${dir}/${entry}`;
                if (statSync(full).isDirectory()) {
                    stack.push(full);
                    continue;
                }
                if (!full.startsWith("src") || !/\.tsx$/.test(full)) continue;
                if (/\.test\.tsx$/.test(full)) continue;
                const lines = readFileSync(full, "utf8").split("\n");
                for (let i = 0; i < lines.length; i += 1) {
                    if (!/<Text/.test(lines[i])) continue;
                    // 往回看 3 行：槽位 prop 就在附近
                    const context = lines.slice(Math.max(0, i - 3), i).join("\n");
                    if (/(?:description|message)=\{/.test(context)) {
                        offenders.push(`  ${full}:${i + 1}`);
                    }
                }
            }
        }
        expect(
            offenders.length === 0
                ? []
                : ["以下槽位里套了 <Text>，会让同一槽位出现两种字号：", ...offenders].join("\n"),
        ).toEqual([]);
    });

    /*
     * Radix `<Text>` 的字号取值（14px/12px）与角色层并存了多轮 —— 同一面板两种
     * "正文"。第七轮按语义映射表全量收编，本门禁钉住清零进程：白名单只允许
     * **变小**；自检要求清单里的文件必须仍在用 `<Text`，迁完就从清单删除，
     * 否则清单本身就在撒谎。清单清空后此断言退化为"全仓禁用"。
     */
    test("src/ui 之外禁止 Radix <Text>（白名单棘轮）", () => {
        const whitelist = new Set([
            "src/App.tsx",
            "src/components/dock/DockLayoutSettingsDialog.tsx",
            "src/components/layout/AboutDialog.tsx",
            "src/components/layout/ActionBar.tsx",
            "src/components/layout/AutoBackupDialog.tsx",
            "src/components/layout/BenchmarkDialog.tsx",
            "src/components/layout/ChannelImportDialog.tsx",
            "src/components/layout/CustomScaleDialog.tsx",
            "src/components/layout/FileBrowserPanel.tsx",
            "src/components/layout/ImportProjectDialog.tsx",
            "src/components/layout/KeybindingsDialog.tsx",
            "src/components/layout/PianoRollPanel.tsx",
            "src/components/layout/QuickClipExportDialog.tsx",
            "src/components/layout/QuickSearchPopup.tsx",
            "src/components/layout/RecordingSettingsDialog.tsx",
            "src/components/layout/RenderCacheDialog.tsx",
            "src/components/layout/SplitTransitionSettingsDialog.tsx",
            "src/components/layout/TimelineDisplaySettingsDialog.tsx",
            "src/components/layout/notebook/NotebookDialogs.tsx",
            "src/components/layout/timeline/SilenceDetectionDialog.tsx",
            "src/components/layout/timeline/TempoMapCornerButton.tsx",
            "src/components/layout/timeline/TempoMapRulerRow.tsx",
            "src/components/layout/timeline/TrackList.tsx",
            "src/components/layout/timeline/clip/ClipFormantToolWindow.tsx",
            "src/components/layout/timeline/kernel/KernelUnavailableNotice.tsx",
        ]);
        const offenders: string[] = [];
        const stack: string[] = ["src"];
        while (stack.length > 0) {
            const dir = stack.pop()!;
            for (const entry of readdirSync(dir)) {
                const full = `${dir}/${entry}`;
                if (statSync(full).isDirectory()) {
                    if (entry !== "node_modules") stack.push(full);
                    continue;
                }
                if (!/\.tsx$/.test(entry) || /\.test\.tsx$/.test(entry)) continue;
                if (full.startsWith("src/ui/")) continue;
                // 掐掉注释行：文档里提一句 `<Text` 不算采用
                const source = readFileSync(full, "utf8")
                    .split("\n")
                    .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
                    .join("\n");
                if (/<Text[\s/>]/.test(source)) offenders.push(full);
            }
        }
        const outside = offenders.filter((file) => !whitelist.has(file));
        expect(
            outside.length === 0
                ? []
                : ["以下白名单之外的文件仍在使用 Radix <Text>：", ...outside].join("\n"),
        ).toEqual([]);
        const stale = [...whitelist].filter(
            (file) => !/<Text[\s/>]/.test(readFileSync(file, "utf8")),
        );
        expect(
            stale.length === 0
                ? []
                : ["白名单里的文件已不再使用 <Text>，请从清单删除：", ...stale].join("\n"),
        ).toEqual([]);
    });

    /*
     * 【本文件门禁的边界】上面的断言读的是 `src/index.css` 的**声明**，
     * 不是浏览器里的层叠结果：它保证"角色存在、且彼此关系正确"，**不保证**
     * "某处元素真的用了角色、且没被别处的 font-size 覆盖"。
     * 后半句由 `designSystemGates.test.ts` 的采用率门禁承担（禁止裸 px 字号
     * + 角色采用率棘轮）。两道门禁合起来才覆盖"声明正确"与"确实被用"。
     */
});
