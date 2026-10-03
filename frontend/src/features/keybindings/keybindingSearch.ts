/**
 * 快捷键的检索索引与匹配。
 *
 * 【解决什么问题】快捷键设置窗口里有 114 条绑定、14 个分组，纵向约 4000px。
 * 用户想改一条绑定时通常已经知道自己要找什么（"撤销"）或记得按键（`Ctrl+Z`），
 * 但窗口里只有线性滚动一种定位手段。这里提供第三种：输入几个字符把候选缩到个位。
 *
 * 【为什么单独成文件】两个导出都是**纯函数**，`keybindingSearch.test.ts` 可以
 * 脱离 React 与 i18n 直接测（传一个恒等的 `resolveLabel` 即可）。能否搜到、
 * 结果按什么顺序排，判断逻辑都集中在这一个文件里，便于单独审阅与调整。
 */
import {
    ACTION_META,
    ALL_ACTION_IDS,
    DEFAULT_KEYBINDINGS,
    GROUP_LABEL_KEYS,
} from "./defaultKeybindings";
import type { ActionId, ActionMeta, Keybinding } from "./types";

/**
 * 一条动作的可检索形态。
 */
export interface KeybindingSearchEntry {
    id: ActionId;
    /**
     * 参与匹配的词条，**按来源分桶**（已去重、小写）。
     *
     * 【为什么要分桶】同一段文本可以从多个来源被命中。若把所有来源混成一个大数组、
     * 命中即累加，词条多的条目会靠**数量**取胜（详见 `scoreToken` 的注释）。
     * 分桶后"每个来源只贡献它的最高一档"，排序才反映"这条像不像"，而不是"这条
     * 有多少地方碰巧含这个字"。
     */
    sources: {
        /** 本地化后的操作名，再按空格切词。 */
        label: string[];
        /** 本地化后的分组名。 */
        group: string[];
        /** 动作 id 的 `.` 分段，如 `clip.split` → ["clip","split"]。 */
        idParts: string[];
        /** 绑定的主键（`z` / `space` / `arrowup`）及其别名。**强信号**。 */
        primaryKeys: string[];
        /** 修饰键名及其别名（`ctrl` / `cmd` / `alt` / `shift`）。**弱信号**。 */
        modifiers: string[];
        /**
         * 操作名的转写形态（拼音全拼 / 初声 / 罗马字 / 谚文分解）。
         *
         * 【为什么要单独成桶而不是并进 `label`】并进 label 就无法对「转写命中」单独
         * 设门槛 —— 单字符 token（`z`）会命中所有拼音以 z 开头的条目，把原本按字面
         * 匹配得到的结果冲散。分桶后可以要求 token 长度 ≥ 2 才参与转写匹配。
         *
         * 【分桶不会让「词条多」取胜】`scoreToken` 对**每个 token** 取各来源的最高档
         * （`Math.max`），不是累加；多一个桶只是多一个候选档位，不会叠加。
         */
        translit: string[];
    };
    /** 排序与展示用的主名称（已本地化）。 */
    label: string;
    group: ActionMeta["group"];
}

/**
 * 按键别名表：把 `formatKeybinding` 渲染成符号/缩写、但用户会用单词搜的键补上词条。
 *
 * 【为什么需要】UI 里 `arrowup` 显示成 `↑`（见 `keybindingsSlice.ts` 的
 * `prettifyKey`），`escape` 显示成 `Escape`。索引 `formatKeybinding()` 的输出会让
 * 打 "up" / "esc" 的用户一无所获；而只索引 `kb.key` 又漏掉这些习惯叫法。
 * 这里补上别名 —— 规模故意压到 10 条，其余键 `kb.key` 本身已经是好词条
 * （`space` / `enter` / `delete` / `tab`）。
 *
 * 【为什么别名同时含两套修饰键叫法】同一个 `ctrl` 字段在 macOS 上渲染成 `⌘`、
 * 在其他平台渲染成 `Ctrl`。两套叫法都进词条，用户按自己看到的那种输入都能命中。
 */
const KEY_SEARCH_ALIASES: Record<string, string[]> = {
    arrowup: ["up", "arrow"],
    arrowdown: ["down", "arrow"],
    arrowleft: ["left", "arrow"],
    arrowright: ["right", "arrow"],
    escape: ["esc"],
    delete: ["del"],
    backspace: ["back"],
    control: ["ctrl", "control", "cmd", "command"],
    meta: ["cmd", "command", "meta", "win"],
    alt: ["alt", "option", "opt"],
};

/**
 * 单个查询 token 在单个来源上的命中评分；0 表示未命中。
 *
 * 【为什么要分档】搜 `z` 时会有两类命中：一类是操作名含 z 的，另一类是按键就绑在
 * Z 上的（撤销 `Ctrl+Z`）。不打分的话顺序由数据定义顺序决定 —— 用户看到的
 * "最像的那条"落在哪个位置是随机的。
 *
 * 【为什么修饰键只值 1 分，和 id / 分组同档】修饰键（`ctrl` / `alt` / `shift`）
 * 是**泛匹配**：一个 token 是否命中它们，几乎独立于这条快捷键是什么。若让它们
 * 值 2 分，那么一个在每个来源都只是泛泛沾边的条目会在**每个 token**上得满
 * （如 `modifier.pianoRollVerticalZoom` 的标签、id、分组里全有字母 z），
 * 靠 token 数量累加压过真正绑在 Z 上的 `Ctrl+Z`。压到 1 分后，
 * 「绑定本身就是查询串」这一最强信号才不会被稀释。
 */
/*
 * 【为什么主键是最高档】用户搜 `ctrl z` 时，最强的信号是"这条绑定**就是** Z"。
 * 名称里含 z 的条目（如 "Canvas vertical zoom"）在任何含 z 的查询里都会来凑热闹，
 * 但它并不是用户说的那个键。要让 `Ctrl+Z` 排在 "zoom" 之前，主键必须高于名称。
 */
const SCORE_PRIMARY_KEY = 4;
const SCORE_LABEL_PREFIX = 3;
const SCORE_LABEL_SUBSTRING = 2;
const SCORE_MODIFIER_OR_ID = 1;

/*
 * 【转写命中与标签同档】拼音/罗马字就是「这条操作名的另一种写法」，用户为它付出的
 * 输入成本与打原文相同（`chexiao` 与「撤销」都是 5~7 次击键），因此同档而非降档。
 * 同档时由原有的「同分保持原分组顺序」兜底，结果仍然稳定。
 */
const SCORE_TRANSLIT_PREFIX = 3;
const SCORE_TRANSLIT_SUBSTRING = 2;

/*
 * 【转写匹配的最小 token 长度】单字符 token 不该参与转写匹配：`z` 会命中所有拼音
 * 含 z 的操作名（几十条），把用户真正想找的那条冲散。字面匹配不受此限 ——
 * 单字符的字面命中与转写功能上线前一致。
 */
const MIN_TRANSLIT_TOKEN_LEN = 2;

/*
 * 【每个 token 内部不许多个来源累加 —— 只取最高一档】
 *
 * 累加会让词条多的条目靠**数量**取胜：`edit.quantize`（`Ctrl+P`）的标签
 * "Quantize"、id 片段 `quantize`、修饰键别名三处都含字母 `z`，累加得分会追平
 * 甚至超过真正绑定在 Z 上的条目。取最高一档之后，"这条有多像"由最强的那个信号
 * 决定，而不是由沾边的地方有多少决定。
 */
function hasAnyMatch(source: string[], token: string): boolean {
    return source.some((term) => term.includes(token));
}
/** 与 `hasAnyMatch` 同形，但要求词条以 token 开头。 */
function hasAnyPrefix(source: string[], token: string): boolean {
    return source.some((term) => term.startsWith(token));
}
/**
 * 查询分隔符：空白、`+`、`,`。
 *
 * 【为什么 `+` 也算】修饰键组合的自然写法是 `ctrl+shift+z`（用户也会照抄按键按钮
 * 上的显示文本）。把它当普通字符会让整串构成一个 token，在任何词条里都找不到。
 * 逗号同理：用户会一次输入多个条件。
 */
const TOKEN_SPLIT = /[\s+,]+/;

/** 去掉 UTF-16 代理对之外的空白并小写；不做其它归一化（见 `termsFromKeybinding`）。 */
function normalize(text: string): string {
    return text.toLowerCase();
}

/**
 * 把文本切成参与匹配的词。
 *
 * 【为什么中文整串保留】中文没有词分隔符，切到单字会产生大量误命中
 * （"分割音频块" 里的每个字各自去匹配，搜"分割"以外的任何单字都会命中一部分）。
 * 整串作为单一词条、由 `includes` 判定子串，才符合中文的输入习惯 ——
 * 输"撤销"命中"撤销"，"重做"不命中。英文则按空格切词，使 "split clip" 这样的
 * 多词输入能用 AND 语义分别命中。
 */
function tokenizeText(text: string): string[] {
    const lower = normalize(text);
    const words = lower.split(/[\s]+/).filter(Boolean);
    // 中文（及其他无空格语言）整串也是一个词条。
    const hasSpace = /[\s]/.test(lower);
    return hasSpace ? words : lower ? [lower] : [];
}

/** 追加词条，跳过重复与空白 —— 索引里同一词条出现多次只会徒增匹配成本。 */
function pushUnique(target: string[], value: string): void {
    const trimmed = value.trim();
    if (trimmed && !target.includes(trimmed)) target.push(trimmed);
}

/**
 * 把一条绑定拆成"主键"与"修饰键"两组词条。
 *
 * 【为什么不索引 `formatKeybinding()` 的输出】它是**平台相关**的：macOS 上
 * `ctrl:true` 渲染成 `⌘`，其余平台渲染成 `Ctrl`。同一份索引在两套平台上行为不一致，
 * 而这是同一种绑定、同一个数据。改为索引 `kb.key`（恒为小写 ASCII，如 `space` /
 * `z` / `arrowup`）后跨平台稳定，平台差异由 `KEY_SEARCH_ALIASES` 显式承担。
 *
 * 【为什么要分成两组】两组在排序里值不同的分：主键是"这条快捷键绑在哪"，修饰键
 * 是"它带什么前缀"。混在一组里就无法区别对待，见 `scoreToken` 的注释。
 */
function termsFromKeybinding(binding: Keybinding): {
    primaryKeys: string[];
    modifiers: string[];
} {
    const primaryKeys: string[] = [];
    const modifiers: string[] = [];

    /*
     * 【`modifierOnly` 的手势要把它的"主键"算作修饰键】
     *
     * 纯修饰键手势（如 "Canvas vertical zoom"）的 `key` 本身就是 `control` /
     * `alt` / `shift`。若把它算进 `primaryKeys`，它会在 `ctrl` 这个 token 上
     * 拿到主键档的 2 分 —— 而这只表示"这条手势含 Ctrl"，跟任何 Ctrl 系绑定
     * 一样普通，不该值主键的分。归到 `modifiers` 后它与所有 Ctrl 系绑定同档，
     * 排序回到"绑定本身像不像"这一个信号上。
     */
    const MODIFIER_KEY_NAMES = new Set(["control", "shift", "alt", "meta"]);
    if (binding.key && binding.key !== "__none__") {
        const key = binding.key.toLowerCase();
        const target = MODIFIER_KEY_NAMES.has(key) ? modifiers : primaryKeys;
        pushUnique(target, key);
        for (const alias of KEY_SEARCH_ALIASES[key] ?? []) pushUnique(target, alias);
    }
    /*
     * 修饰键本身也是可搜的："ctrl" 应该能找出所有 Ctrl 系绑定。
     *
     * 【为什么要走别名表而不是直接写字面量】同一个 `ctrl` 字段在 macOS 上渲染成
     * ⌘、在其余平台渲染成 Ctrl。人们习惯的叫法有五种（ctrl / control / cmd /
     * command / 以及 mac 上口述的 "command"），只写 "control" 的话搜 "cmd"
     * 会一无所获 —— 而 "cmd z" 正是 mac 用户描述"撤销"的自然方式。
     */
    if (binding.ctrl) {
        for (const alias of KEY_SEARCH_ALIASES.control) pushUnique(modifiers, alias);
    }
    if (binding.alt) {
        for (const alias of KEY_SEARCH_ALIASES.alt) pushUnique(modifiers, alias);
    }
    if (binding.shift) pushUnique(modifiers, "shift");
    return { primaryKeys, modifiers };
}

/**
 * 构建全量检索索引。
 *
 * @param resolveLabel 把 i18n 词典键解析成本地化文案。注入而非直接 import
 *   `useI18n()`，使本函数是纯函数（测试里传恒等映射即可），也让索引的重建时机
 *   由调用方按语系控制。
 * @param translitTermsOf 取一段文案的**转写形态**（拼音全拼 / 初声 / 罗马字）。
 *   同样注入：转写规则在后端（`transliterate` 命令），本函数不认识任何语言。
 *   缺省（或后端不可用）时该来源为空，匹配退化为纯字面 —— 功能不中断。
 */
export function buildKeybindingSearchEntries(
    resolveLabel: (key: string) => string,
    translitTermsOf?: (text: string) => readonly string[],
): KeybindingSearchEntry[] {
    return ALL_ACTION_IDS.map((id) => {
        const meta = ACTION_META[id];
        /*
         * 动作 id 的片段参与匹配（`clip.split` → `clip` / `split`）。
         *
         * 【为什么要它】id 是稳定标识符：**英文语系下它本身就是好线索**，且
         * 不会因为词典调整而漂移。用户搜 "split" 时，即使某天词典把标签改成了
         * "Divide clip"，仍然能命中。
         */
        const idParts = normalize(id).split(".").filter(Boolean);

        const label = resolveLabel(meta.labelKey);
        /*
         * 转写只索引**操作名**，不索引分组名。
         *
         * 【为什么不做分组】分组名是「播放与导航」这类场景词，同一分组下几十条
         * 动作共享它 —— 索引它等于给这几十条同时加一个共同词条，对「找到某一条」
         * 没有帮助，只会让按分组词搜出来的结果更拥挤。分组的定位由左侧导航栏承担。
         */
        const translit = translitTermsOf ? [...new Set(translitTermsOf(label))] : [];

        return {
            id,
            sources: {
                label: tokenizeText(label),
                group: tokenizeText(resolveLabel(GROUP_LABEL_KEYS[meta.group])),
                idParts,
                translit,
                ...termsFromKeybinding(DEFAULT_KEYBINDINGS[id]),
            },
            label,
            group: meta.group,
        };
    });
}

/**
 * 单条对一个 token 的得分；0 表示未命中。
 *
 * 【为什么每个 token 内不许多个来源累加】累加会让词条多的条目靠**数量**取胜：
 * 一个在标签、id 片段、分组名里都泛泛沾边的条目，能在每个 token 上叠出高分，
 * 压过真正绑在该键上的条目。取最高一档之后，"这条有多像"由最强的那个信号决定，
 * 而不是由沾边的地方有多少决定。
 */
function scoreToken(entry: KeybindingSearchEntry, token: string): number {
    const { sources } = entry;
    const translitScore =
        token.length >= MIN_TRANSLIT_TOKEN_LEN
            ? hasAnyPrefix(sources.translit, token)
                ? SCORE_TRANSLIT_PREFIX
                : hasAnyMatch(sources.translit, token)
                  ? SCORE_TRANSLIT_SUBSTRING
                  : 0
            : 0;
    return Math.max(
        normalize(entry.label).startsWith(token) ||
            sources.label.some((term) => term.startsWith(token))
            ? SCORE_LABEL_PREFIX
            : hasAnyMatch(sources.label, token)
              ? SCORE_LABEL_SUBSTRING
              : 0,
        hasAnyMatch(sources.primaryKeys, token) ? SCORE_PRIMARY_KEY : 0,
        hasAnyMatch(sources.idParts, token) ? SCORE_MODIFIER_OR_ID : 0,
        hasAnyMatch(sources.group, token) ? SCORE_MODIFIER_OR_ID : 0,
        hasAnyMatch(sources.modifiers, token) ? SCORE_MODIFIER_OR_ID : 0,
        translitScore,
    );
}

/**
 * 按查询过滤并排序。
 *
 * - 空查询：返回全量、**保持原有分组顺序**（条例有序，结果不跳动）。
 * - 非空：多 token 取 **AND**（每个 token 都必须命中），按各 token 得分之和降序；
 *   同分保持原顺序，使同样查询的结果稳定。
 *
 * 【为什么是 AND 而不是 OR】OR 会让 `ctrl z` 返回全部 Ctrl 系绑定 —— 那是 30 多条。
 * AND 才是用户在输入第二个词时的预期：**继续缩小**，而不是撒开。
 */
export function matchKeybindingEntries(
    entries: KeybindingSearchEntry[],
    query: string,
): KeybindingSearchEntry[] {
    const tokens = query.trim().toLowerCase().split(TOKEN_SPLIT).filter(Boolean);
    if (tokens.length === 0) return entries;

    const scored: Array<{ entry: KeybindingSearchEntry; index: number; score: number }> = [];
    entries.forEach((entry, index) => {
        let total = 0;
        for (const token of tokens) {
            const score = scoreToken(entry, token);
            if (score === 0) return; // AND：一个 token 不命中即整条淘汰
            total += score;
        }
        scored.push({ entry, index, score: total });
    });

    // 同分时按原序（`index`），保证多次相同查询得到同一份排序。
    scored.sort((a, b) => b.score - a.score || a.index - b.index);
    return scored.map((item) => item.entry);
}
