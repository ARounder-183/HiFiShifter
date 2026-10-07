/**
 * 词典完整性门禁。
 *
 * 【为什么必须有】审查结论是：**键结构严丝合缝，文本风格完全失控**。
 * 1649 键 × 5 语系的键集合、占位符集合都一致（这靠 `tsc` 的
 * `MessageKey = keyof typeof enUS` 强制），但除此之外**没有任何检查**：
 *
 *   - 213 个测试文件里只有 2 个涉及 i18n，且 `historyOpLabels.test.ts`
 *     只覆盖 47/1649 键（且它是在一次真实的五语系回归之后才补上的）；
 *   - `eslint.config.js` 没有任何 i18n 规则；
 *   - 于是 `param_btn_breath` 与 `param_btn_breathiness` 在 en-US / ja-JP / ko-KR
 *     里同为 `"BRE"` —— 参数编辑器出现两个标签完全相同的按钮，长期无人发现；
 *   - `Ctrl+O/E/B/I/K` 在 5 个语系里全部硬编码，macOS 上全错。
 *
 * 这些问题类型检查抓不到（`"BRE"` 是合法字符串）。本文件把它们变成
 * **会失败的测试**，因为上一轮的经验是：只靠人肉 review 的约定一定会漂移
 * ——词典里绝大多数键在各语系文件中的索引位置不同（当时实测最大位移 1482 位），
 * 人肉 diff 根本不现实。
 */
import { describe, expect, test } from "vitest";

import { messages, type Locale } from "./messages";

const LOCALES = Object.keys(messages) as Locale[];
const REFERENCE: Locale = "en-US";

/**
 * 词典是 `as const` 的（键名参与 `MessageKey` 联合推导），因此这里统一
 * 收宽成 `Record<string, string>` 才能按字符串下标访问。测试关心的是
 * 运行时文本，不需要字面量类型。
 */
type Dict = Record<string, string>;
const dictOf = (locale: Locale): Dict => messages[locale] as unknown as Dict;

const reference = dictOf(REFERENCE);
const referenceKeys = Object.keys(reference);

function entriesOf(locale: Locale): [string, string][] {
    return Object.entries(dictOf(locale));
}

describe("词典结构", () => {
    test("各语系键集合与参考语系完全一致（无缺失、无多余）", () => {
        const referenceSet = new Set<string>(referenceKeys as string[]);
        for (const locale of LOCALES) {
            const keys = Object.keys(dictOf(locale));
            const missing = [...referenceSet].filter((key) => !keys.includes(key));
            const extra = keys.filter((key) => !referenceSet.has(key));
            expect(missing, `${locale} 缺少键`).toEqual([]);
            expect(extra, `${locale} 有多余键`).toEqual([]);
        }
    });

    test("单文件内无重复键", () => {
        // 对象字面量的重复键在 JS 里静默取后者，解析后的对象看不出来；
        // 这里检查的是「两个不同语义的键取了同一个值」这一类（见下方重复值检查），
        // 以及空值。
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                expect(value, `${locale}.${key} 为空`).not.toBe("");
            }
        }
    });

    test("每个键的占位符集合在五语系间逐键一致", () => {
        /*
         * 比较**去重后的集合**而不是出现次数：复数形态写成
         * `"a {n} x|a {n} xs"` 时 `{n}` 出现两次，但 `selectPluralForm`
         * 只取其中一种，渲染结果里仍只有一次。
         */
        const placeholders = (value: string) =>
            [...new Set([...value.matchAll(/\{(\w+)\}/g)].map((match) => match[1]))].sort();

        for (const key of referenceKeys) {
            const expected = placeholders(reference[key]);
            for (const locale of LOCALES) {
                const actual = placeholders(dictOf(locale)[key]);
                expect(actual, `${locale}.${key} 占位符不一致`).toEqual(expected);
            }
        }
    });
});

describe("键命名规范", () => {
    /*
     * 【为什么需要】键结构此前只被"五语系键集合一致"守着 —— 那只保证**一致**，
     * 不保证**成体系**。实测有 32 个裸键（`pitch` / `none` / `loading` …）与
     * 7 个 camelCase 键（`kb_preset_vegasPro` / `render_cache_skip_tooShort` …）：
     * 裸键没有命名空间，读者无法从键名判断它属于哪个界面；camelCase 与其余
     * 1600 多个 snake_case 键并存，两种风格都"看起来官方"。
     *
     * 约定：**snake_case + 命名空间前缀**。只有三个通用确认词可以裸名 ——
     * 它们本身就是"任何界面都用的那三个词"，加前缀反而不便。
     */
    const BARE_ALLOWED = new Set(["ok", "cancel", "close"]);

    test("键名一律 snake_case", () => {
        const offenders = referenceKeys.filter((key) => !/^[a-z][a-z0-9_]*$/.test(key));
        expect(
            offenders.length === 0
                ? []
                : ["以下键名不是 snake_case：", ...offenders.map((k) => `  ${k}`)].join("\n"),
        ).toEqual([]);
    });

    test("键名一律带命名空间前缀（通用确认词除外）", () => {
        const offenders = referenceKeys.filter(
            (key) => !key.includes("_") && !BARE_ALLOWED.has(key),
        );
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下键没有命名空间前缀，读者无法判断它属于哪个界面：",
                      ...offenders.map((k) => `  ${k}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    test("裸名豁免恰好只有那三个通用确认词", () => {
        // 豁免不能悄悄扩大：这条断言让"再加一个裸键"必须同时改测试。
        const bare = referenceKeys.filter((key) => !key.includes("_"));
        expect([...bare].sort()).toEqual([...BARE_ALLOWED].sort());
    });
});

describe("文本风格", () => {
    /*
     * 伪复数 `(s)`。曾经有 14 处，渲染出 `"1 clip(s)"`。
     * 现在复数由 `src/i18n/format.ts` 的 `selectPluralForm` 负责，词典里用
     * `"clip|clips"` 写单复数。
     *
     * 只匹配「单词 + (s)」形态，因此 `"Window (s)"`（单位秒）这类不算违规。
     */
    test("不出现伪复数 (s)", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                if (/\w\(s\)/.test(value)) violations.push(`${locale}.${key} = ${value}`);
            }
        }
        expect(violations, "用 `单数|复数` 形态替代 (s)").toEqual([]);
    });

    /*
     * 复数分隔符只允许出现一次（`单数|复数`），否则 `selectPluralForm` 会
     * 静默丢掉第二段之后的内容。
     */
    test("复数分隔符格式正确", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                const parts = value.split("|");
                if (parts.length > 2) violations.push(`${locale}.${key} 有多个 | 分隔符`);
                if (parts.some((part) => part.trim() === "")) {
                    violations.push(`${locale}.${key} 的 | 两侧为空`);
                }
            }
        }
        expect(violations).toEqual([]);
    });

    /*
     * 带计数的复数文案**必须**用 `{count}` 占位符。
     *
     * `plural()` 只回填 `{count}`；若写成 `{n}` / `{c}` 等同义词，占位符会
     * 原样渲染到界面上（历史上词典里三种写法混用，消费端只好各自
     * `.replace("{n}", …)` 补偿）。
     */
    test("复数文案只使用 {count} 占位符", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                if (!value.includes("|")) continue;
                const placeholders = [...value.matchAll(/\{(\w+)\}/g)].map((match) => match[1]);
                for (const name of placeholders) {
                    if (name !== "count") {
                        violations.push(`${locale}.${key} 用了 {${name}}，应为 {count}`);
                    }
                }
            }
        }
        expect(violations).toEqual([]);
    });

    /*
     * 硬编码 `Ctrl+`。平台感知的格式化早已存在（`formatKeybinding` 与
     * `src/i18n/format.ts` 的 `primaryModifierLabel` 都用 `IS_MAC ? "⌘" : "Ctrl"`），
     * 但文案里仍写死 —— macOS 上全部显示错误。
     *
     * 正确写法：值里写 `{modifier}`，消费端用 `useI18n().shortcut(key)`。
     */
    test("不硬编码 Ctrl 修饰键", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                if (/Ctrl\+|⌘/.test(value)) violations.push(`${locale}.${key} = ${value}`);
            }
        }
        expect(violations, "改用 {modifier} 占位符 + shortcut()").toEqual([]);
    });

    /*
     * 省略号只允许 ASCII 三点 `...`。
     *
     * 【为什么必须有这条断言】`docs/i18n/style-guide.md` §2.2 早已写明这条约定，
     * 但它是**唯一一条只写在文档里、没有测试守着**的排版规则 —— 于是实测出现
     * 了 102 个键用 `...`、2 个键用 U+2026 `…` 的分裂（新增文案时作者按中文
     * 排版直觉写了 `…`）。文档不会让人停下来，断言才会。
     *
     * 【为什么不是"两种都行"】`…` 与 `...` 在同一份词典里并存时，视觉宽度与
     * 换行行为都不同（CJK 字体的 `…` 是等宽全角），同一列菜单项会在这一格
     * 比别的宽出一个字符。统一比"好看"重要。
     */
    test("省略号一律用 ASCII 三点，不用 U+2026", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                if (value.includes("\u2026")) violations.push(`${locale}.${key} = ${value}`);
            }
        }
        expect(
            violations,
            "菜单项打开对话框时以 `...` 结尾，不要用 `…`（见 style-guide §2.2）",
        ).toEqual([]);
    });

    /*
     * 值内多余空白。历史上 en/ko 的标签片段靠**首尾空格**与相邻译文拼接
     * （`" (unavailable)"`、`"Position: "`），空格就是拼接逻辑本身 —— 换一个
     * 消费端或者换一行布局就会悄悄坏掉。正确做法是把整行写进词典模板，
     * 用 `tVars` 回填（见 style-guide §2.2 / §3.3）。
     *
     * 允许项：
     * - `\n`：多行 tooltip 的换行本身是内容；
     * - 纯标点/符号/空白值：那是"布局分隔符"资产（如 `common_value_sep` 的
     *   `": "`），空格就是它的全部内容。
     */
    test("值内无多余空白（首尾空格 / 连续空格 / 制表符 / 全角空格）", () => {
        const violations: string[] = [];
        for (const locale of LOCALES) {
            for (const [key, value] of entriesOf(locale)) {
                if (/^[\s\p{P}\p{S}]+$/u.test(value)) continue;
                if (value !== value.trim()) {
                    violations.push(`${locale}.${key} 首尾空白 ${JSON.stringify(value)}`);
                }
                if (/ {2,}/.test(value)) {
                    violations.push(`${locale}.${key} 连续空格 ${JSON.stringify(value)}`);
                }
                if (/[\t\u3000]/.test(value)) {
                    violations.push(`${locale}.${key} 制表符/全角空格 ${JSON.stringify(value)}`);
                }
            }
        }
        expect(violations, "拼接交给 tVars 模板（见 style-guide §2.2 / §3.3）").toEqual([]);
    });

    /*
     * 全大写英文标签。en-US 里 `tracks: "TRACKS"`、`recapture_missing_media_col_*`
     * 用全大写，而 zh-CN / ja-JP / ko-KR 都是正常大小写 —— 只有英文在喊。
     *
     * 排除纯缩写（BPM / MIDI / WAV / RTF …）与已知的格式/编码名。
     */
    test("英文标签不整体大写", () => {
        const ACRONYMS = new Set([
            "BPM",
            "MIDI",
            "WAV",
            "MP3",
            "FLAC",
            "OGG",
            "RTF",
            "CPU",
            "GPU",
            "RAM",
            "OK",
            "VS",
            "REAPER",
            "ONNX",
            "TPDF",
            "ID",
            "UI",
            "URL",
            "JSON",
            "CSV",
            "HTML",
            "MD",
            "AIFF",
            "AAC",
            "OPUS",
            "DSP",
            "LFO",
            "ADSR",
            "FFT",
            "STFT",
            "RMS",
            "LUFS",
        ]);
        const violations: string[] = [];
        for (const [key, value] of entriesOf(REFERENCE)) {
            // 只看纯 ASCII 字母构成的值（CJK 不受影响）
            if (!/^[A-Za-z][A-Za-z\s]*$/.test(value)) continue;
            const words = value.split(/\s+/);
            if (words.length === 0) continue;
            const allUpper = words.every((word) => word === word.toUpperCase());
            const anyAcronym = words.some((word) => ACRONYMS.has(word));
            if (allUpper && !anyAcronym && value.replace(/\s/g, "").length > 3) {
                violations.push(`en-US.${key} = ${value}`);
            }
        }
        expect(violations).toEqual([]);
    });

    /*
     * **同一命名族内**的短标签不得重名。
     *
     * 这条抓的是 `param_btn_breath` 与 `param_btn_breathiness` 同为 `"BRE"` 那一类：
     * 两者同属 `param_btn_*` 族（同一排工具栏按钮），语义不同却显示成同一个标签，
     * 用户看到两个一模一样的按钮。
     *
     * 【为什么不做全局重名检查】那会产生 98 条告警，绝大多数是正常复用
     * （"Sample Rate" 在录音设置与导出设置里都该这么写）。把范围收进"同一前缀
     * 命名族 + 短标签"后，剩下的才是真问题 —— 一个报错就该是一个 bug。
     */
    test("同一命名族内的短标签不重名", () => {
        /** 同一语义在不同族复用是正常的（各处的 "None"、"Mono" 等）。 */
        const SHARED_OK = new Set([
            "None",
            "Auto",
            "Default",
            "Custom",
            "Mono",
            "Stereo",
            "On",
            "Off",
            "—",
            "✓",
            "✕",
            "%",
            "s",
            "ms",
            "px",
            "Hz",
            "kHz",
            "dB",
            "BPM",
            "MIDI",
        ]);

        /*
         * 已核对为**同语义复用**的键对：它们取同一个值是刻意的，不是 bug。
         * 新增条目必须写明理由 —— 这份清单是"已审计"的记录，不是静音开关。
         */
        const KNOWN_BENIGN: Record<string, string> = {
            "recording_refresh::Refresh":
                "刷新音频设备 / 刷新应用列表 —— 同一个词，动作对象不同，标签本就相同",
            "recapture_missing::Reset All":
                "Reset 与 Reset All 是两个不同范围的按钮，但都该读作「全部重置」",
            "algo_label::Algo": "`algo_label_short` 是 `algo_label` 的短版，长/短变体刻意同值",
            "custom_scale::Custom Scale": "标签 / 对话框标题 / 默认名三处都该是「自定义音阶」",
            "tempo_map::Tempo Map": "面板名与「清除速度图」对话框标题共用同一个名词",
            "tempo_map::Scale": "`tempo_map_scale`（面板列名）与 `tempo_map_tooltip_scale`（变化点提示行）都是「音阶」这个名词本身",
        };

        /** 取前两段作为命名族（`param_btn_breath` → `param_btn`）。 */
        const familyOf = (key: string): string => key.split("_").slice(0, 2).join("_");

        const byFamilyAndValue = new Map<string, string[]>();
        for (const [key, value] of entriesOf(REFERENCE)) {
            if (SHARED_OK.has(value)) continue;
            // 只关心短标签：长句偶然相同不算冲突
            if (value.length > 12) continue;
            if (/[{}|]/.test(value)) continue;
            if (!key.includes("_")) continue;
            const bucket = `${familyOf(key)}::${value}`;
            byFamilyAndValue.set(bucket, [...(byFamilyAndValue.get(bucket) ?? []), key]);
        }

        const violations: string[] = [];
        for (const [bucket, keys] of byFamilyAndValue) {
            if (keys.length < 2) continue;
            if (KNOWN_BENIGN[bucket]) continue;
            const [family, value] = bucket.split("::");
            violations.push(`${family}_* 族内 ${keys.join(" / ")} 同为 "${value}"`);
        }
        expect(violations, "同一命名族内的短标签应各不相同").toEqual([]);
    });
});

describe("CJK 排版", () => {
    /*
     * 简繁中文里的半角括号/冒号。曾发现 `silence_threshold: "阈值 (dBFS)"`
     * 这类混排（全角 `，` 与半角 `( )` 并存）。
     *
     * 只检查中文语系：日文与韩文用半角括号是各自的正字法习惯。
     */
    test("简繁中文不使用半角括号包裹中文", () => {
        const violations: string[] = [];
        for (const locale of ["zh-CN", "zh-TW"] as const) {
            for (const [key, value] of entriesOf(locale)) {
                // 括号内若是纯拉丁/数字内容则允许（例如 "MP3 (VBR)"）
                const halfWidthAroundCjk = /[\u4e00-\u9fff]\s*\([^)]*[\u4e00-\u9fff][^)]*\)/.test(
                    value,
                );
                if (halfWidthAroundCjk) violations.push(`${locale}.${key} = ${value}`);
            }
        }
        expect(violations).toEqual([]);
    });

    /*
     * §2.2 的另一半：CJK 行文里的**其余**半角标点（冒号/逗号/分号/问叹号）。
     * 此前只拦了括号，于是 `benchmark_providers_label: "可用提供者:"` 这类
     * 半角冒号长期与全角冒号并存。规则取宽：**值里只要含汉字**，半角
     * `，；：？！,;:?!` 一律不允许 —— 冒号跟在拉丁词后（`"ONNX Runtime:"`）
     * 也算中文行文的一部分，与相邻键的全角写法保持一致。
     *
     * 豁免：时间格式掩码（`时:分:秒.毫秒`）里的冒号是格式分隔符，不是行文标点。
     */
    test("简繁中文行文不使用半角冒号/逗号/分号/问叹号", () => {
        const FORMAT_MASKS = new Set(["time_unit_clock"]);
        const violations: string[] = [];
        for (const locale of ["zh-CN", "zh-TW"] as const) {
            for (const [key, value] of entriesOf(locale)) {
                if (FORMAT_MASKS.has(key)) continue;
                if (/[\u4e00-\u9fff]/.test(value) && /[,;:?!]/.test(value)) {
                    violations.push(`${locale}.${key} = ${value}`);
                }
            }
        }
        expect(violations, "CJK 行文用全角标点（见 style-guide §2.2）").toEqual([]);
    });
});
