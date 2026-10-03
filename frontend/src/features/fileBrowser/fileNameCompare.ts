/**
 * 文件名的自然序比较 —— 目标是与 Windows 资源管理器的顺序一致。
 *
 * 【为什么不能用 `Intl.Collator` 直接排】实测（Node/ICU，与 WebView2 同源）：
 *
 * 1. **汉字与拉丁的先后相反**。`Intl.Collator("zh-CN").compare("a", "中")` 返回 1 ——
 *    ICU 的 zh 排序把汉字排在拉丁字母**之前**，而资源管理器把拉丁排在前面。
 *    于是"先英文后中文"变成了"先中文后英文"。
 * 2. **标点顺序不同**。`compare("_", ".")` 返回 -1（ICU 的 DUCET 把下划线排在句点
 *    之前），而资源管理器按码点把 `.` 排在 `_` 之前。后果是 `song.wav` 排到了
 *    `song_extra.wav` **之后** —— 前缀更短的文件反而在后面，整组看上去像是倒序。
 *
 * 这两条都不是"配置一下 collator 就能改"的选项：ICU 的标点权重与脚本先后是
 * 排序表的固定内容。因此这里自己做逐字符比较，只在**字母之间**借语系排序器
 * （中文按拼音、忽略大小写与变音符号），其余按明确规则处理。
 *
 * 规则（对齐资源管理器）：
 *   - 字符分三类，符号/空白 < 数字 < 字母；
 *   - 符号之间按**码点**（`.` 在 `_` 之前）；
 *   - 数字串按**数值**（`2` 在 `10` 之前）；
 *   - 字母之间：非 CJK 在 CJK 之前，同类内交给语系排序器；
 *   - 一方是另一方的前缀时，短的在前。
 */

/**
 * 构造语系排序器：只用于**字母之间**的比较。
 *
 * `numeric` 在这里用不上（单字符比较），但保留它与面板的取值一致；
 * `sensitivity: "base"` 让大小写与变音符号不参与比较 —— 资源管理器同样忽略它们。
 * 语系缺省取运行环境默认值（= 系统区域），与资源管理器"按系统区域排序"一致。
 */
function buildLetterCollator(locale?: string | string[]): Intl.Collator {
    return new Intl.Collator(locale, {
        numeric: true,
        sensitivity: "base",
    });
}

/** 文件名比较器签名（可直接交给 `Array.prototype.sort`）。 */
export type FileNameComparator = (a: string, b: string) => number;

/** 类别：0 = 符号/空白，1 = 数字，2 = 字母（含其它数字系统与组合记号）。 */
const SYMBOL = 0;
const DIGIT = 1;
const LETTER = 2;

/** ASCII 走查表：绝大多数字符都在这里，省掉正则与 Map 查找。 */
const ASCII_RANK = new Uint8Array(128);
for (let code = 0; code < 128; code += 1) {
    if (code >= 0x30 && code <= 0x39) ASCII_RANK[code] = DIGIT;
    else if ((code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a))
        ASCII_RANK[code] = LETTER;
    else ASCII_RANK[code] = SYMBOL;
}

/** 非 ASCII 的分类缓存（同一个字符在一次排序里会被问到很多次）。 */
const nonAsciiRank = new Map<number, number>();
const LETTER_LIKE = /[\p{L}\p{M}\p{N}]/u;

function rankOf(codePoint: number): number {
    if (codePoint < 128) return ASCII_RANK[codePoint];
    const cached = nonAsciiRank.get(codePoint);
    if (cached !== undefined) return cached;
    const char = String.fromCodePoint(codePoint);
    // 只有 ASCII 的 0-9 走"数值比较"；全角数字等按字母处理，交给语系排序器。
    const rank = LETTER_LIKE.test(char) ? LETTER : SYMBOL;
    nonAsciiRank.set(codePoint, rank);
    return rank;
}

/** 是否 CJK 字符（汉字 / 假名 / 谚文）。用于把拉丁排到汉字之前。 */
function isCjk(codePoint: number): boolean {
    return (
        (codePoint >= 0x2e80 && codePoint <= 0x9fff) || // 部首扩展 ~ CJK 统一表意文字
        (codePoint >= 0xf900 && codePoint <= 0xfaff) || // CJK 兼容表意文字
        (codePoint >= 0x20000 && codePoint <= 0x3ffff) || // 扩展 B 及以后
        (codePoint >= 0x3040 && codePoint <= 0x30ff) || // 平假名 / 片假名
        (codePoint >= 0xac00 && codePoint <= 0xd7af) || // 谚文音节
        (codePoint >= 0x1100 && codePoint <= 0x11ff) // 谚文字母
    );
}

function step(codePoint: number): number {
    return codePoint > 0xffff ? 2 : 1;
}

/**
 * 用给定语系排序器比较两个文件名。返回负数表示 `a` 在前，正数表示 `b` 在前，
 * 0 表示等价（例如只有大小写不同 —— 与资源管理器一致，二者视为同级）。
 */
function compareWithCollator(collator: Intl.Collator, a: string, b: string): number {
    let i = 0;
    let j = 0;

    while (i < a.length && j < b.length) {
        const ca = a.codePointAt(i) as number;
        const cb = b.codePointAt(j) as number;
        const ra = rankOf(ca);
        const rb = rankOf(cb);

        // 类别不同：符号 < 数字 < 字母。
        if (ra !== rb) return ra - rb;

        if (ra === DIGIT) {
            let endA = i;
            let endB = j;
            while (endA < a.length && rankOf(a.codePointAt(endA) as number) === DIGIT) endA += 1;
            while (endB < b.length && rankOf(b.codePointAt(endB) as number) === DIGIT) endB += 1;
            // 去前导零后按"位数 → 字典序"比较，即按数值比较且不受前导零干扰。
            const numA = a.slice(i, endA).replace(/^0+(?=\d)/, "");
            const numB = b.slice(j, endB).replace(/^0+(?=\d)/, "");
            if (numA.length !== numB.length) return numA.length - numB.length;
            if (numA !== numB) return numA < numB ? -1 : 1;
            // 数值相同（`1` 与 `01`）：位数少者在前。
            if (endA - i !== endB - j) return endA - i - (endB - j);
            i = endA;
            j = endB;
            continue;
        }

        if (ca !== cb) {
            if (ra === SYMBOL) {
                // 符号按码点：资源管理器把 `.` 排在 `_` 之前，而 ICU 相反。
                return ca < cb ? -1 : 1;
            }
            // 字母：非 CJK 在前（资源管理器"先英文后中文"），同类内按语系排序。
            const cjkA = isCjk(ca);
            const cjkB = isCjk(cb);
            if (cjkA !== cjkB) return cjkA ? 1 : -1;
            const order = collator.compare(String.fromCodePoint(ca), String.fromCodePoint(cb));
            if (order !== 0) return order < 0 ? -1 : 1;
            // 等价（例如只有大小写不同）：继续比后面的字符。
        }

        i += step(ca);
        j += step(cb);
    }

    // 一方是另一方的前缀：短的在前（`a.wav` 在 `a_1.wav` 之前）。
    return a.length - i - (b.length - j);
}

/**
 * 用指定区域构造文件名比较器。
 *
 * 生产代码请用下方的 `compareFileNames`（跟随系统区域，与资源管理器一致）。
 * 本工厂用于让依赖语系的分支可被确定性测试：`"zh-CN"` 下 ICU 对汉字按拼音排序
 * （甲 jiǎ 在 乙 yǐ 之前），`"en-US"` 下退化为码点序（乙 U+4E59 在 甲 U+7532
 * 之前），两者结果相反 —— 不固定区域时，断言会随宿主默认区域漂移（CI runner
 * 是 en-US，开发机常见 zh-CN）。
 */
export function createFileNameComparator(locale?: string | string[]): FileNameComparator {
    const collator = buildLetterCollator(locale);
    return (a, b) => compareWithCollator(collator, a, b);
}

/** 生产用比较器：跟随系统区域（与资源管理器一致）。 */
export const compareFileNames: FileNameComparator = createFileNameComparator();
