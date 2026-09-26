//! 状态行文案解析：把后端/切片写出的英文原文映射成当前语言的文案。
//!
//! 状态行是"后端产出 + 前端翻译"的混合体：有些状态是固定串（查 `statusKey`
//! 表即可），有些带**数量**甚至**多个数量**，需要先把数字抠出来再回填 i18n
//! 模板的占位符。
//!
//! 抽成纯函数是为了可测：这些正则一旦写错，表现是"非英语语系下状态行变成
//! 英文原文"（不会报错、不会崩溃，只会静默降级），只有单元测试能钉住。

/** 把状态原文映射到 i18n 键；`undefined` 表示没有对应键。 */
export type StatusKeyMap = Record<string, string | undefined>;

/** 翻译函数：把键翻成当前语言的模板（含 `{n}` / `{m}` / `{p}` 占位符）。 */
export type TranslateFn = (key: string) => string;

/**
 * 解析状态行。
 *
 * 匹配顺序即优先级：精确键 → 多数量模板 → 单数量模板 → 前缀键 → 原样返回。
 * 顺序不能随意调换：`status_fake_stereo_scan_folded` 的原文同时能被单数量
 * 正则的部分前缀命中，先走多数量分支才能拿到 `{m}` / `{p}`。
 */
export function resolveStatusText(
    status: string,
    statusKey: StatusKeyMap,
    t: TranslateFn,
): string {
    if (!status) return status;
    const exact = statusKey[status];
    if (exact) return t(exact);

    // "一条都没折叠"时的差异量级提示（见 sessionSlice 的写入侧）：独立成句，
    // 与计数句分开解析 —— 若把"最接近的差异"并进计数模板，键的数量会随
    //（试扫/实扫 × 有/无读不到 × 有/无提示）组合爆炸。
    const HINT_SEPARATOR = " — nearest ";
    const hintAt = status.indexOf(HINT_SEPARATOR);
    const hint =
        hintAt >= 0
            ? t("status_fake_stereo_scan_nearest_hint").replace(
                  "{d}",
                  status.slice(hintAt + HINT_SEPARATOR.length),
              )
            : "";
    const base = hintAt >= 0 ? status.slice(0, hintAt) : status;
    if (hint && base !== status) {
        const resolved = resolveStatusText(base, statusKey, t);
        // base 没能被翻译（原样返回）时不要把英文原文和中文提示拼在一起。
        if (resolved !== base) return resolved + hint;
    }

    // 假立体声扫描：两个数量 + 两种变体（试扫 / 实扫）+ 可选的"本次没读到"。
    // 形如 "Fake-stereo scan: 5 take(s), 3 folded to mono, 2 unreadable"。
    const scan = status.match(
        /^Fake-stereo scan: (\d+) take\(s\), (\d+) (foldable|folded to mono)(?:, (\d+) unreadable)?$/,
    );
    if (scan) {
        const foldable = scan[3] === "foldable";
        const unreadable = Number(scan[4] ?? 0);
        const key = foldable
            ? unreadable > 0
                ? "status_fake_stereo_scan_foldable_pending"
                : "status_fake_stereo_scan_foldable"
            : unreadable > 0
              ? "status_fake_stereo_scan_folded_pending"
              : "status_fake_stereo_scan_folded";
        return t(key)
            .replace("{n}", scan[1] ?? "0")
            .replace("{m}", scan[2] ?? "0")
            .replace("{p}", String(unreadable));
    }

    // 带数量的状态：提取数字回填占位符模板（如 "Waveform cache cleared (3 files)"）。
    const counted = status.match(/^(.+?)\s*\((\d+)\s*\w+\)$/);
    if (counted) {
        const baseKey = statusKey[counted[1]];
        if (baseKey && t(baseKey).includes("{n}")) {
            return t(baseKey).replace("{n}", counted[2] ?? "0");
        }
    }

    // 前缀匹配：支持 "Export done — path" 等带后缀的状态。
    for (const key of Object.keys(statusKey)) {
        if (status.startsWith(key) && status.length > key.length) {
            return t(statusKey[key] as string) + status.slice(key.length);
        }
    }
    return status;
}
