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

/**
 * 翻译函数：把键翻成当前语言的模板（含 `{n}` / `{m}` / `{p}` 占位符）。
 *
 * `count` 给出时按键选复数形态（词典里写作 `"单数|复数"`）。状态行里有相当
 * 一部分是"数量 + 名词"，此前用字面 `(s)` 糊过去，会渲染出 `"1 clip(s)"`。
 */
export type TranslateFn = (key: string, count?: number) => string;

/**
 * 解析状态行。
 *
 * 匹配顺序即优先级：精确键 → 假立体声扫描的多数量模板 → 单数量模板 → 前缀键
 * → 原样返回。
 */
export function resolveStatusText(status: string, statusKey: StatusKeyMap, t: TranslateFn): string {
    if (!status) return status;
    const exact = statusKey[status];
    if (exact) return t(exact);

    // 假立体声扫描**一个候选都没有**时的原因。全部按后端返回的真实计数成句，
    // 不做猜测 —— 猜出来的原因会随口断言"你的选区在工程里找不到"。
    const noClips = status === "Fake-stereo scan: project has no clips";
    if (noClips) return t("status_fake_stereo_scan_no_clips");
    const unmatched = status.match(
        /^Fake-stereo scan: range matches no clip \((\d+) in project\)$/,
    );
    if (unmatched) {
        return t("status_fake_stereo_scan_range_unmatched").replace("{p}", unmatched[1] ?? "0");
    }
    const noTakes = status.match(/^Fake-stereo scan: (\d+) clip\(s\) have no takes$/);
    if (noTakes) {
        return t("status_fake_stereo_scan_no_takes", Number(noTakes[1] ?? 0));
    }
    const noSource = status.match(/^Fake-stereo scan: (\d+) take\(s\) have no source$/);
    if (noSource) {
        return t("status_fake_stereo_scan_no_source", Number(noSource[1] ?? 0));
    }
    if (status === "Fake-stereo scan: nothing to decide") {
        return t("status_fake_stereo_scan_nothing");
    }

    // 假立体声扫描：分数形计数（{m}=已折叠/可折叠数，{n}=总数）+ 试扫/实扫两种
    // 变体。形如 "Fake-stereo scan: 3/5 folded"。
    const scan = status.match(/^Fake-stereo scan: (\d+)\/(\d+) (foldable|folded)$/);
    if (scan) {
        const key =
            scan[3] === "foldable"
                ? "status_fake_stereo_scan_foldable"
                : "status_fake_stereo_scan_folded";
        return t(key)
            .replace("{m}", scan[1] ?? "0")
            .replace("{n}", scan[2] ?? "0");
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
