/**
 * 颤音预设文件（导入 / 导出）的序列化与解析。
 *
 * 【格式】
 * ```json
 * { "kind": "hifishifter-vibrato-presets", "version": 1, "presets": [ ... ] }
 * ```
 *
 * 【为什么有 `kind`】这个应用的 `.json` 文件至少有三种（布局 / 外观主题 / 预设），
 * 用户拿错文件是最常见的失误。外观主题导入当时只靠"字段形状像不像"来判断，
 * 一个恰好有 `name`/`colors` 的别的文件会被误收。显式的 kind 让"这是不是我们的
 * 文件"成为一次精确比较。
 *
 * 【为什么有 `version`】主题格式经历过 v1→v2 演进，预设大概率也会。有版本号，
 * 将来加字段时才能区分"旧文件补默认值"与"新文件比应用新"——两者的提示语完全
 * 不同（后者是"请升级应用"，不是"文件不对"）。
 *
 * 【单预设与多预设同一格式】导出当前预设 = 数组里一个元素。一个解析器吃两种
 * 文件，不存在两套代码。
 */

import { createVibratoPresetId, sanitizeVibratoPreset } from "./vibratoPresets";
import type { VibratoPreset } from "./vibratoTypes";

/** 文件类型标记。全等匹配，用于拒收拿错的文件。 */
export const VIBRATO_PRESET_FILE_KIND = "hifishifter-vibrato-presets";

/** 当前可读写的文件格式版本。 */
export const VIBRATO_PRESET_FILE_VERSION = 1;

/** 解析失败的分类。UI 据此给不同的话（见 `vibrato_io_*` 词条）。 */
export type VibratoPresetFileError =
    /** 不是合法 JSON。 */
    | "badJson"
    /** `kind` 不匹配 —— 极可能是拿错了文件。 */
    | "wrongKind"
    /** `version` 大于本应用支持的版本：文件比应用新。 */
    | "newerVersion"
    /** 头部对，但 `presets` 缺失或不是数组。 */
    | "noPresets";

export type VibratoPresetFileResult =
    | { ok: true; presets: VibratoPreset[] }
    | { ok: false; error: VibratoPresetFileError };

/**
 * 序列化预设为文件文本（2 空格缩进，便于手工查看与 diff）。
 *
 * id 原样写出：导出是为了完整回放，导入侧会重新生成 id —— 两边各司其职，
 * 谁都不欠谁的。
 */
export function serializeVibratoPresets(presets: readonly VibratoPreset[]): string {
    return JSON.stringify(
        {
            kind: VIBRATO_PRESET_FILE_KIND,
            version: VIBRATO_PRESET_FILE_VERSION,
            presets,
        },
        null,
        2,
    );
}

/**
 * 解析预设文件文本。
 *
 * 【id 一律丢弃并重新生成】两个理由：文件里的 id 可能与本地撞车（`upsert` 按 id
 * 覆盖，会**替换掉用户已有的**同名预设）；文件里可能带着 `builtin.` 前缀
 * （`sanitize` 会把它标成系统预设，而 `upsertVibratoPreset` 拒收系统预设 ——
 * 那条目就静默丢了）。导入的**永远**成为新的用户预设。
 *
 * 【逐条净化】复用 `sanitizeVibratoPreset` —— 与设置加载走同一条净化路径，
 * 手改过的文件、旧版本导出的文件全部收敛到合法值域，不会出现"导入一套规则、
 * 设置加载另一套"的漂移。
 *
 * @returns `ok: false` 时 `error` 指明原因；`ok: true` 而 `presets` 为空表示
 *   文件合法但里面没有预设（UI 应说"文件里没有预设"，而不是当成功）。
 */
export function parseVibratoPresets(json: string): VibratoPresetFileResult {
    let parsed: unknown;
    try {
        parsed = JSON.parse(json);
    } catch {
        return { ok: false, error: "badJson" };
    }
    if (typeof parsed !== "object" || parsed === null) return { ok: false, error: "wrongKind" };

    const record = parsed as { kind?: unknown; version?: unknown; presets?: unknown };
    if (record.kind !== VIBRATO_PRESET_FILE_KIND) return { ok: false, error: "wrongKind" };
    if (typeof record.version !== "number" || !Number.isInteger(record.version)) {
        return { ok: false, error: "wrongKind" };
    }
    if (record.version > VIBRATO_PRESET_FILE_VERSION) {
        return { ok: false, error: "newerVersion" };
    }
    if (!Array.isArray(record.presets)) return { ok: false, error: "noPresets" };

    // sanitize 会给空 id / builtin 前缀 id 各自兜底；这里再把 id **强制**换成
    // 新生成的 —— sanitize 保留输入 id 的行为对"设置加载"是对的（不能凭空改
    // 用户的预设 id），对"从文件导入"是错的（见上）。
    //
    // 【为什么 sanitize 两次】`builtin` 标记由 id 前缀派生。第一次净化可能把
    // `builtin.natural` 标成 builtin:true；只改 id 不重算标记的话，这个预设会
    // 带着 builtin:true 以 custom_ 的 id 入库 —— 编辑器把它当只读系统预设，
    // 用户改不了也删不掉。第二次净化以新 id 重算标记，顺带保证值域依旧合法。
    const presets = record.presets.map((entry) =>
        sanitizeVibratoPreset({
            ...sanitizeVibratoPreset(entry as Partial<VibratoPreset>),
            id: createVibratoPresetId(),
        }),
    );
    return { ok: true, presets };
}

/**
 * 判断一个预设是否与既有用户预设**完全等价**（导入去重用）。
 *
 * 比较净化后的 JSON：sanitize 固定了字段与取值域，因此字符串相等即参数相等。
 * 没有它，"导入同一个文件两次"会翻倍出一组同名预设；有了它，导入是幂等的。
 */
export function vibratoPresetSignature(preset: VibratoPreset): string {
    const { id: _id, builtin: _builtin, ...rest } = preset;
    void _id;
    void _builtin;
    return JSON.stringify(rest);
}

/**
 * 把文件里的预设并入既有用户预设列表，返回**新增**与**跳过**的数量及新增的预设。
 *
 * 跳过判定：与任一既有预设签名相同（名字与全部参数一致）。名字撞了但参数不同
 * 仍会导入 —— 用户可能就是想要两个不同参数的同名预设，删哪个由用户决定。
 *
 * @param capacity 还能再收几条（`MAX_VIBRATO_PRESETS - 现有`）。超出部分截断，
 *   数量反映在返回值里 —— 静默截断不如让 UI 把准确数字说出来。
 */
export function mergeImportedPresets(
    incoming: readonly VibratoPreset[],
    existing: readonly VibratoPreset[],
    capacity: number,
): { imported: VibratoPreset[]; skipped: number } {
    const existingSignatures = new Set(existing.map(vibratoPresetSignature));
    const imported: VibratoPreset[] = [];
    let skipped = 0;
    for (const preset of incoming) {
        const signature = vibratoPresetSignature(preset);
        if (
            existingSignatures.has(signature) ||
            imported.some((p) => vibratoPresetSignature(p) === signature)
        ) {
            skipped += 1;
            continue;
        }
        if (imported.length >= Math.max(0, capacity)) {
            skipped += 1;
            continue;
        }
        imported.push(preset);
        existingSignatures.add(signature);
    }
    return { imported, skipped };
}

/** 文件名里不允许出现的字符（Windows 保留集 + 路径分隔符）。 */
const FORBIDDEN_FILENAME_CHARS = new Set(["/", "\\", ":", "*", "?", '"', "<", ">", "|"]);

/** 该字符能否出现在文件名里（控制字符一律不行）。 */
function isAllowedInFilename(ch: string): boolean {
    const code = ch.charCodeAt(0);
    // 不用正则：控制字符类会触发 no-control-regex（那条规则防的是"无意间用
    // 控制字符匹配输入"，这里恰恰是刻意清洗，语义相反）。逐字符判断同样直白。
    return code > 0x1f && code !== 0x7f && !FORBIDDEN_FILENAME_CHARS.has(ch);
}

/**
 * 由预设名导出**安全的默认文件名**。
 *
 * 【为什么由前端给】预设名是用户起的、几乎必然含非 ASCII —— 与主题导出
 * `defaultFileName` 由前端给的理由相同（后端不掌握这个知识，也不该掌握）。
 * 空名回退 `preset`，保证对话框里永远有一个可用的名字。
 */
export function vibratoPresetFileName(preset: Pick<VibratoPreset, "name">): string {
    const cleaned = [...preset.name]
        .filter((ch) => isAllowedInFilename(ch))
        .join("")
        .trim()
        .replace(/\s+/g, "-");
    return `hifishifter-vibrato-${cleaned.length > 0 ? cleaned : "preset"}.json`;
}
