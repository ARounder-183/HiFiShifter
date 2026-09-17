import type { KeybindingOverrides } from "./types";

const STORAGE_KEY = "hifishifter.keybindings";

/**
 * 旧的「拉伸」修饰键 action id（时间轴 clip 边缘与参数选区边缘**共用**一个）。
 *
 * 拆分后：
 * - `modifier.clipStretch` → 只作用于**时间轴轨道视图**的 clip 边缘；
 * - `modifier.paramStretch` → 只作用于**参数编辑器**的选区边缘。
 */
const LEGACY_STRETCH_ACTION = "modifier.clipStretch";
const PARAM_STRETCH_ACTION = "modifier.paramStretch";

/**
 * 一次性迁移标记位（与 actionId 同层存放）。
 *
 * 【为什么需要它】迁移只读旧键、不删旧键，两个新动作与旧键同值。没有标记位时，
 * 若用户此后把 `modifier.clipStretch` 改回默认（reducer 会删除该覆盖项）而
 * `modifier.paramStretch` 也回到默认，下一次启动会**重新**把"旧值"迁移过去 ——
 * 用户的"重置"被静默撤销。标记位保证每个存储只迁移一次。
 */
const MIGRATION_FLAG = "__stretchSplitMigrated";

/** localStorage 里允许出现的非 actionId 元数据键（不参与快捷键合并）。 */
const META_KEYS = new Set<string>([MIGRATION_FLAG]);

function stripMetaKeys(raw: Record<string, unknown>): KeybindingOverrides {
    const overrides: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (META_KEYS.has(key)) continue;
        overrides[key] = value;
    }
    return overrides as KeybindingOverrides;
}

/** 读取 localStorage 原始对象（含元数据键）；不可用时返回 null。 */
function readRaw(): Record<string, unknown> | null {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
        return parsed as Record<string, unknown>;
    } catch {
        return null;
    }
}

/**
 * 应用「拉伸拆分」一次性迁移（纯函数，便于单测）。
 *
 * 【拆分前 vs 拆分后】拆分前 `modifier.clipStretch` 同时服务时间轴 clip 边缘与
 * 参数编辑器选区边缘。用户若改绑过它，localStorage 里只有那一条覆盖项；拆分后
 * 参数编辑器读 `modifier.paramStretch`，不迁移就会退回默认 Alt —— 用户明明改过
 * 却"设置丢了"。
 *
 * 【迁移策略】旧值**同时**落到两个新动作：升级后两个表面的键位与拆分前完全一致，
 * 用户可再分别改绑。
 *
 * @param raw localStorage 解析出的原始对象（可能含元数据键）。
 * @returns 迁移后的**纯净**覆盖项（已剥离元数据键）与是否发生了迁移。
 *   调用方在 `migrated` 为 true 时应回写（带标记位），使迁移只发生一次。
 */
export function migrateStretchSplit(raw: Record<string, unknown>): {
    overrides: KeybindingOverrides;
    migrated: boolean;
} {
    const overrides = stripMetaKeys(raw);
    if (raw[MIGRATION_FLAG] === true) {
        // 已迁移过：不再改动。
        return { overrides, migrated: false };
    }
    const legacy = overrides[LEGACY_STRETCH_ACTION as keyof KeybindingOverrides];
    if (legacy != null && overrides[PARAM_STRETCH_ACTION as keyof KeybindingOverrides] == null) {
        (overrides as Record<string, unknown>)[PARAM_STRETCH_ACTION] = legacy;
    }
    // 即便没有旧覆盖项也要写标记位：否则每次启动都要重新判定一次。
    return { overrides, migrated: true };
}

/**
 * 从 localStorage 加载用户自定义的快捷键覆盖项（含一次性迁移）。
 */
export function loadKeybindingOverrides(): KeybindingOverrides {
    const raw = readRaw();
    if (raw === null) return {};
    const { overrides, migrated } = migrateStretchSplit(raw);
    if (migrated) {
        // 立即回写（带标记位），避免每次启动重复迁移。
        saveKeybindingOverrides(overrides, { migrated: true });
    }
    return overrides;
}

/**
 * 将用户自定义的快捷键覆盖项保存到 localStorage。
 *
 * 【标记位粘性】已迁移的存储必须一直带着标记位，否则后续任意一次 `setKeybinding`
 * 触发的保存都会把它丢掉，下次启动又会重放迁移。因此本函数默认**继承**已有标记位；
 * `opts.migrated === false` 时才清除（本工程没有该用法，留给显式重置场景）。
 *
 * @param overrides 覆盖项。
 * @param opts.migrated true = 本次即为迁移回写（写入标记位）。
 */
export function saveKeybindingOverrides(
    overrides: KeybindingOverrides,
    opts?: { migrated?: boolean },
): void {
    try {
        const cleaned = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null));
        const alreadyMigrated = readRaw()?.[MIGRATION_FLAG] === true;
        const keepFlag = opts?.migrated ?? alreadyMigrated;
        const payload: Record<string, unknown> = {
            ...cleaned,
            ...(keepFlag ? { [MIGRATION_FLAG]: true } : {}),
        };
        if (Object.keys(cleaned).length === 0 && !keepFlag) {
            localStorage.removeItem(STORAGE_KEY);
        } else {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        }
    } catch {
        // localStorage 不可用时静默失败
    }
}
