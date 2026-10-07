import type { Keybinding, KeybindingOverrides } from "./types";

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

/**
 * 存储中的覆盖项（**未归一**的原始形状）。
 *
 * v1 的值是单个 `Keybinding` 对象；v2 起是 `Keybinding[]`。归一化由
 * `normalizeStoredBindingShape` 负责，这里刻意用 `unknown` 承载 —— 存储里的
 * 内容是外部输入（可能被手改、可能来自旧版本），不该让类型系统假装它可信。
 */
export type StoredKeybindingOverrides = Record<string, unknown>;

function stripMetaKeys(raw: Record<string, unknown>): StoredKeybindingOverrides {
    const overrides: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(raw)) {
        if (META_KEYS.has(key)) continue;
        overrides[key] = value;
    }
    return overrides;
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

/** 已写入的元数据标记位（保存时原样继承，保证标记位粘性）。 */
function metaFlagsFrom(raw: Record<string, unknown> | null): Record<string, true> {
    const flags: Record<string, true> = {};
    if (!raw) return flags;
    for (const key of META_KEYS) {
        if (raw[key] === true) flags[key] = true;
    }
    return flags;
}

/** 值是否是一个形状合法的绑定对象。 */
function isBindingShaped(value: unknown): value is Keybinding {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    return typeof (value as { key?: unknown }).key === "string";
}

/**
 * 把存储里的覆盖项归一为**列表形状**（v1 单对象 → v2 数组）。
 *
 * 【为什么不需要一次性标记位（对比 `migrateStretchSplit`）】那一次迁移会**造出**
 * 一个新键的值，重放会把用户后来的"重置"撤销，所以必须只跑一次。本迁移只改
 * **形状**，天然幂等：数组再过一遍还是数组，不可能凭空复活任何设置。因此这里
 * 是纯粹的归一化函数，每次加载都跑，没有标记位、也没有"迁移过了吗"的状态。
 *
 * 归一的三种情形：
 * - 数组 → 逐项过滤掉形状不合法的元素后保留；`__none__` 只在"整条都是无"时
 *   留一个（与 `KeybindingMap` 声明的不变式一致：`__none__` 仅作为唯一元素）；
 * - 单个绑定对象 → 包装为单元素数组（老版本写入的形态）；
 * - 其它（数字 / 字符串 / 数组套数组…）→ 丢弃该键，回落到默认值。
 *
 * 逐项过滤而不是整体丢弃，是为了让"一个键写坏了"只损失那一个绑定，而不是
 * 把整个动作的设置清空。
 */
export function normalizeStoredBindingShape(
    stored: StoredKeybindingOverrides,
): KeybindingOverrides {
    const out: KeybindingOverrides = {};
    for (const [actionId, value] of Object.entries(stored)) {
        if (Array.isArray(value)) {
            const shaped = value.filter(isBindingShaped);
            const real = shaped.filter((binding) => binding.key !== "__none__");
            if (real.length > 0) {
                out[actionId as keyof KeybindingOverrides] = real;
            } else if (shaped.length > 0) {
                // 整条都是"无" → 保留为显式无绑定（保留首个元素的形状，
                // 例如 modifierOnly 标志）。
                out[actionId as keyof KeybindingOverrides] = [shaped[0]];
            }
            continue;
        }
        if (isBindingShaped(value)) {
            out[actionId as keyof KeybindingOverrides] = [value];
        }
    }
    return out;
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
 * 注意返回值仍是**未归一**的存储形状（v1 单对象 / 已迁移的数组混存），由
 * `normalizeStoredBindingShape` 统一收口 —— 两次迁移因此互不依赖、可分别测试。
 *
 * @param raw localStorage 解析出的原始对象（可能含元数据键）。
 * @returns 迁移后的**纯净**覆盖项（已剥离元数据键）与是否发生了迁移。
 *   调用方在 `migrated` 为 true 时应回写（带标记位），使迁移只发生一次。
 */
export function migrateStretchSplit(raw: Record<string, unknown>): {
    overrides: StoredKeybindingOverrides;
    migrated: boolean;
} {
    const overrides = stripMetaKeys(raw);
    if (raw[MIGRATION_FLAG] === true) {
        // 已迁移过：不再改动。
        return { overrides, migrated: false };
    }
    const legacy = overrides[LEGACY_STRETCH_ACTION];
    if (legacy != null && overrides[PARAM_STRETCH_ACTION] == null) {
        overrides[PARAM_STRETCH_ACTION] = legacy;
    }
    // 即便没有旧覆盖项也要写标记位：否则每次启动都要重新判定一次。
    return { overrides, migrated: true };
}

/**
 * 从 localStorage 加载用户自定义的快捷键覆盖项（含一次性迁移 + 形状归一）。
 */
export function loadKeybindingOverrides(): KeybindingOverrides {
    const raw = readRaw();
    if (raw === null) return {};
    const { overrides, migrated } = migrateStretchSplit(raw);
    if (migrated) {
        // 立即回写（带标记位），避免每次启动重复迁移。
        saveKeybindingOverrides(normalizeStoredBindingShape(overrides), {
            flags: { [MIGRATION_FLAG]: true },
        });
    }
    return normalizeStoredBindingShape(overrides);
}

/**
 * 将用户自定义的快捷键覆盖项保存到 localStorage。
 *
 * 【标记位粘性】已迁移的存储必须一直带着标记位，否则后续任意一次
 * `setKeybindings` 触发的保存都会把它丢掉，下次启动又会重放迁移。因此本函数
 * 默认**继承**存储里已有的全部元数据标记位；`opts.flags` 只做追加（本次迁移
 * 新置的位）。
 *
 * @param overrides 覆盖项。
 * @param opts.flags 本次要额外写入的标记位（如 `{ __stretchSplitMigrated: true }`）。
 */
export function saveKeybindingOverrides(
    overrides: KeybindingOverrides,
    opts?: { flags?: Record<string, true> },
): void {
    try {
        const cleaned = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v != null));
        const flags = { ...metaFlagsFrom(readRaw()), ...(opts?.flags ?? {}) };
        const payload: Record<string, unknown> = { ...cleaned, ...flags };
        if (Object.keys(cleaned).length === 0 && Object.keys(flags).length === 0) {
            localStorage.removeItem(STORAGE_KEY);
        } else {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        }
    } catch {
        // localStorage 不可用时静默失败
    }
}
