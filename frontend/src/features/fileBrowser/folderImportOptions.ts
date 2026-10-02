/**
 * 目录导入选项：类型、默认值与归一化。
 *
 * 【为什么单独成文件】这三个选项被两处消费 —— 导入选项对话框（用户改）与拖放落点
 * （直接执行时读上一次的值），并且要持久化到 `app_config.json`。类型与"缺字段算什么"
 * 集中在这里，两处才不会各自解释。
 *
 * 【为什么 mode 与两个开关是分开的字段】它们回答的是两个正交的问题：
 *   - `mode`：文件**怎么落到轨道上**（首尾相接 / 一条一条 / 叠成 Take）；
 *   - `recursive`：要导入的**文件集合是什么**（只本层 / 含子目录）；
 *   - `createFolderTracks`：落位时**要不要反映目录结构**（仅 `across-tracks` 有效）。
 *
 * 合成一个枚举会让它们变成 2×3 的组合值，而其中一维只对三分之一的取值有效 ——
 * 正是那种半年后没人敢动的枚举。
 */

import type { MessageKey } from "../../i18n/messages";

/** 与 `importMultipleAudioAtPosition` 的 `mode` 一字不差。 */
export type FolderImportMode = "across-time" | "across-tracks" | "as-takes";

export const FOLDER_IMPORT_MODES: readonly FolderImportMode[] = [
    "across-time",
    "across-tracks",
    "as-takes",
];

export interface FolderImportOptions {
    /** 排布方式。 */
    mode: FolderImportMode;
    /** 递归导入子目录中的文件。默认关（与 REAPER 的默认一致）。 */
    recursive: boolean;
    /**
     * 为每个文件夹创建轨道组。默认开。
     *
     * 仅在 `mode === "across-tracks"` 时有效 —— 另外两种模式下文件根本不在
     * 各自的轨道上，没有"组"可言。
     */
    createFolderTracks: boolean;
}

/**
 * 默认值。
 *
 * 【为什么 mode 默认 `across-tracks`】它是唯一能让"创建轨道组"生效的模式，
 * 而后者默认开启 —— 默认值之间必须自洽，否则用户第一次导入看到的就不是默认选项
 * 所描述的结果。
 */
export const DEFAULT_FOLDER_IMPORT_OPTIONS: FolderImportOptions = {
    mode: "across-tracks",
    recursive: false,
    createFolderTracks: true,
};

function asMode(value: unknown, fallback: FolderImportMode): FolderImportMode {
    return typeof value === "string" && (FOLDER_IMPORT_MODES as readonly string[]).includes(value)
        ? (value as FolderImportMode)
        : fallback;
}

function asBool(value: unknown, fallback: boolean): boolean {
    return typeof value === "boolean" ? value : fallback;
}

/** 归一化一份可能来自旧配置 / 手改文件的选项。永不抛错，逐字段回退。 */
export function normalizeFolderImportOptions(input: unknown): FolderImportOptions {
    if (!input || typeof input !== "object") return { ...DEFAULT_FOLDER_IMPORT_OPTIONS };
    const raw = input as Partial<Record<keyof FolderImportOptions, unknown>>;
    const d = DEFAULT_FOLDER_IMPORT_OPTIONS;
    return {
        mode: asMode(raw.mode, d.mode),
        recursive: asBool(raw.recursive, d.recursive),
        createFolderTracks: asBool(raw.createFolderTracks, d.createFolderTracks),
    };
}

/**
 * 排布方式的显示名键。
 *
 * 【为什么复用时间轴那三个键而不是另起一套】用户看到的是同一个选择 —— 同一个
 * `mode` 值在两个入口下必须叫同一个名字，否则"跨轨道添加"和"分配到多条轨道"
 * 会被当成两件事。穷举 Record 让词典少一个键在编译期就报错。
 */
export const FOLDER_IMPORT_MODE_LABEL_KEY: Record<FolderImportMode, MessageKey> = {
    "across-time": "import_across_time",
    "across-tracks": "import_across_tracks",
    "as-takes": "import_as_takes",
};
