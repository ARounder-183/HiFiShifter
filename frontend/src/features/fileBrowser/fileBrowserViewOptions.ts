/**
 * 文件浏览器的视图选项 —— 类型、默认值、归一化与标签表的唯一来源。
 *
 * 【为什么要单独成文件】这套选项有 10 个字段，且要从用户配置（`app_config.json`，
 * 运行期是 `unknown`）读回来。把"合法取值是什么、缺项落到哪"和"界面怎么渲染"
 * 混在面板组件里，等于让每个字段的默认值散落在 JSX 里 —— 一旦后端返回了旧版本
 * 的配置（少字段）或手改过的配置（错类型），表现是面板静默变形而不是回落默认。
 *
 * 本文件模仿 `features/search/searchSettings.ts` 的形状：
 * 类型 → 默认值 → 白名单守卫 → `normalize*` → 派生值 → 标签表。
 *
 * 【与 `audioOnly` 的历史关系】此前"仅显示媒体文件"与"上次目录"用 localStorage
 * 存在 `fileBrowserSlice` 里，`sortMode` 甚至没持久化。现在视图选项统一进这里，
 * 由 `normalize*` 负责把旧的 localStorage 布尔值迁移进来（见 `migrateLegacy*`）。
 */

import type { MessageKey } from "../../i18n/messages";

/** 行密度：紧凑（22px，文件浏览器历史取值）/ 舒适（24px）。 */
export type FileBrowserDensity = "compact" | "comfortable";

/** 排序依据。 */
export type FileBrowserSortKey = "name" | "date" | "size";

/** 行右侧的详情列显示什么；`none` 给窄面板让位。 */
export type FileBrowserDetailsColumn = "size" | "date" | "none";

export interface FileBrowserViewOptions {
    /** 行高密度。直接接 `AppListRow` 的 `density` prop。 */
    density: FileBrowserDensity;
    /** 排序依据。 */
    sortMode: FileBrowserSortKey;
    /** 是否降序。 */
    sortDescending: boolean;
    /** 目录是否始终排在文件之前。 */
    foldersFirst: boolean;
    /**
     * 是否显示隐藏文件。
     *
     * 【注意这不是纯前端开关】后端 `list_directory` 目前**硬跳过**点开头的文件，
     * 因此这一项会作为参数下发给 `list_directory`。Windows 上还应包含带
     * `FILE_ATTRIBUTE_HIDDEN` 的文件（见后端实现）。
     */
    showHiddenFiles: boolean;
    /** 行右侧显示什么。 */
    detailsColumn: FileBrowserDetailsColumn;
    /** 非搜索模式下是否也在第二行显示文件所在目录。 */
    showPathHint: boolean;
    /** 仅显示可导入的媒体文件（音频/视频 + MIDI）。 */
    mediaOnly: boolean;
    /**
     * 点击（左键单击 / 键盘回车）音频文件时试听。
     *
     * 【为什么默认开启】它是文件浏览器的招牌动作 —— 挑素材时"点一下听一下"是
     * 最自然的路径，关掉应当是"我明确不想要"，而不是默认。
     *
     * 【为什么与 `previewOnNavigate` 分成两个字段】两者回答的是不同的问题：
     * 这是"点它会发生什么"，那是"光标移到它会发生什么"。默认值也正交
     * （点击默认开、移动默认关）—— 合成一个枚举就无法表达"点要响、移不响"。
     */
    previewOnClick: boolean;
    /**
     * 键盘光标移动时自动试听。
     *
     * 【为什么默认关闭】浏览素材时每一次方向键都出声是打扰；但挑 take 时它是最
     * 省事的方式，所以给开关而不是不给。
     */
    previewOnNavigate: boolean;
    /** 底部状态行（项数 / 选中数 / 总大小）。 */
    statusBarVisible: boolean;
}

/**
 * 各排序依据的"自然方向"。
 *
 * 【为什么要按模式给默认】按名称是升序（A→Z）符合直觉，按日期/大小则几乎总是
 * 想先看最新的/最大的。用单一布尔量表达"是否降序"时，如果不带这层默认，
 * 用户从"名称"切到"日期"就会看到最旧的文件排在最前 —— 与今天的行为相反。
 */
export const DEFAULT_SORT_DESCENDING: Record<FileBrowserSortKey, boolean> = {
    name: false,
    date: true,
    size: true,
};

export const DEFAULT_FILE_BROWSER_VIEW_OPTIONS: FileBrowserViewOptions = {
    density: "compact",
    sortMode: "name",
    sortDescending: DEFAULT_SORT_DESCENDING.name,
    foldersFirst: true,
    showHiddenFiles: false,
    detailsColumn: "size",
    showPathHint: false,
    mediaOnly: false,
    previewOnClick: true,
    previewOnNavigate: false,
    statusBarVisible: true,
};

const DENSITIES: readonly FileBrowserDensity[] = ["compact", "comfortable"];
const SORT_KEYS: readonly FileBrowserSortKey[] = ["name", "date", "size"];
const DETAILS_COLUMNS: readonly FileBrowserDetailsColumn[] = ["size", "date", "none"];

/** 守卫式取值：值在允许集合内才采用，否则回落。不用裸 `as`。 */
function asMember<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
    return typeof value === "string" && (allowed as readonly string[]).includes(value)
        ? (value as T)
        : fallback;
}

function asBool(value: unknown, fallback: boolean): boolean {
    return typeof value === "boolean" ? value : fallback;
}

/**
 * 把任意输入归一化为一份完整、合法的视图选项。
 *
 * 永不抛：`undefined` / `null` / 数字 / 字符串 / 空对象一律返回默认值的一份拷贝；
 * 部分合法时保留合法字段、其余回落。这是配置来自磁盘（可能被手改、可能是旧版本）
 * 的前提。
 */
export function normalizeFileBrowserViewOptions(input: unknown): FileBrowserViewOptions {
    const d = DEFAULT_FILE_BROWSER_VIEW_OPTIONS;
    if (!input || typeof input !== "object") {
        return { ...d };
    }
    const raw = input as Record<string, unknown>;
    return {
        density: asMember(raw.density, DENSITIES, d.density),
        sortMode: asMember(raw.sortMode, SORT_KEYS, d.sortMode),
        sortDescending: asBool(raw.sortDescending, d.sortDescending),
        foldersFirst: asBool(raw.foldersFirst, d.foldersFirst),
        showHiddenFiles: asBool(raw.showHiddenFiles, d.showHiddenFiles),
        detailsColumn: asMember(raw.detailsColumn, DETAILS_COLUMNS, d.detailsColumn),
        showPathHint: asBool(raw.showPathHint, d.showPathHint),
        mediaOnly: asBool(raw.mediaOnly, d.mediaOnly),
        previewOnClick: asBool(raw.previewOnClick, d.previewOnClick),
        previewOnNavigate: asBool(raw.previewOnNavigate, d.previewOnNavigate),
        statusBarVisible: asBool(raw.statusBarVisible, d.statusBarVisible),
    };
}

/**
 * 从旧的 `localStorage` 键迁移"仅显示媒体文件"。
 *
 * 【为什么需要】`fileBrowser.audioOnly` 是历史存储位置（`fileBrowserSlice.ts`），
 * 老用户的偏好在那里。迁移只在"新配置里没有这一项"时生效，之后以新配置为准 ——
 * 否则用户改了新设置、旧键还留着，下次启动又被打回去。
 *
 * @param input 从配置读到的原始对象（可能是旧版本、可能没有 `mediaOnly`）。
 * @param legacyRaw 旧 localStorage 键的原始字符串（`"true"` / `"false"` / `null`）。
 */
export function migrateLegacyMediaOnly(input: unknown, legacyRaw: string | null): unknown {
    if (!input || typeof input !== "object") return input;
    const raw = input as Record<string, unknown>;
    if (raw.mediaOnly !== undefined) return input;
    if (legacyRaw === null) return input;
    return { ...raw, mediaOnly: legacyRaw === "true" };
}

/** 该排序依据的自然方向（切换排序依据时用）。 */
export function defaultSortDescending(mode: FileBrowserSortKey): boolean {
    return DEFAULT_SORT_DESCENDING[mode];
}

/**
 * 视图密度 → `AppListRow` 的 `density` 取值。
 *
 * 【为什么要有这层映射】本模块的取值是**用户可见的语义**（"紧凑 / 舒适"），而
 * 行原语的取值是它的历史命名（"compact / default"）。让视图选项直接叫
 * `default` 会让设置界面出现一个叫"默认"的密度档 —— 用户不知道它比什么更默认。
 * 两个词表各自表达自己的意思，在边界上翻译一次。
 */
export function rowDensityOf(density: FileBrowserDensity): "compact" | "default" {
    return density === "comfortable" ? "default" : "compact";
}

/** 排序依据的显示名键。穷举 Record：词典少一个键会在编译期报错。 */
export const FILE_BROWSER_SORT_LABEL_KEY: Record<FileBrowserSortKey, MessageKey> = {
    name: "fb_sort_name",
    date: "fb_sort_date",
    size: "fb_sort_size",
};

/** 行密度的显示名键。 */
export const FILE_BROWSER_DENSITY_LABEL_KEY: Record<FileBrowserDensity, MessageKey> = {
    compact: "fb_density_compact",
    comfortable: "fb_density_comfortable",
};

/** 详情列的显示名键。 */
export const FILE_BROWSER_DETAILS_LABEL_KEY: Record<FileBrowserDetailsColumn, MessageKey> = {
    size: "fb_details_size",
    date: "fb_details_date",
    none: "fb_details_none",
};
