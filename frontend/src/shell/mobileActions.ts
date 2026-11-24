/**
 * 移动端抽屉的条目来源。
 *
 * 【为什么不抄一份 `MenuBar` 的菜单定义】`MenuBar` 有 700+ 行手写菜单树，每一项都
 * 带着自己的 onClick、禁用条件与图标。再抄一份给移动端，意味着**每个新功能都要在
 * 两个地方登记**，而漏掉的那一处只会表现为"手机上找不到这个功能"—— 没有任何报错。
 *
 * 这里改从 `ACTION_META` 生成：它是全仓**唯一**的动作注册表（135 条），快捷键、
 * 设置页、菜单栏的快捷键提示全都读它。移动端读同一份，于是"新增动作"这件事只需要
 * 加一条 `ACTION_META` 记录。
 *
 * 【为什么按 `group` 分段】`ACTION_META.group` 本来就是"按使用场景分组"（播放 / 模式 /
 * 编辑 / 片段 / 工程…），与设置页的分组一致。移动端抽屉直接复用这个分组，用户在两处
 * 看到的是同一个心智模型。
 *
 * 【为什么要有排除表】有些动作在移动端**结构上做不到**，而不是"暂时没接"：
 * 从 VocalShifter 剪贴板粘贴（移动端还没有系统剪贴板通道）、把键盘交给宿主
 * （没有宿主）。把它们摆出来，用户点下去只会得到一个失败或一条静默 no-op ——
 * 本仓明令禁止的形状。
 *
 * 【为什么"从任意路径导入"不在排除表里了】它此前被排除的理由是"移动端没有可浏览
 * 的目录树"。现在导入走系统选择器（SAF），选中的文件被收编进工作区，路径对下游
 * 完全有效 —— 继续排除等于把已经能用的功能藏起来。
 */

import { ACTION_META } from "../features/keybindings/defaultKeybindings";
import type { ActionId } from "../features/keybindings/types";
import type { MessageKey } from "../i18n/messages";
import { surfaceAllows, type SurfaceCapabilityId } from "../services/surfaceCapabilities";

/** 抽屉里的一条动作。 */
export interface MobileActionItem {
    id: ActionId;
    /** 已本地化的名称。 */
    label: string;
    /** 当前绑定的快捷键（无绑定时为 `null`）。移动端只作为"这是个常用动作"的旁证。 */
    shortcut: string | null;
}

/** 抽屉里的一段。 */
export interface MobileActionSection {
    id: string;
    title: string;
    items: MobileActionItem[];
}

/**
 * 移动端**不提供**的动作，以及为什么。
 *
 * 键是动作 id，值是它所缺的能力位（`null` 表示"与能力无关，是这个形态本身没有
 * 意义"）。这条信息会出现在抽屉的隐藏统计里 —— 排查"为什么手机上找不到 XX"时，
 * 不需要再去读一遍代码。
 */
const MOBILE_UNAVAILABLE: Partial<Record<ActionId, SurfaceCapabilityId | null>> = {
    // 系统剪贴板：移动端的剪贴板读写还没有原生通道（hifishifter-clipboard 在
    // Android/iOS 上返回 `clipboard_unsupported_platform`），粘贴 VocalShifter
    // 载荷必然失败。
    "edit.pasteVocalShifter": null,
    // 宿主相关：移动端没有 DAW。
    "host.handKeyboardToHost": null,
    // 快速搜索的导航键属于弹窗内部，不该出现在抽屉里。
    "quickSearch.navigate.up": null,
    "quickSearch.navigate.down": null,
    "quickSearch.preview": null,
    "quickSearch.confirm": null,
    "quickSearch.close": null,
};

/**
 * 抽屉顶部的「窗口」分区 id。
 *
 * 【为什么它不在 `SECTION_ORDER` 里】那一份顺序只决定 `ACTION_META` 生成的分段；
 * 窗口分区的条目来自**停靠布局**（哪个面板开着、哪个关了），是运行期状态而不是
 * 动作注册表。两者混在同一个生成函数里会让"动作清单"依赖停靠状态。
 */
export const MOBILE_WINDOW_SECTION_ID = "windows";

/**
 * 按搜索词过滤抽屉内容。
 *
 * 【为什么必须有搜索】抽屉在移动端是**唯一**的功能入口：135 条动作加上窗口分区
 * 之后接近 150 行，而 390px 宽的屏幕一次只能显示 8 行。没有搜索时，"打开日志
 * 文件夹"这类低频项要滑十几屏才看得到 —— 与"根本没有这个功能"没有区别。
 *
 * 【匹配规则】只做大小写无关的子串匹配，且**只匹配已本地化的标签**：用户输入的
 * 是他屏幕上看到的字，拿内部 id 去匹配只会命中他看不见的东西。
 * 空查询原样返回（保持引用，避免无谓重渲染）。
 */
export function filterMobileSections(
    sections: MobileActionSection[],
    query: string,
): MobileActionSection[] {
    const needle = query.trim().toLowerCase();
    if (needle === "") return sections;
    const filtered: MobileActionSection[] = [];
    for (const section of sections) {
        const items = section.items.filter((item) => item.label.toLowerCase().includes(needle));
        if (items.length > 0) filtered.push({ ...section, items });
    }
    return filtered;
}

/** 抽屉的分段顺序。与 `MenuBar` 的菜单顺序一致：先"我要做什么"，最后"界面怎么摆"。 */
const SECTION_ORDER: Array<{ id: string; titleKey: MessageKey }> = [
    { id: "playback", titleKey: "mobile_section_playback" },
    { id: "mode", titleKey: "mobile_section_mode" },
    // 编辑 / 视图沿用桌面菜单的标题键：语义完全一致，另起一个键只会让两个形态
    // 对同一件事用两个词。
    { id: "edit", titleKey: "menu_edit" },
    { id: "clip", titleKey: "mobile_section_clip" },
    // 【这两段曾经整段消失】`paramEditor`（9 条）与 `pianoRoll`（22 条）在
    // `SECTION_ORDER` 里没有对应条目，于是 `buildMobileActionSections` 把这两个
    // bucket 直接丢掉 —— 31 条动作在抽屉里**搜都搜不到**，其中
    // `pianoRoll.shiftParam*` 那 12 条更是**没有任何按钮**（只有键盘能触发）。
    // 症状是"手机上找不到平滑 / 移调"，而界面上看不出任何异常。
    { id: "paramEditor", titleKey: "mobile_section_param_editor" },
    { id: "pianoRoll", titleKey: "mobile_section_piano_roll" },
    { id: "paramBlock", titleKey: "mobile_section_param_block" },
    { id: "project", titleKey: "mobile_section_project" },
    { id: "layout", titleKey: "menu_view" },
];

/**
 * 生成抽屉内容。
 *
 * @param translate 本地化函数（由调用方从 `useI18n` 取，本模块不碰 React）。
 * @param shortcutFor 取某动作的主绑定显示文本；没有绑定时返回 `null`。
 */
export function buildMobileActionSections(
    translate: (key: MessageKey) => string,
    shortcutFor: (id: ActionId) => string | null,
): MobileActionSection[] {
    const buckets = new Map<string, MobileActionItem[]>();

    for (const [rawId, meta] of Object.entries(ACTION_META) as Array<
        [ActionId, (typeof ACTION_META)[ActionId]]
    >) {
        // 修饰键动作是**手势的修饰**，不是可执行命令 —— 它们去修饰键行（ModifierRail）。
        //
        // 【判据是 `modifierOperationType` 而不是 group 前缀】"mode" 与 "mod*" 都以
        // "mod" 开头，用前缀判断会把整个「模式」分组（选择/绘制/直线/颤音工具）
        // 一并当成修饰键丢掉 —— 移动端抽屉里就再也切不了工具，而界面上看不出
        // 任何异常。`modifierOperationType` 是类型定义里"存在即表示这是修饰键绑定"
        // 的**语义**标记（见 `features/keybindings/types.ts`）。
        if (meta.modifierOperationType !== undefined) continue;
        if (rawId in MOBILE_UNAVAILABLE) {
            const capability: SurfaceCapabilityId | null | undefined = MOBILE_UNAVAILABLE[rawId];
            // `null` / `undefined` = 形态上没意义，永远不出现；否则按能力位判断
            // （能力为真时出现 —— 例如 `fs.pathAccess` 在移动端恒为假，因此那些
            // 走路径的动作实际总是隐藏）。
            if (capability === null || capability === undefined) continue;
            if (!surfaceAllows(capability)) continue;
        }
        const bucket = buckets.get(meta.group) ?? [];
        bucket.push({
            id: rawId,
            label: translate(meta.labelKey),
            shortcut: shortcutFor(rawId),
        });
        buckets.set(meta.group, bucket);
    }

    const sections: MobileActionSection[] = [];
    for (const { id, titleKey } of SECTION_ORDER) {
        const items = buckets.get(id);
        if (!items || items.length === 0) continue;
        sections.push({ id, title: translate(titleKey), items });
    }
    return sections;
}

/**
 * 修饰键动作（给 `ModifierRail` 用）。
 *
 * 【为什么单独导出】它们是**状态**（按住/锁定）而不是**命令**，交互模型与抽屉里的
 * 条目完全不同（一次点击 vs 切换）。混在同一个列表里会让两种语义在同一个 UI 里
 * 长得一样 —— 而它们的区别恰恰是"点完之后会发生什么"。
 */
export function buildModifierChips(
    translate: (key: MessageKey) => string,
): Array<{ id: ActionId; label: string; group: string }> {
    const chips: Array<{ id: ActionId; label: string; group: string }> = [];
    for (const [rawId, meta] of Object.entries(ACTION_META) as Array<
        [ActionId, (typeof ACTION_META)[ActionId]]
    >) {
        if (meta.modifierOperationType === undefined) continue;
        chips.push({ id: rawId, label: translate(meta.labelKey), group: meta.group });
    }
    return chips;
}
