/**
 * 「拼音匹配」开关。
 *
 * 【为什么是开关而不是菜单】搜索行上的三个按钮必须能用同一种方式操作 —— 正则、
 * 仅显示媒体文件都是「点击 = 开/关」。原先这里是个三选一菜单，同一行里混着两种
 * 交互：用户得先猜「这个按钮点下去是会切换，还是会弹出一层东西」。现在三个按钮
 * 的点击语义完全一致。
 *
 * 三档宽严（关闭 / 智能 / 模糊）与各语言子开关都收进「选项 → 搜索与匹配」——
 * 那份设置本来就该有一个常驻入口，因为它同时作用于文件浏览器、快速搜索、
 * 快捷键面板与字体列表，不属于任何一个面板。
 *
 * 【图标为什么是「文A」而不是「Aa」】`Aa` 在 VS Code / JetBrains 一类编辑器里
 * 是**区分大小写**，放在搜索框旁边会被直接读错。「文A」是维基百科语言切换器用的
 * 那个记号，表示「不同文字之间」，与本功能（打拉丁字母找中日韩文字）正好对应；
 * 也和本行正则按钮的 `.*` 一样，是「一看就知道是什么的字符记号」而不是图形。
 *
 * 【为什么悬停提示里要写档位】开关只能表达「开 / 关」，表达不了「智能还是模糊」。
 * 不写出来，用户就得打开设置才知道当前是哪一档。
 */

import { useI18n } from "../../../i18n/I18nProvider";
import {
    SEARCH_MODE_LABEL_KEY,
    effectiveSearchMode,
    type SearchSettings,
} from "../../../features/search/searchSettings";
import { AppIconButton } from "../../../ui";

export interface SearchTranslitToggleProps {
    settings: SearchSettings;
    /** 部分更新；调用方负责持久化。 */
    onChange: (patch: Partial<SearchSettings>) => void;
    /**
     * 正则模式是否激活。正则作用于**原文**，与转写互斥 —— 此时开关仍然显示真实
     * 设置（点击它切换的也仍是那个设置），只在提示里说明「此刻不生效」。
     * 让按钮显示成「关」会是假话：用户点一下它反而打开了转写。
     */
    regexActive?: boolean;
    /** 按钮边长（px）。同一行里的按钮应取同一尺寸。 */
    size?: number;
}

export function SearchTranslitToggle({
    settings,
    onChange,
    regexActive = false,
    size = 22,
}: SearchTranslitToggleProps) {
    const { t } = useI18n();
    const mode = effectiveSearchMode(settings);
    const tooltip = `${t("search_match_mode")}: ${
        regexActive ? t("search_regex_disables_translit") : t(SEARCH_MODE_LABEL_KEY[mode])
    }`;

    return (
        <AppIconButton
            icon="文A"
            tooltip={tooltip}
            active={settings.translit}
            // 只翻总开关：`mode` 原样保留，所以关掉再打开会回到用户上次的宽严。
            onClick={() => onChange({ translit: !settings.translit })}
            className="search-translit-toggle"
            style={{ fontSize: "var(--qt-fs-micro)", width: size, height: size }}
        />
    );
}
