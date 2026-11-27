/**
 * 上下文菜单原语 —— 全应用手写右键菜单的**唯一**实现。
 *
 * 【为什么手写而不是用 Radix ContextMenu】这是既有决策，保留：时间轴与
 * 记事本的菜单需要与画布的手势系统协同（例如指针捕获、右键拖拽守卫），
 * Radix 的 portal 菜单会打断这条链路。项目已把该决策写在 `DockTabMenu` 头部。
 *
 * 【它解决的三个问题】
 *
 * 1. **漂移的菜单项外观**。审查发现同一个"菜单项"在 6 个文件里有 6 套
 *    class：`text-qt-sm` / `text-qt-md` / `text-qt-xs`，`hover:bg-qt-button-hover` /
 *    `hover:bg-qt-hover` / `hover:bg-qt-highlight hover:text-white`，
 *    以及 `py-1` / `py-1.5` / `py-2` 三种行内边距。
 *
 * 2. **缺失的键盘可达性**。12 个手写菜单**没有一个有方向键导航**，只有 Esc。
 *    键盘用户打开右键菜单后只能再按 Esc 关掉，无法选择任何一项。
 *
 * 3. **各不相同的定位与层级**。有的按估算尺寸夹紧、有的不夹紧，
 *    z-index 从 `z-50` 到 `z-9999` 有 4 档（其中两处反而高过对话框）。
 *
 * 本组件把这三件事各做一次：单一 class 来源、完整键盘模型、按实测尺寸夹紧 +
 * `--qt-z-menu` 层级。
 *
 * 【壳与项的样式来源】本文件的 className 只剩结构类：壳 `hs-menu`、项
 * `hs-menu__item`、标题 `hs-menu__label`、分隔 `hs-menu__separator`。取值
 * （底色 / 圆角 / 阴影 / 行高 / 悬停色 / 禁用色）全部由 `src/index.css` 的
 * 「上下文菜单样式模型」块决定 —— 手写菜单（需要内联滑杆 / 输入框 / 双列的那些）
 * 挂同一套类，因此两边不可能再漂移。这也是 `ITEM_BASE` 常量被删掉的原因：
 * 它把取值复制到了 TypeScript 里，CSS 那份改不到它。
 *
 * 【滚动】三个原语的内容都包在 `<MenuScrollScope>` 里：壳没有滚动条，只有两条
 * sticky 提示行。内边距因此搬到了 `.hs-menu__content`（见 `index.css` 的说明）。
 *
 * 【两个 data 标记的分工】`data-hs-context-menu` 有**两种**写法，含义不同：
 *   - `="1"`：这是一个**已打开**的菜单表面。时间轴的关闭逻辑据此判断"点在菜单
 *     里面"，别处不得用它当"菜单开着"的判据（常驻元素带这个值会让判据永远为真）。
 *   - 无值：常驻的**锚点**容器（工具栏按钮的 `position: relative` 外壳），只表示
 *     "这里会长出菜单"。`AppTooltip` 的注释记录了为什么不能把两者混为一谈。
 *
 * 【菜单表面必须挂在主题作用域、且在文档流之外】`AppContextMenu` 由调用方 portal，
 * `AppAnchoredMenu` 自己 portal。留在触发它的布局盒里会被沿途任何一层
 * `overflow: hidden` 裁掉 —— 实测参数编辑器工具栏下拉要穿过 9 层，可视高度 0px
 * （详见 `src/index.css` 的 `.hs-menu--submenu`）。挂载目标用
 * `menuPortalContainer()`：优先 `.radix-themes` 以拿全 Radix 令牌，无主题时退回 body。
 */
import {
    Fragment,
    useCallback,
    useEffect,
    useId,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import type { CSSProperties, ReactNode, RefObject } from "react";
import { createPortal } from "react-dom";
import { CheckIcon } from "@radix-ui/react-icons";

import { EDGE_GAP, clampAxisPosition } from "../components/appTooltipPosition";
import { cx } from "./cx";
import { MenuScrollScope } from "./MenuScrollScope";
import { menuItemHeightOf, revealMenuItem } from "./menuScrollDom";
import {
    handleMenuDismissEscape,
    handleMenuDismissPointerDown,
    menuDismissLayerCount,
    pushMenuDismissLayer,
} from "./menuDismissStack";
import { computeMenuPageItemCount, placeMenuWithinViewport } from "./menuScrollModel";
import {
    EMPTY_MENU_TYPEAHEAD,
    matchMenuTypeahead,
    menuItemTypeaheadText,
    menuItemsOf,
    menuTypeaheadChar,
    pushMenuTypeahead,
    type MenuTypeaheadState,
} from "./menuTypeahead";
import { menuPortalContainer } from "./portalContainer";
import { ownsArrowKeys, useMenuKeyboard } from "./useMenuKeyboard";

export interface AppMenuItemSpec {
    key: string;
    label: ReactNode;
    /** 选择回调。`heading: true` 的标题行可省略。 */
    onSelect?: () => void;
    /** 展示用快捷键文本（不参与绑定，与 `data-tooltip` 同源）。 */
    shortcut?: string;
    /**
     * 悬停提示文本，渲染成 `data-tooltip`。
     *
     * 【它什么时候有意义】只用于"这一项到底会做什么"说不清的地方（例如「另存为」
     * 子菜单里每种格式的说明：zip 需要解压后才能打开）。菜单弹出期间项目的自定义
     * 气泡通道被全局抑制（见 `docs/i18n/style-guide.md` §3.5），因此这条**不能**
     * 用来承载"为什么点不了" —— 那件事必须走 `description`，它是正立可见的。
     */
    tooltip?: string;
    /**
     * 尾部**弱化标注**（不是快捷键）。
     *
     * 【它解决什么】有些项自身说不清会发生什么，需要一处"结果预告"跟在标签后面：
     * "自动"要说明它此刻会落到哪一处，"默认（当前）"要说明默认值是什么。塞进
     * `label` 会让标注跟着标签一起被截断，也会被读屏念成名字的一部分；塞进
     * `shortcut` 则是在借一个语义不同的字段（那列还承担"按什么键"的约定）。
     * 与 `shortcut` 同列、同样式，但语义独立。
     */
    hint?: string;
    /** 破坏性操作：悬停变红底红字。 */
    danger?: boolean;
    disabled?: boolean;
    /**
     * 激活后**不关闭**菜单。
     *
     * 【为什么需要】菜单的默认语义是"点一条命令就完事，收起菜单"。但**多选**
     * 不是一条命令：参考轨道组要连着勾几条轨道、逐条看画布上的曲线变化，
     * 每勾一次就关掉菜单等于让用户重新打开 N 次。这与原生菜单里复选项不收起
     * 是同一条约定 —— 复选框是"开关"，不是"命令"。
     *
     * 单选（`radio`）与普通条目保持默认收起。
     */
    keepOpen?: boolean;
    /** 该项上方加一条分隔线，用于视觉分组。 */
    separatorBefore?: boolean;
    /** 右侧勾选标记（用于"当前选中项"这类菜单）。 */
    checked?: boolean;
    /**
     * 单选态：把项渲染成 `role="menuitemradio"` 而不是 `menuitemcheckbox`。
     *
     * 【为什么必须有】`aria-checked` 只允许出现在 `menuitemcheckbox` /
     * `menuitemradio` / `option` / `switch` 上。互斥档位（语言、推理设备、
     * 日志级别、子轨道音高组）此前只有"勾选"这一个角色可用，读屏会把它念成
     * "复选框"——用户听到的是"可以多选"，而实际只能选一个。
     */
    radio?: boolean;
    /** 条目左侧图标（take 操作、菜单按钮等）。 */
    icon?: ReactNode;
    /**
     * 指针进入本项时的回调。
     *
     * 【它解决什么】有些菜单项与菜单**之外**的东西联动：参考轨道组的每一项要在
     * 指针悬停时高亮画布上那条参考曲线。那个联动不属于"菜单内部状态"，必须由
     * 调用方拿到事件；而声明式条目模型此前只有 `onSelect`，悬停无从表达。
     */
    onHover?: () => void;
    /**
     * 指针离开本项时的回调。
     *
     * 【为什么与 `onHover` 成对】只有进入没有离开，高亮会**粘住**：指针从第 3 条
     * 轨道划到"全部"（那一项没有联动）时，画布上第 3 条曲线仍然亮着，而指针底下
     * 已经不是它。两者必须成对出现才是一份完整的"悬停联动"契约。
     */
    onHoverEnd?: () => void;
    /**
     * 禁用理由，渲染成标签下的**第二行弱化小字**。
     *
     * 【为什么不是 tooltip】菜单弹出期间项目的气泡通道被全局抑制（见
     * `docs/i18n/style-guide.md` §3.5），挂在菜单项上的 tooltip 一次都不会显示 ——
     * 而"为什么点不了"正是最需要被看见的那句话。它曾经叫 `tooltip`，名字对、
     * 通道错，于是 `dawControlledReason()` 与整族解释**从未被渲染过**。
     *
     * 只在 `disabled` 时渲染：可用项不需要解释。
     */
    description?: string;
    /**
     * 分组标题行：渲染为不可选中的小标题（大写弱化色），不参与键盘导航。
     * 收编自 ActionBar 录音菜单的手写分组行 —— 平面菜单也常有分段需求。
     */
    heading?: boolean;
    /**
     * 子菜单项。**给了它，这一项就变成"展开子菜单"的触发项**：
     * `onSelect` 不再被调用，`shortcut` / `danger` / `checked` 对触发项也没有意义
     * （它们属于子项）。子项是同样的 `AppMenuItemSpec`，因此可以任意嵌套。
     *
     * 【什么时候该用它】当一组项**同质且可枚举**时（"行 ▸ 上方 / 下方"、
     * "格式 ▸ 加粗 / 倾斜 / …"）。反过来，**右键点开就想直接点的那些**不要塞进
     * 子菜单 —— 多一次悬停就是多一次成本，`heading` 平铺反而是对的。
     */
    items?: AppMenuItemSpec[];
}

/**
 * 渲染一层菜单条目（顶层与子面板共用）。
 *
 * 【为什么抽到模块级】`AppContextMenu`（指针锚定）与 `AppAnchoredMenu`（按钮锚定）
 * 必须渲染出**逐字一致**的条目 —— 分隔线、禁用态、图标列、勾选标记、快捷键列。
 * 各自实现一遍就会出现"工具栏菜单的项比时间轴菜单的项矮一行"这类分裂。
 *
 * `activeIndex` 只对**拥有它**的那一层有意义：`AppContextMenu` 用 `data-active`
 * 表示键盘高亮（焦点不在项上），子面板与锚定菜单则用真实 DOM 焦点导航，因此传 -1。
 */
function renderMenuEntries(
    entries: AppMenuItemSpec[],
    layerActiveIndex: number,
    reserveIcon: boolean,
    /**
     * 本层是否**拥有** `activeIndex`：顶层传写入函数，子面板传 `null`。
     *
     * 【为什么必须区分】`activeIndex` 是**外层**的状态，而递归渲染时每一层都会
     * 生成自己的 `index`。若子面板也用它汇报悬停，子项的下标就写进了外层：
     * 实测鼠标停在子菜单第 1 项，外层第 1 项（撤销）跟着亮；停在第 3 项
     * （删除线），外层第 3 项（剪切）亮 —— **两层的高亮序号一一对应**，正是
     * 这个越界写入。
     *
     * 子面板本来也不需要它：它由 `useMenuKeyboard` 用真实 DOM 焦点导航，
     * 悬停高亮由 CSS `:hover` 负责，没有 `data-active` 这回事。
     */
    onHoverItem: ((index: number) => void) | null,
    /**
     * 本层项的 id 前缀：只有拥有 `activeIndex` 的顶层传（供容器的
     * `aria-activedescendant` 引用）；子面板走 DOM 焦点导航，不需要。
     */
    idPrefix: string | null,
    onClose: () => void,
): ReactNode {
    return entries.map((item, index) => (
        <Fragment key={item.key}>
            {/*
              分组分隔线是**独立元素**，不是首项自己的上边框 —— 加在项上会
              让分隔处那一行比别的行高一截（见 `hs-menu__separator` 的说明）。
            */}
            {item.separatorBefore ? <div className="hs-menu__separator" role="separator" /> : null}
            {item.items?.length ? (
                <AppSubMenu
                    label={item.label}
                    icon={item.icon}
                    itemKey={item.key}
                    disabled={item.disabled}
                    active={index === layerActiveIndex}
                    reserveIcon={reserveIcon}
                    // 触发项与子面板都要把"指针在哪"汇报给本层 —— 否则本层的高亮
                    // 会停在最后一个被划过的普通项上（见 `onHoverChange`）。
                    onHoverChange={
                        onHoverItem
                            ? (onTrigger) => onHoverItem(onTrigger && !item.disabled ? index : -1)
                            : undefined
                    }
                >
                    {renderMenuEntries(item.items, -1, reserveIcon, null, null, onClose)}
                </AppSubMenu>
            ) : (
                <AppContextMenuItem
                    item={item}
                    active={index === layerActiveIndex}
                    reserveIcon={reserveIcon}
                    id={idPrefix ? `${idPrefix}-item-${index}` : undefined}
                    onActive={
                        onHoverItem ? () => onHoverItem(item.disabled ? -1 : index) : undefined
                    }
                    onSelect={() => {
                        item.onSelect?.();
                        if (!item.keepOpen) onClose();
                    }}
                />
            )}
        </Fragment>
    ));
}

export interface AppContextMenuProps {
    /** 视口坐标（`clientX` / `clientY`）。 */
    x: number;
    y: number;
    items: AppMenuItemSpec[];
    onClose: () => void;
    /**
     * 最小宽度覆盖。**省略即用 `--qt-menu-min-w`（推荐）** —— 各面板历史取值
     * 140–248 五档，统一后只有一档；只有内容确实更宽的表单型菜单才显式给值。
     */
    minWidth?: number;
    /** 无障碍名称：菜单是弹出表面，需要有可读名称。 */
    ariaLabel?: string;
    /**
     * 顶部自定义区：渲染在条目列表之上、参与同一次定位测量。
     * `DockTabMenu` 的行内重命名、ActionBar 录音菜单的分组头由此承载。
     * 注意：这里是**非条目**内容，不参与键盘导航 —— 需要可聚焦控件的
     * 内容（如重命名输入框）自己处理焦点与 Enter/Esc。
     */
    header?: ReactNode;
    /**
     * 是否标记为「时间轴浮动菜单」（`data-hs-floating-menu="1"`）。
     *
     * 这是时间轴侧的既有契约，有三个独立读取方，缺了它会静默失去豁免：
     *   - `AppTooltip`：悬停抑制（菜单打开时不该冒提示气泡）；
     *   - `TimelinePanel`：内联编辑器的失焦提交判断（点菜单不应提交输入框）；
     *   - `ClipHeader`：角标编辑的提交守卫。
     *
     * 因此凡是挂在时间轴/剪辑上的菜单都要显式打开它；普通面板菜单不需要。
     */
    floating?: boolean;
    /**
     * 打开时把键盘焦点收到菜单上。
     *
     * 【为什么需要】方向键导航有一道守卫：`ownsArrowKeys(document.activeElement)`
     * 为真时让路（那是为**菜单里**的内联输入框 / 滑杆准备的）。但守卫看的是"当前
     * 焦点元素"，不区分它在不在菜单里 —— 于是当触发菜单的表面本身是可编辑元素
     * （记事本的 contenteditable）时，焦点仍在编辑器上，方向键被判给编辑器，菜单
     * 高亮**一格都不动**：键盘用户按 `ContextMenu` 键打开菜单后无法选择任何一项。
     *
     * 【为什么是可选】大多数菜单的触发面不是可编辑元素（时间轴、文件列表、停靠
     * 标签），焦点本来就不在会吞方向键的元素上，加这一手只会平白改掉焦点归属。
     * 默认关闭，由"触发面可编辑"的调用方显式打开。
     */
    autoFocus?: boolean;
}

/**
 * 上下文菜单。
 *
 * @example
 * {menu ? (
 *   <AppContextMenu
 *     x={menu.x}
 *     y={menu.y}
 *     ariaLabel={t("ctx_clip")}
 *     items={[
 *       { key: "cut", label: t("cut"), shortcut: "Ctrl+X", onSelect: cut },
 *       { key: "del", label: t("delete"), danger: true, separatorBefore: true, onSelect: del },
 *     ]}
 *     onClose={() => setMenu(null)}
 *   />
 * ) : null}
 */
export function AppContextMenu({
    x,
    y,
    items,
    onClose,
    minWidth,
    ariaLabel,
    header,
    floating = false,
    autoFocus = false,
}: AppContextMenuProps) {
    const ref = useRef<HTMLDivElement | null>(null);
    const [position, setPosition] = useState<{ left: number; top: number; ready: boolean }>({
        left: x,
        top: y,
        ready: false,
    });
    /** 键盘焦点所在的下标（指向**可选项**，跳过禁用项与分隔符）。 */
    const [activeIndex, setActiveIndex] = useState(-1);
    /** 打字跳项的缓冲（每个菜单各一份，跨菜单不串）。 */
    const typeaheadRef = useRef<MenuTypeaheadState>(EMPTY_MENU_TYPEAHEAD);

    /**
     * 打开菜单前的焦点元素。
     *
     * 【为什么需要】菜单是弹出表面：关闭后若不归还焦点，键盘用户会被丢回
     * `<body>`，下一次 Tab 从文档头重新开始 —— 对右键菜单而言"关闭"等于"迷路"。
     * 卸载时归还；触发者可能已随菜单一起消失（例如菜单项删掉了它），故先查
     * `isConnected`，不存在就什么都不做。
     */
    const openerRef = useRef<HTMLElement | null>(null);
    useEffect(() => {
        openerRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null;
        return () => {
            const opener = openerRef.current;
            if (opener?.isConnected) opener.focus();
        };
    }, []);

    /*
     * 焦点收到菜单上（可选）。
     *
     * 【顺序要紧】必须排在上面那个"记住触发者"的 effect **之后**：它先记下当前
     * 焦点（编辑器），这里才把焦点搬走；关闭时它再把焦点还回去。反过来写就会把
     * 菜单自己记成触发者，关闭后焦点落在已卸载的节点上、直接掉回 `<body>`。
     *
     * 【为什么延后一个宏任务】同步 `focus()` 会被浏览器**回滚**：右键那一刻的
     * 焦点变化发生在 `contextmenu` 手势内，手势收尾时浏览器把焦点还给了被点的
     * 表面（实测：effect 里同步聚焦后 250ms 再查，焦点仍在 contenteditable 上，
     * 于是方向键又被它吃掉；改成下一宏任务后焦点稳稳停在菜单上）。这与"自动聚焦
     * 要用 setTimeout 0 躲开浏览器自己的焦点处理"是同一类问题。
     */
    useEffect(() => {
        if (!autoFocus) return;
        const timer = window.setTimeout(() => ref.current?.focus(), 0);
        return () => window.clearTimeout(timer);
    }, [autoFocus]);

    /** 可被键盘选中的项下标（禁用项与标题行不参与）。 */
    const selectableIndexes = useMemo(
        () =>
            items.reduce<number[]>(
                (acc, item, index) => (item.disabled || item.heading ? acc : [...acc, index]),
                [],
            ),
        [items],
    );
    /**
     * 是否有任一项带图标 —— 决定是否给**所有**项预留图标列。
     *
     * 【为什么要预留】图标列是定宽的。若只让"有图标的项"占位，同一张菜单里
     * 两类项的文字左缘会差一个列宽（文件浏览器的位置列表就是混合的：只有
     * "收藏/取消收藏"那一项带星标）。
     */
    const reserveIconColumn = useMemo(() => items.some((item) => item.icon), [items]);

    /**
     * 按实测尺寸夹紧，装得下时翻到指针的左上。
     *
     * 【为什么推翻了原先"固定估算高度"的决策】被替换掉的 `NotebookContextMenu`
     * 里有一段注释，说明它刻意用估算而非测量，理由是"在 layout effect 里同步
     * setState 会触发级联渲染（React Compiler 会告警）"。那条理由针对的是**告警**，
     * 不是正确性；而估算的代价是菜单一旦出现换行项（长标签、勾选标记、快捷键列）
     * 就会被裁掉或跑出视口 —— 用户看到的菜单缺一项，且没有任何报错。
     *
     * 现在统一为测量：`useLayoutEffect` 在**绘制前**完成，因此没有视觉闪动；
     * 多付一次渲染的代价远小于"菜单被裁"。同时消除了同一仓库里两种做法并存
     * （本组件测量、`DockTabMenu` 估算），后者也已改为测量。
     *
     * 【为什么不再只是夹紧】锚点在指针上、菜单往右下展开，于是菜单第一行**永远
     * 盖住用户刚点的那个位置**。装得下时翻到左上（原生菜单就是这样）；装不下时
     * 落回 `clampAxisPosition`，与迁移前逐值相同。
     */
    useLayoutEffect(() => {
        const el = ref.current;
        if (!el) return;
        const reclamp = () => {
            const rect = el.getBoundingClientRect();
            const placement = placeMenuWithinViewport({
                anchorX: x,
                anchorY: y,
                width: rect.width,
                height: rect.height,
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight,
                edgeGap: EDGE_GAP,
            });
            setPosition((prev) =>
                prev.left === placement.left && prev.top === placement.top && prev.ready
                    ? prev
                    : { left: placement.left, top: placement.top, ready: true },
            );
        };
        reclamp();
        /*
         * 打开期间视口可能变小（窗口 resize / 缩放 / 停靠布局挤压）：原先只在
         * [x, y, items.length] 变化时夹紧，resize 后菜单可能整张跑出屏幕外。
         * 跟随 resize 重夹；items 本身（而非 length）进依赖，覆盖"标签变长
         * 换行 → 菜单变高"这类条数不变但几何变了的情况。
         */
        window.addEventListener("resize", reclamp);
        return () => window.removeEventListener("resize", reclamp);
    }, [x, y, items]);

    /*
     * 关闭回调走 ref。
     *
     * 【为什么不能把它放进 effect 依赖】调用方几乎都传内联箭头函数，它的身份每次
     * 渲染都变。放进依赖会让"登记 → 注销 → 再登记"跟着每次渲染跑，而**层栈的顺序
     * 就是"谁在最上面"** —— 外层菜单只要重渲染一次就会排到里层之上，于是点里层
     * 菜单时外层先被关掉，正是层栈要解决的问题本身。
     */
    const closeRef = useRef(onClose);
    useLayoutEffect(() => {
        closeRef.current = onClose;
    });

    /**
     * 登记到 dismiss 层栈（只在挂载时登记一次）。
     *
     * 【为什么右键菜单也要登记】"菜单里住着另一个菜单的触发器"确实存在（参数编辑器
     * 工具栏的「⋯」溢出面板里就放着两个自带下拉的组件）。层栈让"点内层不关外层"
     * 成为全局规则，而不是每个菜单各自判断自己是不是外层。
     */
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        return pushMenuDismissLayer(el, () => closeRef.current());
    }, []);

    const step = useCallback(
        (delta: 1 | -1) => {
            if (selectableIndexes.length === 0) return;
            setActiveIndex((current) => {
                const at = selectableIndexes.indexOf(current);
                if (at === -1)
                    return delta === 1 ? selectableIndexes[0] : selectableIndexes.at(-1)!;
                const next = (at + delta + selectableIndexes.length) % selectableIndexes.length;
                return selectableIndexes[next];
            });
        },
        [selectableIndexes],
    );

    /**
     * 全局监听：Esc 关闭、外部指针按下关闭、方向键导航、打字跳项。
     *
     * 【pointerdown 用捕获阶段】编辑器/画布自身的 `pointerdown` 会先改变选择，
     * 若用冒泡阶段，菜单可能先关闭又被下层重新打开。
     *
     * 【keydown 用冒泡阶段】header 槽里的可交互内容（如 DockTabMenu 的重命名
     * 输入框）需要先吃到 Enter/Esc：监听在冒泡阶段时，目标元素的处理先于
     * 本监听，输入框 `stopPropagation()` 即可优先。全局快捷键分发器挂在
     * window 捕获层，无论如何都先于这里 —— 时序不受影响。
     */
    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) {
                // 交给层栈：它知道"点在里层菜单里"不等于"点在所有菜单之外"。
                // 栈为空时（例如只开着某个不登记的手写壳）退回关掉自己。
                if (menuDismissLayerCount() === 0) onClose();
                else handleMenuDismissPointerDown(event.target as Node);
            }
        }
        function onKeyDown(event: KeyboardEvent) {
            // 内层表面（`AppSubMenu` 的子面板）用捕获阶段先处理并 preventDefault；
            // 这里必须让路，否则外层高亮会跟着内层一起动，出现两处高亮。
            if (event.defaultPrevented) return;
            // 焦点在 header 槽的输入框 / 滑杆里时，方向键与 Home/End 属于它们
            // 自己（移动光标 / 改值），不得劫持成菜单导航。Escape 不在其列：
            // 文本控件不消费它，任何位置按 Esc 都应关闭菜单。
            if (event.key !== "Escape" && ownsArrowKeys(document.activeElement)) {
                return;
            }
            // 焦点已在另一层菜单表面时，方向键 / Home / End 也属于那一层。
            // 父层若继续按过期的 activeIndex 移动，会出现两处高亮；子菜单键盘
            // 用户因此无法稳定走到目标项。
            const focused = document.activeElement;
            if (event.key !== "Escape" && focused instanceof HTMLElement) {
                const surface = focused.closest<HTMLElement>('[data-hs-context-menu="1"]');
                if (surface !== null && surface !== ref.current) return;
            }
            switch (event.key) {
                case "Escape": {
                    event.preventDefault();
                    // 【只关最内层】层栈按打开顺序排列，一次 Escape 只弹一层 ——
                    // 否则"视图 → 布局 → 预设"三层会被一次按键全部关掉。
                    if (handleMenuDismissEscape(event)) return;
                    onClose();
                    return;
                }
                case "ArrowDown":
                    event.preventDefault();
                    step(1);
                    return;
                case "ArrowUp":
                    event.preventDefault();
                    step(-1);
                    return;
                case "Home":
                    event.preventDefault();
                    if (selectableIndexes.length) setActiveIndex(selectableIndexes[0]);
                    return;
                case "End":
                    event.preventDefault();
                    if (selectableIndexes.length) setActiveIndex(selectableIndexes.at(-1)!);
                    return;
                /*
                 * PageUp/PageDown = **移动高亮一页**，而不是只滚。
                 *
                 * 【为什么不只滚】只滚会让高亮停在屏幕外（违反"当前项永远完整
                 * 可见"），而 `activeIndex` 与 DOM 焦点这两种高亮表示各要一份特例。
                 */
                case "PageDown":
                case "PageUp": {
                    event.preventDefault();
                    if (selectableIndexes.length === 0) return;
                    const page = computeMenuPageItemCount({
                        clientHeight: ref.current?.clientHeight ?? 0,
                        itemHeight: menuItemHeightOf(ref.current),
                    });
                    const at = selectableIndexes.indexOf(activeIndex);
                    const delta = event.key === "PageDown" ? page : -page;
                    const nextAt =
                        at < 0
                            ? event.key === "PageDown"
                                ? 0
                                : selectableIndexes.length - 1
                            : Math.min(selectableIndexes.length - 1, Math.max(0, at + delta));
                    setActiveIndex(selectableIndexes[nextAt]);
                    return;
                }
                /*
                 * ArrowRight = 展开当前项的子菜单。ArrowLeft 在**顶层**没有去处
                 * （原生菜单里它只是把焦点交给菜单栏），因此不处理。
                 */
                case "ArrowRight": {
                    const item = items[activeIndex];
                    if (!item?.items?.length) return;
                    event.preventDefault();
                    const trigger = Array.from(
                        ref.current?.querySelectorAll<HTMLElement>("[data-menu-key]") ?? [],
                    ).find((element) => element.dataset.menuKey === item.key);
                    if (trigger?.getAttribute("aria-expanded") !== "true") trigger?.click();
                    return;
                }
                default: {
                    /*
                     * 打字跳项。
                     *
                     * 【为什么在 DOM 上匹配】`label` 是 `ReactNode`（图标 + 文本 +
                     * 角标），从 props 里取不出干净字符串；而渲染出来的
                     * `.hs-menu__label-text` 是唯一权威文本，手写菜单也共用它。
                     */
                    const char = menuTypeaheadChar(event.key, event);
                    if (char === null) return;
                    const surface = ref.current;
                    if (!surface) return;
                    const state = pushMenuTypeahead(typeaheadRef.current, char, performance.now());
                    typeaheadRef.current = state;
                    const domItems = menuItemsOf(surface);
                    // DOM 里的可导航项与 `selectableIndexes` 是**同一个顺序**
                    // （渲染顺序 = items 顺序，两者都跳过禁用与标题行）。
                    const match = matchMenuTypeahead(
                        domItems.map(menuItemTypeaheadText),
                        state.query,
                        selectableIndexes.indexOf(activeIndex),
                    );
                    if (match < 0 || match >= selectableIndexes.length) return;
                    event.preventDefault();
                    setActiveIndex(selectableIndexes[match]);
                    return;
                }
            }
        }
        document.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("keydown", onKeyDown);
        return () => {
            document.removeEventListener("pointerdown", onPointerDown, true);
            document.removeEventListener("keydown", onKeyDown);
        };
    }, [activeIndex, items, onClose, selectableIndexes, step]);

    /** 键盘激活：与鼠标点击走同一条路径，保证行为不分叉。（冒泡阶段，理由同上） */
    useEffect(() => {
        function onKeyActivate(event: KeyboardEvent) {
            if (event.key !== "Enter" && event.key !== " ") return;
            // 内层表面先处理的激活让路；焦点在输入框里时，Enter/Space 是在
            // 打字 —— 悬停残留的 activeIndex 不得把打字劫持成"激活菜单项"
            // （表现为输入一个空格反而触发了某条菜单并关掉整窗）。
            if (event.defaultPrevented) return;
            if (ownsArrowKeys(document.activeElement)) return;
            // 焦点已经坐进**另一层菜单表面**（嵌套子面板用真实 DOM 焦点导航并
            // 自己处理激活）：本层的 activeIndex 还停在触发项上 —— 不让路的
            // 话，Enter 会被当成"再点一次触发项"，把刚展开的子菜单关回去，
            // 子菜单项从此无法用键盘激活（键盘用户只能用鼠标开子菜单）。
            const focused = document.activeElement;
            if (focused instanceof HTMLElement) {
                const surface = focused.closest<HTMLElement>('[data-hs-context-menu="1"]');
                if (surface !== null && surface !== ref.current) return;
            }
            const index = activeIndex;
            if (index < 0) return;
            const item = items[index];
            if (!item || item.disabled) return;
            /*
             * 子菜单触发项：**不关闭菜单**，改为把它展开。
             *
             * 触发项的展开状态住在 `AppSubMenu` 内部，父层够不着，所以这里找到它的
             * 按钮并点一下（等同鼠标点击）。不能像普通项那样走 `onSelect` +
             * `onClose` —— 那会让回车变成"整张菜单关掉"。
             *
             * 必须显式 `.click()`，不能指望原生 Enter：本层的键盘高亮是
             * `activeIndex`（`data-active`），**焦点并不在触发项上**，没有按钮去
             * 产生默认的 click。
             */
            if (item.items?.length) {
                event.preventDefault();
                const trigger = Array.from(
                    ref.current?.querySelectorAll<HTMLElement>("[data-menu-key]") ?? [],
                ).find((element) => element.dataset.menuKey === item.key);
                trigger?.click();
                return;
            }
            event.preventDefault();
            item.onSelect?.();
            onClose();
        }
        document.addEventListener("keydown", onKeyActivate);
        return () => document.removeEventListener("keydown", onKeyActivate);
    }, [activeIndex, items, onClose]);

    /*
     * 键盘高亮的可访问名称：`role="menu"` 的键盘契约要求容器报告"当前项"
     *（焦点不落到项上 —— 见 renderEntries 的说明），否则方向键移动时读屏静默。
     * 仅覆盖顶层：子面板由 useMenuKeyboard 用真实 DOM 焦点导航，天然可读。
     */
    const menuId = useId();
    const activeDescendant =
        activeIndex >= 0 && items[activeIndex] && !items[activeIndex].items?.length
            ? `${menuId}-item-${activeIndex}`
            : undefined;

    /*
     * 键盘高亮滚动跟随。
     *
     * 【为什么需要】壳是 `max-height` + 滚动容器：菜单项多到超出视口高度时，方向键
     * 走到可视区之外的项上，高亮存在（`data-active` 已写）却看不见 —— 键盘用户
     * 等于在盲走。
     *
     * 【为什么不用 `scrollIntoView({ block: "nearest" })`】它只知道"客户端可见区域"，
     * 不知道那里盖着一条提示行 —— 于是走到最后一项时，项**技术上可见**、
     * **实际被渐变吃掉**。`revealMenuItem` 按"扣掉提示行"的窗口计算。
     */
    useEffect(() => {
        if (activeIndex < 0) return;
        revealMenuItem(ref.current, document.getElementById(`${menuId}-item-${activeIndex}`));
    }, [activeIndex, menuId]);

    return (
        <div
            ref={ref}
            role="menu"
            aria-label={ariaLabel}
            aria-activedescendant={activeDescendant}
            data-hs-context-menu="1"
            data-hs-floating-menu={floating ? "1" : undefined}
            className="hs-menu"
            // 只有需要收焦点时才可聚焦（`tabIndex=-1` 不进 Tab 序列，仅可编程聚焦）。
            tabIndex={autoFocus ? -1 : undefined}
            style={{
                left: position.left,
                top: position.top,
                minWidth,
                visibility: position.ready ? undefined : "hidden",
                // 聚焦到菜单上时不该出现焦点圈：它是弹出表面，不是可交互控件本身。
                outline: autoFocus ? "none" : undefined,
            }}
            // 阻止冒泡到 document 的 pointerdown 关闭逻辑：面板自身的容器
            // 通常也监听 pointerdown 来清除选择，菜单内点击不应触发它。
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            <MenuScrollScope>
                {header ? <div className="hs-menu__header">{header}</div> : null}
                {renderMenuEntries(
                    items,
                    activeIndex,
                    reserveIconColumn,
                    setActiveIndex,
                    menuId,
                    onClose,
                )}
            </MenuScrollScope>
        </div>
    );
}

function AppContextMenuItem({
    item,
    active,
    reserveIcon,
    id,
    onActive,
    onSelect,
}: {
    item: AppMenuItemSpec;
    active: boolean;
    reserveIcon: boolean;
    /** 顶层项的 id（aria-activedescendant 引用）；子层为 undefined。 */
    id?: string;
    /**
     * 把"指针在这一项上"汇报给**拥有 activeIndex 的那一层**。
     * 子面板不传 —— 它的下标不能写进外层（见 `renderMenuEntries` 的 `onHoverItem`）。
     */
    onActive?: () => void;
    onSelect: () => void;
}) {
    /* 见 `AppMenuItemSpec.onHover` / `onHoverEnd`：调用方的联动与"本层高亮"是两件事。 */
    if (item.heading) {
        return <div className="hs-menu__label">{item.label}</div>;
    }

    /*
     * 【三种角色的分工】`aria-checked` 只允许出现在 `menuitemcheckbox` /
     * `menuitemradio` 上。互斥档位必须是 radio —— 挂成 checkbox 时读屏会念
     * "复选框"，用户听到的是"可以多选"。
     */
    const role = item.radio
        ? "menuitemradio"
        : item.checked === undefined
          ? "menuitem"
          : "menuitemcheckbox";

    return (
        <button
            type="button"
            id={id}
            disabled={item.disabled}
            role={role}
            aria-checked={item.checked === undefined ? undefined : item.checked ? "true" : "false"}
            // 键盘高亮与鼠标悬停在 CSS 里是**同一条规则**（`[data-active]` 与
            // `:hover` 并列），因此这里只需如实标出"当前高亮的是这一项"。
            data-active={active && !item.disabled ? "1" : undefined}
            data-danger={item.danger ? "1" : undefined}
            data-tooltip={item.tooltip}
            className="hs-menu__item"
            onMouseEnter={() => {
                onActive?.();
                // 调用方的联动（例如高亮画布上的参考曲线）与"本层高亮"是两件事：
                // 前者即使在没有 activeIndex 的子面板里也要照常触发。
                item.onHover?.();
            }}
            onMouseLeave={() => item.onHoverEnd?.()}
            onClick={() => {
                if (item.disabled) return;
                onSelect();
            }}
        >
            <span className="flex min-w-0 flex-col items-start">
                <span className="flex min-w-0 items-center gap-2">
                    {item.icon ? (
                        <span className="hs-menu__icon" aria-hidden>
                            {item.icon}
                        </span>
                    ) : reserveIcon ? (
                        <span className="hs-menu__icon" aria-hidden />
                    ) : null}
                    <span className="hs-menu__label-text">{item.label}</span>
                </span>
                {/* 【为什么只在 disabled 时】可用项的第二行是噪声；而"点不了"的
                    理由必须在点之前就能读到（菜单里没有别的通道）。 */}
                {item.disabled && item.description ? (
                    <span className="hs-menu__description">{item.description}</span>
                ) : null}
            </span>
            <span className="hs-menu__trail">
                {item.checked ? (
                    <span className="hs-menu__check" aria-hidden>
                        <CheckIcon width={12} height={12} />
                    </span>
                ) : null}
                {item.shortcut ? <span>{item.shortcut}</span> : null}
                {item.hint ? <span>{item.hint}</span> : null}
            </span>
        </button>
    );
}

export interface AppSubMenuProps {
    /** 触发项文案。 */
    label: ReactNode;
    /** 触发项右侧的计数角标（如 Take 数量）。 */
    badge?: string;
    /**
     * 触发项左侧图标。
     *
     * 【为什么与 `AppContextMenu` 的 `icon` 同款】子菜单的触发项本身就是一级菜单
     * 里的一行，图标列必须和它的兄弟行对齐 —— 否则"有子菜单的那一行"文字会往左
     * 多出一个列宽（工具栏工具菜单里三种工具 + 预设入口并列，一眼就能看出错位）。
     */
    icon?: ReactNode;
    /**
     * 该项在**本层**的 key，渲染成 `data-menu-key`。
     *
     * 【为什么需要】`AppContextMenu` 的键盘高亮是 `activeIndex`（`data-active`），
     * 它**不移动 DOM 焦点**（与 `useMenuKeyboard` 相反）。于是按 Enter 时焦点还在
     * 菜单外壳上，没有一个"已获得焦点的按钮"去产生原生 click —— 回车对子菜单
     * 触发项会**毫无反应**。父层靠这个属性把触发项找出来，显式 `.click()`。
     */
    itemKey?: string;
    /**
     * 指针在**本层哪一处**：触发项上（`true`）还是子面板里（`false`）。
     *
     * 【为什么必须有】`AppContextMenu` 的"当前项"（`activeIndex`）跟着鼠标走 ——
     * 这样"划过去再按回车"激活的才是划到的那一条。但触发项此前**从不汇报悬停**，
     * 于是 `activeIndex` 停在上一个被划过的普通项上：指针移到触发项、或进了子面板
     * 之后，那条普通项仍然带着 `data-active` 亮着，与指针底下那一条**同时高亮**。
     *
     * 传 `false` 表示"指针已不在触发项上"，父层据此清掉自己的高亮。
     */
    onHoverChange?: (onTrigger: boolean) => void;
    disabled?: boolean;
    /**
     * 键盘高亮态（`data-active`）。
     *
     * 【为什么由外面传】`AppContextMenu` 用 `activeIndex` 表示"当前高亮项"（鼠标
     * 悬停与方向键共用），而本组件原本只服务手写菜单 —— 那些菜单用**焦点**表示
     * 高亮。两者混用时，若不把父层的下标传进来，方向键走到子菜单触发项上会
     * **一点反馈都没有**：`activeIndex` 变了，但触发项没有 `data-active`。
     */
    active?: boolean;
    /**
     * 即使没有图标也占住图标列（与 `AppContextMenuItem` 的同名参数一致）。
     *
     * 同一层里只要有一项带图标，整层的文字左缘就要对齐；触发项若不自占位，
     * 它会比兄弟行少一个列宽。
     */
    reserveIcon?: boolean;
    /**
     * 子面板壳的追加类。
     *
     * 【为什么需要】子面板默认与主菜单同一套壳（`max-height` + 滚动）。子面板里若
     * 要放**自带滚动**的内容（长列表 + 钉住的页脚，如颤音预设的飞行面板），面板再
     * 滚一次就是双滚动条；此时需要 `hs-menu--no-scroll` 让面板放手，由内容自己
     * 约束高度。
     */
    panelClassName?: string;
    /**
     * 子面板**内容层**的追加类。
     *
     * 【为什么与 `panelClassName` 分开】内边距搬到了 `.hs-menu__content`（见
     * `index.css` 的说明），于是"壳是 flex 容器"不再等于"内容层是 flex 容器"。
     * `flex flex-col` 这类**布局**类必须落在内容层上，否则它们作用在包装元素上，
     * 里面的 `flex: 1 1 auto` 子项就再也拿不到剩余高度。
     */
    panelContentClassName?: string;
    /**
     * 子面板内容。
     *
     * 【为什么是 children 而不是 items】子面板里经常要放 `AppContextMenu` 装不下的
     * 东西 —— 带尾随按钮的行、滑杆、内联输入框。这与本文件顶部"手写菜单存在
     * 的理由"是同一条。（`AppContextMenu` 的 `items` API 另有 `items` 字段做嵌套，
     * 走的是本组件。）
     */
    children: ReactNode;
}

/**
 * 一级菜单里的二级子菜单（悬停或点击展开）。
 *
 * 【为什么收进本文件】它原本是 `ClipContextMenu` 的局部组件，而"菜单里要有
 * 子菜单"并不是 Clip 特有的需求 —— 颤音预设列表同样需要（十几个预设平铺会把
 * 菜单撑得比屏幕高）。放进原语层，第二次需要它的人不必再写一遍定位夹紧与
 * 键盘导航，也顺带让 `useMenuKeyboard` 的嵌套菜单支持有了第二个消费者。
 *
 * 【导航】`useMenuKeyboard` 按 `closest('[role="menu"]')` 分层，因此外层菜单与
 * 子面板各按各的方向键走，互不串门。子面板额外接受 `ArrowLeft` 退出。
 */
export function AppSubMenu({
    label,
    badge,
    icon,
    itemKey,
    onHoverChange,
    disabled = false,
    active = false,
    reserveIcon = false,
    panelClassName,
    panelContentClassName,
    children,
}: AppSubMenuProps) {
    const [open, setOpen] = useState(false);
    const panelRef = useRef<HTMLDivElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    // 子面板是**条件渲染**的：挂载瞬间 `panelRef.current` 还是 null。把 `open`
    // 作为钩子的 `active` 传下去，面板真正出现时 effect 才会重跑并注册导航
    // （否则只跑一次就命中 `if (!container) return`，方向键永远由外层菜单响应）。
    useMenuKeyboard(panelRef, open, { nested: true, onDismiss: () => setOpen(false) });

    /*
     * 子面板也进 dismiss 层栈：它比外层菜单**后打开**，所以是栈顶。
     * 没有这一条时，点子面板会被外层判成"点在菜单外面"。
     */
    useEffect(() => {
        if (!open) return;
        const panel = panelRef.current;
        if (!panel) return;
        return pushMenuDismissLayer(panel, () => setOpen(false));
    }, [open]);

    useLayoutEffect(() => {
        if (!open) return;
        const panel = panelRef.current;
        const wrapper = wrapperRef.current;
        if (!panel || !wrapper) return;
        /*
         * 【为什么面板是 fixed 定位】子面板以 `position: fixed` 相对**视口**展开
         * （见 `index.css` 的 `.hs-menu--submenu`）：它不再是父壳里被
         * `overflow-y: auto` 裁切的绝对定位后代，因此父壳可以保持
         * `max-height` + 滚动 —— 带子菜单的菜单与扁平菜单在窗口高度不够时
         * 行为一致（都能滚）。代价是面板不再跟随壳滚动，由下面的滚动收起兜住。
         *
         * 坐标全部从触发项的**视口矩形**推出，量与夹的循环与壳的定位同一套纪律：
         * 每一步都实测 `getBoundingClientRect`，不估算。
         */
        const place = () => {
            const trigger = wrapper.getBoundingClientRect();
            // 与触发项重叠 4px 展开：指针从触发项滑进面板时中间不能出现"两者都
            // 不在"的缝隙 —— 缝隙里 mouseleave 会把面板收掉（与旧的
            // `calc(100% - 4px)` 同一个理由）。
            panel.style.left = `${trigger.right - 4}px`;
            panel.style.top = `${trigger.top - 5}px`;
            // 宽度随内容展开：fixed 定位面板的宽度默认由内容决定，长文本以
            // max-content 展开后若超出视口，按最终锚定侧的可用空间收口 ——
            // 行内标签以 truncate 兜底。
            panel.style.width = "max-content";
            panel.style.maxWidth = "none";

            const vw = window.innerWidth;
            const vh = window.innerHeight;
            let rect = panel.getBoundingClientRect();
            // 右侧放不下就翻到左侧（菜单贴着视口右缘时必然如此）。
            let anchoredLeft = true;
            if (rect.right > vw - 4) {
                panel.style.left = `${trigger.left - rect.width + 4}px`;
                anchoredLeft = false;
            }
            rect = panel.getBoundingClientRect();
            const availableWidth = anchoredLeft ? vw - 8 - rect.left : rect.right - 8;
            if (rect.width > availableWidth) {
                panel.style.maxWidth = `${Math.max(160, Math.floor(availableWidth))}px`;
                rect = panel.getBoundingClientRect();
            }
            // 下方放不下就向上对齐（面板底边与触发项底边齐平）。
            if (rect.bottom > vh - 4) {
                panel.style.top = `${trigger.bottom - rect.height + 5}px`;
            }
        };
        place();
        // 打开期间视口可能变小（窗口 resize / 缩放）：壳会重夹紧、触发项会移动，
        // 面板跟着重摆一次。
        window.addEventListener("resize", place);
        return () => window.removeEventListener("resize", place);
    }, [open]);

    /*
     * 【滚动即收起】面板不再随壳滚动：壳（或任何祖先滚动容器）滚动时，触发项
     * 移走了而面板停在原地 —— 指针与键盘都会"指着空气"。原生菜单在宿主滚动时
     * 同样收起子菜单，取舍一致：滚动一次 = 子菜单收起，再悬停即可重新展开。
     *
     * 面板**自己**的滚动除外 —— 长列表（Take、预设）在面板内滚动是正常操作，
     * 不该把面板收掉。
     */
    useEffect(() => {
        if (!open) return;
        function onScroll(event: Event) {
            if (panelRef.current?.contains(event.target as Node)) return;
            setOpen(false);
        }
        document.addEventListener("scroll", onScroll, true);
        return () => document.removeEventListener("scroll", onScroll, true);
    }, [open]);

    return (
        <div
            ref={wrapperRef}
            className="relative"
            onMouseEnter={() => {
                // 指针在触发项上 —— 父层的"当前项"也要跟过来，否则它会留在
                // **上一个**被划过的普通项上，与这里同时高亮（见 onHoverChange）。
                onHoverChange?.(true);
                if (!disabled) setOpen(true);
            }}
            onMouseLeave={() => setOpen(false)}
        >
            <button
                type="button"
                role="menuitem"
                className="hs-menu__item"
                disabled={disabled}
                data-menu-key={itemKey}
                // 父层的键盘高亮（`activeIndex`）与鼠标悬停走同一条 CSS 规则，
                // 因此这里只要如实标出"当前高亮的是这一项"。
                data-active={active && !disabled ? "1" : undefined}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!disabled) setOpen((value) => !value);
                }}
                aria-haspopup="menu"
                aria-expanded={open}
            >
                <span className="flex min-w-0 items-center gap-2">
                    {icon ? (
                        <span className="hs-menu__icon">{icon}</span>
                    ) : reserveIcon ? (
                        <span className="hs-menu__icon" aria-hidden />
                    ) : null}
                    <span className="hs-menu__label-text">{label}</span>
                    {badge ? (
                        <span className="text-qt-micro leading-none rounded bg-black/20 px-1 py-0.5 opacity-70 max-w-36 truncate">
                            {badge}
                        </span>
                    ) : null}
                </span>
                <span className="hs-menu__trail">
                    <svg
                        width="12"
                        height="12"
                        viewBox="0 0 15 15"
                        fill="none"
                        aria-hidden="true"
                        className="shrink-0 opacity-50"
                    >
                        <path
                            d="M6 3.5L10 7.5L6 11.5"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                </span>
            </button>
            {open && !disabled ? (
                <div
                    ref={panelRef}
                    role="menu"
                    data-hs-context-menu="1"
                    // 【不做移动端的底部 sheet】面板的坐标由上面的 layout effect
                    // 逐条写在**内联样式**里；sheet 变换会用 `!important` 把它们
                    // 全部覆盖成"贴底通栏"，于是子菜单脱离触发项、跑到屏幕底部，
                    // 而父菜单还开着 —— 两层各自都不知道对方在哪。
                    data-hs-menu-inline="1"
                    // 子面板与主菜单**共用同一个表面**；定位由上面的 layout effect
                    // 逐条覆盖（翻左 / 对齐 / 收宽），因此只借 `--submenu` 的
                    // `position: fixed`（相对视口展开，不被父壳裁切 —— 见
                    // `index.css` 的 `.hs-menu--submenu`）。
                    className={cx("hs-menu hs-menu--submenu", panelClassName)}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                    // 指针进了子面板：父层那个触发项**不该再亮着** —— 亮着的应该是
                    // 指针底下这一条。回到触发项上时（`mouseleave` 落在包装盒内）
                    // 再交还给它。
                    onMouseEnter={() => onHoverChange?.(false)}
                    onMouseLeave={() => onHoverChange?.(true)}
                >
                    <MenuScrollScope contentClassName={panelContentClassName}>
                        {children}
                    </MenuScrollScope>
                </div>
            ) : null}
        </div>
    );
}

/** 锚定菜单与触发元素之间的呼吸（与 `--qt-space-2` 同一个值）。 */
const ANCHORED_MENU_GAP_PX = 4;

export interface AppAnchoredMenuProps {
    /** 触发元素：菜单在它**下方**展开（`align` 决定左右对齐）。 */
    anchorRef: RefObject<HTMLElement | null>;
    /** 是否展开。`false` 时本组件返回 `null`，调用方不必自己判空。 */
    open: boolean;
    /**
     * 菜单容器的 ref（**必填**）。
     *
     * 【为什么必填】它同时承担两件事：① 外部"点在菜单内部就不关闭"的判定
     * （`ref.current.contains(target)`）；② 本组件按实测尺寸做视口夹紧。
     * 菜单现在挂在 `document.body` 下，**不再是锚点的 DOM 后代** —— 拿锚点的 ref
     * 去 `contains` 会永远为假，那会让"点菜单里的按钮反而把菜单关掉"。
     */
    menuRef: RefObject<HTMLDivElement | null>;
    /** 追加类（布局用，如 `flex flex-col`）。外观类由壳提供，不要在这里重写。 */
    className?: string;
    /** 内容层的追加类（内边距与布局都在这一层，见 `MenuScrollScope`）。 */
    contentClassName?: string;
    /** 内联样式（如各面板自己算出的 `maxHeight`）。 */
    style?: CSSProperties;
    /**
     * 与锚点的水平对齐：`"start"`（默认，左对齐）或 `"end"`（右对齐）。
     *
     * 【为什么需要】工具栏最右侧的下拉（语言、溢出菜单）左对齐展开时会横跨整个
     * 工具栏、甚至被视口夹回来；原生下拉在右端一律右对齐。
     */
    align?: "start" | "end";
    /**
     * 声明式条目（与 `AppContextMenu` **同一套** `AppMenuItemSpec`）。
     *
     * 【为什么锚定菜单也要有它】工具栏上的"右键还有几个模式可选"这类菜单内容完全
     * 是数据（标题 + 几个单选项），手写一遍 `<button className="hs-menu__item">`
     * 只会让同一个菜单在两处渲染出不同的行高与勾选位置。给了 `items` 就不需要
     * `children`；需要内联滑杆 / 子菜单的手写菜单仍然走 `children`（那是它存在的
     * 理由，见文件头）。
     */
    items?: AppMenuItemSpec[];
    /**
     * 关闭回调。**给了它就由本组件接管 Esc 与"点外面关闭"**。
     *
     * 【为什么是可选的】手写菜单的调用方（参数编辑器的三个工具菜单）自己管着一份
     * 关闭逻辑（它们要同时关掉同组的其它菜单）。不给就等于"本组件只负责摆位置"，
     * 与改动前逐字一致。
     */
    onClose?: () => void;
    /**
     * 打开时把键盘焦点收到菜单上。
     *
     * 【与 `AppContextMenu.autoFocus` 的分工】那个解决的是"触发面自己会吞方向键"
     * （contenteditable）；这里解决的是"**键盘打开**的下拉菜单焦点应当进入菜单"
     * —— 鼠标打开时焦点留在触发器上（与 VS Code / Windows 菜单栏一致）。
     * 由触发器在 `ArrowDown` / `Enter` 打开时传 `true`。
     */
    autoFocus?: boolean;
    /** 无障碍名称（`role="menu"` 需要一个可读名字）。 */
    ariaLabel?: string;
    children?: ReactNode;
}

/**
 * 锚定在触发元素下方的菜单表面。
 *
 * 【与 `AppContextMenu` 的分工】那个锚在**指针**上（右键菜单），这个锚在**控件**
 * 上（工具栏按钮下拉）。两者共用同一个壳与同一套条目样式，区别只有坐标从哪来。
 *
 * 【为什么必须 portal 到 `document.body`】菜单只要留在触发它的布局盒里，就会被
 * 沿途任何一层 `overflow: hidden` 裁掉 —— 参数编辑器的工具按钮下拉要穿过 9 层，
 * 实测可视高度 0px（详见 `src/index.css` 的 `.hs-menu--submenu` 说明）。挂到
 * body 之后，布局盒里再加多少层裁切/滚动/变换都不影响它。
 *
 * 【坐标怎么来】一次 `useLayoutEffect` 同时做三件事：按锚点矩形取期望坐标、
 * 按菜单**实测尺寸**夹紧进视口、按"锚点是否住在移动端菜单 sheet 里"决定要不要
 * 跳过底部 sheet 变换。它在浏览器绘制**之前**跑完，因此菜单首帧那一次临时坐标
 * （`0,0`）用户看不到 —— 这也是不需要"先渲染、再夹紧"两趟状态机的原因。
 * 竖直方向只夹紧、不翻转：菜单属于它所在的面板，翻到按钮上方只会盖住本面板自己
 * 的工具栏（见 `menuPlacement.ts` 的同名说明）。
 */
export function AppAnchoredMenu({
    anchorRef,
    open,
    menuRef,
    className,
    contentClassName,
    style,
    align = "start",
    items,
    onClose,
    autoFocus = false,
    ariaLabel,
    children,
}: AppAnchoredMenuProps) {
    const [position, setPosition] = useState({ left: 0, top: 0 });

    /*
     * 方向键 / Home / End / PageUp·PageDown / 打字跳项。
     *
     * 【为什么锚定菜单也要有】它与 `AppContextMenu` 是同一类表面（工具栏按钮的
     * "更多选项"），键盘用户没有任何理由在两处得到两套手感。用真实 DOM 焦点导航
     * 而不是 `activeIndex`：这里的项就是真实 `<button>`，不需要再造一套高亮。
     */
    useMenuKeyboard(menuRef, open);

    /*
     * Esc / 点外面关闭（仅在调用方给了 `onClose` 时接管）。
     *
     * pointerdown 用**捕获**阶段：面板自身的 pointerdown 会先改变选择，冒泡阶段
     * 处理会让"点外面"变成"先选中、再关菜单"。与 `AppContextMenu` 同一理由。
     *
     * 【为什么不直接 `close()`】层栈里可能还压着里层菜单（参数工具栏的「⋯」面板
     * 里就住着两个自带下拉的组件）。由层栈判断"点在谁身上"，本层才不会在用户点
     * 内层菜单时被误关。
     */
    useEffect(() => {
        if (!open || !onClose) return;
        const close = onClose;
        function onPointerDown(event: PointerEvent) {
            if (menuRef.current?.contains(event.target as Node)) return;
            /*
             * 【触发器自己不算"外面"】点触发器是**切换**（开着就关、关着就开），
             * 由触发器自己的 onClick 负责。若这里先关一次，切换就会变成"关了再
             * 开"——用户看到菜单闪一下又回来。
             */
            if (anchorRef.current?.contains(event.target as Node)) return;
            if (menuDismissLayerCount() === 0) close();
            else handleMenuDismissPointerDown(event.target as Node);
        }
        function onKeyDown(event: KeyboardEvent) {
            if (event.key !== "Escape") return;
            event.preventDefault();
            if (handleMenuDismissEscape(event)) return;
            close();
        }
        document.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("keydown", onKeyDown, true);
        return () => {
            document.removeEventListener("pointerdown", onPointerDown, true);
            document.removeEventListener("keydown", onKeyDown, true);
        };
    }, [open, onClose, menuRef, anchorRef]);

    /*
     * 进层栈：让"点内层菜单不关外层"对锚定菜单同样成立。
     *
     * 【依赖只有 open】与 AppContextMenu 同一理由：onClose 的身份每次渲染都变，
     * 放进依赖会让层栈顺序跟着渲染次数抖 —— 而顺序就是"谁在最上面"。
     */
    const anchoredCloseRef = useRef(onClose);
    useLayoutEffect(() => {
        anchoredCloseRef.current = onClose;
    });
    useEffect(() => {
        if (!open) return;
        const element = menuRef.current;
        if (!element) return;
        return pushMenuDismissLayer(element, () => anchoredCloseRef.current?.());
    }, [open, menuRef]);

    /*
     * 关闭时把焦点还给触发它的元素。
     *
     * 【为什么需要】方向键导航会把焦点搬进菜单项；菜单一卸载，焦点就掉回 `<body>`，
     * 键盘用户下一次 Tab 从文档头重新开始 —— 与 `AppContextMenu` 的 `openerRef`
     * 同一件事，只是这里的触发者恒为锚点里的按钮。
     */
    const openerRef = useRef<HTMLElement | null>(null);
    useEffect(() => {
        if (!open) return;
        openerRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null;
        return () => {
            const opener = openerRef.current;
            if (opener?.isConnected) opener.focus();
        };
    }, [open]);

    /*
     * 键盘打开时把焦点收进菜单。
     *
     * 【为什么延后一个宏任务】与 `AppContextMenu.autoFocus` 同一原因：触发键的
     * 默认行为（以及浏览器的焦点收尾）会在本帧之后才落定，同步 `focus()` 会被
     * 回滚。
     */
    useEffect(() => {
        if (!open || !autoFocus) return;
        const timer = window.setTimeout(() => menuRef.current?.focus(), 0);
        return () => window.clearTimeout(timer);
    }, [open, autoFocus, menuRef]);

    useLayoutEffect(() => {
        if (!open) return;
        const el = menuRef.current;
        const anchor = anchorRef.current;
        if (!el || !anchor) return;
        /*
         * 【移动端：住在 sheet 里的菜单不做 sheet 变换】
         *
         * 移动形态的菜单栏住在 `MobileMenuSheet` 里（`shell/MobileChrome.tsx`）。
         * 若它的下拉也命中"底部 sheet"那条规则，点一次「视图」会**叠出第二层
         * sheet**，返回路径从一层变成两层。
         *
         * 判据取**结构事实**（锚点是否住在 sheet 里），而不是 `isMobileForm()`
         * 分支：形态是 CSS 令牌层的一等公民，多一个 JS 分支就等于给"这是哪种
         * 形态"开了第二个答案。
         */
        if (anchor.closest("[data-mobile-menu-sheet]") !== null) {
            el.dataset.menuInline = "1";
        } else {
            delete el.dataset.menuInline;
        }
        const reclamp = () => {
            const anchorRect = anchor.getBoundingClientRect();
            const rect = el.getBoundingClientRect();
            const desiredLeft = align === "end" ? anchorRect.right - rect.width : anchorRect.left;
            const next = {
                left: clampAxisPosition(desiredLeft, rect.width, window.innerWidth, 0, EDGE_GAP),
                top: clampAxisPosition(
                    anchorRect.bottom + ANCHORED_MENU_GAP_PX,
                    rect.height,
                    window.innerHeight,
                    0,
                    EDGE_GAP,
                ),
            };
            // 值不变就不落新对象：调用方的 items 数组常是逐渲染重建的（如录音菜单），
            // items 进依赖后重夹紧会跟着每次渲染跑；无守卫的 setPosition 会反过来
            // 再触发渲染，形成"渲染 → 重夹 → 渲染"的循环。
            setPosition((prev) => (prev.left === next.left && prev.top === next.top ? prev : next));
        };
        reclamp();
        /*
         * 打开期间**内容**与**视口**都可能变：录音菜单在打开的同一拍里异步枚举
         * 设备/应用，列表落地的瞬间 items 变长 —— 原先只在 open 翻真时测一次，
         * 菜单底部可能顶出视口。items 本身（而非 length）进依赖，覆盖"标签变长
         * 换行 → 菜单变高"这类条数不变但几何变了的情况；resize 重夹与
         * AppContextMenu 同一理由（窗口缩小后菜单不得整张跑出屏幕）。
         */
        window.addEventListener("resize", reclamp);
        // 内容是异步枚举 / 本地化文案换行时，bounding rect 也会变；用元素的
        // 真实尺寸变化再夹一次，而不是把每次渲染重建的 items 数组塞进依赖。
        const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(reclamp);
        observer?.observe(el);
        return () => {
            window.removeEventListener("resize", reclamp);
            observer?.disconnect();
        };
    }, [open, anchorRef, menuRef, align]);

    if (!open) return null;

    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            aria-label={ariaLabel}
            data-hs-context-menu="1"
            className={cx("hs-menu", className)}
            tabIndex={autoFocus ? -1 : undefined}
            style={{
                left: position.left,
                top: position.top,
                // 聚焦到菜单上时不该出现焦点圈：它是弹出表面，不是可交互控件本身。
                outline: autoFocus ? "none" : undefined,
                ...style,
            }}
        >
            <MenuScrollScope contentClassName={contentClassName}>
                {items
                    ? renderMenuEntries(
                          items,
                          -1,
                          items.some((item) => item.icon),
                          null,
                          null,
                          onClose ?? (() => undefined),
                      )
                    : children}
            </MenuScrollScope>
        </div>,
        menuPortalContainer(),
    );
}
