/**
 * 文件浏览器拖拽期间的「悬停副作用抑制」标记。
 *
 * 【要修的问题】在文件浏览器里按住一行开始拖拽时，浮标会进入"钉住"模式
 * （`AppTooltipProvider.onPointerDown` 见到 `data-tooltip` 元素就钉住），
 * 而钉住期间气泡跟随指针 —— 于是把它拖进时间轴时，文件浏览器那条 ToolTip
 * 会一路跟着鼠标飘过去。
 *
 * 【为什么放在外部 store 而不是 Redux】同 `dockDragStore`：拖拽中每个
 * `pointermove` 都会惊动订阅全量 store 的组件（菜单栏、33Hz 播放轮询）。
 * 本标记只在指针事件里被同步读取，不参与任何渲染，没有订阅者 —— 因此这里
 * 只需要一个模块级布尔量，连 `useSyncExternalStore` 的订阅机制都不必带。
 *
 * 【为什么只有"是否激活"一个字段】本标记只回答一个问题：**浮标该不该消失**。
 * 拖拽的内容与坐标都由 `hifi-file-drag` 事件承载，这里不复制第二份。
 *
 * 【为什么只在越过阈值后置真】未越过阈值时用户只是在点击（选中 / 试听），
 * 此时气泡该照常显示 —— 提前抑制会让"点一下文件名"时气泡莫名消失。
 */

let active = false;

/**
 * 置真 / 置假。只在拖拽越过启动阈值时置真，在所有结束路径
 * （drop / cancel / 失焦）置假。
 */
export function setFileBrowserDragActive(next: boolean): void {
    active = next;
}

/** 当前是否处于"已越过阈值的文件浏览器拖拽"中。 */
export function isFileBrowserDragActive(): boolean {
    return active;
}

/** 仅测试用：把标记复位，避免用例之间互相污染。 */
export function resetFileBrowserDragStoreForTests(): void {
    active = false;
}
