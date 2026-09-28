/*
 * 面板 DOM 宿主的注册表 —— 停靠系统"不重挂载"的关键。
 *
 * 【要解决的问题】时间轴轨道区走 WebGL2 自绘内核（`timelineKernelHost`）。
 * 如果停靠重排导致 React 卸载再挂载这个组件，代价是：GL 上下文销毁重建、
 * 波形 mipmap 缓存全部失效、滚动位置与缩放丢失，用户会看到数秒卡顿。参数
 * 编辑器同理（7945 行，自带重试收敛循环）。
 *
 * 【解法】每个窗体的 React 子树**全局只挂载一次**，渲染进一个由本模块创建、
 * 并由本模块搬来搬去的 `<div>`。布局树变化时只把这个 div `appendChild` 到
 * 新的槽位 —— React 组件从不卸载，GL 上下文、滚动位置、内部状态全部幸存。
 * 因为节点是带着 `data-hs-surface` 一起搬的，`uiFocus/focusSurface` 的
 * `closest()` 就近解析也天然继续正确。
 *
 * 【停泊区为什么要保留尺寸】非活动标签的宿主被移进停泊区。若停泊区用
 * `display:none`，面板的 `clientWidth` 会变成 0，ResizeObserver 报 0×0 ——
 * 内核按 0 视口计算滚动范围与最小缩放，容易算出退化值。因此停泊区放在视口外
 * （`left:-20000px`）并给宿主写上**最后一次实测尺寸**，让面板始终看到合理的
 * 尺寸；`visibility:hidden` 保证它不参与绘制。
 */

const hosts = new Map<string, HTMLDivElement>();
const listeners = new Set<() => void>();
let version = 0;
let parking: HTMLDivElement | null = null;

function notify(): void {
    version += 1;
    for (const listener of listeners) listener();
}

export function subscribePanelHosts(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** 快照版本号：`useSyncExternalStore` 用它判断"宿主集合变了"。 */
export function getPanelHostVersion(): number {
    return version;
}

function ensureParking(): HTMLDivElement {
    if (parking?.isConnected) return parking;
    const element = document.createElement("div");
    element.dataset.dockParking = "1";
    element.setAttribute("aria-hidden", "true");
    // 视口外 + 不绘制：既不闪，也不产生滚动条。
    element.style.cssText =
        "position:fixed;left:-20000px;top:0;visibility:hidden;pointer-events:none;z-index:-1;";
    document.body.appendChild(element);
    parking = element;
    return element;
}

/**
 * 取得（必要时创建）窗体的宿主元素。
 *
 * 幂等：同一个 formId 永远拿到同一个元素 —— 这正是"React 不重挂载"的前提。
 *
 * 【刻意不在此通知订阅者】本函数会在**渲染期**被调用（`PanelMount` 需要宿主
 * 才能在本次渲染里建 portal）。在渲染期唤醒别的组件的订阅，会触发 React 的
 * "渲染另一个组件时更新状态"告警。通知改由两个提交后时机负责：`PanelMount`
 * 的 effect（`notifyPanelHosts`）与 `releasePanelHost`。
 */
export function acquirePanelHost(formId: string): HTMLDivElement {
    const existing = hosts.get(formId);
    if (existing) return existing;

    const host = document.createElement("div");
    host.dataset.dockHost = formId;
    // 默认填满槽位；停泊时会被写成显式像素（见 `parkPanelHost`）。
    host.style.cssText = "width:100%;height:100%;min-width:0;min-height:0;overflow:hidden;";
    hosts.set(formId, host);
    ensureParking().appendChild(host);
    return host;
}

/**
 * 通知订阅者"宿主集合变了"。
 *
 * 只允许在提交后（effect / 事件回调）调用，见 `acquirePanelHost` 的说明。
 */
export function notifyPanelHosts(): void {
    notify();
}

export function getPanelHost(formId: string): HTMLDivElement | undefined {
    return hosts.get(formId);
}

/**
 * 槽位归属：哪个 `useDockSlot` 实例当前"拥有"这个宿主。
 *
 * 【为什么必须有它】槽位切换（停靠 ⇄ 浮动、换标签组、换标签）时，旧槽位的
 * **被动 effect 清理**晚于新槽位的 **layout effect** 执行 —— 也就是说新槽位
 * 刚把宿主搬进自己的槽，旧槽位的清理才跑，若它无条件把宿主移回停泊区，面板
 * 就会"刚停靠好就消失"，用户必须手动重开。归属令牌让清理只在"宿主仍归我"
 * 时才动手。
 */
const owners = new Map<string, symbol>();

/** 生成一个槽位归属令牌。 */
export function createSlotOwner(): symbol {
    return Symbol("dock-slot");
}

/**
 * 把宿主搬进一个槽位。
 *
 * 【为什么要先测尺寸】搬出去之前记下它当前的实测尺寸，供将来停泊时使用 ——
 * 面板被移进停泊区时若拿不到"上次有多大"，就只能猜，而猜错会让内核按错误
 * 视口算一遍滚动范围。
 */
export function attachPanelHost(formId: string, slot: HTMLElement, owner: symbol): void {
    const host = acquirePanelHost(formId);
    owners.set(formId, owner);
    if (host.parentElement === slot) return;

    const rect = host.getBoundingClientRect();
    if (rect.width > 1 && rect.height > 1) {
        lastSize.set(formId, { w: Math.round(rect.width), h: Math.round(rect.height) });
    }

    host.style.width = "100%";
    host.style.height = "100%";
    slot.appendChild(host);
}

const lastSize = new Map<string, { w: number; h: number }>();

/**
 * 把宿主移回停泊区，并按最后一次实测尺寸给它一个明确的尺寸。
 *
 * 传 `fallback` 是为了让"从未显示过"的窗体也有合理尺寸（取面板定义的默认值）。
 */
export function parkPanelHost(
    formId: string,
    fallback: { w: number; h: number },
    owner: symbol,
): void {
    const host = hosts.get(formId);
    if (!host) return;
    // 已被别的槽位接手：这里**不是**它的主人，绝不能把它搬走（见 `owners` 的说明）。
    if (owners.get(formId) !== owner) return;
    owners.delete(formId);

    const rect = host.getBoundingClientRect();
    const measured =
        rect.width > 1 && rect.height > 1
            ? { w: Math.round(rect.width), h: Math.round(rect.height) }
            : (lastSize.get(formId) ?? fallback);

    host.style.width = `${Math.max(80, measured.w)}px`;
    host.style.height = `${Math.max(60, measured.h)}px`;
    const element = ensureParking();
    if (host.parentElement !== element) element.appendChild(host);
}

/**
 * 彻底释放宿主（窗体被永久移除时，例如将来的插件卸载）。
 *
 * 【不要从 `PanelMount` 的 effect 清理里调用】宿主元素被 `useMemo` 缓存着，
 * 销毁后缓存引用会指向已脱离文档的 div，面板就此隐形（`StrictMode` 的双调用
 * 必然触发）。关闭面板只需把宿主停泊起来 —— 见 `parkPanelHost`。
 */
export function releasePanelHost(formId: string): void {
    const host = hosts.get(formId);
    if (!host) return;
    host.remove();
    hosts.delete(formId);
    lastSize.delete(formId);
    owners.delete(formId);
    notify();
}

/** 仅测试用。 */
export function resetPanelHostsForTests(): void {
    for (const host of hosts.values()) host.remove();
    hosts.clear();
    lastSize.clear();
    owners.clear();
    listeners.clear();
    parking?.remove();
    parking = null;
    version = 0;
}
