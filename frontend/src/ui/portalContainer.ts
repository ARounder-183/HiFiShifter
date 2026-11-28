/**
 * 弹出面挂载容器。
 *
 * Radix Themes 的 --accent-9 / --gray-a5 / --space-* 只定义在 .radix-themes 上。
 * 挂到 document.body 的弹出面落在那个元素之外，所有 var(--accent-*) 都会整条失效，
 * 滑块头、实心按钮等背景回落成 transparent。挂进 .radix-themes 后既能拿到完整令牌，
 * 又因为弹出面是 position: fixed，不受祖先 overflow: hidden 裁切。
 *
 * 不存在 .radix-themes（极端早期渲染 / 无主题单测）时退回 body。
 */
export function menuPortalContainer(): HTMLElement {
    return document.querySelector<HTMLElement>(".radix-themes") ?? document.body;
}
