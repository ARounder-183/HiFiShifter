/**
 * 指针跟随浮层的定位样式。
 *
 * 【它解决什么】这类浮层的坐标直接来自指针，而 `clientX` / `clientY` 在
 * **`devicePixelRatio > 1` 的屏幕上带小数**（HiDPI / 系统缩放 125% / 150% 时，
 * 鼠标每一帧都可能落在 0.4 这种位置）。若把小数坐标交给 `left` / `top`，浮层每帧
 * 都换一个**次像素相位**，浏览器只能把里面的文字重新栅格化一遍，字形抗锯齿逐帧
 * 变化 —— 用户看到的就是"气泡跟着指针走，可里面的文字在抖"，且**字最多的一行
 * 抖得最明显**。
 *
 * 【为什么走合成层 transform 而不是 `left` / `top`】把位移放进 `transform` 并声明
 * `will-change`，浮层就成为独立合成层：文字只栅格化一次，之后由合成器**平移这份
 * 栅格**，不再逐帧重栅格化 —— 抖动消失。合成器还会把层对齐到**设备像素**，这正
 * 是"文字保持清晰"所需的最细粒度。
 *
 * 【为什么不在这里 `Math.round` 到整 CSS 像素】曾经这么修过，结果是**更糟**：
 * 取整把连续位移量化成 1 CSS 像素的台阶，在 HiDPI 上比设备像素粗（125% 缩放时
 * 1 CSS 像素 = 1.25 设备像素），浮层里的静态行开始可见地一跳一跳。文字清晰要求
 * 落在整**设备**像素上，而不是整 CSS 像素上；该对齐交给合成器，别用 JS 粗化。
 *
 * 【为什么不分别取整到设备像素】那需要读 `devicePixelRatio` 并把小数残差写回样式，
 * 而合成器本来就在做同一件事（且做得更准，因为它知道真正的缩放与吸附规则）。
 */
export function pointerOverlayStyle(
    anchor: { clientX: number; clientY: number },
    rect: { left: number; top: number },
): {
    left: number;
    top: number;
    transform: string;
    willChange: "transform";
} {
    const x = Number.isFinite(anchor.clientX - rect.left) ? anchor.clientX - rect.left : 0;
    const y = Number.isFinite(anchor.clientY - rect.top) ? anchor.clientY - rect.top : 0;
    return {
        // 布局位置归零，位移全部由 transform 承担 —— 这样它才是合成层的属性，
        // 而不是每帧触发重排 / 重绘的 `left` / `top`。
        left: 0,
        top: 0,
        // `translateY(-100%)` 把浮层的**底边**对齐到指针（百分比相对自身高度）。
        transform: `translate3d(${x}px, ${y}px, 0) translateY(-100%)`,
        willChange: "transform",
    };
}
