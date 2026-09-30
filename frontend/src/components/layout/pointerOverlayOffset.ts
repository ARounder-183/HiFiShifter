/**
 * 指针跟随浮层的整像素定位。
 *
 * 【为什么必须取整】这类浮层的 `left` / `top` 直接取自指针坐标，而
 * `clientX` / `clientY` 在 **HiDPI 屏（`devicePixelRatio > 1`）上是小数** ——
 * 鼠标每动一帧，浮层就落在新的**次像素相位**上，浏览器只能把里面的文字
 * 重新栅格化一遍，字形边缘的抗锯齿随之逐帧变化。用户看到的现象正是"气泡跟着
 * 指针走，可里面的文字在抖"，而且**字越多的一行抖得越明显**（同一次相位变化下
 * 参与重栅格化的字形更多）。
 *
 * 取整之后，整段手势里浮层与文字的相位是**恒定**的：`translate(0, -100%)`
 * 带来的那个小数偏移（浮层高度是小数）不随指针变化，栅格化结果因此可以复用，
 * 文字不再抖。
 *
 * 【为什么在浮层侧取整，而不是在指针侧】指针坐标还要用于命中测试与值换算，
 * 那里的小数是有意义的；只有"画在哪里"需要落到整像素上。
 *
 * 【为什么放在浮层这一层而不是调用方】两个读数气泡（曲线浮窗、纵轴浮窗）共用
 * 同一条规则；写成一处、测一处，才不会出现"改了一个、另一个又开始抖"。
 */
export function pointerOverlayOffset(
    anchor: { clientX: number; clientY: number },
    rect: { left: number; top: number },
): { left: number; top: number } {
    return {
        left: Math.round(anchor.clientX - rect.left),
        top: Math.round(anchor.clientY - rect.top),
    };
}
