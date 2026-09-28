export const AXIS_W = 56;

export const PITCH_MIN_MIDI = 36; // C2
export const PITCH_MAX_MIDI = 96; // C8

/**
 * 参数编辑器底部为**自绘水平滚动条**预留的行高（CSS px）。
 *
 * 【为什么必须是显式像素值】滚动条要"独占一行"，就得让滚动容器在底边收边
 * （`bottom: PARAM_EDITOR_BOTTOM_BAR_PX`），而**纵轴列**仍占满整列 —— 于是纵轴
 * 列比滚动视口高出正好这一行。纵轴画布把这一行当作刻度文字的落笔空间
 * （见内核的 `AXIS_TICK_LABEL_DESCENT_PX`），因此两者是同一个量：
 *
 * - 写死像素而不是用 Tailwind 的 `bottom-2`：后者的实际高度取决于根字号
 *   （`0.5rem`，默认根字号 16px 才是 8px）。根字号一旦变化，预留行就会小于
 *   刻度文字需要的下探量，最下方刻度值又会被裁掉 —— 这正是本常量要消除的耦合。
 * - 变更本值时须复核 `AXIS_TICK_LABEL_DESCENT_PX` 是否仍 ≤ 本值（有单测钉住）。
 */
export const PARAM_EDITOR_BOTTOM_BAR_PX = 8;
