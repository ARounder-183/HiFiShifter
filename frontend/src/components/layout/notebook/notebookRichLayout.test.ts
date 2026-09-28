/*
 * 富文本视图"可点击性"的标记契约。
 *
 * 【问题】点正文下方那片空白曾经既不落光标也不能输入 —— 因为可编辑元素只有
 * 内容那么高，那片空白属于滚动容器而不是编辑器。记事本为空时最刺眼：下方
 * 一大片空间全是死的。
 *
 * 【修法】可编辑区撑满面板，留白搬进可编辑区内部。链条见 `notebook.css`
 * 的 `.hs-notebook-rich` 注释：滚动容器（定高）→ EditorContent 包装层
 * （`min-height:100%` + flex 列）→ 可编辑元素（`flex:1 1 auto` + `padding`）。
 *
 * 【真实行为已在浏览器里实测过（临时探针页面，验完即删）】
 * - 改动前：点正文下方空白，事件目标是滚动容器 div，编辑器不聚焦、
 *   `posAtCoords` 返回 null，打字不进文档；空文档下同样无效。
 * - 改动后：事件目标是 ProseMirror 本身，编辑器聚焦、光标落到文档末尾
 *   （有内容时 pos=30，空文档 pos=2），打字直接进正文；可编辑区高度从
 *   66px（内容高）变成 418px（撑满 420px 的面板），外观与内边距不变。
 *
 * 【本测试覆盖什么、不覆盖什么】浏览器行为进不了 vitest，这里锁定的是**面板
 * 侧的标记契约** —— 它是最容易被"顺手改回去"的两处（把内边距挪回滚动容器、
 * 或漏掉包装层上的类名）。另一半是 `notebook.css` 里那两条规则；CSS 在本仓库
 * 的 vitest 里读不到（`?raw` / `?inline` / raw glob 都返回空串，node 类型也没
 * 进 app 的 tsconfig），因此没有自动守卫 —— 改动那两条规则时请连同上面的探针
 * 结论一起看。
 */

import { expect, test } from "vitest";

import notebookPanelSource from "./NotebookPanel.tsx?raw";

test("富文本视图：可编辑区由包装层类名撑满，留白不在容器上", () => {
    // 1. 包装层必须拿到版式类 —— 没有它，.hs-notebook-rich 的规则不生效，
    //    可编辑区又会退回"只有内容那么高"，正文下方重新变成点不动的死区。
    expect(notebookPanelSource).toContain('className="hs-notebook-rich"');

    // 2. 留白必须待在可编辑区**内部**：滚动容器若再出现内边距，面板边缘就会
    //    多出一圈点不动的死边（本次修复要消除的正是"点不动"）。
    expect(notebookPanelSource).not.toContain("px-3 py-3");
});
