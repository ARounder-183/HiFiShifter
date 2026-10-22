/**
 * 边缘平滑度：标签 + 滑块 + 百分比读数。
 *
 * 三者是一组**按工具栏层级依次让位**的控件（见各 `ToolbarSlot` 的 `hideAt`）：
 * 全称（5）→ 百分比读数（6）→ 滑块（7）。顺序是按"信息价值"排的：名字可以省，
 * 读数不能省，控件留到最后。
 *
 * 【为什么单独成文件】这组控件里塞着工具栏最密集的一段布局知识：
 * - 滑块 `flex-basis` 必须是 0（不是 `auto`），否则收缩量按基准比例分摊，
 *   窄容器下它只缩到 102px，仍占着大头；基准为 0 时它只取"别人用剩下的"；
 * - 下限 16px 是**实际能到的最小宽度**（0 × shrink 恒为 0），再窄就只剩滑块、
 *   看不到轨道；
 * - 百分比读数标了 `data-toolbar-volatile`：它拖滑块时每帧都变，布局签名对它
 *   只记长度，否则每帧触发一次测量。
 *
 * 这些约束混在工具栏行内布局里时，任何一处改动都可能把"滑块无法缩小"的老缺陷
 * 重新引回来（见滑块上的注释）。
 */
import { Flex } from "@radix-ui/themes";
import {
    persistUiSettings,
    setEdgeSmoothnessPercent,
} from "../../../../features/session/sessionSlice";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ToolbarSlot } from "./toolbarParts";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        "attachEdgeSmoothnessWheel" | "dispatch" | "s" | "tf" | "toolbarTier"
    >;
};

export function SmoothingSlider({ m }: Props) {
    const { attachEdgeSmoothnessWheel, dispatch, s, tf, toolbarTier } = m;

    return (
        <Flex align="center" gap="1" ml="2" style={{ minWidth: 0, flexShrink: 1 }}>
            {/* 标签允许被压缩裁切（完整名称在悬停提示里）：横向极窄时
                            应当由它先让位，而不是把整行撑到溢出。省略号让"让位"
                            看起来是有意的降级，而不是渲染出错的半截字。
                            第 5 级隐藏：滑块与百分比数值仍在，全称在 ToolTip 里。 */}
            <ToolbarSlot hideAt={5} tier={toolbarTier}>
                <span
                    className="hs-type-label"
                    data-tooltip={tf("edge_smoothness")}
                    style={{
                        minWidth: 0,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                    }}
                >
                    {tf("edge_smoothness_short")}:
                </span>{" "}
            </ToolbarSlot>
            {/* 第 7 级才隐藏滑块：它是**控件**，比只读的百分比数值（第 6 级）
                            更该留到最后。悬停时用 ToolTip 报出当前百分比 —— 数值被隐藏
                            （第 6 级）之后仍能读到。 */}
            <ToolbarSlot hideAt={7} tier={toolbarTier}>
                <input
                    ref={attachEdgeSmoothnessWheel}
                    className="qt-range"
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={Math.round(s.edgeSmoothnessPercent)}
                    data-tooltip={`${tf("edge_smoothness")}: ${Math.round(
                        s.edgeSmoothnessPercent,
                    )}%`}
                    onChange={(e) => {
                        const next = Number(e.currentTarget.value);
                        dispatch(setEdgeSmoothnessPercent(next));
                    }}
                    onPointerUp={() => {
                        void dispatch(persistUiSettings());
                    }}
                    onKeyUp={() => {
                        void dispatch(persistUiSettings());
                    }}
                    style={{
                        // 【flex-basis 必须取 0，而不是 `auto`（= width: 120）】
                        // 基准为 120 时，横向不足的收缩量会**按基准比例分摊**给
                        // 同排所有项 —— 实测窄容器下它只缩到 102px，仍占着大头。
                        // 基准为 0 时，它先让出空间、只取"别人用剩下的"，宽裕时再由
                        // `maxWidth` 封顶在 120px。它是低频调节项，理应最先让步。
                        //
                        // 【下限取 16】基准为 0 的项在收缩阶段分摊到的是 0（0 × shrink
                        // 恒为 0），因此**它的下限就是它实际能到的最小宽度**：下限多大，
                        // 拥挤时它就让出多少。16 = 12px 滑块 + 两侧各 2px，是仍能拖动的
                        // 最小值；再窄就只剩滑块本身、看不到轨道了。极窄时它本就该让位
                        // （数值仍可读，滚轮调值照常可用）。
                        flex: "1 1 0",
                        // 非 flex 上下文（理论上不会发生）时的兜底宽度。
                        width: 120,
                        minWidth: 16,
                        maxWidth: 120,
                    }}
                />{" "}
            </ToolbarSlot>
            {/* 数值需要完整可读（"100%"），因此给它一个较小的固定下限，
                            但不再是 36px 那种"宁可溢出也不缩"的宽度。
                            第 6 级隐藏：它只是**读数**，比滑块（第 7 级）先让位。 */}
            <ToolbarSlot hideAt={6} tier={toolbarTier}>
                {/* data-toolbar-volatile：这个读数在拖滑块时每帧都变，
                                布局签名对它只记长度，否则每一帧都会触发一次测量。 */}
                <span
                    className="hs-type-label"
                    data-toolbar-volatile
                    style={{ minWidth: 28, textAlign: "right" }}
                >
                    {Math.round(s.edgeSmoothnessPercent)}%
                </span>
            </ToolbarSlot>
        </Flex>
    );
}
