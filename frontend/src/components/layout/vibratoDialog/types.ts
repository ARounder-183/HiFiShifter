// VibratoDialog 的公开类型与布局常量。
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";

export interface VibratoApplyTarget {
    /**
     * 取选区（无选区时取曲线头部）的原始帧值。
     *
     * 由宿主提供：只有面板手里有 `rootTrackId` / `editParam` / 选区。返回 `null`
     * 表示取不到数据（例如刚打开还没有参数曲线）。
     */
    loadOriginal: () => Promise<{ values: number[]; framePeriodMs: number } | null>;
    /** 应用：把（可能已微调的）完整预设交给编辑管线。 */
    onApply: (preset: VibratoPreset) => void;
    /**
     * 从选区提取颤音预设。
     *
     * 【为什么由宿主做】提取要读参数帧（选区 + 轨道 + 参数），只有宿主够得着。成功
     * 返回**已入库**的那条预设（本窗口随即选中它），失败返回 `null`（本窗口给出
     * 行内提示）。入库动作留在宿主，是因为它还要顺带更新"当前使用"。
     */
    onExtract?: () => Promise<VibratoPreset | null>;
}

export interface VibratoDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 预设编辑时的"当前参数"：深度按它换算成原生单位显示与编辑。 */
    editParam?: string;
    paramRange?: { min: number; max: number };
    /**
     * 打开时**编辑**哪个预设；省略则用当前活动预设。
     *
     * 从选区提取出来的那条预设走这里：用户刚把它提出来，当然是要接着编辑它。
     */
    initialPresetId?: string;
    /**
     * 选区宿主；省略 = 纯预设库。
     *
     * 【这一个 prop 决定窗口的两副面孔】给了它才有「套用到选区」页签、「从选区提取」
     * 与「应用」—— 也就是「添加颤音」那一次打开的全部专属能力；不给就是"改库"那一面
     * （工具栏 / 菜单栏的「管理预设…」），与合并前的预设管理器一致。
     *
     * 打开时落在哪个预览页签也由它决定：有宿主就落在「套用到选区」（那次打开的目的
     * 就是"看套上去什么样、然后应用"），没有就落在「预设波形」。
     */
    applyTarget?: VibratoApplyTarget;
}

/** 列表列的宽度（CSS 像素）。 */

/** 列表列的宽度（CSS 像素）。 */
export const LIST_WIDTH = 208;

/** 预设列表的两个分组。 */
export type PresetGroup = "system" | "user";

/** 对话框内容的定高。 */
export const CONTENT_HEIGHT = "min(60vh, 560px)";
