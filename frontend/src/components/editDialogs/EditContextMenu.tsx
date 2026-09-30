import { useI18n } from "../../i18n/I18nProvider";
import { AppContextMenu, useMenuShortcut, type AppMenuItemSpec } from "../../ui";

interface EditContextMenuProps {
    x: number;
    y: number;
    isPitchParam: boolean;
    /**
     * 当前参数是否为「音量」：为 true 时显示"转换为动态"。
     * 音量与动态同为 0..4 的乘性增益，搬迁是纯拷贝 + 源参数归一化。
     */
    isVolumeParam?: boolean;
    /** 当前参数是否为「动态」：为 true 时显示"转换为音量"。 */
    isDynParam?: boolean;
    onClose: () => void;
    onCopy?: () => void;
    onCut?: () => void;
    onPaste?: () => void;
    onSelectAll?: () => void;
    onDeselect?: () => void;
    onInitialize?: () => void;
    onTransposeCents?: () => void;
    onTransposeDegrees?: () => void;
    onSetPitch?: () => void;
    onAverage?: () => void;
    onSmooth?: () => void;
    /**
     * 打开「添加颤音」弹窗（`Ctrl+B`）：在弹窗里选预设、调旋钮、看套用预览。
     *
     * 【为什么菜单里不再直接铺预设列表】预设的选择与微调是弹窗的事 —— 那里有
     * 波形缩略图与套用预览，菜单行给不了。菜单只保留这一个入口。
     */
    onAddVibrato?: () => void;
    onQuantize?: () => void;
    onMeanQuantize?: () => void;
    onSaveAsPitchRef?: () => void;
    onExportMidi?: () => void;
    /** 音量 → 动态（源参数归位到 1.0）。 */
    onConvertVolumeToDyn?: () => void;
    /** 动态 → 音量（源参数归位到「沿用原声」）。 */
    onConvertDynToVolume?: () => void;
    /** 从选区提取预设（作用于当前选区，与预设选择无关，故留在菜单）。 */
    onExtractVibratoPreset?: () => void;
}

export function EditContextMenu({
    x,
    y,
    isPitchParam,
    isVolumeParam = false,
    isDynParam = false,
    onClose,
    onCopy,
    onCut,
    onPaste,
    onSelectAll,
    onDeselect,
    onInitialize,
    onTransposeCents,
    onTransposeDegrees,
    onSetPitch,
    onAverage,
    onSmooth,
    onAddVibrato,
    onQuantize,
    onMeanQuantize,
    onSaveAsPitchRef,
    onExportMidi,
    onConvertVolumeToDyn,
    onConvertDynToVolume,
    onExtractVibratoPreset,
}: EditContextMenuProps) {
    const { tf } = useI18n();

    // 菜单项右侧的快捷键提示：从快捷键注册表读取当前生效的绑定。
    // 参数编辑器与时间轴共用 Ctrl+C/X/V（复制/剪切/粘贴按「活动编辑
    // 表面」定向派发，见 focusRouting.resolveEditOpRoute）。
    const copyShortcut = useMenuShortcut("clip.copy");
    const cutShortcut = useMenuShortcut("clip.cut");
    const pasteShortcut = useMenuShortcut("clip.paste");
    const selectAllShortcut = useMenuShortcut("edit.selectAll");
    const deselectShortcut = useMenuShortcut("edit.deselect");
    const initializeShortcut = useMenuShortcut("edit.initialize");
    const transposeCentsShortcut = useMenuShortcut("edit.transposeCents");
    const transposeDegreesShortcut = useMenuShortcut("edit.transposeDegrees");
    const setPitchShortcut = useMenuShortcut("edit.setPitch");
    const averageShortcut = useMenuShortcut("edit.average");
    const smoothShortcut = useMenuShortcut("edit.smooth");
    const addVibratoShortcut = useMenuShortcut("edit.addVibrato");
    const quantizeShortcut = useMenuShortcut("edit.quantize");
    const meanQuantizeShortcut = useMenuShortcut("edit.meanQuantize");

    // 菜单项映射到共享原语的 AppMenuItemSpec：分组分隔线由每段首项的
    // `separatorBefore` 表达；`onSelect` 只调用业务动作，关闭由原语负责
    //（原语在 onSelect 之后自行调用 onClose）。
    const items: AppMenuItemSpec[] = [
        {
            key: "copy",
            label: tf("menu_copy"),
            shortcut: copyShortcut,
            onSelect: () => onCopy?.(),
        },
        {
            key: "cut",
            label: tf("menu_cut"),
            shortcut: cutShortcut,
            onSelect: () => onCut?.(),
        },
        {
            key: "paste",
            label: tf("menu_paste"),
            shortcut: pasteShortcut,
            onSelect: () => onPaste?.(),
        },
        {
            key: "selectAll",
            label: tf("menu_select_all"),
            shortcut: selectAllShortcut,
            separatorBefore: true,
            onSelect: () => onSelectAll?.(),
        },
        {
            key: "deselect",
            label: tf("menu_deselect"),
            shortcut: deselectShortcut,
            onSelect: () => onDeselect?.(),
        },
        {
            key: "initialize",
            label: tf("menu_initialize"),
            shortcut: initializeShortcut,
            separatorBefore: true,
            onSelect: () => onInitialize?.(),
        },
        ...(isPitchParam
            ? ([
                  {
                      key: "transposeCents",
                      label: tf("menu_transpose_cents"),
                      shortcut: transposeCentsShortcut,
                      separatorBefore: true,
                      onSelect: () => onTransposeCents?.(),
                  },
                  {
                      key: "transposeDegrees",
                      label: tf("menu_transpose_degrees"),
                      shortcut: transposeDegreesShortcut,
                      onSelect: () => onTransposeDegrees?.(),
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        {
            key: "setPitch",
            label: isPitchParam ? tf("menu_set_pitch") : tf("menu_set_value"),
            shortcut: setPitchShortcut,
            onSelect: () => onSetPitch?.(),
        },
        {
            key: "average",
            label: tf("menu_average"),
            shortcut: averageShortcut,
            separatorBefore: true,
            onSelect: () => onAverage?.(),
        },
        {
            key: "smooth",
            label: tf("menu_smooth"),
            shortcut: smoothShortcut,
            onSelect: () => onSmooth?.(),
        },
        // 颤音：唯一入口是弹窗 —— 选预设、微调、看套用预览都在那里。
        {
            key: "addVibrato",
            label: tf("menu_add_vibrato"),
            shortcut: addVibratoShortcut,
            onSelect: () => onAddVibrato?.(),
        },
        ...(onExtractVibratoPreset
            ? ([
                  {
                      key: "extractVibratoPreset",
                      label: tf("vibrato_extract_action"),
                      onSelect: onExtractVibratoPreset,
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        {
            key: "quantize",
            label: tf("menu_quantize"),
            shortcut: quantizeShortcut,
            separatorBefore: true,
            onSelect: () => onQuantize?.(),
        },
        {
            key: "meanQuantize",
            label: tf("menu_mean_quantize"),
            shortcut: meanQuantizeShortcut,
            onSelect: () => onMeanQuantize?.(),
        },
        // 音量 ↔ 动态 互转：仅在当前参数是其一、且回调可用时显示。
        // 换算（基线补偿 + 源参数归位）在后端 convert_mix_param 内完成，
        // 前端只传选区 —— 转换是响度等效的，不是简单复制。
        ...(isVolumeParam && onConvertVolumeToDyn
            ? ([
                  {
                      key: "convertVolumeToDyn",
                      label: tf("menu_convert_volume_to_dyn"),
                      separatorBefore: true,
                      onSelect: onConvertVolumeToDyn,
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        ...(isDynParam && onConvertDynToVolume
            ? ([
                  {
                      key: "convertDynToVolume",
                      label: tf("menu_convert_dyn_to_volume"),
                      separatorBefore: true,
                      onSelect: onConvertDynToVolume,
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        ...(isPitchParam && onSaveAsPitchRef
            ? ([
                  {
                      key: "saveAsPitchRef",
                      label: tf("menu_save_as_pitch_ref"),
                      separatorBefore: true,
                      onSelect: onSaveAsPitchRef,
                  },
                  ...(onExportMidi
                      ? [
                            {
                                key: "exportMidi",
                                label: tf("menu_export_midi"),
                                onSelect: onExportMidi,
                            } satisfies AppMenuItemSpec,
                        ]
                      : []),
              ] satisfies AppMenuItemSpec[])
            : []),
    ];

    return <AppContextMenu x={x} y={y} items={items} onClose={onClose} />;
}
