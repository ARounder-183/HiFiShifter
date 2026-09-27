import { useI18n } from "../../i18n/I18nProvider";
import { useAppSelector } from "../../app/hooks";
import { selectKeybinding, formatKeybinding } from "../../features/keybindings/keybindingsSlice";
import type { ActionId } from "../../features/keybindings/types";
import { AppContextMenu, type AppMenuItemSpec } from "../../ui/Menu";

/**
 * 读取动作当前生效的快捷键文本（跟随用户在快捷键设置中的自定义绑定）。
 * 未绑定（None binding）时返回 undefined，菜单项不显示快捷键。
 */
function useMenuShortcut(actionId: ActionId): string | undefined {
    const kb = useAppSelector((state) => selectKeybinding(state, actionId));
    return formatKeybinding(kb, "") || undefined;
}

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
    onAddVibrato?: () => void;
    onQuantize?: () => void;
    onMeanQuantize?: () => void;
    onSaveAsPitchRef?: () => void;
    onExportMidi?: () => void;
    /** 音量 → 动态（源参数归位到 1.0）。 */
    onConvertVolumeToDyn?: () => void;
    /** 动态 → 音量（源参数归位到「沿用原声」）。 */
    onConvertDynToVolume?: () => void;
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
}: EditContextMenuProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

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
            label: tAny("menu_copy"),
            shortcut: copyShortcut,
            onSelect: () => onCopy?.(),
        },
        {
            key: "cut",
            label: tAny("menu_cut"),
            shortcut: cutShortcut,
            onSelect: () => onCut?.(),
        },
        {
            key: "paste",
            label: tAny("menu_paste"),
            shortcut: pasteShortcut,
            onSelect: () => onPaste?.(),
        },
        {
            key: "selectAll",
            label: tAny("menu_select_all"),
            shortcut: selectAllShortcut,
            separatorBefore: true,
            onSelect: () => onSelectAll?.(),
        },
        {
            key: "deselect",
            label: tAny("menu_deselect"),
            shortcut: deselectShortcut,
            onSelect: () => onDeselect?.(),
        },
        {
            key: "initialize",
            label: tAny("menu_initialize"),
            shortcut: initializeShortcut,
            separatorBefore: true,
            onSelect: () => onInitialize?.(),
        },
        ...(isPitchParam
            ? ([
                  {
                      key: "transposeCents",
                      label: tAny("menu_transpose_cents"),
                      shortcut: transposeCentsShortcut,
                      separatorBefore: true,
                      onSelect: () => onTransposeCents?.(),
                  },
                  {
                      key: "transposeDegrees",
                      label: tAny("menu_transpose_degrees"),
                      shortcut: transposeDegreesShortcut,
                      onSelect: () => onTransposeDegrees?.(),
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        {
            key: "setPitch",
            label: isPitchParam ? tAny("menu_set_pitch") : tAny("menu_set_value"),
            shortcut: setPitchShortcut,
            onSelect: () => onSetPitch?.(),
        },
        {
            key: "average",
            label: tAny("menu_average"),
            shortcut: averageShortcut,
            separatorBefore: true,
            onSelect: () => onAverage?.(),
        },
        {
            key: "smooth",
            label: tAny("menu_smooth"),
            shortcut: smoothShortcut,
            onSelect: () => onSmooth?.(),
        },
        {
            key: "addVibrato",
            label: tAny("menu_add_vibrato"),
            shortcut: addVibratoShortcut,
            onSelect: () => onAddVibrato?.(),
        },
        {
            key: "quantize",
            label: tAny("menu_quantize"),
            shortcut: quantizeShortcut,
            onSelect: () => onQuantize?.(),
        },
        {
            key: "meanQuantize",
            label: tAny("menu_mean_quantize"),
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
                      label: tAny("menu_convert_volume_to_dyn"),
                      separatorBefore: true,
                      onSelect: onConvertVolumeToDyn,
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        ...(isDynParam && onConvertDynToVolume
            ? ([
                  {
                      key: "convertDynToVolume",
                      label: tAny("menu_convert_dyn_to_volume"),
                      separatorBefore: true,
                      onSelect: onConvertDynToVolume,
                  },
              ] satisfies AppMenuItemSpec[])
            : []),
        ...(isPitchParam && onSaveAsPitchRef
            ? ([
                  {
                      key: "saveAsPitchRef",
                      label: tAny("menu_save_as_pitch_ref"),
                      separatorBefore: true,
                      onSelect: onSaveAsPitchRef,
                  },
                  ...(onExportMidi
                      ? [
                            {
                                key: "exportMidi",
                                label: tAny("menu_export_midi"),
                                onSelect: onExportMidi,
                            } satisfies AppMenuItemSpec,
                        ]
                      : []),
              ] satisfies AppMenuItemSpec[])
            : []),
    ];

    return <AppContextMenu x={x} y={y} items={items} onClose={onClose} />;
}
