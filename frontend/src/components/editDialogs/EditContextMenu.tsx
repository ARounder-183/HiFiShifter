import { useI18n } from "../../i18n/I18nProvider";
import { AppContextMenu, AppSubMenu, useMenuShortcut, type AppMenuItemSpec } from "../../ui";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";
import { vibratoPresetLabel } from "../layout/vibrato/vibratoDialogLogic";

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
    /** 套用当前活动的颤音预设（`Ctrl+B`）。 */
    onAddVibrato?: () => void;
    onQuantize?: () => void;
    onMeanQuantize?: () => void;
    onSaveAsPitchRef?: () => void;
    onExportMidi?: () => void;
    /** 音量 → 动态（源参数归位到 1.0）。 */
    onConvertVolumeToDyn?: () => void;
    /** 动态 → 音量（源参数归位到「沿用原声」）。 */
    onConvertDynToVolume?: () => void;
    /**
     * 可用颤音预设（系统 + 用户），用于在菜单里直接切换。
     *
     * 【为什么在菜单里铺开而不是开对话框】预设的用途就是"一键套用"。
     * 打开对话框再选一次，等于把两步的操作变成四步。
     */
    vibratoPresets?: readonly VibratoPreset[];
    /** 当前活动的颤音预设 id（列表里打勾）。 */
    activeVibratoPresetId?: string;
    /** 选中某条预设：设为活动预设并立即套用到选区。 */
    onSelectVibratoPreset?: (presetId: string) => void;
    /** 打开预设管理器。 */
    onManageVibratoPresets?: () => void;
    /** 从选区提取预设（选区里已有一段颤音时可用）。 */
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
    vibratoPresets,
    activeVibratoPresetId,
    onSelectVibratoPreset,
    onManageVibratoPresets,
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

    // 预设列表存在时才显示这一整组；否则「添加颤音」保持为唯一的入口。
    /**
     * 颤音预设二级菜单。
     *
     * 【为什么不平铺】系统预设已有 12 个，加上用户预设会把菜单撑得比屏幕还高，
     * 而"添加颤音"本身只是一项动作 —— 十几行预设把它挤到需要滚动才能看见。
     * 收进子菜单后主菜单长度与预设数量无关。
     */
    const vibratoSubmenu =
        onSelectVibratoPreset && vibratoPresets && vibratoPresets.length > 0 ? (
            <AppSubMenu label={tf("vibrato_menu_presets")}>
                {vibratoPresets.map((preset) => {
                    const active = preset.id === activeVibratoPresetId;
                    return (
                        <button
                            key={preset.id}
                            type="button"
                            role="menuitemradio"
                            aria-checked={active}
                            className="hs-type-body flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left transition-colors hover:bg-qt-hover"
                            style={{
                                paddingLeft: "var(--qt-space-5)",
                                paddingRight: "var(--qt-space-5)",
                            }}
                            onPointerDown={(event) => event.stopPropagation()}
                            onClick={(event) => {
                                event.stopPropagation();
                                onSelectVibratoPreset(preset.id);
                                onClose();
                            }}
                        >
                            <span className="truncate">{vibratoPresetLabel(preset, tf)}</span>
                            {active ? <span aria-hidden>✓</span> : null}
                        </button>
                    );
                })}
                {(onExtractVibratoPreset || onManageVibratoPresets) && (
                    <div
                        className="my-1 border-t border-qt-border"
                        style={{
                            marginLeft: "var(--qt-space-5)",
                            marginRight: "var(--qt-space-5)",
                        }}
                    />
                )}
                {onExtractVibratoPreset ? (
                    <button
                        type="button"
                        role="menuitem"
                        className="hs-type-body flex w-full items-center px-3 py-1.5 text-left transition-colors hover:bg-qt-hover"
                        style={{
                            paddingLeft: "var(--qt-space-5)",
                            paddingRight: "var(--qt-space-5)",
                        }}
                        onPointerDown={(event) => event.stopPropagation()}
                        onClick={(event) => {
                            event.stopPropagation();
                            onExtractVibratoPreset();
                            onClose();
                        }}
                    >
                        <span className="truncate">{tf("vibrato_extract_action")}</span>
                    </button>
                ) : null}
                {onManageVibratoPresets ? (
                    <button
                        type="button"
                        role="menuitem"
                        className="hs-type-body flex w-full items-center px-3 py-1.5 text-left transition-colors hover:bg-qt-hover"
                        style={{
                            paddingLeft: "var(--qt-space-5)",
                            paddingRight: "var(--qt-space-5)",
                        }}
                        onPointerDown={(event) => event.stopPropagation()}
                        onClick={(event) => {
                            event.stopPropagation();
                            onManageVibratoPresets();
                            onClose();
                        }}
                    >
                        <span className="truncate">{tf("vibrato_manager_open")}</span>
                    </button>
                ) : null}
            </AppSubMenu>
        ) : null;

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
        // 颤音：一键套用当前预设。预设列表、提取与管理都在二级菜单里 ——
        // 十几行预设平铺会把这一项挤出视野（见上方 `vibratoSubmenu`）。
        {
            key: "addVibrato",
            label: tf("menu_add_vibrato"),
            shortcut: addVibratoShortcut,
            onSelect: () => onAddVibrato?.(),
        },
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

    return (
        <AppContextMenu
            x={x}
            y={y}
            items={items}
            // 二级子菜单：`items` 是纯数据，装不下需要真实 React 节点的内容
            //（勾选行 + 分组线 + 两个动作）。见 `AppContextMenu.extraItems` 的说明。
            extraItems={vibratoSubmenu}
            onClose={onClose}
        />
    );
}
