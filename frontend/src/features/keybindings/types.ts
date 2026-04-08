/**
 * 快捷键管理系统 — 类型定义
 */

/** 所有可绑定操作的 ID */
export type ActionId =
    // 播放控制
    | "playback.toggle" // 播放/暂停
    | "playback.stop" // 停止播放
    | "playback.focusCursor" // 聚焦播放光标
    | "playback.seekLeft" // 播放光标左移
    | "playback.seekRight" // 播放光标右移
    | "timeline.zoomIn" // 焦点时间轴横向放大
    | "timeline.zoomOut" // 焦点时间轴横向缩小
    // 编辑
    | "edit.undo" // 撤销
    | "edit.redo" // 重做
    | "edit.selectAll" // 全选
    | "edit.deselect" // 取消选择
    | "edit.initialize" // 初始化
    | "edit.transposeCents" // 按指定音分移调
    | "edit.transposeDegrees" // 按指定度数移调
    | "edit.setPitch" // 音高设置
    | "edit.average" // 平均化
    | "edit.smooth" // 平滑化
    | "edit.addVibrato" // 添加颤音
    | "edit.quantize" // 量化
    | "edit.meanQuantize" // 均值量化
    | "edit.pasteReaper" // 粘贴 Reaper 剪贴板数据
    | "edit.pasteVocalShifter" // 粘贴 VocalShifter 剪贴板数据
    // 工程
    | "project.new" // 新建工程
    | "project.open" // 打开工程
    | "project.save" // 保存
    | "project.saveAs" // 另存为
    | "project.export" // 导出音频
    // 轨道
    | "track.add" // 新建轨道
    | "track.selectUp" // 选择上一条轨道
    | "track.selectDown" // 选择下一条轨道
    // Clip 操作
    | "clip.delete" // 删除选中 clip
    | "clip.copy" // 复制 clip
    | "clip.cut" // 剪切 clip
    | "clip.paste" // 粘贴 clip
    | "clip.split" // 分割 clip
    | "clip.normalize" // 规格化选中 clip
    // PianoRoll 操作
    | "pianoRoll.copy" // PianoRoll 内复制参数帧
    | "pianoRoll.paste" // PianoRoll 内粘贴参数帧
    | "pianoRoll.shiftParamUp" // 选中 clip 参数线整体上移
    | "pianoRoll.shiftParamDown" // 选中 clip 参数线整体下移
    | "pianoRoll.shiftParamUpSelection" // 参数编辑器选择范围内参数线上移
    | "pianoRoll.shiftParamDownSelection" // 参数编辑器选择范围内参数线下移
    // 模式切换
    | "mode.toggle" // 模式切换（正向）
    | "mode.selectTool" // 切换到选择工具
    | "mode.drawTool" // 切换到绘制工具
    | "mode.lineTool" // 切换到直线/颤音工具
    // 修饰键行为
    | "modifier.clipSlipEdit" // 拖动 clip 时进入 slip edit
    | "modifier.clipStretch" // clip 边缘拖动时从 trim 变为 stretch
    | "modifier.clipNoSnap" // clip 移动/trim/stretch 时切换吸附
    | "modifier.clipCopyDrag" // 拖动 clip 时进入复制模式
    | "modifier.horizontalZoom" // 按住+滚轮水平缩放
    | "modifier.pianoRollVerticalZoom" // PianoRoll Ctrl+滚轮垂直缩放
    | "modifier.scrollHorizontal" // 按住+滚轮水平滚动
    | "modifier.scrollVertical" // 按住+滚轮竖直滚动
    | "modifier.pianoKeysVerticalScroll" // 钢琴键垂直滚动（按住+滚轮）
    | "modifier.pianoKeysVerticalZoom" // 钢琴键垂直缩放（按住+滚轮）
    | "modifier.paramMorph" // 参数编辑器形变模式（按住）
    | "modifier.paramFineAdjust" // 参数微调（按住）
    | "modifier.vibratoAmplitudeAdjust" // 颤音绘制时滚轮调振幅
    | "modifier.vibratoFrequencyAdjust" // 颤音绘制时滚轮调频率
    // 快速搜索
    | "quickSearch.open" // 打开快速搜索弹窗
    | "quickSearch.navigate.up" // 快速搜索：向上切换候选项
    | "quickSearch.navigate.down" // 快速搜索：向下切换候选项
    | "quickSearch.preview" // 快速搜索：预览/试听
    | "quickSearch.confirm" // 快速搜索：确认放置
    | "quickSearch.close"; // 快速搜索：关闭弹窗

/** 单个快捷键绑定 */
export interface Keybinding {
    /** 主键名称（小写），如 "space", "s", "delete", "backspace" */
    key: string;
    /** 是否需要 Ctrl (Windows) / Cmd (Mac) */
    ctrl?: boolean;
    /** 是否需要 Shift */
    shift?: boolean;
    /** 是否需要 Alt */
    alt?: boolean;
    /** 仅作为修饰键使用（无主键），用于 modifier 类绑定 */
    modifierOnly?: boolean;
}

/** 操作元信息（用于 UI 显示） */
export interface ActionMeta {
    /** 国际化文本的 key（用于操作名称显示） */
    labelKey: string;
    /** 分组（用于设置面板分组展示） */
    group:
        | "playback"
        | "edit"
        | "project"
        | "clip"
        | "pianoRoll"
        | "mode"
        | "modifier"
        | "quickSearch";
    /**
     * 修饰键操作类型（仅用于修饰键冲突检测）。
     * 同类型的修饰键绑定才会提示冲突，不同类型不提示。
     */
    modifierOperationType?: "drag" | "wheel";
    /**
     * 作用域上下文（仅用于冲突检测）。
     * 具有不同 scopedContext 的绑定不会视为冲突，
     * 因为它们在不同的 UI 上下文中激活（如 quickSearch 弹窗中）。
     */
    scopedContext?: string;
}

/** 完整的快捷键映射：actionId → Keybinding */
export type KeybindingMap = Record<ActionId, Keybinding>;

/** 用户覆盖项（只存储与默认不同的部分） */
export type KeybindingOverrides = Partial<KeybindingMap>;
