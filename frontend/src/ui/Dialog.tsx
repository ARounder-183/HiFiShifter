/**
 * 对话框组合壳 —— 全应用 42 个对话框的**唯一**结构来源。
 *
 * 【现状与问题】审查时全仓有 42 处 `<Dialog.Root>` / `<Dialog.Content>`，全部
 * 手写，没有共享外壳。手写本身不是问题，**规格散落才是**：
 *
 *   - 宽度用了 14 个不同的字面量（340/360/380/400/420/440/460/480/520/560/
 *     620/720/760/860），且一半走 `maxWidth="620px"`、一半走
 *     `style={{ maxWidth: 460 }}`；
 *   - **Enter 从不确认**：全仓 `<form>` 与 `type="submit"` 数量为 0，
 *     因此 42 个对话框里约 38 个按 Enter 毫无反应 —— 桌面端最基础的
 *     默认按钮约定缺失，只有 5 处手工接了 Enter；
 *   - **Esc 与外部点击没有任何未保存护栏**：42 个里只有 1 个守卫了
 *     外部点击、1 个屏蔽了 Esc，其余 40 个转瞬即关；
 *   - **高度上限有 5 种做法**：Radix ScrollArea、内联 maxHeight+flex、
 *     Tailwind `max-h-[240px]`、`max-h-[50vh]`，以及约 30 个完全没做上限
 *     的（含 400 行的录音设置、8 个编辑对话框）—— 小窗口上内容溢出屏幕；
 *   - **关闭方式不统一**：32 处用 `<Dialog.Close>`，另约 22 处用
 *     `onClick={() => onOpenChange(false)}`；
 *   - `onKeyDown` 冒泡阻断 32/42 有、10 处漏（漏的 6 处都在 `App.tsx`，
 *     会导致对话框内按空格触发播放）。
 *
 * 本壳把这 9 件事各做一次，并且**新对话框默认就是对的** —— 作者不需要
 * 知道 Enter 要接 form、Esc 要护栏、快捷键要抑制。
 *
 * 【对扩展 API 的意义】第三方面板开对话框时，直接得到与内置一致的
 * 交互协议（Enter/Esc/焦点/快捷键抑制），不必重新发明。
 */
import { Dialog } from "@radix-ui/themes";
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { AppButton, type AppButtonIntent } from "./Button";
import { cx } from "./cx";
import { acquireShortcutSuppression, releaseShortcutSuppression } from "./shortcutScope";

/**
 * 对话框宽度档位。四档取代历史上的 14 个字面量。
 *
 * 映射依据：340–440（小确认/单行输入）→ sm；460–560（常规设置）→ md；
 * 620–720（多列表单）→ lg；760–860（含表格/预览的大对话框）→ xl。
 */
export type AppDialogSize = "sm" | "md" | "lg" | "xl";

const SIZE_PX: Record<AppDialogSize, number> = {
    sm: 400,
    md: 520,
    lg: 640,
    xl: 800,
};

export interface AppDialogAction {
    id: string;
    label: ReactNode;
    /**
     * 按钮语义。`primary` 在一张对话框里应当只有一个。
     * 省略时按位置推断：最后一个 `end` 动作视为 primary。
     */
    intent?: AppButtonIntent;
    /**
     * 停靠在页脚左侧还是右侧。
     *
     * 右侧是默认（`[取消] [确认]`）；左侧专供**破坏性/低频动作**
     * （如「删除」「清理附件」），使它们远离主动作，避免误点。
     * 历史上 `CustomScaleDialog` / `TempoMapRulerRow` / `NotebookDialogs`
     * 各有各的左侧按钮做法，这里统一。
     */
    align?: "start" | "end";
    disabled?: boolean;
    /**
     * 点击处理。返回 Promise 时自动进入 pending 态：按钮显示加载中、
     * 全部按钮禁用、并且**不自动关闭**（成功由调用方决定何时关）。
     */
    onClick: () => void | Promise<void>;
    /** 覆盖默认的"点击后自动关闭"。异步动作默认 `false`。 */
    autoClose?: boolean;
}

export interface AppDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    title: ReactNode;
    /** 副标题/说明。Radix 要求对话框有可读描述，缺失时用标题兜底。 */
    description?: ReactNode;
    size?: AppDialogSize;
    /** 页脚动作。省略则渲染无页脚（例如纯进度对话框）。 */
    actions?: AppDialogAction[];
    /**
     * Enter 触发的动作 id。默认取**最后一个 `align="end"` 且非 danger
     * 的动作**（即视觉上的主按钮）。
     */
    defaultActionId?: string;
    /**
     * 关闭前询问。返回 `false` 否决本次关闭（用于未保存护栏）。
     * 仅对 Esc / 外部点击生效 —— 动作按钮走各自的 `onClick`。
     */
    beforeClose?: () => boolean;
    /**
     * 是否允许 Esc / 外部点击关闭。`false` 时只能通过页脚按钮退出
     * （历史上只有诊断导出对话框这么做）。
     */
    dismissible?: boolean;
    /**
     * 是否抑制全局快捷键。默认 `true` —— 否则对话框内按空格会触发播放、
     * 按字母会触发时间轴动作。历史上 39/42 个对话框漏了这一条。
     */
    suppressGlobalShortcuts?: boolean;
    /**
     * 正文。纯确认对话框（如「确定要重置布局吗？」）可以没有正文。
     */
    children?: ReactNode;
    className?: string;
    /** 传 `undefined` 可显式关闭 Radix 的"缺少描述"控制台告警。 */
    ariaDescribedBy?: string | undefined;
}

/**
 * 统一对话框。
 *
 * @example
 * <AppDialog
 *   open={open}
 *   onOpenChange={onOpenChange}
 *   title={t("bitrate")}
 *   description={t("export_dialog_desc")}
 *   size="md"
 *   actions={[
 *     { id: "cancel", label: t("cancel"), onClick: () => onOpenChange(false) },
 *     { id: "save", label: t("save"), intent: "primary", onClick: save },
 *   ]}
 * >
 *   <AppForm>...</AppForm>
 * </AppDialog>
 */
export function AppDialog({
    open,
    onOpenChange,
    title,
    description,
    size = "md",
    actions,
    defaultActionId,
    beforeClose,
    dismissible = true,
    suppressGlobalShortcuts = true,
    children,
    className,
    ariaDescribedBy,
}: AppDialogProps) {
    /**
     * 隐藏的默认提交按钮。用于在 `onSubmit` 里区分「Enter 隐式提交」与
     * 「正文里某个按钮被点击」—— 两者都表现为一次 submit 事件。
     */
    const defaultSubmitRef = useRef<HTMLButtonElement | null>(null);
    /** 正在执行的异步动作 id；非空时禁用全部按钮。 */
    const [pendingActionId, setPendingActionId] = useState<string | null>(null);

    /**
     * 抑制全局快捷键。
     *
     * 只在 `open` 期间持有作用域，且引用计数使多个对话框重叠时不会
     * 提前解除（历史上三个独立 body 属性做不到这一点）。
     */
    useEffect(() => {
        if (!open || !suppressGlobalShortcuts) return;
        const token = acquireShortcutSuppression();
        return () => releaseShortcutSuppression(token);
    }, [open, suppressGlobalShortcuts]);

    /**
     * Esc / 外部点击的统一出口。
     *
     * `open` 是受控的，因此"不调用 `onOpenChange`"就等于"保持打开" ——
     * 护栏只需在这一处判一次。
     *
     * 【为什么不能只靠 `preventDefault`】Radix 的 `onEscapeKeyDown` 里
     * `preventDefault()` 在部分路径下并不能阻止它回调 `onOpenChange`。
     * 既有的诊断导出对话框正是因此写成 `onOpenChange={() => {}}`
     * （把回调整个废掉）。这里改成在校验层拦：两条路径都覆盖。
     */
    const requestClose = (next: boolean) => {
        if (next) {
            onOpenChange(true);
            return;
        }
        if (!dismissible) return;
        if (beforeClose && !beforeClose()) return;
        onOpenChange(false);
    };

    const endActions = actions?.filter((action) => (action.align ?? "end") === "end") ?? [];
    const startActions = actions?.filter((action) => action.align === "start") ?? [];
    /** 默认动作：最后一个非 danger 的右侧动作。 */
    const resolvedDefaultId =
        defaultActionId ??
        [...endActions].reverse().find((action) => action.intent !== "danger")?.id ??
        endActions.at(-1)?.id;

    const runAction = async (action: AppDialogAction) => {
        const result = action.onClick();
        if (!(result instanceof Promise)) {
            if (action.autoClose ?? true) onOpenChange(false);
            return;
        }
        setPendingActionId(action.id);
        try {
            await result;
            if (action.autoClose ?? false) onOpenChange(false);
        } finally {
            setPendingActionId(null);
        }
    };

    /**
     * 表单提交处理。
     *
     * 【为什么必须校验 `submitter`】Radix 的 `Button` 不渲染 `type` 属性
     * （已核对 `base-button.js`），因此在 `<form>` 里，**对话框正文中任何一个
     * Radix 按钮都会成为 submit 按钮**。点它就会触发一次 submit；若不校验，
     * 该按钮自己的 `onClick` 与对话框的默认动作会同时执行 —— 例如"浏览文件"
     * 顺带把对话框确认掉。
     *
     * 判定方式：只接受两种来源 ——
     *   - `submitter` 是那个隐藏的默认按钮（Enter 隐式提交）；
     *   - `submitter` 为空（部分引擎对隐式提交不给 submitter）。
     *
     * 其余一律忽略，等于把正文里的按钮自动"降级"为普通按钮，无需在 40 个
     * 调用点逐处补 `type="button"`。这是机制上的一次性修复。
     */
    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (pendingActionId) return;
        // React 的 FormEvent 把 nativeEvent 放宽成 Event，而 submitter 只在
        // SubmitEvent 上定义 —— 这里收窄回真实类型。
        const submitter = (event.nativeEvent as SubmitEvent).submitter;
        if (submitter && submitter !== defaultSubmitRef.current) return;
        const action = actions?.find((candidate) => candidate.id === resolvedDefaultId);
        if (!action || action.disabled) return;
        void runAction(action);
    };

    const busy = pendingActionId !== null;

    return (
        <Dialog.Root open={open} onOpenChange={requestClose}>
            <Dialog.Content
                className={cx("app-dialog flex flex-col", className)}
                style={{
                    maxWidth: SIZE_PX[size],
                    // 高度上限统一：小窗口上内容滚动而不是溢出屏幕。
                    // 历史上约 30 个对话框完全没有上限。
                    maxHeight: "min(86vh, 960px)",
                }}
                aria-describedby={ariaDescribedBy}
                /**
                 * 阻止按键冒泡到全局监听。历史上 10/42 个对话框漏了这一步，
                 * 导致框内按空格触发播放。
                 */
                onKeyDown={(event) => event.stopPropagation()}
                /**
                 * `dismissible` 之外不在这里跑 `beforeClose`：Radix 只会在
                 * 事件未被 preventDefault 时回调 `onOpenChange(false)`，
                 * 而 `requestClose` 已经在那里问过一次了。两处都问会导致
                 * 护栏副作用（如弹确认框）被执行两遍。
                 */
                onPointerDownOutside={(event) => {
                    if (!dismissible) event.preventDefault();
                }}
                onEscapeKeyDown={(event) => {
                    if (!dismissible) event.preventDefault();
                }}
            >
                {/*
                 * 用 `<form>` 包住内容与页脚，使 **Enter 确认**成为默认行为。
                 *
                 * 这是补齐桌面端默认按钮约定的关键一步：全仓此前没有任何
                 * `<form>`，靠浏览器原生提交语义才不需要为每个输入框手工接
                 * keydown。单行输入里 Enter 提交、多行 textarea 里 Enter 换行，
                 * 两种行为都由引擎给出，无需特判。
                 */}
                <form onSubmit={onSubmit} className="flex min-h-0 flex-1 flex-col">
                    {/*
                     * 隐藏的默认提交按钮必须位于**树序最前**。
                     *
                     * 浏览器对「在输入框里按 Enter」的隐式提交，会去点树序上第一个
                     * submit 按钮；而 Radix 的 Button 不渲染 type，正文里任何按钮都
                     * 是 submit 按钮。这个按钮若排在它们后面，Enter 就会点到正文里的
                     * 第一个按钮而不是默认动作。`onSubmit` 里再用 `submitter` 复核。
                     */}
                    {resolvedDefaultId ? (
                        <button
                            ref={defaultSubmitRef}
                            type="submit"
                            hidden
                            tabIndex={-1}
                            aria-hidden="true"
                        />
                    ) : null}

                    <Dialog.Title className="app-dialog__title">{title}</Dialog.Title>
                    {description ? (
                        <Dialog.Description className="app-dialog__description mt-1">
                            {description}
                        </Dialog.Description>
                    ) : (
                        /*
                         * 无描述时仍渲染 `Dialog.Description`（视觉上隐藏）。
                         *
                         * 不能只挂一个 `aria-describedby` 指向自己的 span：
                         * Radix 是通过 `Dialog.Description` 这个**组件**在
                         * 上下文里打标记的，缺了它会在控制台报
                         * "Missing Description"。用组件 + sr-only 既满足
                         * 无障碍（屏幕阅读器读到标题），又不显示多余文字。
                         */
                        <Dialog.Description className="sr-only">{title}</Dialog.Description>
                    )}

                    <div className="app-dialog__body mt-3 min-h-0 flex-1 overflow-y-auto">{children}</div>

                    {actions?.length ? (
                        <div className="app-dialog__footer mt-4 flex shrink-0 items-center gap-2">
                            <div className="flex items-center gap-2">
                                {startActions.map((action) => (
                                    <DialogActionButton
                                        key={action.id}
                                        action={action}
                                        busy={busy}
                                        pending={pendingActionId === action.id}
                                        onRun={() => void runAction(action)}
                                    />
                                ))}
                            </div>
                            <div className="flex flex-1 items-center justify-end gap-2">
                                {endActions.map((action) => (
                                    <DialogActionButton
                                        key={action.id}
                                        action={action}
                                        busy={busy}
                                        pending={pendingActionId === action.id}
                                        onRun={() => void runAction(action)}
                                    />
                                ))}
                            </div>
                        </div>
                    ) : null}
                </form>
            </Dialog.Content>
        </Dialog.Root>
    );
}

function DialogActionButton({
    action,
    busy,
    pending,
    onRun,
}: {
    action: AppDialogAction;
    busy: boolean;
    pending: boolean;
    onRun: () => void;
}) {
    return (
        <AppButton
            intent={action.intent ?? "default"}
            disabled={action.disabled || (busy && !pending)}
            loading={pending}
            onClick={onRun}
        >
            {action.label}
        </AppButton>
    );
}

/**
 * 醒目的确认对话框 —— 供 `window.alert` / `window.confirm` 的替换使用。
 *
 * 【为什么单独给一个】审查时发现 7 处 `window.alert()` 在 Tauri 桌面应用里
 * 弹出浏览器原生弹窗，绕过全部对话框约定（不可主题化、不可本地化按钮、
 * 外观与操作系统绑死）。给一个三行就能用的封装，是让这些调用点愿意迁移的
 * 前提 —— 换成通用 `AppDialog` 需要写 15 行 JSX。
 */
export interface AppConfirmDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    title: ReactNode;
    message: ReactNode;
    /** 确认按钮文案。省略用调用方传入的通用词（如「确定」）。 */
    confirmLabel: ReactNode;
    cancelLabel: ReactNode;
    intent?: "primary" | "danger";
    onConfirm: () => void | Promise<void>;
}

export function AppConfirmDialog({
    open,
    onOpenChange,
    title,
    message,
    confirmLabel,
    cancelLabel,
    intent = "primary",
    onConfirm,
}: AppConfirmDialogProps) {
    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={title}
            size="sm"
            actions={[
                { id: "cancel", label: cancelLabel, onClick: () => onOpenChange(false) },
                { id: "confirm", label: confirmLabel, intent, onClick: onConfirm },
            ]}
        >
            <p className="text-sm leading-5 text-qt-text">{message}</p>
        </AppDialog>
    );
}
