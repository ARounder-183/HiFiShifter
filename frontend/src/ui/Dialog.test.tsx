// @vitest-environment jsdom
/*
 * `AppDialog` 组合壳的契约测试。
 *
 * 【为什么必须有】这个壳要一次性接管 42 个对话框的行为收口，其中两条是
 * **当前完全缺失**的桌面端范式，靠人眼很难稳定复验：
 *
 *   1. **Enter 确认**。全仓此前没有任何 `<form>`，因此约 38 个对话框
 *      按 Enter 毫无反应。这条只能靠模拟真实按键来锁。
 *   2. **关闭护栏**。此前 40/42 个对话框没有未保存确认，Esc 一按就丢草稿。
 *
 * 另外锁定"重叠时先关内层不解除快捷键抑制"这条 —— 它是从三个独立
 * body 属性迁移过来的直接动因。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { AppDialog } from "./Dialog";
import { isShortcutSuppressed, resetShortcutScopesForTests } from "./shortcutScope";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    // Radix 通过 portal 把内容挂到 document.body，不在 container 里。
    // 只移除 container 会留下孤儿 portal，污染下一个用例的查询。
    document.body.innerHTML = "";
    resetShortcutScopesForTests();
});

test("Enter 触发主按钮动作（补齐缺失的默认按钮约定）", async () => {
    const onSave = vi.fn();
    const onOpenChange = vi.fn();

    await act(async () => {
        root.render(
            <AppDialog
                open
                onOpenChange={onOpenChange}
                title="导出"
                actions={[
                    { id: "cancel", label: "取消", onClick: () => onOpenChange(false) },
                    { id: "save", label: "保存", intent: "primary", onClick: onSave },
                ]}
            >
                <input defaultValue="x" />
            </AppDialog>,
        );
    });

    const form = document.body.querySelector("form");
    expect(form).not.toBeNull();

    // jsdom 不实现 Enter 的隐式表单提交，因此验证的是"表单存在且有提交入口"
    // 加上"提交处理会跑到主按钮"这两件事。
    await act(async () => {
        form!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
});

test("默认动作取最后一个右侧非危险动作，而不是危险动作", async () => {
    const onDelete = vi.fn();
    const onApply = vi.fn();

    await act(async () => {
        root.render(
            <AppDialog
                open
                onOpenChange={() => {}}
                title="自定义音阶"
                actions={[
                    { id: "delete", label: "删除", intent: "danger", align: "start", onClick: onDelete },
                    { id: "cancel", label: "取消", onClick: () => {} },
                    { id: "apply", label: "应用", intent: "primary", onClick: onApply },
                ]}
            >
                <span>内容</span>
            </AppDialog>,
        );
    });

    await act(async () => {
        document.body
            .querySelector("form")!
            .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });

    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onDelete).not.toHaveBeenCalled();
});

test("beforeClose 返回 false 时 Esc 被否决（未保存护栏）", async () => {
    const onOpenChange = vi.fn();
    const beforeClose = vi.fn(() => false);

    await act(async () => {
        root.render(
            <AppDialog open onOpenChange={onOpenChange} title="设置" beforeClose={beforeClose}>
                <span>内容</span>
            </AppDialog>,
        );
    });

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });

    expect(beforeClose).toHaveBeenCalled();
    expect(onOpenChange).not.toHaveBeenCalled();
});

test("dismissible=false 时 Esc 不关闭（诊断导出对话框的行为）", async () => {
    const onOpenChange = vi.fn();

    await act(async () => {
        root.render(
            <AppDialog open onOpenChange={onOpenChange} title="导出诊断" dismissible={false}>
                <span>进行中</span>
            </AppDialog>,
        );
    });

    await act(async () => {
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });

    expect(onOpenChange).not.toHaveBeenCalled();
});

test("打开期间抑制全局快捷键，关闭后释放", async () => {
    await act(async () => {
        root.render(
            <AppDialog open onOpenChange={() => {}} title="设置">
                <span>内容</span>
            </AppDialog>,
        );
    });
    expect(isShortcutSuppressed()).toBe(true);

    await act(async () => {
        root.render(
            <AppDialog open={false} onOpenChange={() => {}} title="设置">
                <span>内容</span>
            </AppDialog>,
        );
    });
    expect(isShortcutSuppressed()).toBe(false);
});

test("宽度档位映射到四档令牌值，不再是 14 种字面量", async () => {
    const sizes = { sm: 400, md: 520, lg: 640, xl: 800 } as const;

    for (const [size, expected] of Object.entries(sizes)) {
        await act(async () => {
            root.render(
                <AppDialog open onOpenChange={() => {}} title="t" size={size as keyof typeof sizes}>
                    <span>内容</span>
                </AppDialog>,
            );
        });
        const content = document.body.querySelector(".app-dialog") as HTMLElement | null;
        expect(content, `size=${size} 应渲染出 Dialog.Content`).not.toBeNull();
        expect(content!.style.maxWidth).toBe(`${expected}px`);
    }
});

test("异步动作进入 pending 且不自动关闭", async () => {
    const onOpenChange = vi.fn();
    let resolveAction: () => void = () => {};
    const pending = new Promise<void>((resolve) => {
        resolveAction = resolve;
    });

    await act(async () => {
        root.render(
            <AppDialog
                open
                onOpenChange={onOpenChange}
                title="导出"
                actions={[{ id: "run", label: "开始", intent: "primary", onClick: () => pending }]}
            >
                <span>内容</span>
            </AppDialog>,
        );
    });

    await act(async () => {
        (document.body.querySelectorAll("button")[0] as HTMLButtonElement).click();
    });

    // 未完成前不得关闭
    expect(onOpenChange).not.toHaveBeenCalled();

    await act(async () => {
        resolveAction();
        await pending;
    });

    // 异步动作默认 autoClose=false：由调用方决定何时关
    expect(onOpenChange).not.toHaveBeenCalled();
});
