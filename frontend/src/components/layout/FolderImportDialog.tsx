/**
 * 「导入文件夹」选项对话框。
 *
 * 【为什么目录导入需要一个对话框，而多文件导入不需要】多文件导入的选择只有"怎么排"
 * 一件事，三个选项一句话说得完，用菜单点一下最快。目录导入多出两个**正交**的问题：
 * 要不要下钻子目录（决定"要导入的文件集合是什么"）、要不要为文件夹建轨道组
 * （决定"落位时反不反映目录结构"）。后者还只在"跨轨道添加"下有效。这不是菜单能
 * 表达的形状，而 REAPER 在这件事上也是弹对话框 —— 用户对它的预期就是"问一次"。
 *
 * 【为什么两个选项的条件显隐不同】递归是"这次没得选"（没有子目录），直接不显示；
 * 建轨道组是"这个模式下调不了"，显示但禁用 —— 让用户看得见它、也看得见为什么。
 */

import { useI18n } from "../../i18n/I18nProvider";
import type { FolderMediaScan } from "../../services/api/fileBrowser";
import type { FolderImportPlan } from "../../features/fileBrowser/folderImportPlan";
import {
    FOLDER_IMPORT_MODES,
    FOLDER_IMPORT_MODE_LABEL_KEY,
    type FolderImportMode,
    type FolderImportOptions,
} from "../../features/fileBrowser/folderImportOptions";
import { AppSelect } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 当前扫描结果；`null` 表示还没扫完（或扫描失败）。 */
    scan: FolderMediaScan | null;
    /** 扫描是否进行中（切换递归选项会重新扫描）。 */
    scanning: boolean;
    /** 按当前选项展开出的计划。 */
    plan: FolderImportPlan;
    /** 当前选项（受控）。 */
    options: FolderImportOptions;
    onOptionsChange: (patch: Partial<FolderImportOptions>) => void;
    onConfirm: () => void;
}

export function FolderImportDialog({
    open,
    onOpenChange,
    scan,
    scanning,
    plan,
    options,
    onOptionsChange,
    onConfirm,
}: Props) {
    const { t, plural } = useI18n();

    // 文件夹数取自**计划**（已剔除空目录），而不是扫描结果的组数 —— 后者会把
    // 不会建出轨道的空目录也算进去，汇总文案于是与实际导入结果不符。
    const folderCount = plan.totalFolders;
    const fileCount = plan.totalFiles;
    const hasSubdirs = scan?.groups.some((group) => group.hasSubdirs) ?? false;
    const rejectedCount = scan?.rejected.length ?? 0;
    const truncated = scan?.truncated ?? false;
    const canImport = fileCount > 0 && !scanning;

    const summary = scanning
        ? t("folder_import_scanning")
        : fileCount === 0
          ? t("folder_import_no_media")
          : `${plural("folder_import_summary_folders", folderCount)} · ${plural(
                "folder_import_summary_files",
                fileCount,
            )}`;

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("folder_import_title")}
            message={summary}
            size="sm"
            actions={[
                { id: "cancel", label: t("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "import",
                    label: t("folder_import_import"),
                    intent: "primary",
                    disabled: !canImport,
                    onClick: () => {
                        if (!canImport) return;
                        onConfirm();
                    },
                },
            ]}
        >
            <AppForm booleanRow="leading">
                <AppField label={t("folder_import_mode")}>
                    <AppSelect
                        value={options.mode}
                        onValueChange={(value) =>
                            onOptionsChange({ mode: value as FolderImportMode })
                        }
                        options={FOLDER_IMPORT_MODES.map((mode) => ({
                            value: mode,
                            label: t(FOLDER_IMPORT_MODE_LABEL_KEY[mode]),
                        }))}
                    />
                </AppField>

                {/* 没有子目录时"递归"是这次没得选，直接不显示。 */}
                {hasSubdirs && (
                    <AppSwitchRow
                        control="checkbox"
                        label={t("folder_import_recursive")}
                        checked={options.recursive}
                        onCheckedChange={(checked) => onOptionsChange({ recursive: checked })}
                    />
                )}

                {/*
                  建轨道组只在"跨轨道添加"下有效（另外两种模式下文件根本不在各自的
                  轨道上）。这里**显示但禁用**而不是隐藏：让用户看得见它存在、也看得见
                  为什么现在调不了。
                */}
                <AppSwitchRow
                    control="checkbox"
                    label={t("folder_import_create_tracks")}
                    hint={
                        options.mode === "across-tracks"
                            ? t("folder_import_create_tracks_hint")
                            : t("folder_import_create_tracks_unavailable")
                    }
                    disabled={options.mode !== "across-tracks"}
                    checked={options.createFolderTracks && options.mode === "across-tracks"}
                    onCheckedChange={(checked) => onOptionsChange({ createFolderTracks: checked })}
                />

                {/* 截断与跳过都必须说出来：不说就等于静默少导入。 */}
                {truncated && (
                    <p className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                        {plural("folder_import_truncated", fileCount)}
                    </p>
                )}
                {rejectedCount > 0 && (
                    <p className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                        {plural("folder_import_rejected", rejectedCount)}
                    </p>
                )}
            </AppForm>
        </AppDialog>
    );
}
