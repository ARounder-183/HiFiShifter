import React from "react";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppContextMenu } from "../../../ui/Menu";

/** 无调用方传入的关闭回调时使用的占位：保持旧实现「菜单不自关闭」的行为。 */
const noop = () => {};

export const GlueContextMenu: React.FC<{
    x: number;
    y: number;
    disabled: boolean;
    onGlue: () => void;
    /**
     * 旧实现没有任何关闭路径（无 Esc、无外部点击），调用方也从不传入关闭
     * 回调。迁移到共享原语后该回调可选：不传时 Esc / 外部点击仍不关闭，
     * 与旧行为一致。
     */
    onClose?: () => void;
}> = ({ x, y, disabled, onGlue, onClose }) => {
    const { t } = useI18n();

    return (
        <AppContextMenu
            x={x}
            y={y}
            onClose={onClose ?? noop}
            items={[{ key: "glue", label: t("glue"), disabled, onSelect: onGlue }]}
        />
    );
};
