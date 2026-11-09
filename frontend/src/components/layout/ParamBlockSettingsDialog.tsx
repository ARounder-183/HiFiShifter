/**
 * 参数块设置：编辑区下拉菜单里的全局配置迁到这里。
 *
 * 即时动作（合并 / 删除 / 恢复）仍由块右键菜单承载；本对话框只放跨工程
 * 生效的策略，改动即写 UI 设置并持久化。
 */
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { persistUiSettings } from "../../features/session/sessionSlice";
import { setParamBlockFollowPitch } from "../../features/session/sessionSlice";
import { useI18n } from "../../i18n/I18nProvider";
import { AppDialog } from "../../ui/Dialog";
import { AppForm, AppSwitchRow } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function ParamBlockSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const followPitch = useAppSelector((state) => state.session.paramBlockFollowPitch);
    const { tf } = useI18n();

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("param_block_settings_title")}
            description={tf("param_block_settings_desc")}
            size="sm"
            actions={[
                {
                    id: "ok",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => onOpenChange(false),
                },
            ]}
        >
            <AppForm>
                <AppSwitchRow
                    label={tf("param_block_follow_pitch")}
                    hint={tf("param_block_follow_pitch_hint")}
                    checked={followPitch}
                    onCheckedChange={(checked) => {
                        dispatch(setParamBlockFollowPitch(checked));
                        void dispatch(persistUiSettings());
                    }}
                />
            </AppForm>
        </AppDialog>
    );
}
