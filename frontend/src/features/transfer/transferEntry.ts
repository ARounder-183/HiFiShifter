/**
 * Transfer 入口的可见性。
 *
 * 【为什么入口必须**无条件**可见（只要在插件里）】入口此前按"本实例是不是 Transfer"
 * 判定，而形态要等 `setActive` 之后 `ARA_BINDING_GRACE`（5 秒）才落定，bootstrap 又是
 * **窗口创建那一刻**算的 —— 于是窗口开得早的会话里根本没有答案，用户实测"非 ARA 宿主
 * 里看不到任何 Transfer 入口"。宿主形态（独立 App / 插件）与"形态落没落定"无关，
 * 永远知道答案，因此判据只能是它。
 *
 * 【为什么在 ARA 宿主里也显示】Melodyne 的 Transfer 按钮同样常驻。让入口消失会让
 * "这个宿主到底能不能用"变成一个谜；控件与面板会如实说明"宿主正在通过 ARA 供音，
 * 不需要 Transfer"（见 `transfer_status.araBound`）。
 *
 * 【为什么没有"本实例是不是 Transfer"的前端判据了】它此前唯一的消费者是面板的自动
 * 浮出。录制的主入口现在长在传输栏上（`TransferControl`），而面板退成设置与诊断 ——
 * 不再需要按形态分流。留一个没有消费者的判据只会让下一个人以为它在某处生效。
 */
import { isPluginMode } from "../../services/hostCapabilities";

/** 「视图」菜单里是否显示 Transfer 入口。插件形态下**恒真**（见文件头）。 */
export function showsTransferEntry(): boolean {
    return isPluginMode();
}
