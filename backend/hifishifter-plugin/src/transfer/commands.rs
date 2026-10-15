//! Transfer 命令：UI 驱动录制与 take 管理的唯一入口。
//!
//! 【为什么这些命令不走 `editor/commands.rs` 的会话分发】那条路面向 `EditorSession`，
//! 而 `EditorSession` 只持有 `Weak<DocumentSession>` —— 它到不了 `ExtensionOwner`，
//! 也就到不了捕获存储。Transfer 的全部状态都在 owner 上，所以入口挂在**视图层**
//! （那里已经有 `link.owner()`，`set_transport` 走的是同一条路）。
//!
//! 【为什么返回 `Option`】视图层先问一次"这是不是 Transfer 命令"；不是就照旧交给会话。
//! 于是这个模块的接入对既有命令**零影响**。

use crate::render::extension::ExtensionOwner;
use crate::transfer::{TransferOptions, TransferStatus};
use serde_json::{json, Value};
use std::sync::Arc;

/// 本命令是否属于 Transfer。视图层据此决定要不要拦下来。
pub(crate) fn is_transfer_command(command: &str) -> bool {
    command.starts_with("transfer_")
}

/// 处理一条 Transfer 命令。调用方必须先确认 [`is_transfer_command`]。
pub(crate) fn handle(
    owner: &Arc<ExtensionOwner>,
    command: &str,
    args: &Value,
) -> Result<Value, String> {
    let transfer = owner.transfer();
    match command {
        "transfer_status" => Ok(status_json(&transfer.status(), owner.is_ara_bound())),
        "transfer_start" => {
            // 【为什么必须先挡这一条】宿主已经把音频经 ARA 交给我们时，再从输入总线录一遍
            // 是**另一条**素材来源，两者会互相覆盖。此前它静默地允许了 —— 用户会在 ARA
            // 宿主里录出一份与界面显示无关的东西。
            if owner.is_ara_bound() {
                return Err("transfer_not_applicable_ara".into());
            }
            transfer.start()?;
            Ok(status_json(&transfer.status(), owner.is_ara_bound()))
        }
        "transfer_stop" => {
            let take = transfer.stop()?;
            // 录完立刻重投影并排一次合成：用户按下停止之后期待听到处理结果，而不是等
            // 下一次某个不相关的动作把它带起来。
            owner.request_transfer_reload();
            Ok(json!({ "take": take, "status": status_json(&transfer.status(), owner.is_ara_bound()) }))
        }
        "transfer_cancel" => {
            transfer.cancel();
            Ok(status_json(&transfer.status(), owner.is_ara_bound()))
        }
        "transfer_clear" => {
            transfer.clear();
            // 时间线要跟着变空（否则界面还留着已经删掉的片段）。
            owner.request_transfer_reload();
            Ok(status_json(&transfer.status(), owner.is_ara_bound()))
        }
        "transfer_select_take" => {
            let id = args["takeId"]
                .as_u64()
                .ok_or("transfer_select_take needs a numeric takeId")?;
            transfer.select_take(id)?;
            owner.request_transfer_reload();
            Ok(status_json(&transfer.status(), owner.is_ara_bound()))
        }
        "transfer_set_options" => {
            let current = transfer.options();
            transfer.set_options(TransferOptions {
                max_duration_sec: args["maxDurationSec"]
                    .as_f64()
                    .unwrap_or(current.max_duration_sec),
                segment_sec: args["segmentSec"].as_f64().unwrap_or(current.segment_sec),
                monitor: args["monitor"].as_bool().unwrap_or(current.monitor),
            });
            Ok(status_json(&transfer.status(), owner.is_ara_bound()))
        }
        other => Err(format!("unknown transfer command: {other}")),
    }
}

/// 状态的线上形状。
///
/// 【为什么 camelCase】与既有 bootstrap / 命令载荷一致；前端只认这一种命名。
/// 状态载荷的线上形状。
///
/// `ara_bound` 让界面能说清"这个宿主不需要 Transfer"，而不是让按钮点了没反应 ——
/// 入口现在是**无条件**可见的（见 `transferEntry`：形态判定要等 5 秒宽限期，不能拿它
/// 当可见性判据），所以面板必须自己回答"我在这里有没有用"。
fn status_json(status: &TransferStatus, ara_bound: bool) -> Value {
    let recording = status.recording.map(|recording| {
        json!({
            "capturedFrames": recording.captured_frames,
            "sealedSegments": recording.sealed_segments,
            "position": recording.position,
            "peak": [recording.peak[0], recording.peak[1]],
            "gaps": recording.gaps,
            "spares": recording.spares,
        })
    });
    json!({
        "state": status.state,
        "recording": recording,
        "options": {
            "maxDurationSec": status.options.max_duration_sec,
            "segmentSec": status.options.segment_sec,
            "monitor": status.options.monitor,
        },
        "takeCount": status.take_count,
        "activeTake": status.active_take,
        // 【为什么这三项是"观察到过"而不是"声明过"】它们描述宿主**实际**做了什么，
        // 而宿主不会声明。前端据此决定显示哪一条引导语：见过位置就能录，没见过位置
        // 就该说"这个宿主没有把播放位置交给插件"，而不是让用户对着按钮猜。
        "sawContext": status.saw_context,
        "sawInput": status.saw_input,
        "sawOffline": status.saw_offline,
        // 录制的**唯一**硬前提是工程位置：音频按 `projectTimeSamples` 对齐，
        // 没有它整条通路不成立。
        "canCapture": status.saw_context,
        // 宿主是否**正在**通过 ARA 供音（那时 Transfer 不适用，见 `transfer_start`）。
        "araBound": ara_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner() -> Arc<ExtensionOwner> {
        let owner = Arc::new(ExtensionOwner::default());
        owner.set_setup_sample_rate(48_000.0);
        owner
    }

    /// 命令名前缀是视图层分流的唯一依据。
    #[test]
    fn only_the_transfer_prefix_is_claimed() {
        assert!(is_transfer_command("transfer_start"));
        assert!(is_transfer_command("transfer_status"));
        assert!(!is_transfer_command("set_transport"));
        assert!(!is_transfer_command("get_timeline_state"));
    }

    /// 宿主正在通过 ARA 供音时，Transfer 必须**明确拒绝**而不是偷偷再录一份。
    ///
    /// 【为什么这条必须有】`transfer_start` 此前不检查这件事：在 ARA 宿主里按下录制会
    /// 从输入总线再采一份**与界面显示无关**的素材，两条来源互相覆盖，用户听到的东西
    /// 与他看到的时间线对不上。入口现在无条件可见（见 `transferEntry`），所以后端必须
    /// 是那道真正的门。
    #[test]
    fn arming_is_refused_while_the_host_provides_audio_through_ara() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        assert!(owner.is_ara_bound(), "这个夹具的宿主是绑定了 ARA 的");
        let error = handle(&owner, "transfer_start", &json!({})).unwrap_err();
        assert_eq!(error, "transfer_not_applicable_ara");
        assert!(!owner.transfer().is_armed(), "拒绝之后不能留下已武装状态");
        // 状态里也要如实报出来，面板才能说清"这个宿主不需要 Transfer"。
        let status = handle(&owner, "transfer_status", &json!({})).unwrap();
        assert_eq!(status["araBound"], true);
    }

    /// 未录制时的状态如实报告"还不能录"，而不是假装可用。
    #[test]
    fn an_untouched_instance_reports_that_it_cannot_capture_yet() {
        let owner = owner();
        let status = handle(&owner, "transfer_status", &json!({})).unwrap();
        assert_eq!(status["state"], "empty");
        assert_eq!(status["takeCount"], 0);
        assert_eq!(status["activeTake"], Value::Null);
        assert_eq!(
            status["canCapture"], false,
            "没见过 ProcessContext 之前不能声称可以录制"
        );
        assert_eq!(status["recording"], Value::Null);
        assert_eq!(status["araBound"], false, "没绑定 ARA 的实例就是 Transfer 的适用对象");
    }

    /// 选项补丁是**逐字段**的：只传一个字段不该把其余字段清掉。
    #[test]
    fn an_options_patch_only_touches_the_fields_it_names() {
        let owner = owner();
        handle(
            &owner,
            "transfer_set_options",
            &json!({ "maxDurationSec": 120.0 }),
        )
        .unwrap();
        let options = owner.transfer().options();
        assert_eq!(options.max_duration_sec, 120.0);
        assert!(options.monitor, "没提 monitor 就不能改它");
        assert!(options.segment_sec > 0.0);
    }

    /// 未知命令明确报错，而不是静默成功。
    #[test]
    fn an_unknown_transfer_command_is_an_error() {
        let owner = owner();
        assert!(handle(&owner, "transfer_frobnicate", &json!({})).is_err());
    }

    /// 选择不存在的 take 是错误，而不是静默改成"没有活动 take"。
    #[test]
    fn selecting_an_unknown_take_is_rejected() {
        let owner = owner();
        assert!(handle(&owner, "transfer_select_take", &json!({ "takeId": 99 })).is_err());
    }

    /// 没声明工程率时不能开始录制，且错误要说明**怎么才能行**。
    #[test]
    fn starting_without_a_declared_rate_explains_what_to_do() {
        let owner = Arc::new(ExtensionOwner::default());
        let error = handle(&owner, "transfer_start", &json!({})).unwrap_err();
        assert!(
            error.contains("setupProcessing"),
            "错误必须指向可执行的下一步，而不是只说'失败'：{error}"
        );
    }
}
