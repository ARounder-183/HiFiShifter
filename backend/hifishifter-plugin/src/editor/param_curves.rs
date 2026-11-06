//! GUI / 私有分组用的纯曲线工具。
//!
//! 【为什么单独成文件】这三个函数原先住在 `parameter_atlas` 里，而它们与"图集"
//! （源域曲线、锚、复制种子）没有任何关系 —— 只碰 `TrackParamsState` 的数组。
//! 图集删除时它们必须留下，于是搬到这里，成为唯一一份"曲线键 → 数组"的映射。
//!
//! 【为什么键是字符串而不是枚举】`extra:{name}` 是**开放**集合（用户可加任意参数
//! 曲线），枚举会逼出一个"其它"分支，而那个分支迟早与真实的键漂移。

use hifishifter_kernel::editor::param_edit::default_pad_for_key;
use hifishifter_kernel::state::TrackParamsState;

/// 某个曲线键的"未编辑"值。
///
/// 【为什么必须集中在一处】它是"这一帧到底有没有编辑"的唯一判据：写入侧用它决定
/// 要不要造一条 run，显示侧用它决定要不要标 `user_modified`。两处各写一份，就会出现
/// "画了一条等于默认值的线，于是整条曲线被标成已编辑"。
pub(super) fn pad(key: &str) -> f32 {
    default_pad_for_key(key)
}

/// GUI / 分组**可编辑**的曲线集合（不含分析派生数组）。
pub(super) fn editable_parameter_curves(params: &TrackParamsState) -> Vec<(String, &[f32])> {
    let mut curves = vec![
        ("pitch_edit".into(), params.pitch_edit.as_slice()),
        ("tension_edit".into(), params.tension_edit.as_slice()),
    ];
    curves.extend(
        params
            .extra_curves
            .iter()
            .map(|(key, values)| (format!("extra:{key}"), values.as_slice())),
    );
    curves
}

/// 按曲线键写回一份数组（`extra:{name}` 落到 `extra_curves`）。
pub(super) fn set_curve(params: &mut TrackParamsState, key: &str, values: Vec<f32>) {
    match key {
        "pitch_orig" => params.pitch_orig = values,
        "pitch_edit" => params.pitch_edit = values,
        "tension_orig" => params.tension_orig = values,
        "tension_edit" => params.tension_edit = values,
        "dyn_orig" => params.dyn_orig = values,
        _ => {
            if let Some(key) = key.strip_prefix("extra:") {
                params.extra_curves.insert(key.into(), values);
            }
        }
    }
}
