//! 原UI部分配置补丁合并；插件与独立app都不能因单项保存重置其余设置。

/// 需要做**深度合并**（逐个嵌套子键覆盖）的顶层键。
///
/// 【为什么是公开常量】独立 App 的 `commands/ui_settings.rs` 有**第二份**同内容的
/// 清单（它不经过本模块的 `merge`）。两份清单没有编译保护，只能靠
/// `ui_settings.rs` 里的一条测试断言两者相等 —— 因此这里必须可被它读到。
pub const DEEP_MERGE_KEYS: &[&str] = &[
    "timelineSnap",
    "renderCache",
    "channelImportPolicy",
    "notebook",
    "dock",
    // ARA 形态的布局与 App 的 `dock` 同形同语义，因此同样需要子键合并 ——
    // 漏登记这一项，用户在插件里改一个行为开关就会把整份布局抹掉。
    "dockPlugin",
    "search",
    "penInput",
];

/// 保留原app的一层深合并语义；不触碰设备、文件路径或进程全局模型配置。
pub fn merge(mut base: serde_json::Value, patch: &serde_json::Value) -> serde_json::Value {
    const DEEP: &[&str] = DEEP_MERGE_KEYS;
    if let (serde_json::Value::Object(base), serde_json::Value::Object(patch)) = (&mut base, patch)
    {
        for (key, value) in patch {
            if DEEP.contains(&key.as_str()) {
                match base.get_mut(key) {
                    Some(serde_json::Value::Object(current)) => {
                        if let serde_json::Value::Object(next) = value {
                            for (key, value) in next {
                                current.insert(key.clone(), value.clone());
                            }
                        }
                    }
                    _ => {
                        base.insert(key.clone(), value.clone());
                    }
                }
            } else {
                base.insert(key.clone(), value.clone());
            }
        }
    }
    base
}
