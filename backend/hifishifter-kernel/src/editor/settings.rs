//! 原UI部分配置补丁合并；插件与独立app都不能因单项保存重置其余设置。
/// 保留原app的一层深合并语义；不触碰设备、文件路径或进程全局模型配置。
pub fn merge(mut base:serde_json::Value,patch:&serde_json::Value)->serde_json::Value {
    const DEEP:&[&str]=&["timelineSnap","renderCache","channelImportPolicy","notebook","dock","search","penInput"];
    if let (serde_json::Value::Object(base),serde_json::Value::Object(patch))=(&mut base,patch) {
        for (key,value) in patch {
            if DEEP.contains(&key.as_str()) {
                match base.get_mut(key) {
                    Some(serde_json::Value::Object(current))=>{
                        if let serde_json::Value::Object(next)=value {for (key,value) in next {current.insert(key.clone(),value.clone());}}
                    },
                    _=>{base.insert(key.clone(),value.clone());},
                }
            } else {base.insert(key.clone(),value.clone());}
        }
    }
    base
}
