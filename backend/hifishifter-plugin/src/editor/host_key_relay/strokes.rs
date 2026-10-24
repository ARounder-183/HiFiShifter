//! 让路用的按键词汇：token 与各平台键码之间的**唯一**一张表。
//!
//! 【为什么 token 是字符串而不是键码】键码是平台相关的（VK / keyCode / KeySym），
//! 而"哪些键可以让路"是**跨平台**的产品决定。先把前端归一化后的 token 收进来，
//! 再由各平台模块翻成自己的键码 —— 判据只有一份，换算各写各的。
//!
//! 【为什么只有具名键】字母/数字要靠 `VkKeyScanW`（Windows）、
//! `charactersIgnoringModifiers`（macOS）、`XStringToKeysym`（X11）换算，三套换算在
//! 非美式布局上都会给出不同结果。而"让路"的全部价值在于**用户按什么、宿主就收到什么**，
//! 一个被布局改写的键恰好违背它。默认绑定里要用的（Space / Enter）都是具名键。

use serde_json::Value;
use std::collections::HashSet;

/// 一个可让路的具名键。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum KeyToken {
    Space,
    Enter,
    Tab,
    Escape,
    Backspace,
    Insert,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowLeft,
    ArrowUp,
    ArrowRight,
    ArrowDown,
}

/// token 与键的对应表。**唯一**一张 —— `parse` 读它，跨语言比对也读它。
///
/// 【为什么表与 `parse` 不各写一份】两处各写一张表时，最容易发生的漂移是"token 收进
/// 来了、分支忘了加"，表现是那个键按了毫无反应、且没有任何提示 —— 最难查的一类。
///
/// 【为什么前端还有一份】前端要在**发 IPC 之前**就判掉"这个键根本不让"，否则每次按
/// 都要多一次往返。两份表的漂移由 `hostKeyRelay.table.test.ts` 逐项比对（它读这段源码），
/// 因此这里改动必须同步改 `frontend/src/services/hostKeyRelay.ts` 的 `RELAYABLE_KEYS`。
const TOKENS: &[(&str, KeyToken)] = &[
    ("space", KeyToken::Space),
    ("enter", KeyToken::Enter),
    ("tab", KeyToken::Tab),
    ("escape", KeyToken::Escape),
    ("backspace", KeyToken::Backspace),
    ("insert", KeyToken::Insert),
    ("delete", KeyToken::Delete),
    ("home", KeyToken::Home),
    ("end", KeyToken::End),
    ("pageup", KeyToken::PageUp),
    ("pagedown", KeyToken::PageDown),
    ("arrowleft", KeyToken::ArrowLeft),
    ("arrowup", KeyToken::ArrowUp),
    ("arrowright", KeyToken::ArrowRight),
    ("arrowdown", KeyToken::ArrowDown),
];

impl KeyToken {
    /// token → 键。表外一律 `None`（不猜、不近似）。
    pub(crate) fn parse(token: &str) -> Option<Self> {
        TOKENS
            .iter()
            .find(|(name, _)| *name == token)
            .map(|(_, key)| *key)
    }
}

/// 表里的全部 token。
///
/// 【为什么只在测试构建里存在】生产代码读这张表的唯一方式是 `KeyToken::parse`；
/// 把名字列出来只对"表是否自洽""与前端那张表是否一致"这两条断言有用。做成
/// `cfg(test)` 而不是 `allow(dead_code)`，是为了让"没人用"这件事在编译期就成立，
/// 而不是靠一句注释维持。
#[cfg(test)]
pub(crate) fn relayable_tokens() -> impl Iterator<Item = &'static str> {
    TOKENS.iter().map(|(name, _)| *name)
}

/// 一次按键：具名键 + 四个修饰位。
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Stroke {
    pub(crate) token: KeyToken,
    pub(crate) ctrl: bool,
    pub(crate) shift: bool,
    pub(crate) alt: bool,
    pub(crate) meta: bool,
}

impl Stroke {
    /// 裸键（无任何修饰）。
    ///
    /// 【为什么修饰键一律不让】重投窗口消息 / 合成 X 事件都**不更新线程的异步键状态**，
    /// 宿主的 `GetKeyState` 会看到"没按修饰键" —— 那样的 `S` 不是用户按的 `S`。
    pub(crate) fn is_bare(&self) -> bool {
        !self.ctrl && !self.shift && !self.alt && !self.meta
    }

    /// 从 IPC 载荷（`{ key, ctrl, shift, alt, meta }`）解析。缺字段按 `false`。
    pub(crate) fn from_json(value: &Value) -> Option<Self> {
        let token = KeyToken::parse(value.get("key")?.as_str()?)?;
        let flag = |name: &str| value.get(name).and_then(Value::as_bool).unwrap_or(false);
        Some(Self {
            token,
            ctrl: flag("ctrl"),
            shift: flag("shift"),
            alt: flag("alt"),
            meta: flag("meta"),
        })
    }
}

/// "属于宿主"的按键集合（`host_key_escape_set` 的产物）。
///
/// 【为什么原生也要一份】`PassThrough` 平台（macOS）必须在**页面看到按键之前**决定让不让，
/// 而原生不知道用户的键位（键位在 Redux / localStorage 里）。集合由前端算好推过来，
/// 两端因此共用同一套判据。Windows 上也存着：它是一致性闸 —— 页面不能凭一次 `invoke`
/// 就借宿主的手按一个它没声明过的键。
#[derive(Default, Clone, Debug)]
pub(crate) struct StrokeSet(HashSet<Stroke>);

impl StrokeSet {
    /// 解析 `{ strokes: [ { key, ctrl, shift, alt, meta } ] }`。
    ///
    /// 上限 64 条：这是"用户绑定了哪些播放键"的规模（实际是 1–4），
    /// 没有理由接受一份任意长的清单。
    pub(crate) fn from_json(value: &Value) -> Self {
        let mut set = HashSet::new();
        for item in value.as_array().into_iter().flatten().take(64) {
            if let Some(stroke) = Stroke::from_json(item) {
                set.insert(stroke);
            }
        }
        Self(set)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// 集合是否还是空的（= 前端还没发布过）。
    ///
    /// 【为什么要区分"空"与"非空"】空集合意味着"前端还没告诉我们哪些键属于宿主"，
    /// 此时按"前端已经判定过"处理（与引入集合之前的行为一致）；非空时它就是一份白名单。
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn contains(&self, stroke: &Stroke) -> bool {
        self.0.contains(stroke)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// **不变量：表里的每一项都能解析回自己。**
    #[test]
    fn every_token_round_trips_through_parse() {
        for (name, key) in TOKENS {
            assert_eq!(KeyToken::parse(name), Some(*key), "{name}");
        }
    }

    /// **不变量：表外一律拒绝。**
    ///
    /// 【为什么钉死】"不猜、不近似"是这个模块唯一的防御：一次猜测就是把一个用户没按的键
    /// 送到宿主手里。
    #[test]
    fn tokens_outside_the_table_are_refused() {
        for token in ["f5", "a", "", "SPACE", "space ", "control"] {
            assert_eq!(KeyToken::parse(token), None, "{token:?}");
        }
    }

    /// 表内 token 必须两两不同，且顺序稳定（前端逐项比对依赖它）。
    #[test]
    fn the_table_has_no_duplicates() {
        let mut names: Vec<&str> = relayable_tokens().collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "token 表里出现了重复项");
    }

    /// 载荷解析：缺修饰字段按"没按"处理，未知 token 直接拒绝。
    #[test]
    fn strokes_parse_from_ipc_payloads() {
        assert_eq!(
            Stroke::from_json(&json!({"key": "space"})),
            Some(Stroke {
                token: KeyToken::Space,
                ctrl: false,
                shift: false,
                alt: false,
                meta: false,
            })
        );
        assert_eq!(
            Stroke::from_json(&json!({"key": "enter", "shift": true})),
            Some(Stroke {
                token: KeyToken::Enter,
                ctrl: false,
                shift: true,
                alt: false,
                meta: false,
            })
        );
        assert_eq!(Stroke::from_json(&json!({"key": "f5"})), None);
        assert_eq!(Stroke::from_json(&json!({})), None);
    }

    /// 集合只收表内、去重，且超长载荷被截断而不是被接受。
    #[test]
    fn the_escape_set_is_bounded_and_deduplicated() {
        let set = StrokeSet::from_json(&json!([
            {"key": "space"},
            {"key": "space"},
            {"key": "enter"},
            {"key": "f5"},
            {"key": "space", "ctrl": true},
        ]));
        assert_eq!(set.len(), 3, "space / enter / ctrl+space 各一条，f5 被拒");
        assert!(set.contains(&Stroke {
            token: KeyToken::Space,
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }));
        assert!(set.contains(&Stroke {
            token: KeyToken::Space,
            ctrl: true,
            shift: false,
            alt: false,
            meta: false,
        }));

        let many = json!((0..128)
            .map(|_| json!({"key": "space"}))
            .collect::<Vec<_>>());
        assert_eq!(StrokeSet::from_json(&many).len(), 1);
        assert!(StrokeSet::default().is_empty());
    }
}
