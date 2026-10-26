//! 让路用的按键词汇：一个按键的**跨平台身份**。
//!
//! 【为什么用 DOM `KeyboardEvent.code` 而不是键码或"具名键 token"】`code` 是
//! **与键盘布局无关的物理键名**（US 布局按 Q 键，AZERTY 上按同一个物理键，`code` 都是
//! `KeyQ`）。它同时是页面**唯一**能稳定拿到的东西：`e.key` 会随布局与 Shift 变，
//! `e.code` 不会。原生侧再把它翻成平台键码，因此判据只有一份、换算各写各的。
//!
//! 【为什么表里只有"具名键"】可打印字符是**打字**的领地：它们在插件里有自己的绑定
//! （`s` 分割、`g` 编组），而未绑定的那些如果也让，插件窗口就变成一个透明的键盘
//! 转发器 —— 用户在一个没有输入框的表面敲字，宿主要么没反应，要么执行一个他完全没
//! 预期的动作。具名键没有这个歧义：没人会"在时间轴上打字"。
//!
//! 【为什么要有这张表】它同时是**白名单**（表外一律拒绝，不猜、不近似）与**跨语言
//! 契约**（`frontend/src/services/hostKeyRelay.table.test.ts` 逐项比对前端的
//! `RELAYABLE_CODES`）。两处漂移的表现是"按了没反应且没有任何提示" —— 最难查的一类。

use serde_json::Value;
use std::collections::HashSet;

/// 可让路的按键，取值为 DOM `KeyboardEvent.code`。
///
/// 【为什么是字符串常量而不是枚举】枚举要求这里再写一遍"名字 → 变体"的映射，
/// 于是"表里加了、映射忘了"成为一类只能靠测试发现的 bug。字符串表只有一处。
const CODES: &[&str] = &[
    "Space",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "Insert",
    "Delete",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "ArrowLeft",
    "ArrowUp",
    "ArrowRight",
    "ArrowDown",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "Numpad0",
    "Numpad1",
    "Numpad2",
    "Numpad3",
    "Numpad4",
    "Numpad5",
    "Numpad6",
    "Numpad7",
    "Numpad8",
    "Numpad9",
    "NumpadEnter",
    "NumpadAdd",
    "NumpadSubtract",
    "NumpadMultiply",
    "NumpadDivide",
    "NumpadDecimal",
];

/// 一个可让路的按键。
///
/// 【为什么是 `&'static str` 而不是 `String`】它来自上面那张表，因此是**已经校验过**
/// 的：类型本身保证"值一定在表里"，比一个到处可能被构造成表外值的 `String` 强。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct KeyCode(&'static str);

impl KeyCode {
    /// DOM `code` → 键。表外一律 `None`（不猜、不近似）。
    pub(crate) fn parse(code: &str) -> Option<Self> {
        CODES
            .iter()
            .find(|known| **known == code)
            .map(|known| KeyCode(known))
    }

    /// 回读原始 `code`（日志与 IPC 用）。
    pub(crate) fn as_str(self) -> &'static str {
        self.0
    }
}

/// 表里的全部取值。
///
/// 【为什么只在测试构建里存在】生产代码读这张表的唯一方式是 `KeyCode::parse`；
/// 把名字列出来只对"表是否自洽""与前端那张表是否一致"这两条断言有用。
#[cfg(test)]
pub(crate) fn relayable_codes() -> impl Iterator<Item = &'static str> {
    CODES.iter().copied()
}

/// 一次按键：具名键 + 四个修饰位。
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Stroke {
    pub(crate) code: KeyCode,
    pub(crate) ctrl: bool,
    pub(crate) shift: bool,
    pub(crate) alt: bool,
    pub(crate) meta: bool,
}

impl Stroke {
    /// 裸键（无任何修饰）。
    ///
    /// 【为什么修饰键一律不让】重投窗口消息**不更新线程的异步键状态**：宿主读
    /// `GetKeyState(VK_CONTROL)` 时看到的是"没按 Ctrl"，于是 `Ctrl+Home` 会被解释成
    /// `Home`。异步键状态只由真实输入事件更新，而那条路（`SendInput`）要求先抢焦点 ——
    /// 那是本功能明确不做的。所以"带修饰的键不能忠实转交"是 Win32 的语义限制，
    /// 不是实现偷懒：与其送一个错的键，不如不送。
    pub(crate) fn is_bare(&self) -> bool {
        !self.ctrl && !self.shift && !self.alt && !self.meta
    }

    /// 从 IPC 载荷（`{ code, ctrl, shift, alt, meta }`）解析。缺修饰字段按 `false`。
    pub(crate) fn from_json(value: &Value) -> Option<Self> {
        let code = KeyCode::parse(value.get("code")?.as_str()?)?;
        let flag = |name: &str| value.get(name).and_then(Value::as_bool).unwrap_or(false);
        Some(Self {
            code,
            ctrl: flag("ctrl"),
            shift: flag("shift"),
            alt: flag("alt"),
            meta: flag("meta"),
        })
    }
}

/// **插件自己占用的**按键集合（`host_key_owned_set` 的产物）。
///
/// 【为什么是"占用"而不是"可让"】可让的集合是**未绑定**的那些键 —— 它是一个补集，
/// 没法枚举（用户随时可能改键，字母数字的取值空间也不是有限表）。而"插件占用了哪些键"
/// 恰好是有限的、且前端本来就算得出来的那份（合并后的键位表）。
///
/// 于是原生的闸门是：**拒绝投递一个在已占用集合里的键**。它防的是"页面凭一次 invoke
/// 就借宿主的手按一个它自己也在用的键"（最坏的情形是 `Delete`：插件删片段，宿主删
/// 它自己的对象）。它**只拦入口 B**（未绑定键），理由见
/// [`OwnedStrokes::blocks_unbound_relay`]。
#[derive(Default, Clone, Debug)]
pub(crate) struct OwnedStrokes(HashSet<Stroke>);

impl OwnedStrokes {
    /// 解析 `[ { code, ctrl, shift, alt, meta } ]`。
    ///
    /// 上限 128 条：默认表是 81 条真实绑定，用户再加也到不了这个量级；
    /// 没有理由接受一份任意长的清单。
    pub(crate) fn from_json(value: &Value) -> Self {
        let mut set = HashSet::new();
        for item in value.as_array().into_iter().flatten().take(128) {
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
    /// 【为什么空集合要"拒绝一切"】空意味着"我们还不知道插件占用了什么"。此时放行
    /// 等于把闸门整个拆掉（页面出 bug 时会借宿主的手按任何键）；拒绝只会让让路暂时
    /// 不生效 —— 而前端在挂载时就会发布，这个窗口是毫秒级的。保守的方向只有一种。
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn contains(&self, stroke: &Stroke) -> bool {
        self.0.contains(stroke)
    }

    /// 【一致性闸】这条投递要不要在原生侧拦下。
    ///
    /// 只拦**入口 B**（`action` 为空 = 未绑定键）："没有任何绑定命中"是前端刚判定完
    /// 的事实，这里复核一次，防的是两侧键表漂移。**入口 A（命中了绑定但能力矩阵拒绝）
    /// 豁免**：它的键按定义都在本集合里 —— 两者来自同一份键位表，闸它等于把入口 A
    /// 全部拒死。键虽然挂在插件的动作上，插件却已经声明"这件事归宿主做"
    /// （transport / 节拍器 / 本地录音三族），此刻它并没有被插件占用。
    pub(crate) fn blocks_unbound_relay(&self, action: &str, stroke: &Stroke) -> bool {
        action.is_empty() && (self.is_empty() || self.contains(stroke))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// **不变量：表里的每一项都能解析回自己。**
    #[test]
    fn every_code_round_trips_through_parse() {
        for code in CODES {
            let parsed = KeyCode::parse(code).expect(code);
            assert_eq!(parsed.as_str(), *code);
        }
    }

    /// **不变量：表外一律拒绝。**
    ///
    /// 【为什么钉死】"不猜、不近似"是这个模块唯一的防御：一次猜测就是把一个用户没按的
    /// 键送到宿主手里。可打印字符（`KeyA` / `Digit1`）与旧版的小写 token 都在表外。
    #[test]
    fn codes_outside_the_table_are_refused() {
        for code in [
            "KeyA",
            "Digit1",
            "Semicolon",
            "space",
            "SPACE",
            "F25",
            "F0",
            "Numpad10",
            "",
        ] {
            assert_eq!(KeyCode::parse(code), None, "{code:?}");
        }
    }

    /// 表内取值必须两两不同，且顺序稳定（前端逐项比对依赖它）。
    #[test]
    fn the_table_has_no_duplicates() {
        let mut names: Vec<&str> = relayable_codes().collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "键表里出现了重复项");
    }

    /// 载荷解析：缺修饰字段按"没按"处理，未知 code 直接拒绝。
    #[test]
    fn strokes_parse_from_ipc_payloads() {
        assert_eq!(
            Stroke::from_json(&json!({"code": "Space"})),
            Some(Stroke {
                code: KeyCode::parse("Space").unwrap(),
                ctrl: false,
                shift: false,
                alt: false,
                meta: false,
            })
        );
        assert_eq!(
            Stroke::from_json(&json!({"code": "F2", "shift": true})),
            Some(Stroke {
                code: KeyCode::parse("F2").unwrap(),
                ctrl: false,
                shift: true,
                alt: false,
                meta: false,
            })
        );
        assert_eq!(Stroke::from_json(&json!({"code": "KeyA"})), None);
        assert_eq!(Stroke::from_json(&json!({})), None);
        // 旧协议的 `key` 字段不再被接受：宁可让前端回落到旧文案，也不猜它的含义。
        assert_eq!(Stroke::from_json(&json!({"key": "space"})), None);
    }

    /// 集合只收表内、去重，且超长载荷被截断而不是被接受。
    #[test]
    fn the_owned_set_is_bounded_and_deduplicated() {
        let set = OwnedStrokes::from_json(&json!([
            {"code": "Space"},
            {"code": "Space"},
            {"code": "Enter"},
            {"code": "KeyA"},
            {"code": "Space", "ctrl": true},
        ]));
        assert_eq!(set.len(), 3, "Space / Enter / Ctrl+Space 各一条，KeyA 被拒");
        assert!(set.contains(&Stroke {
            code: KeyCode::parse("Space").unwrap(),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }));

        let many = json!((0..256)
            .map(|_| json!({"code": "Space"}))
            .collect::<Vec<_>>());
        assert_eq!(OwnedStrokes::from_json(&many).len(), 1);
        assert!(OwnedStrokes::from_json(&json!([])).is_empty());
    }

    /// **一致性闸只拦入口 B。**
    ///
    /// 【为什么钉死】入口 A 的键按定义都在占用集合里（两者来自同一份键位表）；
    /// 闸门若不分入口，Space/方向键的"能力被拒就还给宿主"就会被 100% 拒死，
    /// 让路的主场景整体失效 —— 这是真实的回归形状（见
    /// `docs/plans/2026-10-27-audit-repair-round2.md` §1.2），必须由测试钉住。
    #[test]
    fn the_gate_blocks_unbound_relays_only() {
        let space = Stroke {
            code: KeyCode::parse("Space").unwrap(),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        };
        let home = Stroke {
            code: KeyCode::parse("Home").unwrap(),
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        };
        let owned = OwnedStrokes::from_json(&json!([{"code": "Space"}]));

        // 入口 A（带 action）：键即使已占用也放行 —— 能力矩阵显式审过的白名单。
        assert!(!owned.blocks_unbound_relay("transport.playPause", &space));
        // 入口 A 且集合未发布：同样放行（豁免与集合内容无关）。
        assert!(!OwnedStrokes::default().blocks_unbound_relay("transport.playPause", &space));

        // 入口 B：已占用 → 拒（闸门存在的意义：防两侧键表漂移）。
        assert!(owned.blocks_unbound_relay("", &space));
        // 入口 B：未占用 → 放行。
        assert!(!owned.blocks_unbound_relay("", &home));
        // 入口 B 且集合未发布 → 保守拒绝一切（方向不变）。
        assert!(OwnedStrokes::default().blocks_unbound_relay("", &home));
    }
}
