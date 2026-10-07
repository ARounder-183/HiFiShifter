/*
 * 链接地址的归一化（记事本）。
 *
 * 【要解决的问题】用户在"链接地址"里填 `www.bilibili.com`，这个字符串会被原样
 * 存进链接标记。而它没有协议，浏览器/WebView 会把它当**相对地址**，按当前页面
 * 的 origin 解析 —— 于是点开跑到 `tauri.localhost/www.bilibili.com`。导出成
 * HTML 之后同样坏掉（相对导出文件的位置）。
 *
 * 【为什么不能只靠 Link 扩展的 `defaultProtocol`】那个选项只作用于
 * **自动识别**路径（粘贴、输入规则里的 linkify），`setLink` 存的是调用方给的
 * 原字符串，一个字符都不会补。因此补协议这件事必须由写入方自己做。
 *
 * 【判定顺序】先认"明确的相对写法"（`/` `#` `?` `.` 开头），再认协议，最后才
 * 猜测裸主机名。顺序不能反：`example.com:8080` 里的 `example.com:` 长得像一个
 * 协议（`^[a-z][a-z0-9+-]*:`），先认协议就会把它放过去；而协议名按 RFC 不含
 * 点，因此"带点的冒号前缀"不算协议 —— 这条区别正是顺序的判据。
 */

/** 明确带层级协议的写法（`http://`、`hifi://`…）。 */
const HIERARCHICAL_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;

/**
 * 其它协议（`mailto:`、`tel:`…）。
 *
 * **不含点**：RFC 3986 允许协议名里有 `+` `-` `.`，但真实协议没有一个带点，
 * 而裸主机名 + 端口（`example.com:8080`）恰好长这样。排除点即可把两者分开。
 */
const NAMED_SCHEME = /^[a-z][a-z0-9+-]*:/i;

/** 明确的相对写法：绝对路径、锚点、查询串、`./` `../`。 */
const RELATIVE_PREFIX = /^[/#?.]/;

/**
 * 补协议时用的默认协议。
 *
 * 用 `https` 而不是 Link 扩展默认的 `http`：现在没有几个站点还只跑 http，而
 * 地址是**用户手填**的（不是自动识别来的），补成 https 不会让任何真实站点打不开
 * （http 站点会 301 到 https，反之则不会）。
 */
const DEFAULT_PROTOCOL = "https://";

/**
 * 把用户填的地址归一化成**绝对地址**；认不出来的原样返回。
 *
 * 归一化只做一件事：给缺协议的裸主机名补上 `https://`。它**不**做别的事 ——
 * 不改大小写、不补尾部斜杠、不编码、不删参数，因为那些都会让"我填的地址"与
 * "文档里存的地址"看起来不一样，而用户没法从界面上看出被改了什么。
 */
export function normalizeLinkHref(href: string): string {
    const value = href.trim();
    if (!value) return "";
    // 1) 用户明确写成了相对地址 / 锚点：那是他的选择，不动。
    if (RELATIVE_PREFIX.test(value)) return value;
    // 2) 已经有协议（`http://`、`hifi://`、`mailto:`、`tel:`…）：不动。
    if (HIERARCHICAL_SCHEME.test(value) || NAMED_SCHEME.test(value)) return value;
    // 3) 邮箱：`x@y.com` 补 `mailto:`，而不是补成 `https://x@y.com`（那会变成
    //    一个带用户名、主机为 y.com 的 URL，含义完全不同）。
    if (value.includes("@")) return `mailto:${value}`;
    // 4) 裸主机名：取第一个 `/` `?` `#` `:` 之前的部分，含点才算主机
    //    （`assets/img.png` 的 "assets" 不含点，因此不会被误判）。
    const hostname = value.split(/[/?#:]/)[0];
    if (hostname.includes(".")) return `${DEFAULT_PROTOCOL}${value}`;
    // 5) 认不出来（`foo`、`localhost:3000` 这类）：原样留着，别猜。
    return value;
}
