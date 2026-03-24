// i18n 翻譯訊息匯總入口，匯出所有語系的翻譯物件與類型定義
import { enUS } from "./en-US";
import { zhCN } from "./zh-CN";
import { zhTW } from "./zh-TW";
import { jaJP } from "./ja-JP";
import { koKR } from "./ko-KR";

export const messages = {
    "en-US": enUS,
    "zh-CN": zhCN,
    "zh-TW": zhTW,
    "ja-JP": jaJP,
    "ko-KR": koKR,
} as const;

export type Locale = keyof typeof messages;
export type MessageKey = keyof (typeof messages)["en-US"];
