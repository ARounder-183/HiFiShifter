/** 把文本拆为连续数字段 / 非数字段，顺序与原文一致。 */
export function splitDigitRuns(text: string): Array<{ text: string; digits: boolean }> {
    const parts: Array<{ text: string; digits: boolean }> = [];
    let current = "";
    let currentDigits = false;
    const flush = () => {
        if (current.length > 0) {
            parts.push({ text: current, digits: currentDigits });
            current = "";
        }
    };
    for (const ch of text) {
        const isDigit = ch >= "0" && ch <= "9";
        if (isDigit !== currentDigits) {
            flush();
            currentDigits = isDigit;
        }
        current += ch;
    }
    flush();
    return parts;
}
