/**
 * 文件浏览器行内图标。
 *
 * 【为什么从面板里搬出来】这些图标此前与 `FileBrowserPanel` 的编排逻辑挤在同一个
 * 文件里（该文件当时已近 1300 行）。面板还要继续长（右键菜单、属性对话框、导航），
 * 把纯展示的部分搬走，面板才只留"编排"。
 */

/** 文件夹图标。 */
export function FolderIcon({ className }: { className?: string }) {
    return (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" className={className}>
            <path
                d="M1 3.5C1 3.22386 1.22386 3 1.5 3H5.29289L6.64645 4.35355C6.74021 4.44732 6.86739 4.5 7 4.5H13.5C13.7761 4.5 14 4.72386 14 5V12.5C14 12.7761 13.7761 13 13.5 13H1.5C1.22386 13 1 12.7761 1 12.5V3.5Z"
                fill="currentColor"
            />
        </svg>
    );
}

/** 视频媒体图标。 */
export function VideoIcon({ className }: { className?: string }) {
    return (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" className={className}>
            <rect
                x="1.5"
                y="2.5"
                width="12"
                height="10"
                rx="1.5"
                stroke="currentColor"
                strokeWidth="1.2"
            />
            <path d="M6 5.5V9.5L9.5 7.5L6 5.5Z" fill="currentColor" />
        </svg>
    );
}

/** 音频文件图标。 */
export function AudioIcon({ className }: { className?: string }) {
    return (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" className={className}>
            <path
                d="M7.5 0.75L7.5 14.25M10.5 3L10.5 12M4.5 3L4.5 12M13.5 5.5L13.5 9.5M1.5 5.5L1.5 9.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
            />
        </svg>
    );
}

/** MIDI 文件图标（双音符）。 */
export function MidiIcon({ className }: { className?: string }) {
    return (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" className={className}>
            <path
                d="M5 2.5V9.5M5 9.5C5 8.39543 4.10457 7.5 3 7.5C1.89543 7.5 1 8.39543 1 9.5C1 10.6046 1.89543 11.5 3 11.5C4.10457 11.5 5 10.6046 5 9.5ZM12.5 3.5V9.5M12.5 9.5C12.5 8.39543 11.6046 7.5 10.5 7.5C9.39543 7.5 8.5 8.39543 8.5 9.5C8.5 10.6046 9.39543 11.5 10.5 11.5C11.6046 11.5 12.5 10.6046 12.5 9.5Z"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
            />
            <path d="M5 2.5L12.5 1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
    );
}

/** 工程文件图标（文档 + 星标），用于高亮 hshp/hsp/rpp/vshp/vsp。 */
export function ProjectIcon({ className }: { className?: string }) {
    return (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" className={className}>
            <path
                d="M2.5 1.5H6.5L9 4V13.5H2.5V1.5Z"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinejoin="round"
            />
            <path d="M6.5 1.5V4H9" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
            <path
                d="M7.75 6.75L8.36 8.02L9.75 8.18L8.72 9.14L8.96 10.52L7.75 9.84L6.54 10.52L6.78 9.14L5.75 8.18L7.14 8.02L7.75 6.75Z"
                fill="currentColor"
            />
        </svg>
    );
}
