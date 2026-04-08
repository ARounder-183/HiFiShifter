export const MISSING_FILE_CONFIRM_EVENT = "hifi:confirmMissingFileReplacement";

type MissingFileConfirmDetail = {
    missingPath: string;
    resolve: (shouldPick: boolean) => void;
};

export async function requestMissingFileReplacement(missingPath: string): Promise<boolean> {
    if (typeof window === "undefined") return false;
    return new Promise<boolean>((resolve) => {
        window.dispatchEvent(
            new CustomEvent<MissingFileConfirmDetail>(MISSING_FILE_CONFIRM_EVENT, {
                detail: { missingPath, resolve },
            }),
        );
    });
}
