import { configureStore } from "@reduxjs/toolkit";
import sessionReducer from "../features/session/sessionSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import notebookReducer from "../features/notebook/notebookSlice";
import recordingReducer from "../features/recording/recordingSlice";

export const store = configureStore({
    reducer: {
        session: sessionReducer,
        fileBrowser: fileBrowserReducer,
        keybindings: keybindingsReducer,
        notebook: notebookReducer,
        recording: recordingReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {
                ignoredPaths: ["session"],
                ignoredActions: ["session/setTimelineState"],
            },
            immutableCheck: {
                ignoredPaths: ["session.timeline"],
            },
        }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
