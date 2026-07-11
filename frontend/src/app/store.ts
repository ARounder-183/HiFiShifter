import { configureStore } from "@reduxjs/toolkit";
import sessionReducer from "../features/session/sessionSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../features/keybindings/keybindingsSlice";
import notebookReducer from "../features/notebook/notebookSlice";

export const store = configureStore({
    reducer: {
        session: sessionReducer,
        fileBrowser: fileBrowserReducer,
        keybindings: keybindingsReducer,
        notebook: notebookReducer,
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
