import { configureStore } from "@reduxjs/toolkit";
import sessionReducer from "../features/session/sessionSlice";
import fileBrowserReducer from "../features/fileBrowser/fileBrowserSlice";
import keybindingsReducer from "../features/keybindings/keybindingsSlice";

export const store = configureStore({
    reducer: {
        session: sessionReducer,
        fileBrowser: fileBrowserReducer,
        keybindings: keybindingsReducer,
    },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
