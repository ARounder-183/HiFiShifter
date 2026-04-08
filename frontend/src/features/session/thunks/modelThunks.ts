import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";

export const loadModel = createAsyncThunk("session/loadModel", async (modelDir: string) => {
    return webApi.loadModel(modelDir);
});

export const loadDefaultModel = createAsyncThunk("session/loadDefaultModel", async () => {
    return webApi.loadDefaultModel();
});
