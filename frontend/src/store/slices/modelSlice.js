import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { modelService } from '../../services/modelService';

// Async thunks
export const fetchModelInfo = createAsyncThunk(
  'model/fetchModelInfo',
  async (modelName, { rejectWithValue }) => {
    try {
      const data = await modelService.getModelInfo(modelName);
      return data;
    } catch (error) {
      return rejectWithValue({
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
      });
    }
  },
);

export const fetchLatestTraining = createAsyncThunk(
  'model/fetchLatestTraining',
  async (modelName, { rejectWithValue }) => {
    try {
      const data = await modelService.getLatestTraining(modelName);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const startTraining = createAsyncThunk(
  'model/startTraining',
  async (_, { dispatch, getState, rejectWithValue }) => {
    try {
      const data = await modelService.trainModel();
      const { model } = getState();
      if (model.selectedModelName) {
        // Refetch training history to show the new 'running' state
        dispatch(
          fetchTrainingHistory({ modelName: model.selectedModelName, limit: 10 }),
        );
      }
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchTrainingHistory = createAsyncThunk(
  'model/fetchTrainingHistory',
  async ({ modelName, limit = 10 }, { rejectWithValue }) => {
    try {
      const data = await modelService.getTrainingHistory(modelName, limit);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchExperimentHistory = createAsyncThunk(
  'model/fetchExperimentHistory',
  async ({ limit = 10 }, { rejectWithValue }) => {
    try {
      const data = await modelService.getExperimentHistory(limit);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchAvailableModels = createAsyncThunk(
  'model/fetchAvailableModels',
  async (_, { rejectWithValue }) => {
    try {
      const data = await modelService.getAvailableModels();
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchModelVersions = createAsyncThunk(
  'model/fetchModelVersions',
  async (modelName, { rejectWithValue }) => {
    try {
      const data = await modelService.getModelVersions(modelName);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const promoteModel = createAsyncThunk(
  'model/promoteModel',
  async ({ modelName, version }, { dispatch, rejectWithValue }) => {
    try {
      const data = await modelService.promoteModelToProduction(
        modelName,
        version,
      );
      // After successful promotion, dispatch actions to refetch all relevant data
      dispatch(fetchModelInfo(modelName));
      dispatch(fetchModelVersions(modelName));
      dispatch(fetchTrainingHistory({ modelName, limit: 10 }));
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const registerModel = createAsyncThunk(
  'model/registerModel',
  async ({ runId, modelName }, { dispatch, rejectWithValue }) => {
    try {
      const data = await modelService.registerModel(runId, modelName);
      dispatch(fetchAvailableModels()); // Refresh available models list
      dispatch(fetchModelVersions(modelName)); // Refresh model versions
      dispatch(fetchTrainingHistory({ modelName, limit: 10 })); // Refresh training history
      dispatch(fetchModelInfo(modelName)); // Refresh current model info
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

const initialState = {
  currentModel: null,
  latestTraining: null,
  trainingHistory: [],
  experimentHistory: [],
  availableModels: [],
  modelVersions: [],
  loading: false,
  trainingLoading: false,
  error: null,
  trainingMessage: null,
  selectedModelName: null,
  registrationStatus: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
  registrationMessage: null,
  registrationError: null,
};

const modelSlice = createSlice({
  name: 'model',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    clearTrainingMessage: (state) => {
      state.trainingMessage = null;
    },
    setSelectedModel: (state, action) => {
      state.selectedModelName = action.payload;
    },
    clearModelData: (state) => {
      state.currentModel = null;
      state.latestTraining = null;
      state.trainingHistory = [];
      state.modelVersions = [];
    },
    clearRegistrationStatus: (state) => {
      state.registrationStatus = 'idle';
      state.registrationMessage = null;
      state.registrationError = null;
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch model info
      .addCase(fetchModelInfo.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchModelInfo.fulfilled, (state, action) => {
        state.loading = false;
        state.currentModel = action.payload;
      })
      .addCase(fetchModelInfo.rejected, (state, action) => {
        state.loading = false;
        if (action.payload?.status === 404) {
          state.error = null; // Gracefully handle no production model
        } else {
          state.error =
            action.payload?.data?.detail ||
            action.payload?.message ||
            'An unknown error occurred.';
        }
        state.currentModel = null;
      })
      // Fetch latest training
      .addCase(fetchLatestTraining.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchLatestTraining.fulfilled, (state, action) => {
        state.loading = false;
        state.latestTraining = action.payload;
      })
      .addCase(fetchLatestTraining.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Start training
      .addCase(startTraining.pending, (state) => {
        state.trainingLoading = true;
        state.error = null;
        state.trainingMessage = null;
      })
      .addCase(startTraining.fulfilled, (state, action) => {
        state.trainingLoading = false;
        state.trainingMessage = action.payload.message;
      })
      .addCase(startTraining.rejected, (state, action) => {
        state.trainingLoading = false;
        state.error = action.payload;
      })
      // Fetch training history
      .addCase(fetchTrainingHistory.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchTrainingHistory.fulfilled, (state, action) => {
        state.loading = false;
        state.trainingHistory = action.payload.training_history || [];
      })
      .addCase(fetchTrainingHistory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Fetch experiment history
      .addCase(fetchExperimentHistory.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchExperimentHistory.fulfilled, (state, action) => {
        state.loading = false;
        state.experimentHistory = action.payload.training_history?.runs || [];
        console.log(state.experimentHistory);
        console.log(JSON.stringify(state.experimentHistory));
      })
      .addCase(fetchExperimentHistory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Fetch available models
      .addCase(fetchAvailableModels.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchAvailableModels.fulfilled, (state, action) => {
        state.loading = false;
        state.availableModels = action.payload.models || [];
      })
      .addCase(fetchAvailableModels.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Fetch model versions
      .addCase(fetchModelVersions.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchModelVersions.fulfilled, (state, action) => {
        state.loading = false;
        state.modelVersions = action.payload.versions || [];
      })
      .addCase(fetchModelVersions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Promote model
      .addCase(promoteModel.pending, (state) => {
        state.loading = true;
      })
      .addCase(promoteModel.fulfilled, (state, action) => {
        state.loading = false;
        state.trainingMessage = action.payload.message;
      })
      .addCase(promoteModel.rejected, (state, action) => {
        state.loading = false;
        state.error =
          action.payload?.detail ||
          action.payload?.message ||
          'An unknown error occurred.';
      })
      // Register model
      .addCase(registerModel.pending, (state) => {
        state.registrationStatus = 'loading';
        state.registrationMessage = null;
        state.registrationError = null;
      })
      .addCase(registerModel.fulfilled, (state, action) => {
        state.registrationStatus = 'succeeded';
        state.registrationMessage = action.payload.message;
      })
      .addCase(registerModel.rejected, (state, action) => {
        state.registrationStatus = 'failed';
        state.registrationError =
          action.payload?.detail ||
          action.payload?.message ||
          'An unknown error occurred.';
      });
  },
});

export const { clearError, clearTrainingMessage, setSelectedModel, clearModelData, clearRegistrationStatus } = modelSlice.actions;
export default modelSlice.reducer;
