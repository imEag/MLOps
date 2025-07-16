import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { modelService } from '../../services/modelService';

// Async thunks
export const makePrediction = createAsyncThunk(
  'prediction/makePrediction',
  async ({ modelName, data }, { rejectWithValue }) => {
    try {
      const result = await modelService.predict(modelName, data);
      return result;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchPredictionHistory = createAsyncThunk(
  'prediction/fetchHistory',
  async ({ limit = 50, offset = 0 }, { rejectWithValue }) => {
    try {
      const data = await modelService.getPredictionHistory(limit, offset);
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

export const fetchPredictionStats = createAsyncThunk(
  'prediction/fetchStats',
  async (_, { rejectWithValue }) => {
    try {
      const data = await modelService.getPredictionStats();
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

const initialState = {
  predictions: [],
  predictionHistory: {
    predictions: [],
    total_count: 0,
  },
  predictionStats: {},
  latestPrediction: null,
  loading: false,
  error: null,
  makingPrediction: false,
};

const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    clearLatestPrediction: (state) => {
      state.latestPrediction = null;
    },
    addPrediction: (state, action) => {
      state.predictions.unshift(action.payload);
      state.latestPrediction = action.payload;
    },
    clearPredictions: (state) => {
      state.predictions = [];
      state.predictionHistory = initialState.predictionHistory;
      state.latestPrediction = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Make prediction
      .addCase(makePrediction.pending, (state) => {
        state.makingPrediction = true;
        state.error = null;
      })
      .addCase(makePrediction.fulfilled, (state, action) => {
        state.makingPrediction = false;
        state.latestPrediction = action.payload;
        state.predictions.unshift(action.payload);
      })
      .addCase(makePrediction.rejected, (state, action) => {
        state.makingPrediction = false;
        state.error = action.payload;
      })
      // Fetch prediction history
      .addCase(fetchPredictionHistory.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchPredictionHistory.fulfilled, (state, action) => {
        state.loading = false;
        state.predictionHistory = action.payload;
      })
      .addCase(fetchPredictionHistory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Fetch prediction stats
      .addCase(fetchPredictionStats.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchPredictionStats.fulfilled, (state, action) => {
        state.loading = false;
        state.predictionStats = action.payload;
      })
      .addCase(fetchPredictionStats.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { clearError, clearLatestPrediction, addPrediction, clearPredictions } = predictionSlice.actions;
export default predictionSlice.reducer;
