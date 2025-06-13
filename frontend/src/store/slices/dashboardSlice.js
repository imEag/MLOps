import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { modelService } from '../../services/modelService';

// Async thunk for fetching dashboard summary
export const fetchDashboardSummary = createAsyncThunk(
  'dashboard/fetchSummary',
  async (_, { rejectWithValue }) => {
    try {
      const data = await modelService.getDashboardSummary();
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

// Async thunk for checking services health
export const checkServicesHealth = createAsyncThunk(
  'dashboard/checkHealth',
  async (_, { rejectWithValue }) => {
    try {
      const data = await modelService.checkServicesHealth();
      return data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  },
);

const initialState = {
  summary: {
    models: [],
    total_models: 0,
    recent_predictions_count: 0,
    total_predictions_count: 0,
    recent_trainings_count: 0,
  },
  servicesHealth: {
    overall: 'unknown',
    services: {},
  },
  loading: false,
  error: null,
  lastUpdated: null,
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    clearDashboard: (state) => {
      state.summary = initialState.summary;
      state.servicesHealth = initialState.servicesHealth;
      state.lastUpdated = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch dashboard summary
      .addCase(fetchDashboardSummary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboardSummary.fulfilled, (state, action) => {
        state.loading = false;
        state.summary = action.payload;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchDashboardSummary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Check services health
      .addCase(checkServicesHealth.pending, (state) => {
        state.loading = true;
      })
      .addCase(checkServicesHealth.fulfilled, (state, action) => {
        state.loading = false;
        state.servicesHealth = action.payload;
      })
      .addCase(checkServicesHealth.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { clearError, clearDashboard } = dashboardSlice.actions;
export default dashboardSlice.reducer;
