import { configureStore } from '@reduxjs/toolkit';
import dashboardSlice from './slices/dashboardSlice';
import modelSlice from './slices/modelSlice';
import predictionSlice from './slices/predictionSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardSlice,
    model: modelSlice,
    prediction: predictionSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
});

// Type definitions for TypeScript usage
// export type RootState = ReturnType<typeof store.getState>;
// export type AppDispatch = typeof store.dispatch;

export default store;
