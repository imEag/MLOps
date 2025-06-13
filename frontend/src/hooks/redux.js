import { useDispatch, useSelector } from 'react-redux';

// Custom hooks for typed Redux usage
export const useAppDispatch = () => useDispatch();
export const useAppSelector = useSelector;
