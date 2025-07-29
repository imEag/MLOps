import { useState, useEffect } from 'react';
import { theme } from 'antd';

const { useToken } = theme;

const useBreakpoint = () => {
  const { token } = useToken();
  const [breakpoint, setBreakpoint] = useState('');

  useEffect(() => {
    const checkBreakpoint = () => {
      const { screen } = window;
      if (screen.width < token.screenSM) {
        setBreakpoint('xs');
      } else if (screen.width < token.screenMD) {
        setBreakpoint('sm');
      } else if (screen.width < token.screenLG) {
        setBreakpoint('md');
      } else if (screen.width < token.screenXL) {
        setBreakpoint('lg');
      } else {
        setBreakpoint('xl');
      }
    };

    checkBreakpoint();
    window.addEventListener('resize', checkBreakpoint);

    return () => window.removeEventListener('resize', checkBreakpoint);
  }, [token]);

  return breakpoint;
};

export const useIsMobile = () => {
  const breakpoint = useBreakpoint();
  return breakpoint === 'xs' || breakpoint === 'sm';
};

export default useBreakpoint;
