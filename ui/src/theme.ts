// Chart colours. Categorical slots follow a validated palette (adjacent pairs pass
// colour-vision-deficiency separation in both modes); chrome reads Astryx tokens.
import {useEffect, useState} from 'react';

export interface ChartPalette {
  series: [string, string, string, string];
  grid: string;
  axis: string;
  text: string;
  surface: string;
}

const LIGHT: ChartPalette = {
  series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  text: '#898781',
  surface: '#fcfcfb',
};

const DARK: ChartPalette = {
  series: ['#3987e5', '#d95926', '#199e70', '#c98500'],
  grid: '#2c2c2a',
  axis: '#383835',
  text: '#898781',
  surface: '#1a1a19',
};

const query = () => window.matchMedia('(prefers-color-scheme: dark)');

export function useDarkMode(): boolean {
  const [dark, setDark] = useState(() => query().matches);
  useEffect(() => {
    const media = query();
    const onChange = (event: MediaQueryListEvent) => setDark(event.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);
  return dark;
}

export function usePalette(): ChartPalette {
  return useDarkMode() ? DARK : LIGHT;
}

/** Series hue at a wash opacity for area fills. */
export function wash(hex: string, alpha = 0.12): string {
  const value = parseInt(hex.slice(1), 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
