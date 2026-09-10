import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

// Served by the FastAPI app at /ops; the build lands inside the Python package
// so the container image and `uv run start` pick it up without a separate host.
export default defineConfig({
  plugins: [react()],
  base: '/ops/',
  build: {
    outDir: '../src/vemsa/ops/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ops/api': 'http://localhost:8000',
    },
  },
});
