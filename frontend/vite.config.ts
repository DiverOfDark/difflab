import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

/**
 * The page is built into the Python package, so one `uvicorn` serves the API and the app from the
 * same origin - no CORS, no second port, and a container ships one thing.
 *
 * In development `npm run dev` serves the page and passes everything the API owns through to the
 * Python process, so the page reloads on save while watching a real render.
 */
const API = 'http://localhost:8765'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: '../difflab/api/static',
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(
      ['/api', '/runs', '/output', '/uploads'].map((path) => [path, { target: API, changeOrigin: true }]),
    ),
  },
})
