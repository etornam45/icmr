import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/source': 'http://127.0.0.1:8000',
      '/overlay': 'http://127.0.0.1:8000',
      '/events': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/status': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
