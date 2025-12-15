import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  server: {
    proxy: {
      '/luna-llm': {
        target: 'http://117.50.85.179:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/luna-llm/, '')
      },
      '/luna-tts': {
        target: 'http://117.50.85.179:9880',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/luna-tts/, '')
      },
      '/asahi-llm': {
        target: 'http://117.50.85.179:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/asahi-llm/, '')
      },
      '/asahi-tts': {
        target: 'http://117.50.85.179:9881',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/asahi-tts/, '')
      },
      '/baidu-api': {
        target: 'https://fanyi-api.baidu.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/baidu-api/, '')
      }
    }
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
