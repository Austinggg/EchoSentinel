import { defineConfig } from '@vben/vite-config';

import ElementPlus from 'unplugin-element-plus/vite';
import path from 'path'; 
export default defineConfig(async () => {
  return {
    application: {},
    vite: {
      resolve: {
        alias: { // 新增路径别名配置
          '@': path.resolve(__dirname, './src'),
        },
      },
      plugins: [
        ElementPlus({
          format: 'esm',
        }),
      ],

      server: {
        proxy: {
          '/api': {
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api/, ''),
            // mock代理目标地址
            // target: 'http://localhost:5320/api',
            target: 'http://localhost:8000/api', //flask
            ws: true,
          },
        },
      },
    },
  };
});
