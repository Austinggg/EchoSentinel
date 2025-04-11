import type { RouteRecordRaw } from 'vue-router';

import { $t } from '#/locales';

const routes: RouteRecordRaw[] = [
  {
    meta: {
      icon: 'ic:baseline-view-in-ar',
      keepAlive: true,
      order: 1000,
      title: $t('demos.title'),
    },
    name: 'Demos',
    path: '/demos',
    children: [
      {
        meta: {
          title: $t('demos.elementPlus'),
        },
        name: 'NaiveDemos',
        path: '/demos/element',
        component: () => import('#/views/demos/element/index.vue'),
      },
      {
        meta: {
          title: $t('demos.form'),
        },
        name: 'BasicForm',
        path: '/demos/form',
        component: () => import('#/views/demos/form/basic.vue'),
      },
      //数字人检测
      {
        meta: {
          title: $t('demos.aigcDetection'),
        },
        name: 'AIGCdetection',
        path: '/demos/AIGC-detection',
        component: () => import('#/views/demos/AIGC-detection/index.vue'),
      },
      // 文本处理
      {
        meta: {
          title: $t('demos.contentAnalysis'),
        },
        name: 'VideoUpload',
        path: '/demos/video-upload',
        component: () => import('#/views/demos/content-analysis/index.vue'),
      },
      // 知识图谱
      {
        meta: {
          title: $t('demos.KnowledgeGraph'),
        },
        name: 'KnowledgeGraph',
        path: '/demos/KnowledgeGraph',
        component: () => import('#/views/demos/KnowledgeGraph/index.vue'),
      {
        meta: {
          title: $t('demos.userAnalyse'),
        },
        name: 'userAnalyse',
        path: '/demos/userAnalyse',
        component: () => import('#/views/demos/userAnalyse/index.vue'),
      },
    ],
  },
];

export default routes;
