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
          hideInMenu: true,
        },
        name: 'NaiveDemos',
        path: '/demos/element',
        component: () => import('#/views/demos/element/index.vue'),
      },
      {
        meta: {
          title: $t('demos.form'),
          hideInMenu: true,
        },
        name: 'BasicForm',
        path: '/demos/form',
        component: () => import('#/views/demos/form/basic.vue'),
      },
      // 数字人检测
      {
        meta: {
          title: $t('demos.aigcDetection'),
        },
        name: 'AIGCdetection',
        path: '/demos/AIGC-detection',
        component: () => import('#/views/demos/AIGC-detection/index.vue'),
      },
      // 视频上传
      {
        meta: {
          title: $t('demos.videoUpload'),
        },
        name: 'VideoUpload',
        path: '/demos/video-upload',
        component: () => import('#/views/demos/content-analysis/index.vue'),
      },
      // 分析记录 - 现在作为父路由
      // 分析记录路由
      {
        meta: {
          title: $t('demos.analysisRecords'),
        },
        name: 'AnalysisRecords',
        path: '/demos/analysis-records',
        component: () =>
          import('#/views/demos/content-analysis/analysisRecords.vue'),
        children: [
          // 文本分析作为分析记录的子路由
          {
            meta: {
              title: $t('demos.contentAnalysis'),
              hideInMenu: true,
            },
            name: 'contentAnalysis',
            path: 'analysis',
            component: () =>
              import('#/views/demos/content-analysis/analysis.vue'),
            // 添加评估理由作为文本分析的子路由
            children: [
              {
                meta: {
                  title: '评估理由详情',
                  hideInMenu: true,
                },
                name: 'AssessmentReason',
                path: 'reason', // 相对路径，实际为/demos/analysis-records/analysis/reason
                component: () =>
                  import('#/views/demos/content-analysis/reason.vue'),
              },
            ],
          },
        ],
      },
      // 知识图谱
      {
        meta: {
          title: $t('demos.KnowledgeGraph'),
        },
        name: 'KnowledgeGraph',
        path: '/demos/KnowledgeGraph',
        component: () => import('#/views/demos/KnowledgeGraph/index.vue'),
      },
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
