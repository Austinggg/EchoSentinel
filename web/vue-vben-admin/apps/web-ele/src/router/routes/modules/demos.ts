import type { RouteRecordRaw } from 'vue-router';

import { $t } from '#/locales';

const routes: RouteRecordRaw[] = [
  {
    meta: {
      icon: 'ic:baseline-view-in-ar',  // 保持原有图标
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
          icon: 'ep:element-plus',  // Element Plus图标
          hideInMenu: true,
        },
        name: 'NaiveDemos',
        path: '/demos/element',
        component: () => import('#/views/demos/element/index.vue'),
      },
      {
        meta: {
          title: $t('demos.form'),
          icon: 'mdi:form-select',  // 表单图标
          hideInMenu: true,
        },
        name: 'BasicForm',
        path: '/demos/form',
        component: () => import('#/views/demos/form/basic.vue'),
      },
      {
        meta: {
          title: $t('demos.videoUpload'),
          icon: 'mdi:video-plus',  // 视频上传图标
        },
        name: 'VideoUpload',
        path: '/demos/video-upload',
        component: () => import('#/views/demos/content-analysis/index.vue'),
      },
      // 数字人检测
      {
        meta: {
          title: $t('demos.aigcDetection'),
          icon: 'mdi:robot-outline',  // AI/机器人图标
        },
        name: 'AIGCdetection',
        path: '/demos/AIGC-detection',
        component: () => import('#/views/demos/AIGC-detection/index.vue'),
      },
      // 分析记录 - 现在作为父路由
      {
        meta: {
          title: $t('demos.analysisRecords'),
          icon: 'mdi:file-document-multiple',  // 文档记录图标
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
              icon: 'mdi:text-search',  // 文本分析图标
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
                  icon: 'mdi:clipboard-text-search',  // 评估详情图标
                  hideInMenu: true,
                },
                name: 'AssessmentReason',
                path: 'reason',
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
          title: '事实核查',
          icon: 'mdi:graph',  // 知识图谱图标
        },
        name: 'KnowledgeGraph',
        path: '/demos/KnowledgeGraph',
        component: () => import('#/views/demos/KnowledgeGraph/index.vue'),
      },
      {
        meta: {
          title: $t('demos.userAnalyse'),
          icon: 'mdi:account-search',  // 用户分析图标
        },
        name: 'userAnalyse',
        path: '/demos/userAnalyse',
        component: () => import('#/views/demos/userAnalyse/index.vue'),
      },
    ],
  },
];

export default routes;