import type { RouteRecordRaw } from 'vue-router';
import { $t } from '#/locales';

const routes: RouteRecordRaw[] = [
  {
    meta: {
      icon: 'mdi:home-variant',
      order: 1,
      title: '主页',
    },
    name: 'Main',
    path: '/main',
    children: [
      // 添加账号上传路由
      {
        meta: {
          icon: 'mdi:account-plus',
          title: '添加账号',
        },
        name: 'AddAccount',
        path: '/main/add-account',
        component: () => import('#/views/mainpage/addAccount.vue'),
      },
      // 这里可以添加其他主页功能模块
      // 在路由配置文件中添加
      {
        path: '/main/analysis-tasks',
        name: 'AnalysisTasks',
        component: () => import('#/views/mainpage/analysis-tasks.vue'),
        meta: {
          title: '分析任务列表',
          icon: 'ant-design:ordered-list-outlined',
        },
        children: [
          {
            meta: {
              title: '用户内容列表',
              hideInMenu: true, // 在菜单中隐藏此路由
            },
            name: 'UserContent',
            path: 'user-content',
            component: () => import('#/views/mainpage/user-content.vue'),
            children: [
              // 添加视频处理进度页面作为子路由
              {
                path: 'processing-details',
                name: 'VideoProcessingDetails',
                component: () => import('#/views/mainpage/video-processing-details.vue'),
                meta: {
                  title: '视频处理进度',
                  hideInMenu: true, // 在菜单中隐藏此路由
                }
              }
            ]
          },
        ],
      },
      {
        meta: {
          icon: 'mdi:settings',
          title: '设置',
        },
        name: 'Settings',
        path: '/main/settings',
        component: () => import('#/views/mainpage/settings.vue'),
      },
    ],
  },
];

export default routes;