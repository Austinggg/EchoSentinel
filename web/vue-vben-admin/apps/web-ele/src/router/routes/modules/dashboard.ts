import type { RouteRecordRaw } from 'vue-router';

import { $t } from '#/locales';

const routes: RouteRecordRaw[] = [
  {
    meta: {
      icon: 'lucide:layout-dashboard',
      order: -1,
      title: $t('page.dashboard.title'),
    },
    name: 'Dashboard',
    path: '/dashboard',
    children: [
      {
        name: 'Analytics',
        path: '/analytics',
        component: () => import('#/views/dashboard/analytics/index.vue'),
        meta: {
          affixTab: true,
          icon: 'lucide:area-chart',
          title: $t('page.dashboard.analytics'),
        },
      },
      {
        name: 'Workspace',
        path: '/workspace',
        component: () => import('#/views/dashboard/workspace/index.vue'),
        meta: {
          icon: 'carbon:workspace',
          title: $t('page.dashboard.workspace'),
        },
      },
      {
        path: 'risk-monitor',
        name: 'RiskMonitor',
        component: () => import('#/views/dashboard/risk-monitor/index.vue'),
        meta: {
          title: '风险监控中心',
          icon: 'clarity:alert-line',
        },
      },
      {
        path: 'system-performance',
        name: 'SystemPerformance',
        component: () => import('#/views/dashboard/system-performance/index.vue'),
        meta: {
          title: '系统性能统计',
          icon: 'lucide:activity',
        },
      },
    ],
  },
];

export default routes;
