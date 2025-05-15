<script lang="ts" setup>
import type { AnalysisOverviewItem } from '@vben/common-ui';
import type { TabOption } from '@vben/types';

import { ref, computed, onMounted } from 'vue';
import axios from 'axios';
import { 
  VideoPlay,      // 视频总量
  DataAnalysis,   // 数据分析
  Avatar,         // 数字人
  User            // 用户
} from '@element-plus/icons-vue';

import {
  AnalysisChartCard,
  AnalysisChartsTabs,
  AnalysisOverview,
} from '@vben/common-ui';
import {
  SvgBellIcon,
  SvgCakeIcon,
  SvgCardIcon,
  SvgDownloadIcon,
} from '@vben/icons';

import AnalyticsTrends from './analytics-trends.vue';
import AnalyticsVisitsData from './analytics-visits-data.vue';
import AnalyticsVisitsSales from './analytics-visits-sales.vue';
import AnalyticsVisitsSource from './analytics-visits-source.vue';
import AnalyticsVisits from './analytics-visits.vue';

// 添加数据存储和加载状态
const analyticsData = ref({
  total_videos: 0,
  risk_distribution: { low: 0, medium: 0, high: 0, processing: 0 },
  recent_videos: 0,
  task_stats: { total: 0, completed: 0, failed: 0, completion_rate: 0 },
  total_users: 0,
});
const loading = ref(false);

// 加载数据的函数
const loadAnalyticsData = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/overview');
    if (response.data && response.data.code === 200) {
      analyticsData.value = response.data.data;
    } else {
      console.error('无法获取统计数据:', response.data?.message || '未知错误');
    }
  } catch (error) {
    console.error('获取统计数据失败:', error);
  } finally {
    loading.value = false;
  }
};

// 将静态数据转为计算属性，使用API数据
const overviewItems = computed<AnalysisOverviewItem[]>(() => [
  {
    icon: VideoPlay,
    title: '视频总量',
    totalTitle: '风险视频',
    totalValue:
      analyticsData.value.risk_distribution.high +
      analyticsData.value.risk_distribution.medium,
    value: analyticsData.value.total_videos,
  },
  {
    icon: DataAnalysis,
    title: '本周新增',
    totalTitle: '完成分析',
    totalValue: analyticsData.value.task_stats.completed,
    value: analyticsData.value.recent_videos,
  },
  {
    icon: Avatar,
    title: '数字人用户',
    totalTitle: '占分析用户',
    totalValue:
      Math.round(
        (analyticsData.value.user_stats?.digital_human_users /
          analyticsData.value.user_stats?.analysed_users) *
          100,
      ) || 0,
    value: analyticsData.value.user_stats?.digital_human_users || 0,
    suffix: '%',
  },
  {
    icon: User,
    title: '分析用户',
    totalTitle: '高风险用户',
    totalValue: analyticsData.value.user_stats?.high_risk_users || 0,
    value:
      analyticsData.value.user_stats?.analysed_users ||
      analyticsData.value.total_users ||
      0,
  },
]);

// 组件挂载时加载数据
onMounted(() => {
  loadAnalyticsData();
});

const chartTabs: TabOption[] = [
  {
    label: '趋势分析',
    value: 'trends',
  },
  {
    label: '内容评估',
    value: 'visits',
  },
];
</script>

<template>
  <div class="p-5">
    <AnalysisOverview :items="overviewItems" :loading="loading" />
    <AnalysisChartsTabs :tabs="chartTabs" class="mt-5">
      <template #trends>
        <AnalyticsTrends />
      </template>
      <template #visits>
        <AnalyticsVisits />
      </template>
    </AnalysisChartsTabs>

    <div class="mt-5 w-full md:flex">
      <AnalysisChartCard class="mt-5 md:mr-4 md:mt-0 md:w-1/3" title="风险分布">
        <AnalyticsVisitsData />
      </AnalysisChartCard>
      <AnalysisChartCard class="mt-5 md:mr-4 md:mt-0 md:w-1/3" title="视频来源">
        <AnalyticsVisitsSource />
      </AnalysisChartCard>
      <AnalysisChartCard class="mt-5 md:mt-0 md:w-1/3" title="数字人占比">
        <AnalyticsVisitsSales />
      </AnalysisChartCard>
    </div>
  </div>
</template>
