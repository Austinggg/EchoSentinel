<script lang="ts" setup>
import type { EchartsUIType } from '@vben/plugins/echarts';

import { onMounted, ref } from 'vue';
import axios from 'axios';

import { EchartsUI, useEcharts } from '@vben/plugins/echarts';

const chartRef = ref<EchartsUIType>();
const { renderEcharts } = useEcharts(chartRef);
const loading = ref(false);

// 加载视频来源数据
const loadSourcesData = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/sources');
    
    if (response.data && response.data.code === 200) {
      const { platform_distribution } = response.data.data;
      
      // 渲染图表
      renderEcharts({
        legend: {
          bottom: '2%',
          left: 'center',
        },
        series: [
          {
            animationDelay() {
              return Math.random() * 100;
            },
            animationEasing: 'exponentialInOut',
            animationType: 'scale',
            avoidLabelOverlap: false,
            color: ['#5ab1ef', '#b6a2de', '#67e0e3', '#2ec7c9'], // 保持原有配色
            data: [
              { name: '抖音', value: platform_distribution.douyin },
              { name: 'TikTok', value: platform_distribution.tiktok },
              { name: 'Bilibili', value: platform_distribution.bilibili },
              { name: '用户上传', value: platform_distribution.upload }
            ],
            emphasis: {
              label: {
                fontSize: '12',
                fontWeight: 'bold',
                show: true,
              },
            },
            itemStyle: {
              borderRadius: 10,
              borderWidth: 2,
            },
            label: {
              position: 'center',
              show: false,
            },
            labelLine: {
              show: false,
            },
            name: '视频来源', // 改名为视频来源
            radius: ['40%', '65%'],
            type: 'pie',
          },
        ],
        tooltip: {
          trigger: 'item',
        },
      });
    } else {
      // 如果API调用失败，显示默认数据
      renderDefaultChart();
    }
  } catch (error) {
    console.error('加载视频来源数据失败:', error);
    // 使用默认数据渲染
    renderDefaultChart();
  } finally {
    loading.value = false;
  }
};

// 使用默认数据渲染图表
const renderDefaultChart = () => {
  renderEcharts({
    legend: {
      bottom: '2%',
      left: 'center',
    },
    series: [
      {
        animationDelay() {
          return Math.random() * 100;
        },
        animationEasing: 'exponentialInOut',
        animationType: 'scale',
        avoidLabelOverlap: false,
        color: ['#5ab1ef', '#b6a2de', '#67e0e3', '#2ec7c9'],
        data: [
          { name: '抖音', value: 450 },
          { name: 'TikTok', value: 350 },
          { name: 'Bilibili', value: 280 },
          { name: '用户上传', value: 184 },
        ],
        emphasis: {
          label: {
            fontSize: '12',
            fontWeight: 'bold',
            show: true,
          },
        },
        itemStyle: {
          borderRadius: 10,
          borderWidth: 2,
        },
        label: {
          position: 'center',
          show: false,
        },
        labelLine: {
          show: false,
        },
        name: '视频来源',
        radius: ['40%', '65%'],
        type: 'pie',
      },
    ],
    tooltip: {
      trigger: 'item',
    },
  });
};

onMounted(() => {
  loadSourcesData();
});
</script>

<template>
  <div class="h-56">
    <EchartsUI ref="chartRef" />
  </div>
</template>   