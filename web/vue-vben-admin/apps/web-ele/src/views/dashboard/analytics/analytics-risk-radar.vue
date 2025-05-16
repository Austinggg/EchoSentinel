<script lang="ts" setup>
import type { EchartsUIType } from '@vben/plugins/echarts';

import { onMounted, ref } from 'vue';
import axios from 'axios';

import { EchartsUI, useEcharts } from '@vben/plugins/echarts';

const chartRef = ref<EchartsUIType>();
const { renderEcharts } = useEcharts(chartRef);
const loading = ref(false);

// 加载视频风险分布数据
const loadRiskDistribution = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/risk-distribution');

    if (response.data && response.data.code === 200) {
      const { risk_distribution } = response.data.data;

      // 从API获取数据并转换为雷达图格式
      const digitalData = [
        risk_distribution.high.digital,
        risk_distribution.medium.digital,
        risk_distribution.low.digital,
      ];

      const nonDigitalData = [
        risk_distribution.high.non_digital,
        risk_distribution.medium.non_digital,
        risk_distribution.low.non_digital,
      ];

      // 计算总量用于标题显示
      const totalVideos =
        digitalData.reduce((a, b) => a + b, 0) +
        nonDigitalData.reduce((a, b) => a + b, 0);
      const totalDigital = digitalData.reduce((a, b) => a + b, 0);

      // 渲染图表
      renderEcharts({
        title: {
          subtext: `数字人视频占比: ${Math.round((totalDigital / totalVideos) * 100) || 0}%`,
          left: 'center',
          top: 0,
          textStyle: {
            fontSize: 14,
          },
          subtextStyle: {
            fontSize: 12,
            color: '#F56C6C',
          },
        },
        legend: {
          bottom: 0,
          data: ['数字人视频', '真实视频'],
        },
        radar: {
          indicator: [
            {
              name: '高风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2,
            },
            {
              name: '中风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2,
            },
            {
              name: '低风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2,
            },
          ],
          radius: '60%',
          center: ['50%', '55%'], // 调整中心点位置，为标题腾出空间
          splitNumber: 5,
          axisName: {
            color: '#666',
            fontSize: 12,
          },
        },
        series: [
          {
            areaStyle: {
              opacity: 0.8,
              shadowBlur: 0,
              shadowColor: 'rgba(0,0,0,.2)',
              shadowOffsetX: 0,
              shadowOffsetY: 10,
            },
            data: [
              {
                itemStyle: {
                  color: '#67C23A', // 真实视频使用绿色
                },
                name: '真实视频',
                value: nonDigitalData,
              },
              {
                itemStyle: {
                  color: '#F56C6C', // 数字人使用红色
                },
                name: '数字人视频',
                value: digitalData,
              },
            ],
            itemStyle: {
              borderRadius: 10,
              borderWidth: 2,
            },
            symbolSize: 0,
            type: 'radar',
          },
        ],
        tooltip: {
          trigger: 'item',
          formatter: function (params) {
            return `${params.name}<br/>
                    <span style="color:#F56C6C;">高风险视频:</span> ${params.value[0]}个<br/>
                    <span style="color:#E6A23C;">中风险视频:</span> ${params.value[1]}个<br/>
                    <span style="color:#67C23A;">低风险视频:</span> ${params.value[2]}个`;
          },
        },
      });
    } else {
      throw new Error(response.data?.message || '获取风险分布数据失败');
    }
  } catch (error) {
    console.error('加载风险分布数据失败:', error);

    // 使用默认数据渲染
    renderDefaultChart();
  } finally {
    loading.value = false;
  }
};

// 使用默认数据渲染图表
const renderDefaultChart = () => {
  renderEcharts({
    title: {
      text: '视频风险分布',
      subtext: '数字人视频占比: 35%',
      left: 'center',
      top: 0,
      textStyle: {
        fontSize: 14,
      },
      subtextStyle: {
        fontSize: 12,
        color: '#F56C6C',
      },
    },
    legend: {
      bottom: 0,
      data: ['数字人视频', '真实视频'],
    },
    radar: {
      indicator: [
        {
          name: '高风险',
          max: 100,
        },
        {
          name: '中风险',
          max: 100,
        },
        {
          name: '低风险',
          max: 100,
        },
      ],
      radius: '60%',
      center: ['50%', '55%'],
      splitNumber: 5,
    },
    series: [
      {
        areaStyle: {
          opacity: 0.8,
          shadowBlur: 0,
          shadowColor: 'rgba(0,0,0,.2)',
          shadowOffsetX: 0,
          shadowOffsetY: 10,
        },
        data: [
          {
            itemStyle: {
              color: '#67C23A',
            },
            name: '真实视频',
            value: [25, 60, 80],
          },
          {
            itemStyle: {
              color: '#F56C6C',
            },
            name: '数字人视频',
            value: [65, 40, 20],
          },
        ],
        itemStyle: {
          borderRadius: 10,
          borderWidth: 2,
        },
        symbolSize: 0,
        type: 'radar',
      },
    ],
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        return `${params.name}<br/>
                <span style="color:#F56C6C;">高风险视频:</span> ${params.value[0]}个<br/>
                <span style="color:#E6A23C;">中风险视频:</span> ${params.value[1]}个<br/>
                <span style="color:#67C23A;">低风险视频:</span> ${params.value[2]}个`;
      },
    },
  });
};

onMounted(() => {
  loadRiskDistribution();
});
</script>

<template>
  <div class="h-56">
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
    </div>
    <EchartsUI ref="chartRef" />
  </div>
</template>

<style scoped>
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.7);
  z-index: 10;
}

.loading-spinner {
  width: 30px;
  height: 30px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #409eff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
