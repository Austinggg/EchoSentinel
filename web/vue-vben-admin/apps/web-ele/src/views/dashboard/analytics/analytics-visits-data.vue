<script lang="ts" setup>
import type { EchartsUIType } from '@vben/plugins/echarts';

import { onMounted, ref } from 'vue';
import axios from 'axios';

import { EchartsUI, useEcharts } from '@vben/plugins/echarts';

const chartRef = ref<EchartsUIType>();
const { renderEcharts } = useEcharts(chartRef);
const loading = ref(false);

// 加载风险分布数据
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
        risk_distribution.low.digital
      ];
      
      const nonDigitalData = [
        risk_distribution.high.non_digital,
        risk_distribution.medium.non_digital,
        risk_distribution.low.non_digital
      ];
      
      // 渲染图表
      renderEcharts({
        legend: {
          bottom: 0,
          data: ['数字人', '非数字人'],
        },
        radar: {
          indicator: [
            {
              name: '高风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2
            },
            {
              name: '中风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2
            },
            {
              name: '低风险',
              max: Math.max(...digitalData, ...nonDigitalData) * 1.2
            },
          ],
          radius: '60%',
          splitNumber: 5,
          axisName: {
            color: '#999'
          }
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
                  color: '#F56C6C', // 数字人使用红色
                },
                name: '数字人',
                value: digitalData,
              },
              {
                itemStyle: {
                  color: '#67C23A', // 非数字人使用绿色
                },
                name: '非数字人',
                value: nonDigitalData,
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
          formatter: function(params) {
            return `${params.name}<br/>
                    高风险: ${params.value[0]}<br/>
                    中风险: ${params.value[1]}<br/>
                    低风险: ${params.value[2]}`;
          }
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
    legend: {
      bottom: 0,
      data: ['数字人', '非数字人'],
    },
    radar: {
      indicator: [
        {
          name: '高风险',
          max: 100
        },
        {
          name: '中风险',
          max: 100
        },
        {
          name: '低风险',
          max: 100
        },
      ],
      radius: '60%',
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
              color: '#F56C6C',
            },
            name: '数字人',
            value: [65, 40, 20],
          },
          {
            itemStyle: {
              color: '#67C23A',
            },
            name: '非数字人',
            value: [25, 60, 80],
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
      trigger: 'item'
    },
  });
};

onMounted(() => {
  loadRiskDistribution();
});
</script>

<template>
  <div class="h-56">
    <EchartsUI ref="chartRef" />
  </div>
</template>