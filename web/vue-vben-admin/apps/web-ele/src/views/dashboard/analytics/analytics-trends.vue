<script lang="ts" setup>
import type { EchartsUIType } from '@vben/plugins/echarts';

import { onMounted, ref } from 'vue';
import axios from 'axios';

import { EchartsUI, useEcharts } from '@vben/plugins/echarts';

const chartRef = ref<EchartsUIType>();
const { renderEcharts } = useEcharts(chartRef);
const loading = ref(false);

// 加载趋势数据
const loadTrendsData = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/trends');
    
    if (response.data && response.data.code === 200) {
      const { dates, uploads, high_risk } = response.data.data;
      
      // 使用API数据渲染图表，但保持原有样式
      renderEcharts({
          title: {
          text: '风险总体趋势图',
          left: 'center',
          top: 10,
          textStyle: {
            fontSize: 16
          }
        },
        grid: {
          bottom: 30,
          containLabel: true,
          left: '1%',
          right: '1%',
          top: '2 %',
        },
        series: [
          {
            name: '视频上传数',
            areaStyle: {},
            data: uploads,
            itemStyle: {
              color: '#5ab1ef',
            },
            smooth: true,
            type: 'line',
          },
          {
            name: '高风险视频数',
            areaStyle: {},
            data: high_risk,
            itemStyle: {
              color: '#019680',
            },
            smooth: true,
            type: 'line',
          },
        ],
                legend: {
          data: ['视频上传数', '高风险视频数'],
          bottom: 0,  // 将图例放在底部
        },
        tooltip: {
          axisPointer: {
            lineStyle: {
              color: '#019680',
              width: 1,
            },
          },
          trigger: 'axis',
        },
        xAxis: {
          axisTick: {
            show: false,
          },
          boundaryGap: false,
          data: dates,
          splitLine: {
            lineStyle: {
              type: 'solid',
              width: 1,
            },
            show: true,
          },
          type: 'category',
        },
        yAxis: [
          {
            axisTick: {
              show: false,
            },
            // 动态设置最大值为数据最大值的1.2倍，确保曲线不会触顶
            max: Math.max(...uploads, ...high_risk) * 1.2,
            splitArea: {
              show: true,
            },
            splitNumber: 4,
            type: 'value',
          },
        ],
      });
    } else {
      // 如果API调用失败，回退到使用原始的模拟数据
      console.error('趋势数据获取失败，使用模拟数据');
      renderDefaultChart();
    }
  } catch (error) {
    console.error('加载趋势数据失败:', error);
    // 出错时使用默认数据
    renderDefaultChart();
  } finally {
    loading.value = false;
  }
};

// 使用默认数据渲染图表的备用函数
const renderDefaultChart = () => {
  renderEcharts({
    grid: {
      bottom: 0,
      containLabel: true,
      left: '1%',
      right: '1%',
      top: '2 %',
    },
    series: [
      {
        areaStyle: {},
        data: [
          111, 2000, 6000, 16_000, 33_333, 55_555, 64_000, 33_333, 18_000,
          36_000, 70_000, 42_444, 23_222, 13_000, 8000, 4000, 1200, 333, 222,
          111,
        ],
        itemStyle: {
          color: '#5ab1ef',
        },
        smooth: true,
        type: 'line',
      },
      {
        areaStyle: {},
        data: [
          33, 66, 88, 333, 3333, 6200, 20_000, 3000, 1200, 13_000, 22_000,
          11_000, 2221, 1201, 390, 198, 60, 30, 22, 11,
        ],
        itemStyle: {
          color: '#019680',
        },
        smooth: true,
        type: 'line',
      },
    ],
    tooltip: {
      axisPointer: {
        lineStyle: {
          color: '#019680',
          width: 1,
        },
      },
      trigger: 'axis',
    },
    xAxis: {
      axisTick: {
        show: false,
      },
      boundaryGap: false,
      data: Array.from({ length: 18 }).map((_item, index) => `${index + 6}:00`),
      splitLine: {
        lineStyle: {
          type: 'solid',
          width: 1,
        },
        show: true,
      },
      type: 'category',
    },
    yAxis: [
      {
        axisTick: {
          show: false,
        },
        max: 80_000,
        splitArea: {
          show: true,
        },
        splitNumber: 4,
        type: 'value',
      },
    ],
  });
};

onMounted(() => {
  loadTrendsData();
});
</script>

<template>
  <EchartsUI ref="chartRef" />
</template>