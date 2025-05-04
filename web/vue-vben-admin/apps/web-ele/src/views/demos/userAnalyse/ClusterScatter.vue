<script setup lang="ts">
import type { ScatterSeriesOption } from 'echarts/charts';
import type {
  DatasetComponentOption,
  DataZoomComponentOption,
  GridComponentOption,
  TooltipComponentOption,
  VisualMapComponentOption,
} from 'echarts/components';

import { onMounted, ref, watch } from 'vue';

import { ScatterChart } from 'echarts/charts';
import {
  DatasetComponent,
  DataZoomComponent,
  GridComponent,
  MarkPointComponent,
  TooltipComponent,
  TransformComponent,
  VisualMapComponent,
} from 'echarts/components';
import * as echarts from 'echarts/core';
import { UniversalTransition } from 'echarts/features';
import { CanvasRenderer } from 'echarts/renderers';

import { requestClient } from '#/api/request';

const props = defineProps({
  secUid: {
    type: String,
    required: true,
    default: '',
  },
});
defineExpose({
  markPoint: markPointInChart,
  reDraw,
});
echarts.use([
  MarkPointComponent,
  DatasetComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  TransformComponent,
  ScatterChart,
  CanvasRenderer,
  UniversalTransition,
  DataZoomComponent,
]);

type EChartsOption = echarts.ComposeOption<
  | DatasetComponentOption
  | DataZoomComponentOption
  | GridComponentOption
  | ScatterSeriesOption
  | TooltipComponentOption
  | VisualMapComponentOption
>;
const chartRef = ref<HTMLElement | null>(null);
const rawDataRef = ref<any[]>([]);
let myChart: echarts.ECharts | null = null;
// 获取数据
async function clusterPlotData(sec_uid: string = props.secUid) {
  const r = await requestClient.get('/userAnalyse/clusterPlotData', {
    params: { sec_uid },
  });

  return r.data;
}

onMounted(async () => {
  if (props.secUid) {
    rawDataRef.value = await clusterPlotData(props.secUid);
  }

  const option = createOption(rawDataRef.value);
  if (chartRef.value) {
    myChart = echarts.init(chartRef.value);
    myChart.setOption(option);
    myChart.on('click', (params: any) => {
      const url = `https://www.douyin.com/user/${params.data[3]}`;
      window.open(url, '_blank');
    });
  }
});
watch(
  () => props.secUid,
  async () => {
    // console.log('new id', newId);
    rawDataRef.value = await clusterPlotData(props.secUid);
    reDraw();
  },
);
function reDraw() {
  if (myChart) {
    myChart?.setOption({
      dataset: [
        {
          source: rawDataRef.value,
        },
      ],
      dataZoom: [
        {
          // 针对 X 轴的 dataZoom 组件 (通常 index 为 0)
          start: 0, // 起始百分比
          end: 100, // 结束百分比
          startValue: null, // 清除或覆盖之前设置的 startValue
          endValue: null, // 清除或覆盖之前设置的 endValue
        },
        {
          // 针对 Y 轴的 dataZoom 组件 (通常 index 为 1)
          start: 0,
          end: 100,
          startValue: null,
          endValue: null,
        },
      ],
      series: {
        markPoint: {
          data: [],
        },
      },
    });
  }
}
async function markPointInChart(selectedId: string = props.secUid) {
  if (!myChart) {
    // 确保 myChart 在外部作用域可访问
    console.error('ECharts instance is not ready.');
    return;
  }

  rawDataRef.value = await clusterPlotData(props.secUid);

  if (!rawDataRef.value || rawDataRef.value.length === 0) {
    console.error('No data available to mark point.');
    return;
  }

  const index = rawDataRef.value.findIndex(
    (item) => item && item[3] === selectedId,
  );

  if (index === -1) {
    console.warn(`Point with ID ${selectedId} not found in the dataset.`);
    return;
  }

  const pointData = rawDataRef.value[index];
  const x = Number.parseFloat(pointData[0]); // 假设 X 坐标在索引 0
  const y = Number.parseFloat(pointData[1]); // 假设 Y 坐标在索引 1
  const pointName = pointData[4] || `ID: ${selectedId.slice(0, 6)}`; // 假设昵称在索引 4

  // --- 4. 计算目标缩放区域 ---
  // 这个值决定了缩放的程度，需要根据你的数据尺度进行调整
  const zoomRange = 20;
  const startX = x - zoomRange / 2;
  const endX = x + zoomRange / 2;
  const startY = y - zoomRange / 2;
  const endY = y + zoomRange / 2;

  // --- 5. 使用 setOption 同时更新标记和缩放，实现动画过渡 ---
  myChart.setOption({
    dataZoom: [
      {
        startValue: startX, // 设置 X 轴数据窗口起点
        endValue: endX, // 设置 X 轴数据窗口终点
      },
      {
        startValue: startY, // 设置 Y 轴数据窗口起点
        endValue: endY, // 设置 Y 轴数据窗口终点
      },
    ],
    // 更新系列数据中的 markPoint 配置
    series: {
      markPoint: {
        itemStyle: {
          color: 'red',
        },
        data: [
          {
            name: pointName, // 标签上显示的名称
            coord: [x, y], // 标记点的坐标
          },
        ],
      },
    },
    animationDurationUpdate: 2000, // 更新动画时长 (单位 ms)
    // animationEasingUpdate: 'cubicInOut', // 更新动画的缓动效果
  });
}
function createOption(data: any) {
  let option: EChartsOption;
  const CLUSTER_COUNT = 21;
  const DIENSIION_CLUSTER_INDEX = 2;
  const COLOR_ALL = [
    '#37A2DA', // 水蓝 (原始)
    '#E06343', // 暖橙红 (原始)
    '#37A354', // 森林绿 (原始)
    '#B55DBA', // 薰衣草紫 (原始)
    '#B5BD48', // 芥末黄 (原始)
    '#8378EA', // 淡紫灰 (原始)
    '#96BFFF', // 天蓝 (原始)
    '#8AC6D1', // 雾霾蓝 (新增)
    '#FFB6C1', // 灰粉 (新增)
    '#98FB98', // 薄荷绿 (新增)
    '#FF7F50', // 浅珊瑚 (新增)
    '#FFFFE0', // 香草黄 (新增)
    '#ADD8E6', // 浅天蓝 (新增)
    '#D8BFD8', // 灰紫 (新增)
    '#6B8E23', // 橄榄绿 (新增)
    '#FFDEAD', // 淡杏色 (新增)
    '#2E8B57', // 灰绿 (新增)
    '#B0E0E6', // 浅灰蓝 (新增)
    '#F5F5DC', // 米白 (新增)
    '#B8860B', // 暗金棕 (新增)
    '#F0F0F0', // 珍珠白 (新增)
  ];
  const pieces: { color: string | undefined; label: string; value: number }[] =
    [];
  for (let i = 0; i < CLUSTER_COUNT; i++) {
    pieces.push({
      value: i,
      label: `cluster ${i}`,
      color: COLOR_ALL[i],
    });
  }
  // eslint-disable-next-line prefer-const
  option = {
    dataset: [
      {
        source: data,
      },
    ],
    tooltip: {
      position: 'top',
    },
    visualMap: {
      show: false,
      type: 'piecewise',
      top: 'middle',
      min: 0,
      max: CLUSTER_COUNT,
      left: '0',
      splitNumber: CLUSTER_COUNT,
      dimension: DIENSIION_CLUSTER_INDEX,
      pieces,
    },
    grid: {
      left: 20,
      right: 20,
      top: 20,
      bottom: 20,
    },
    xAxis: {
      axisLabel: { show: false },
      axisLine: { show: false },
      // splitLine: { show: false },
    },
    yAxis: {
      axisLabel: { show: false },
      axisLine: { show: false },
      // splitLine: { show: false },
    },
    dataZoom: [
      {
        type: 'inside', // 启用内部缩放和平移
        xAxisIndex: [0], // 控制第一个 x 轴
        filterMode: 'filter',
        zoomOnMouseWheel: true, // 滚轮缩放
        moveOnMouseMove: true, // 鼠标拖动平移
        moveOnMouseWheel: false, // 通常滚轮不用于平移
      },
      {
        type: 'inside',
        yAxisIndex: [0], // 控制第一个 y 轴
        filterMode: 'filter',
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false,
      },
    ],
    series: {
      type: 'scatter',
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const dataRow = params.data;
          const nickname = dataRow[4];
          const avatarUrl = dataRow[5];

          return `
      <div class="flex items-center gap-1.5">
        <img
          class="w-8 h-8 rounded-full flex-shrink-0"
          src="${avatarUrl}"
          alt="Avatar"
        />
        <span class="overflow-hidden text-ellipsis whitespace-nowrap">
          ${nickname}
        </span>
      </div>
    `;
        },
      },
      symbolSize: 15,
      itemStyle: {
        borderColor: '#555',
      },
      datasetIndex: 0,
    },
  };
  return option;
}
</script>
<template>
  <div class="flex items-center justify-center">
    <div ref="chartRef" class="h-[500px] w-[1000px]"></div>
  </div>
</template>
