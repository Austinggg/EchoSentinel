<script setup lang="ts">
import * as echarts from 'echarts/core';
import {
  DatasetComponent,
  type DatasetComponentOption,
  TooltipComponent,
  type TooltipComponentOption,
  GridComponent,
  type GridComponentOption,
  VisualMapComponent,
  type VisualMapComponentOption,
  TransformComponent,
} from 'echarts/components';
import { ScatterChart, type ScatterSeriesOption } from 'echarts/charts';
import { UniversalTransition } from 'echarts/features';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  DatasetComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  TransformComponent,
  ScatterChart,
  CanvasRenderer,
  UniversalTransition,
]);

type EChartsOption = echarts.ComposeOption<
  | DatasetComponentOption
  | TooltipComponentOption
  | GridComponentOption
  | VisualMapComponentOption
  | ScatterSeriesOption
>;
import { onMounted, ref } from 'vue';
import { requestClient } from '#/api/request';

const chartRef = ref<HTMLElement | null>(null);
const rawDataRef = ref<any[]>([]);
let myChart: echarts.ECharts | null = null;
//获取数据
async function clusterPlotData() {
  let r = await requestClient.get('/userAnalyse/clusterPlotData');
  return r.data;
}
onMounted(async () => {
  rawDataRef.value = await clusterPlotData();

  let option = createOption(rawDataRef.value);
  if (chartRef.value) {
    myChart = echarts.init(chartRef.value);
    myChart.setOption(option);
    myChart.on('click', function (params: any) {
      // console.log(params.data);
      let url = `https://www.douyin.com/user/${params.data[3]}`;
      window.open(url, '_blank');
    });
  }
});
function markPoint(
  selectedId: string = 'MS4wLjABAAAAd9fAgiTlM5gtObeuPaTnm68ID1or-XBmUKpGcprkzw9AOLwMZPCliu8IEsOPIyNd',
) {
  let index = -1;

  if (rawDataRef.value) {
    index = rawDataRef.value.findIndex((item) => item[3] === selectedId);
    console.log('找到的索引:', index);
  }
  if (myChart) {
    // 显示 tooltip
    myChart.dispatchAction({
      type: 'showTip',
      seriesIndex: 0,
      dataIndex: index - 1,
    });
    myChart.dispatchAction({
      type: 'highlight',
      seriesIndex: 0,
      dataIndex: index - 1,
    });
  }
}
function createOption(data: any) {
  var option: EChartsOption;
  var CLUSTER_COUNT = 21;
  var DIENSIION_CLUSTER_INDEX = 2;
  var COLOR_ALL = [
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
  var pieces: { value: number; label: string; color: string | undefined }[] =
    [];
  for (var i = 0; i < CLUSTER_COUNT; i++) {
    pieces.push({
      value: i,
      label: 'cluster ' + i,
      color: COLOR_ALL[i],
    });
  }
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
      pieces: pieces,
    },
    grid: {
      left: 10,
      right: 10,
      top: 10,
      bottom: 10,
    },
    xAxis: {
      axisLabel: { show: false },
      axisLine: { show: false },
      splitLine: { show: false },
    },
    yAxis: {
      axisLabel: { show: false },
      axisLine: { show: false },
      splitLine: { show: false },
    },
    series: {
      type: 'scatter',
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          let dataRow = params.data;
          let nickname = dataRow[4];
          let avatarUrl = dataRow[5];

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

const childMethod = (x: string = 'hello') => {
  console.log('子组件方法被触发');
  console.log(`参数：${x}`);
};
defineExpose({
  childMethod,
  markPoint,
});
</script>
<template>
  <div class="flex items-center justify-center">
    <div ref="chartRef" class="h-[500px] w-[1000px]"></div>
  </div>
</template>
