<template>
  <div class="system-performance">
    <div class="performance-header">
      <h2>系统性能统计</h2>
      <el-button @click="refreshData" :icon="Refresh" type="primary">
        刷新数据
      </el-button>
    </div>

    <!-- 概览卡片 -->
    <div class="overview-cards">
      <el-card class="metric-card">
        <div class="metric-content">
          <div class="metric-icon">
            <el-icon><Cpu /></el-icon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ systemOverview.total_processing_tasks }}</div>
            <div class="metric-label">总处理任务</div>
          </div>
        </div>
      </el-card>

      <el-card class="metric-card">
        <div class="metric-content">
          <div class="metric-icon">
            <el-icon><VideoPlay /></el-icon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ systemOverview.total_videos_processed }}</div>
            <div class="metric-label">已处理本地视频</div>
          </div>
        </div>
      </el-card>

      <el-card class="metric-card">
        <div class="metric-content">
          <div class="metric-icon">
            <el-icon><Timer /></el-icon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ 263.45 }}s</div>
            <div class="metric-label">平均处理时间</div>
          </div>
        </div>
      </el-card>

      <el-card class="metric-card">
        <div class="metric-content">
          <div class="metric-icon token-icon">
            <el-icon><Coin /></el-icon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ formatNumber(totalTokenUsage) }}</div>
            <div class="metric-label">总Token使用量</div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 趋势图 - 单独一行 -->
    <div class="trend-section">
      <el-card class="trend-card">
        <template #header>
          <span>7天处理趋势</span>
        </template>
        <div ref="trendChart" class="trend-chart"></div>
      </el-card>
    </div>

    <!-- 图表区域 - 三个图表一行 -->
    <div class="charts-container">
      <!-- 任务性能统计 -->
      <el-card class="chart-card">
        <template #header>
          <span>任务性能统计</span>
        </template>
        <div ref="taskPerformanceChart" class="chart"></div>
      </el-card>

      <!-- 日志分布 -->
      <el-card class="chart-card">
        <template #header>
          <span>系统日志分布</span>
        </template>
        <div ref="logChart" class="chart"></div>
      </el-card>

      <!-- Token使用分布 -->
      <el-card class="chart-card">
        <template #header>
          <span>Token使用分布</span>
        </template>
        <div ref="tokenChart" class="chart"></div>
      </el-card>
    </div>

    <!-- 详细统计表格 - 左右布局 -->
    <div class="tables-container">
      <!-- 任务详情表 -->
      <el-card class="table-card">
        <template #header>
          <span>任务处理详情</span>
        </template>
        <el-table :data="taskPerformance" style="width: 100%" size="small">
          <el-table-column prop="task_type" label="任务类型" width="100">
            <template #default="{ row }">
              <el-tag size="small">{{ getTaskName(row.task_type) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="avg_duration" label="平均耗时(秒)" width="90" />
          <el-table-column prop="total_tasks" label="总数" width="60" />
          <el-table-column prop="completed_tasks" label="成功" width="60" />
          <el-table-column prop="failed_tasks" label="失败" width="60" />
          <el-table-column prop="success_rate" label="成功率" width="120">
            <template #default="{ row }">
              <el-progress
                :percentage="row.success_rate"
                :color="getSuccessRateColor(row.success_rate)"
                :show-text="false"
                :stroke-width="6"
              />
              <span style="margin-left: 8px; font-size: 12px">{{ row.success_rate }}%</span>
            </template>
          </el-table-column>
          <el-table-column prop="avg_attempts" label="平均重试" width="80" />
        </el-table>
      </el-card>

      <!-- 评估项性能表 -->
      <el-card class="table-card">
        <template #header>
          <span>内容评估性能</span>
        </template>
        <el-table :data="assessmentData" style="width: 100%" size="small">
          <el-table-column prop="name" label="评估项" width="140" />
          <el-table-column prop="avg_time" label="耗时(秒)" width="80" />
          <el-table-column prop="success_rate" label="成功率" width="120">
            <template #default="{ row }">
              <el-progress
                :percentage="row.success_rate"
                :color="getSuccessRateColor(row.success_rate)"
                :show-text="false"
                :stroke-width="6"
              />
              <span style="margin-left: 8px; font-size: 12px">{{ row.success_rate }}%</span>
            </template>
          </el-table-column>
          <el-table-column prop="token_usage" label="Token" width="80">
            <template #default="{ row }">
              <span style="font-size: 12px">{{ formatNumber(row.token_usage) }}</span>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>

    <!-- 错误分析 -->
    <el-card class="error-analysis-card">
      <template #header>
        <span>错误分析</span>
      </template>
      <div v-if="errorAnalysis.recent_errors.length > 0">
        <h4>最近错误日志</h4>
        <el-timeline>
          <el-timeline-item
            v-for="error in errorAnalysis.recent_errors"
            :key="error.timestamp"
            :timestamp="formatDateTime(error.timestamp)"
            type="danger"
          >
            <div class="error-item">
              <el-tag type="danger" size="small">{{ error.task_type }}</el-tag>
              <span class="error-message">{{ error.message }}</span>
              <span class="error-file">文件: {{ error.video_filename }}</span>
            </div>
          </el-timeline-item>
        </el-timeline>
      </div>
      <div v-else class="no-errors">
        <el-result icon="success" title="暂无错误" sub-title="系统运行良好">
        </el-result>
      </div>
    </el-card>
  </div>
</template>

<script setup>
// ... 脚本部分保持不变 ...
import { ref, onMounted, nextTick, computed } from 'vue';
import { 
  ElMessage, 
  ElCard, 
  ElButton, 
  ElTable, 
  ElTableColumn, 
  ElTag, 
  ElProgress, 
  ElTimeline, 
  ElTimelineItem, 
  ElResult,
  ElIcon
} from 'element-plus';
import { Refresh, Cpu, VideoPlay, Timer, Coin } from '@element-plus/icons-vue';
import axios from 'axios';
import * as echarts from 'echarts';

// 数据定义
const loading = ref(false);
const taskPerformance = ref([]);
const dailyTrends = ref({ dates: [], processing_counts: [], avg_durations: [] });
const logDistribution = ref({});
const assessmentPerformance = ref({});
const systemOverview = ref({
  total_processing_tasks: 0,
  total_videos_processed: 0,
  avg_video_processing_time: 0
});
const errorAnalysis = ref({
  error_distribution: [],
  failure_stats: [],
  recent_errors: []
});

// Token使用量数据（前端模拟）
const totalTokenUsage = ref(1248500);
const tokenUsageByTask = ref({
  transcription: 0,           // 视频转录不消耗Token
  extract: 389200,
  summary: 156800,
  assessment: 289400,
  classify: 98300,
  report: 69200,
  digital_human: 180000,
  fact_check: 360000
});

// 图表引用
const taskPerformanceChart = ref(null);
const trendChart = ref(null);
const logChart = ref(null);
const tokenChart = ref(null);

// 加载数据
const loadPerformanceData = async () => {
  try {
    loading.value = true;
    
    // 获取系统性能数据
    const performanceResponse = await axios.get('/api/analytics/system-performance');
    if (performanceResponse.data.code === 200) {
      const data = performanceResponse.data.data;
      taskPerformance.value = data.task_performance;
      dailyTrends.value = data.daily_trends;
      logDistribution.value = data.log_distribution;
      assessmentPerformance.value = data.assessment_performance;
      systemOverview.value = data.system_overview;
    }

    // 获取错误分析数据
    const errorResponse = await axios.get('/api/analytics/error-analysis');
    if (errorResponse.data.code === 200) {
      errorAnalysis.value = errorResponse.data.data;
    }

    // 模拟更新Token使用量
    updateTokenUsage();

    // 等待DOM更新后渲染图表
    await nextTick();
    renderCharts();

  } catch (error) {
    console.error('加载性能数据失败:', error);
    ElMessage.error('加载数据失败');
  } finally {
    loading.value = false;
  }
};

// 模拟Token使用量更新
const updateTokenUsage = () => {
  // 基于任务数量模拟Token使用量
  let total = 0;
  taskPerformance.value.forEach(task => {
    const baseTokens = {
      transcription: 0,         // 视频转录不消耗Token
      extract: 1500,
      summary: 800,
      assessment: 1200,
      classify: 500,
      report: 300,
      digital_human: 2000,      // 数字人检测Token使用量较高
      fact_check: 3500          // 事实核查Token使用量最高
    };
    
    const tokens = (baseTokens[task.task_type] || 500) * task.completed_tasks;
    tokenUsageByTask.value[task.task_type] = tokens;
    total += tokens;
  });
  
  totalTokenUsage.value = total;
};

// 渲染图表
const renderCharts = () => {
  if (taskPerformanceChart.value) renderTaskPerformanceChart();
  if (trendChart.value) renderTrendChart();
  if (logChart.value) renderLogChart();
  if (tokenChart.value) renderTokenChart();
};

// 任务性能图表
const renderTaskPerformanceChart = () => {
  const chart = echarts.init(taskPerformanceChart.value);
  
  const option = {
    title: {
      text: '任务平均处理时间',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    xAxis: {
      type: 'category',
      data: taskPerformance.value.map(item => getTaskName(item.task_type))
    },
    yAxis: {
      type: 'value',
      name: '时间(秒)'
    },
    series: [{
      data: taskPerformance.value.map(item => ({
        value: item.avg_duration,
        itemStyle: {
          color: item.success_rate >= 95 ? '#67C23A' : 
                 item.success_rate >= 90 ? '#E6A23C' : '#F56C6C'
        }
      })),
      type: 'bar',
      showBackground: true,
      backgroundStyle: {
        color: 'rgba(180, 180, 180, 0.2)'
      }
    }]
  };
  
  chart.setOption(option);
};

// 趋势图表
const renderTrendChart = () => {
  const chart = echarts.init(trendChart.value);
  
  const option = {
    title: {
      text: '处理量与平均耗时趋势',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['处理量', '平均耗时'],
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: dailyTrends.value.dates
    },
    yAxis: [
      {
        type: 'value',
        name: '处理量',
        position: 'left'
      },
      {
        type: 'value',
        name: '耗时(秒)',
        position: 'right'
      }
    ],
    series: [
      {
        name: '处理量',
        type: 'bar',
        data: dailyTrends.value.processing_counts,
        itemStyle: { color: '#409EFF' }
      },
      {
        name: '平均耗时',
        type: 'line',
        yAxisIndex: 1,
        data: dailyTrends.value.avg_durations,
        itemStyle: { color: '#67C23A' }
      }
    ]
  };
  
  chart.setOption(option);
};

// 日志分布图表
const renderLogChart = () => {
  const chart = echarts.init(logChart.value);
  
  const logColors = {
    INFO: '#409EFF',
    WARNING: '#E6A23C',
    ERROR: '#F56C6C'
  };
  
  const data = Object.entries(logDistribution.value).map(([level, count]) => ({
    name: level,
    value: count,
    itemStyle: { color: logColors[level] || '#909399' }
  }));
  
  const option = {
    title: {
      text: '系统日志级别分布',
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    series: [{
      type: 'pie',
      radius: '70%',
      data: data,
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  };
  
  chart.setOption(option);
};

// Token使用分布图表
const renderTokenChart = () => {
  const chart = echarts.init(tokenChart.value);
  
  const data = Object.entries(tokenUsageByTask.value).map(([task, tokens]) => ({
    name: getTaskName(task),
    value: tokens
  }));
  
  const option = {
    title: {
      text: 'Token使用分布',
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} tokens ({d}%)'
    },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      data: data,
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  };
  
  chart.setOption(option);
};

// 工具函数
const getTaskName = (taskType) => {
  const names = {
    transcription: '视频转录',
    extract: '信息提取',
    summary: '生成摘要',
    assessment: '内容评估',
    classify: '风险分类',
    report: '威胁报告',
    digital_human: '数字人检测',
    fact_check: '事实核查'
  };
  return names[taskType] || taskType;
};

const getSuccessRateColor = (rate) => {
  if (rate >= 95) return '#67C23A';
  if (rate >= 90) return '#E6A23C';
  return '#F56C6C';
};

const formatNumber = (num) => {
  return num ? num.toLocaleString() : '0';
};

const formatDateTime = (timestamp) => {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
};

// 评估项数据
const assessmentData = computed(() => {
  const assessmentNames = {
    p1: '背景信息充分性',
    p2: '背景信息准确性',
    p3: '内容完整性',
    p4: '意图正当性',
    p5: '发布者信誉',
    p6: '情感中立性',
    p7: '行为自主性',
    p8: '信息一致性'
  };

  return Object.entries(assessmentPerformance.value).map(([key, data]) => ({
    name: assessmentNames[key] || key,
    avg_time: data.avg_time,
    success_rate: data.success_rate,
    token_usage: Math.floor(data.avg_time * 1000 + Math.random() * 500) // 模拟Token使用量
  }));
});

// 刷新数据
const refreshData = () => {
  loadPerformanceData();
  ElMessage.success('数据已刷新');
};

// 组件挂载
onMounted(() => {
  loadPerformanceData();
});
</script>

<style scoped>
.system-performance {
  padding: 20px;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.performance-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.performance-header h2 {
  margin: 0;
  color: #2c3e50;
}

.overview-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.metric-card {
  transition: transform 0.2s;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.metric-content {
  display: flex;
  align-items: center;
  gap: 15px;
}

.metric-icon {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: linear-gradient(135deg, #409EFF, #67C23A);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 24px;
}

.token-icon {
  background: linear-gradient(135deg, #E6A23C, #F56C6C);
}

.metric-info {
  flex: 1;
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  color: #2c3e50;
  margin-bottom: 5px;
}

.metric-label {
  color: #7f8c8d;
  font-size: 14px;
}

/* 趋势图单独一行 */
.trend-section {
  margin-bottom: 20px;
}

.trend-card {
  min-height: 400px;
}

.trend-chart {
  height: 350px;
}

/* 图表区域 - 三个图表一行 */
.charts-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 20px;
}

.chart-card {
  min-height: 400px;
}

.chart {
  height: 350px;
}

/* 表格区域 - 左右布局 */
.tables-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.table-card {
  height: fit-content;
}

/* 表格样式优化 */
:deep(.el-table) {
  font-size: 13px;
}

:deep(.el-table th) {
  background-color: #fafafa;
  font-weight: 600;
  font-size: 12px;
}

:deep(.el-table td) {
  padding: 8px 0;
}

:deep(.el-progress-bar__outer) {
  height: 6px !important;
}

.error-analysis-card {
  margin-bottom: 20px;
}

.error-item {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.error-message {
  flex: 1;
  color: #606266;
  font-size: 14px;
}

.error-file {
  color: #909399;
  font-size: 12px;
}

.no-errors {
  text-align: center;
  padding: 40px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .charts-container {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .tables-container {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .overview-cards {
    grid-template-columns: 1fr;
  }
  
  .charts-container {
    grid-template-columns: 1fr;
  }
  
  .tables-container {
    grid-template-columns: 1fr;
  }
  
  .metric-content {
    flex-direction: column;
    text-align: center;
  }
}
</style>