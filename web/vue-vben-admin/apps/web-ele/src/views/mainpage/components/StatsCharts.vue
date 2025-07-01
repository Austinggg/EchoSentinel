<script lang="ts" setup>
import { ref, onMounted, onBeforeUnmount, nextTick, defineProps, watch } from 'vue';
import axios from 'axios';
import * as echarts from 'echarts/core';
import { PieChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { ElMessage, ElLoading, ElEmpty, ElButton, ElIcon } from 'element-plus';
import { Loading, Refresh } from '@element-plus/icons-vue';

// 注册ECharts组件
echarts.use([
  PieChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);

// Props
const props = defineProps<{
  accountInfo: any;
}>();

// 状态变量
const loadingStats = ref(false);
const statsData = ref({
  total_videos: 0,
  analyzed_videos: 0,
  pending_videos: 0,
  risk_distribution: [],
  analysis_status: [],
});

// 新增数字人检测数据
const digitalHumanData = ref({
  detected_videos: 0,
  total_digital_human_probability: 0.0,
  digital_human_distribution: [],
  detection_status: []
});

// 图表引用
const digitalHumanChart = ref(null);
const riskDistributionChart = ref(null);

// ECharts实例
let digitalHumanChartInstance = null;
let riskChartInstance = null;

// 加载统计数据
const loadStats = async () => {
  if (!props.accountInfo?.id) return;

  try {
    loadingStats.value = true;

    // 加载基础统计数据
    const response = await axios.get(
      `/api/account/${props.accountInfo.id}/stats`,
    );

    if (response.data.code === 200) {
      statsData.value = response.data.data;
    }

    // 加载数字人检测统计数据
    const digitalHumanResponse = await axios.get(
      `/api/account/${props.accountInfo.id}/digital-human-stats`,
    );

    if (digitalHumanResponse.data.code === 200) {
      digitalHumanData.value = digitalHumanResponse.data.data;
    }

    // 初始化图表
    nextTick(() => {
      initCharts();
    });
  } catch (error) {
    console.error('加载统计数据失败:', error);
    ElMessage.error('加载统计数据失败');
  } finally {
    loadingStats.value = false;
  }
};

// 初始化图表
const initCharts = () => {
  console.log('开始初始化图表', statsData.value, digitalHumanData.value);

  // 使用setTimeout确保DOM完全渲染
  setTimeout(() => {
    try {
      // 数字人检测图表
      if (digitalHumanChart.value) {
        console.log('初始化数字人检测图表', digitalHumanChart.value);
        
        // 先销毁旧实例
        if (digitalHumanChartInstance) digitalHumanChartInstance.dispose();

        // 重新创建实例
        digitalHumanChartInstance = echarts.init(digitalHumanChart.value);

        // 应用选项
        const option = {
          tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)',
          },
          legend: {
            orient: 'vertical',
            left: 10,
            data: digitalHumanData.value.digital_human_distribution.map((item) => item.name),
          },
          series: [
            {
              name: '数字人检测',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2,
              },
              data: digitalHumanData.value.digital_human_distribution,
              color: ['#F56C6C', '#67C23A', '#E6A23C'], // AI生成-红色，真实内容-绿色，不确定-橙色
            },
          ],
        };

        digitalHumanChartInstance.setOption(option);
        console.log('数字人检测图表初始化完成');
      }

      // 风险分析图
      if (riskDistributionChart.value) {
        console.log('初始化风险分布图表', riskDistributionChart.value);
        
        if (riskChartInstance) riskChartInstance.dispose();

        riskChartInstance = echarts.init(riskDistributionChart.value);

        const option = {
          tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)',
          },
          legend: {
            orient: 'vertical',
            left: 10,
            data: statsData.value.risk_distribution.map((item) => item.name),
          },
          series: [
            {
              name: '风险分布',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2,
              },
              data: statsData.value.risk_distribution,
              color: ['#67C23A', '#E6A23C', '#F56C6C', '#909399'],
            },
          ],
        };

        riskChartInstance.setOption(option);
        console.log('风险分布图表初始化完成');
      }
    } catch (error) {
      console.error('图表初始化错误:', error);
    }
  }, 300);
};

// 监听窗口大小变化
const resizeCharts = () => {
  if (digitalHumanChartInstance) digitalHumanChartInstance.resize();
  if (riskChartInstance) riskChartInstance.resize();
};

// 监听账号信息变化
watch(
  () => props.accountInfo,
  (newVal) => {
    if (newVal?.id) {
      loadStats();
    }
  },
  { immediate: true }
);

// 组件挂载时的处理
onMounted(() => {
  // 添加窗口大小变化监听
  window.addEventListener('resize', resizeCharts);
});

// 组件卸载时清理
onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts);
  if (digitalHumanChartInstance) digitalHumanChartInstance.dispose();
  if (riskChartInstance) riskChartInstance.dispose();
});
</script>

<template>
  <div class="analysis-overview-section">
    <div class="stats-container">
      <!-- 加载状态 -->
      <div v-if="loadingStats" class="stats-loading">
        <el-icon class="rotating"><Loading /></el-icon>
        <span>加载统计数据...</span>
      </div>

      <!-- 空状态 -->
      <div v-else-if="!statsData.total_videos" class="stats-empty">
        <el-empty description="暂无统计数据">
          <template #description>
            <p>尚未采集到视频数据或尚未进行分析</p>
          </template>
        </el-empty>
      </div>

      <!-- 统计内容 -->
      <div v-else class="stats-content">
        <!-- 统计数字 -->
        <div class="stats-summary">
          <div class="stat-item">
            <div class="stat-value">{{ statsData.total_videos }}</div>
            <div class="stat-label">总视频数</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">{{ statsData.analyzed_videos }}</div>
            <div class="stat-label">已分析</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">{{ statsData.pending_videos }}</div>
            <div class="stat-label">待分析</div>
          </div>
          <!-- 新增数字人检测概率 -->
          <div class="stat-item">
            <div class="stat-value">{{ (digitalHumanData.total_digital_human_probability * 100).toFixed(1) }}%</div>
            <div class="stat-label">平均AI概率</div>
          </div>
        </div>

        <!-- 图表区域 -->
        <div class="charts-container">
          <!-- 数字人检测图表 -->
          <div class="chart-wrapper">
            <h4>数字人检测分布</h4>
            <div ref="digitalHumanChart" class="echarts-container"></div>
            <!-- 数字说明 -->
            <div class="chart-description">
              <div class="desc-item">
                <span class="desc-label">已检测视频：</span>
                <span class="desc-value">{{ digitalHumanData.detected_videos }} 个</span>
              </div>
              <div class="desc-item">
                <span class="desc-label">平均AI概率：</span>
                <span class="desc-value">{{ (digitalHumanData.total_digital_human_probability * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>

          <!-- 风险级别图表 -->
          <div class="chart-wrapper">
            <h4>风险级别分布</h4>
            <div ref="riskDistributionChart" class="echarts-container"></div>
            <!-- 数字说明 -->
            <div class="chart-description">
              <div class="desc-item">
                <span class="desc-label">已分析视频：</span>
                <span class="desc-value">{{ statsData.analyzed_videos }} 个</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 刷新按钮 -->
      <div class="stats-actions">
        <el-button
          type="primary"
          size="small"
          plain
          @click="loadStats"
          :loading="loadingStats"
        >
          <el-icon><Refresh /></el-icon>
          刷新分析数据
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stats-container {
  min-height: 300px;
}

.stats-loading,
.stats-empty {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 300px;
  color: #909399;
}

.rotating {
  animation: rotate 2s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.stats-summary {
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
  padding: 20px;
  background: #f8f8f8;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.stat-item {
  padding: 15px 20px;
  text-align: center;
  transition: all 0.3s ease;
  flex: 1;
}

.stat-item:hover {
  transform: translateY(-5px);
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 13px;
  color: #606266;
}

/* 图表容器样式 - 这是关键的修复 */
.charts-container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: space-around;
  margin-top: 20px;
}

.chart-wrapper {
  flex: 1 1 400px;
  min-width: 300px;
  border: 1px solid #eee;
  border-radius: 8px;
  padding: 16px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chart-wrapper h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #303133;
  text-align: center;
}

.echarts-container {
  height: 300px !important;
  width: 100% !important;
  min-height: 300px;
  background-color: transparent;
}

.chart-description {
  margin-top: 12px;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 4px;
  font-size: 12px;
  border: 1px solid #e9ecef;
}

.desc-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.desc-item:last-child {
  margin-bottom: 0;
}

.desc-label {
  color: #606266;
}

.desc-value {
  color: #409eff;
  font-weight: 500;
}

.stats-actions {
  text-align: center;
  margin-top: 20px;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .stats-summary {
    flex-wrap: wrap;
    gap: 10px;
  }
  
  .stat-item {
    flex: 1 1 45%;
    min-width: 120px;
    padding: 10px 15px;
  }
  
  .stat-value {
    font-size: 24px;
  }

  .charts-container {
    flex-direction: column;
  }

  .chart-wrapper {
    min-width: 100%;
  }

  .echarts-container {
    height: 250px !important;
  }
}

@media (max-width: 480px) {
  .stats-summary {
    flex-direction: column;
    gap: 8px;
  }

  .stat-item {
    flex: none;
    min-width: auto;
  }
}
</style>
