<script lang="ts" setup>
import { computed } from 'vue';
import {
  ElCard,
  ElIcon,
  ElTag,
  ElButton,
  ElTooltip,
  ElRow,
  ElCol,
  ElStatistic,
} from 'element-plus';
import {
  TrendCharts,
  DataAnalysis,
  Document,
  Warning,
  View,
  UserFilled,
  Lock,
  Connection,
  ArrowRight,
} from '@element-plus/icons-vue';

// 全新的评估项配置 - 更简洁的设计
const assessmentItems = [
  { key: 'p1', name: '背景充分性', icon: Document, color: '#3b82f6' },
  { key: 'p2', name: '信息准确性', icon: DataAnalysis, color: '#06b6d4' },
  { key: 'p3', name: '内容完整性', icon: View, color: '#10b981' },
  { key: 'p4', name: '意图正当性', icon: Lock, color: '#8b5cf6' },
  { key: 'p5', name: '发布者信誉', icon: UserFilled, color: '#f59e0b' },
  { key: 'p6', name: '情感中立性', icon: TrendCharts, color: '#ef4444' },
  { key: 'p7', name: '行为自主性', icon: Connection, color: '#84cc16' },
  { key: 'p8', name: '信息一致性', icon: Warning, color: '#f97316' },
];

const props = defineProps({
  assessmentData: {
    type: Object,
    default: () => ({}),
  },
});

const emit = defineEmits(['view-reasoning']);

// 检查是否有评估数据
const hasAssessments = computed(() => {
  return props.assessmentData && Object.keys(props.assessmentData).length > 0;
});

// 获取评估项数据
const getAssessmentItem = (key) => {
  const data = props.assessmentData[key];
  const config = assessmentItems.find(item => item.key === key);
  return {
    ...config,
    score: data?.score || 0,
    reasoning: data?.reasoning,
    hasData: !!data && data.score !== null && data.score !== undefined,
  };
};

// 计算总体评分
const overallScore = computed(() => {
  const validItems = assessmentItems.filter(item => {
    const data = props.assessmentData[item.key];
    return data && data.score !== null && data.score !== undefined;
  });
  
  if (validItems.length === 0) return 0;
  
  const sum = validItems.reduce((acc, item) => {
    return acc + props.assessmentData[item.key].score;
  }, 0);
  
  return (sum / validItems.length) * 100;
});

// 获取等级标识
const getGrade = (score) => {
  if (score >= 80) return { text: 'A', color: '#10b981', bg: '#dcfce7' };
  if (score >= 70) return { text: 'B', color: '#3b82f6', bg: '#dbeafe' };
  if (score >= 60) return { text: 'C', color: '#f59e0b', bg: '#fef3c7' };
  return { text: 'D', color: '#ef4444', bg: '#fee2e2' };
};

const overallGrade = computed(() => getGrade(overallScore.value));

const goToReasoning = (itemKey) => {
  emit('view-reasoning', itemKey);
};

// 新增：用于柱状图的数据
const chartData = computed(() => {
  return assessmentItems.map(item => {
    const data = props.assessmentData[item.key];
    return {
      name: item.name,
      score: data?.score ? data.score * 100 : 0,
      color: item.color,
      hasData: !!(data && data.score !== null && data.score !== undefined),
    };
  });
});

// 获取图表的最大高度
const maxScore = computed(() => {
  return Math.max(...chartData.value.map(item => item.score), 100);
});

// 新增：检查是否有有效数据
const hasValidData = computed(() => {
  return chartData.value.some(item => item.hasData);
});

// 新增：计算雷达图多边形的点坐标
const getPolygonPoints = () => {
  const points = [];
  const centerX = 120;
  const centerY = 120;
  const maxRadius = 100;
  
  chartData.value.forEach((item, index) => {
    const angle = (index * 45 - 90) * (Math.PI / 180); // 从顶部开始，每45度一个点
    const radius = item.hasData ? (item.score / 100) * maxRadius : 0;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    points.push(`${x},${y}`);
  });
  
  return points.join(' ');
};

// 新增：计算雷达图数据点位置
const getPointPosition = (index, score) => {
  const angle = (index * 45 - 90) * (Math.PI / 180);
  const radius = (score / 100) * 100; // 最大半径100px
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  
  return {
    transform: `translate(${x}px, ${y}px)`,
    backgroundColor: chartData.value[index].color,
  };
};

// 新增：计算雷达图标签位置
const getLabelPosition = (index) => {
  const angle = (index * 45 - 90) * (Math.PI / 180);
  const radius = 110; // 标签距离中心的距离
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  
  // 根据位置调整对齐方式
  let transform = `translate(${x}px, ${y}px)`;
  if (x > 50) transform += ' translateX(-100%)';
  else if (Math.abs(x) <= 50) transform += ' translateX(-50%)';
  
  return {
    transform,
    color: chartData.value[index].color,
  };
};
</script>

<template>
  <div class="process-analysis">
    <!-- 简洁的头部概览 -->
    <div class="header-overview" v-if="hasAssessments">
      <div class="overview-left">
        <h2 class="page-title">内容分析结果</h2>
        <p class="page-subtitle">基于8个维度的智能评估分析</p>
      </div>
      <div class="overview-right">
        <div class="score-display">
          <div class="score-circle" :style="{ borderColor: overallGrade.color }">
            <span class="score-number">{{ overallScore.toFixed(0) }}</span>
            <span class="score-unit">分</span>
          </div>
          <div class="grade-badge" :style="{ backgroundColor: overallGrade.bg, color: overallGrade.color }">
            {{ overallGrade.text }} 级
          </div>
        </div>
      </div>
    </div>

    <!-- 新增：可视化图表区域 -->
    <div class="charts-section" v-if="hasAssessments">
      <div class="chart-container">
        <h3 class="chart-title">评估维度雷达图</h3>
        <div class="radar-chart">
          <div class="radar-grid">
            <!-- 雷达图网格 -->
            <div class="radar-axes">
              <div class="radar-axis" v-for="(item, index) in 8" :key="index" 
                   :style="{ transform: `rotate(${index * 45}deg)` }">
                <div class="axis-line"></div>
              </div>
            </div>
            
            <!-- 同心圆 -->
            <div class="radar-circles">
              <div class="radar-circle circle-20"></div>
              <div class="radar-circle circle-40"></div>
              <div class="radar-circle circle-60"></div>
              <div class="radar-circle circle-80"></div>
              <div class="radar-circle circle-100"></div>
            </div>
            
            <!-- 数据多边形连线 -->
            <svg class="radar-polygon" viewBox="0 0 240 240">
              <polygon 
                :points="getPolygonPoints()"
                fill="rgba(59, 130, 246, 0.1)"
                stroke="#3b82f6"
                stroke-width="2"
                v-if="hasValidData"
              />
            </svg>
            
            <!-- 数据点 -->
            <div class="radar-data">
              <div 
                v-for="(item, index) in chartData" 
                :key="index"
                class="radar-point"
                :class="{ 'has-data': item.hasData }"
                :style="getPointPosition(index, item.score)"
              >
                <div class="point-tooltip">
                  <span>{{ item.name }}</span>
                  <span>{{ item.score.toFixed(0) }}分</span>
                </div>
              </div>
            </div>
            
            <!-- 标签 -->
            <div class="radar-labels">
              <div 
                v-for="(item, index) in chartData" 
                :key="index"
                class="radar-label"
                :style="getLabelPosition(index)"
              >
                <span class="label-text">{{ item.name }}</span>
                <span class="label-score" v-if="item.hasData">{{ item.score.toFixed(0) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="chart-container">
        <h3 class="chart-title">各维度得分对比</h3>
        <div class="bar-chart">
          <div class="chart-y-axis">
            <div class="y-axis-line"></div>
            <div class="y-labels">
              <div class="y-label" v-for="i in 6" :key="i">{{ (i - 1) * 20 }}</div>
            </div>
          </div>
          <div class="chart-bars">
            <div 
              v-for="item in chartData" 
              :key="item.name"
              class="bar-item"
              :class="{ 'has-data': item.hasData }"
            >
              <div class="bar-container">
                <div 
                  class="bar-fill" 
                  :style="{ 
                    height: item.hasData ? `${(item.score / 100) * 100}%` : '0%',
                    backgroundColor: item.color
                  }"
                >
                  <div class="bar-value" v-if="item.hasData">{{ item.score.toFixed(0) }}</div>
                </div>
              </div>
              <div class="bar-label">{{ item.name }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 优化的指标卡片网格 - 4列布局 -->
    <div class="indicators-section" v-if="hasAssessments">
      <h3 class="section-title">详细评估指标</h3>
      <div class="indicators-grid-4col">
        <div
          v-for="item in assessmentItems"
          :key="item.key"
          class="indicator-card-compact"
          :class="{ 'has-data': getAssessmentItem(item.key).hasData }"
        >
          <div class="card-header">
            <div class="icon-wrapper" :style="{ backgroundColor: item.color + '15', color: item.color }">
              <el-icon :size="18">
                <component :is="item.icon" />
              </el-icon>
            </div>
            <div class="card-info">
              <h5 class="card-title">{{ item.name }}</h5>
              <div class="card-score" v-if="getAssessmentItem(item.key).hasData">
                <span class="score" :style="{ color: item.color }">
                  {{ (getAssessmentItem(item.key).score * 100).toFixed(0) }}
                </span>
                <span class="unit">分</span>
              </div>
              <span class="no-data" v-else>未评估</span>
            </div>
          </div>
          
          <div class="card-progress" v-if="getAssessmentItem(item.key).hasData">
            <div class="mini-progress">
              <div 
                class="progress-bar-mini" 
                :style="{ 
                  width: (getAssessmentItem(item.key).score * 100) + '%',
                  backgroundColor: item.color 
                }"
              ></div>
            </div>
            <div class="grade-mini">
              <span 
                class="grade-badge-mini" 
                :style="{ 
                  backgroundColor: getGrade(getAssessmentItem(item.key).score * 100).bg,
                  color: getGrade(getAssessmentItem(item.key).score * 100).color 
                }"
              >
                {{ getGrade(getAssessmentItem(item.key).score * 100).text }}
              </span>
            </div>
          </div>

          <div class="card-action" v-if="getAssessmentItem(item.key).hasData && getAssessmentItem(item.key).reasoning">
            <el-button 
              text 
              type="primary" 
              size="small"
              @click="goToReasoning(item.key)"
              class="analysis-btn"
            >
              详情
              <el-icon><ArrowRight /></el-icon>
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <div class="empty-content">
        <div class="empty-icon">📊</div>
        <h3>暂无分析数据</h3>
        <p>视频内容分析尚未完成，请稍后查看</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.process-analysis {
  padding: 20px;
  background: #f8fafc;
  min-height: 100%;
}

/* 头部概览 */
.header-overview {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding: 24px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.page-title {
  margin: 0 0 8px 0;
  font-size: 24px;
  font-weight: 700;
  color: #1e293b;
}

.page-subtitle {
  margin: 0;
  color: #64748b;
  font-size: 14px;
}

.score-display {
  display: flex;
  align-items: center;
  gap: 16px;
}

.score-circle {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 80px;
  height: 80px;
  border: 3px solid;
  border-radius: 50%;
  background: white;
}

.score-number {
  font-size: 20px;
  font-weight: 700;
  line-height: 1;
}

.score-unit {
  font-size: 12px;
  color: #64748b;
}

.grade-badge {
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
}

/* 图表区域样式 */
.charts-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
}

.chart-container {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chart-title {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
  text-align: center;
}

/* 雷达图样式 - 修复显示问题 */
.radar-chart {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}

.radar-grid {
  position: relative;
  width: 240px;
  height: 240px;
}

.radar-axes {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.radar-axis {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 2px;
  height: 50%;
  background: #e2e8f0;
  transform-origin: bottom center;
}

.axis-line {
  width: 100%;
  height: 100%;
  background: #e2e8f0;
}

.radar-circles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.radar-circle {
  position: absolute;
  top: 50%;
  left: 50%;
  border: 1px solid #e2e8f0;
  border-radius: 50%;
  transform: translate(-50%, -50%);
}

.circle-20 {
  width: 20%;
  height: 20%;
}

.circle-40 {
  width: 40%;
  height: 40%;
}

.circle-60 {
  width: 60%;
  height: 60%;
}

.circle-80 {
  width: 80%;
  height: 80%;
}

.circle-100 {
  width: 100%;
  height: 100%;
}

/* 新增：多边形连线 */
.radar-polygon {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.radar-data {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.radar-point {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transform-origin: 0 0;
  transition: all 0.3s ease;
  z-index: 10;
}

.radar-point.has-data {
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);
  border: 2px solid white;
}

.point-tooltip {
  position: absolute;
  top: -30px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  padding: 4px 8px;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  font-size: 12px;
  white-space: nowrap;
}

.radar-labels {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.radar-label {
  position: absolute;
  top: 50%;
  left: 50%;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  transform-origin: 0 0;
  background: rgba(255, 255, 255, 0.9);
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* 柱状图样式 - 修复动画问题 */
.bar-chart {
  position: relative;
  height: 300px;
  display: flex;
  align-items: flex-end;
  padding-left: 40px;
}

.chart-bars {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  height: 100%;
  flex: 1;
  padding-bottom: 30px;
}

.bar-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
}

.bar-container {
  position: relative;
  width: 100%;
  height: 240px;
  display: flex;
  align-items: flex-end;
  background: #f1f5f9;
  border-radius: 4px 4px 0 0;
}

.bar-fill {
  width: 100%;
  border-radius: 4px 4px 0 0;
  transition: height 1s ease;
  position: relative;
  /* 移除可能导致消失的动画 */
}

.bar-value {
  position: absolute;
  top: -20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  font-weight: 600;
  color: #374151;
  background: white;
  padding: 1px 4px;
  border-radius: 2px;
  white-space: nowrap;
}

.bar-label {
  margin-top: 8px;
  font-size: 10px;
  font-weight: 500;
  color: #64748b;
  text-align: center;
  line-height: 1.2;
  max-width: 100%;
  word-break: keep-all;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 紧凑的4列指标网格 */
.indicators-section {
  margin-top: 32px;
}

.section-title {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
}

.indicators-grid-4col {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.indicator-card-compact {
  background: white;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
  border: 2px solid transparent;
}

.indicator-card-compact.has-data:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.indicator-card-compact:not(.has-data) {
  opacity: 0.6;
  background: #f8fafc;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  flex-shrink: 0;
}

.card-info {
  flex: 1;
}

.card-title {
  margin: 0 0 4px 0;
  font-size: 14px;
  font-weight: 600;
  color: #1e293b;
}

.card-score {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

.score {
  font-size: 18px;
  font-weight: 700;
}

.unit {
  font-size: 10px;
  color: #64748b;
}

.no-data {
  font-size: 12px;
  color: #94a3b8;
  font-style: italic;
}

/* 柱状图Y轴样式 */
.chart-y-axis {
  position: absolute;
  left: 0;
  top: 0;
  width: 40px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding-bottom: 30px;
}

.y-axis-line {
  position: absolute;
  left: 35px;
  top: 0;
  width: 1px;
  height: calc(100% - 30px);
  background: #e2e8f0;
}

.y-labels {
  display: flex;
  flex-direction: column-reverse;
  justify-content: space-between;
  height: calc(100% - 30px);
  padding-right: 8px;
}

.y-label {
  font-size: 10px;
  color: #64748b;
  text-align: right;
  line-height: 1;
}

/* 空状态样式 */
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.empty-content {
  text-align: center;
  max-width: 300px;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-content h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
}

.empty-content p {
  margin: 0;
  color: #64748b;
  font-size: 14px;
  line-height: 1.5;
}

/* 动画 */
/* @keyframes barGrow {
  from {
    height: 0%;
  }
  to {
    height: var(--target-height, 0%);
  }
} */

/* 响应式设计 */
@media (max-width: 1200px) {
  .indicators-grid-4col {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .charts-section {
    grid-template-columns: 1fr;
  }
  
  .indicators-grid-4col {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .chart-bars {
    gap: 4px;
  }
  
  .bar-label {
    font-size: 10px;
  }
}

@media (max-width: 480px) {
  .indicators-grid-4col {
    grid-template-columns: 1fr;
  }
}
  </style>

