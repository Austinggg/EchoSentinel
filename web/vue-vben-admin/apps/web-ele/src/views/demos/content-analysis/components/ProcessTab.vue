<script lang="ts" setup>
import { computed } from 'vue';
import {
  ElProgress,
  ElCard,
  ElIcon,
  ElTag,
  ElButton,
  ElDivider,
} from 'element-plus';
import {
  TrendCharts,
  DataAnalysis,
  User,
  Document,
  Warning,
  Connection,
  View,
  Monitor,
  EditPen,
  UserFilled,
} from '@element-plus/icons-vue';

// 定义评估项的语义映射和图标
const assessmentConfig = {
  p1: { name: '背景信息充分性', icon: Document, category: '信息质量' },
  p2: { name: '背景信息准确性', icon: DataAnalysis, category: '信息质量' },
  p3: { name: '内容完整性', icon: View, category: '内容评估' },
  p4: { name: '意图正当性', category: '可信度评估' }, // 使用 Shield
  p5: { name: '发布者信誉', icon: UserFilled, category: '可信度评估' },
  p6: { name: '情感中立性', icon: TrendCharts, category: '内容评估' },
  p7: { name: '行为自主性', icon: Connection, category: '行为分析' },
  p8: { name: '信息一致性', icon: Warning, category: '信息质量' },
};

// 定义组件接收的props
const props = defineProps({
  assessmentData: {
    type: Object,
    default: () => ({}),
  },
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['view-reasoning']);

// 检查是否有评估数据
const hasAssessments = computed(() => {
  return props.assessmentData && Object.keys(props.assessmentData).length > 0;
});

// 格式化评估项数据并按类别分组
const assessmentItems = computed(() => {
  if (!hasAssessments.value) return [];

  return Object.entries(props.assessmentData)
    .filter(
      ([_, item]) => item && item.score !== null && item.score !== undefined,
    )
    .map(([key, item]) => ({
      key,
      name: assessmentConfig[key]?.name || key,
      icon: assessmentConfig[key]?.icon || Document,
      category: assessmentConfig[key]?.category || '其他',
      score: item.score,
      reasoning: item.reasoning,
    }));
});

// 按类别分组
const groupedAssessments = computed(() => {
  const groups = {};
  assessmentItems.value.forEach((item) => {
    if (!groups[item.category]) {
      groups[item.category] = [];
    }
    groups[item.category].push(item);
  });
  return groups;
});

// 计算总体评分
const overallScore = computed(() => {
  if (assessmentItems.value.length === 0) return 0;
  const sum = assessmentItems.value.reduce((acc, item) => acc + item.score, 0);
  return (sum / assessmentItems.value.length) * 100;
});

// 跳转到评估理由详情页
const goToReasoning = (itemKey) => {
  emit('view-reasoning', itemKey);
};

// 根据评分获取等级和颜色
const getScoreLevel = (score) => {
  if (score >= 0.8) return { level: '优秀', color: '#67C23A', bg: '#f0f9ff' };
  if (score >= 0.6) return { level: '良好', color: '#E6A23C', bg: '#fffbeb' };
  if (score >= 0.4) return { level: '一般', color: '#F56C6C', bg: '#fef2f2' };
  return { level: '较差', color: '#909399', bg: '#f9fafb' };
};

// 格式化评分值
const formatScore = (score) => {
  return typeof score === 'number' ? (score * 100).toFixed(0) : 'N/A';
};

// 获取总体评估等级
const overallLevel = computed(() => {
  const score = overallScore.value / 100;
  return getScoreLevel(score);
});
</script>

<template>
  <div class="process-container">
    <!-- 总体评分卡片 -->
    <el-card class="overview-card" shadow="hover" v-if="hasAssessments">
      <div class="overview-content">
        <div class="overview-left">
          <h3 class="overview-title">视频内容分析总评</h3>
          <div class="overview-score">
            <span class="score-number" :style="{ color: overallLevel.color }">
              {{ overallScore.toFixed(0) }}
            </span>
            <span class="score-unit">分</span>
          </div>
          <el-tag
            :type="
              overallLevel.level === '优秀'
                ? 'success'
                : overallLevel.level === '良好'
                  ? 'warning'
                  : 'danger'
            "
            size="large"
          >
            {{ overallLevel.level }}
          </el-tag>
        </div>
        <div class="overview-right">
          <el-progress
            type="dashboard"
            :percentage="overallScore"
            :color="overallLevel.color"
            :stroke-width="8"
            :width="120"
          />
        </div>
      </div>
    </el-card>

    <!-- 分类指标 -->
    <div v-if="hasAssessments" class="categories-container">
      <div
        v-for="(items, category) in groupedAssessments"
        :key="category"
        class="category-section"
      >
        <h4 class="category-title">{{ category }}</h4>
        <div class="assessment-grid">
          <el-card
            v-for="item in items"
            :key="item.key"
            class="assessment-card"
            shadow="hover"
            :style="{
              borderLeft: `4px solid ${getScoreLevel(item.score).color}`,
            }"
          >
            <div class="card-header">
              <div class="card-title">
                <el-icon
                  class="title-icon"
                  :style="{ color: getScoreLevel(item.score).color }"
                >
                  <component :is="item.icon" />
                </el-icon>
                <span>{{ item.name }}</span>
              </div>
              <div class="card-score">
                <span
                  class="score-value"
                  :style="{ color: getScoreLevel(item.score).color }"
                >
                  {{ formatScore(item.score) }}
                </span>
                <span class="score-label">分</span>
              </div>
            </div>

            <div class="progress-container">
              <el-progress
                :percentage="item.score * 100"
                :color="getScoreLevel(item.score).color"
                :stroke-width="8"
                :show-text="false"
                class="custom-progress"
              />
              <div
                class="score-level"
                :style="{ color: getScoreLevel(item.score).color }"
              >
                {{ getScoreLevel(item.score).level }}
              </div>
            </div>

            <div class="card-footer">
              <el-button
                v-if="item.reasoning"
                :icon="View"
                size="small"
                type="primary"
                :text="true"
                @click="goToReasoning(item.key)"
              >
                查看详细分析
              </el-button>
              <div v-else class="no-reasoning">暂无详细分析</div>
            </div>
          </el-card>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <div class="empty-icon">📊</div>
      <h3 class="empty-title">暂无分析数据</h3>
      <p class="empty-description">当前视频还没有进行内容分析评估</p>
    </div>
  </div>
</template>

<style scoped>
.process-container {
  height: 100%;
  overflow: auto;
  padding: 0 4px;
}

/* 总体评分卡片 */
.overview-card {
  margin-bottom: 24px;
  border-radius: 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.overview-card :deep(.el-card__body) {
  padding: 24px;
}

.overview-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.overview-title {
  margin: 0 0 12px 0;
  font-size: 18px;
  font-weight: 600;
  color: white;
}

.overview-score {
  margin-bottom: 12px;
}

.score-number {
  font-size: 36px;
  font-weight: 700;
  color: white;
}

.score-unit {
  font-size: 16px;
  margin-left: 4px;
  color: rgba(255, 255, 255, 0.8);
}

/* 分类容器 */
.categories-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.category-section {
  margin-bottom: 16px;
}

.category-title {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  padding-bottom: 8px;
  border-bottom: 2px solid #e1e8ed;
}

/* 评估卡片网格 */
.assessment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
}

.assessment-card {
  border-radius: 12px;
  transition: all 0.3s ease;
  cursor: pointer;
}

.assessment-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.card-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #2c3e50;
}

.title-icon {
  font-size: 18px;
}

.card-score {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

.score-value {
  font-size: 24px;
  font-weight: 700;
}

.score-label {
  font-size: 12px;
  color: #8492a6;
}

/* 进度条容器 */
.progress-container {
  margin-bottom: 16px;
}

.custom-progress {
  margin-bottom: 8px;
}

.score-level {
  text-align: right;
  font-size: 12px;
  font-weight: 500;
}

/* 卡片底部 */
.card-footer {
  text-align: center;
}

.no-reasoning {
  color: #adb5bd;
  font-size: 12px;
  font-style: italic;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  text-align: center;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-title {
  margin: 0 0 8px 0;
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.empty-description {
  margin: 0;
  color: #8492a6;
  font-size: 14px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .overview-content {
    flex-direction: column;
    gap: 16px;
    text-align: center;
  }

  .assessment-grid {
    grid-template-columns: 1fr;
  }

  .category-title {
    font-size: 14px;
  }
}

/* Element Plus 组件样式覆盖 */
:deep(.el-progress-bar__outer) {
  border-radius: 10px;
}

:deep(.el-progress-bar__inner) {
  border-radius: 10px;
}

:deep(.el-card__body) {
  padding: 20px;
}
</style>
