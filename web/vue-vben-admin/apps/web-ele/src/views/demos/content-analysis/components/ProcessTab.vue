<script lang="ts" setup>
import { computed } from 'vue';
import { ElProgress } from 'element-plus';

// 定义评估项的语义映射
const assessmentNames = {
  p1: '背景信息充分性',
  p2: '背景信息准确性',
  p3: '内容完整性',
  p4: '意图正当性',
  p5: '发布者信誉',
  p6: '情感中立性',
  p7: '行为自主性',
  p8: '信息一致性',
};

// 定义组件接收的props
const props = defineProps({
  assessmentData: {
    type: Object,
    default: () => ({})
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['view-reasoning']);

// 检查是否有评估数据
const hasAssessments = computed(() => {
  return props.assessmentData && Object.keys(props.assessmentData).length > 0;
});

// 格式化评估项数据
const assessmentItems = computed(() => {
  if (!hasAssessments.value) return [];

  return Object.entries(props.assessmentData)
    .filter(
      ([_, item]) => item && item.score !== null && item.score !== undefined,
    )
    .map(([key, item]) => ({
      key,
      name: assessmentNames[key] || key,
      score: item.score,
      reasoning: item.reasoning,
    }));
});

// 跳转到评估理由详情页
const goToReasoning = (itemKey) => {
  emit('view-reasoning', itemKey);
};

// 根据评分获取进度条颜色
const getScoreColor = (score) => {
  if (score >= 0.8) return '#67C23A'; // 绿色
  if (score >= 0.5) return '#E6A23C'; // 橙色
  return '#F56C6C'; // 红色
};

// 格式化评分值（保留1位小数）
const formatScore = (score) => {
  return typeof score === 'number' ? score.toFixed(1) : 'N/A';
};
</script>

<template>
  <div class="process-container">
    <h3 class="section-heading">视频分析过程</h3>

    <!-- 使用hasAssessments和assessmentItems计算属性 -->
    <div v-if="hasAssessments" class="assessment-list">
      <div
        v-for="item in assessmentItems"
        :key="item.key"
        class="assessment-item"
      >
        <div class="assessment-header">
          <div class="assessment-title">
            {{ item.name }} ({{ item.key }})
          </div>
          <div
            class="assessment-score"
            :style="{ color: getScoreColor(item.score) }"
          >
            {{ formatScore(item.score) }}
          </div>
        </div>

        <el-progress
          :percentage="item.score * 100"
          :color="getScoreColor(item.score)"
          :stroke-width="10"
          :show-text="false"
        />

        <!-- 添加点击事件和鼠标悬停样式 -->
        <div
          v-if="item.reasoning"
          class="reasoning-link"
          @click="goToReasoning(item.key)"
        >
          点击查看详细评估理由
        </div>
        <div v-else class="no-reasoning">无评估理由</div>
      </div>
    </div>

    <!-- 没有评估数据时显示提示 -->
    <div v-else class="empty-state">
      <div class="emoji-placeholder">📊</div>
      <div>暂无分析数据</div>
    </div>
  </div>
</template>

<style scoped>
.process-container {
  height: 100%;
  overflow: auto;
}

.section-heading {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.assessment-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.assessment-item {
  border-left-width: 4px;
  border-left-color: #3b82f6;
  border-left-style: solid;
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  padding-left: 1rem;
}

.assessment-header {
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.assessment-title {
  font-weight: 500;
}

.assessment-score {
  font-size: 1.125rem;
  font-weight: 700;
}

.reasoning-link {
  margin-top: 0.5rem;
  cursor: pointer;
  color: #4b5563;
}

.reasoning-link:hover {
  color: #3b82f6;
  text-decoration: underline;
}

.no-reasoning {
  margin-top: 0.5rem;
  color: #9ca3af;
  font-style: italic;
}

.empty-state {
  padding-top: 2rem;
  padding-bottom: 2rem;
  text-align: center;
  color: #6b7280;
}

.emoji-placeholder {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}
</style>