<script lang="ts" setup>
import { ref, computed, onMounted } from 'vue';
import {
  ElButton,
  ElProgress,
  ElCard,
  ElImage,
  ElTag,
  ElDivider,
  ElSteps, // 添加这行
  ElStep, // 添加这行
  ElIcon, // 添加这行
} from 'element-plus';
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  VideoCamera,
  Picture, // 添加这行
} from '@element-plus/icons-vue';
// 定义组件接收的props
const props = defineProps({
  videoData: {
    type: Object,
    default: () => ({}),
  },
});

// 检测步骤
const activeStep = ref(0);
const steps = [
  { title: '信号层检测', description: '分析数字伪造的技术特征' },
  { title: '视觉感知层检测', description: '分析视觉内容的真实性' },
];

// 切换检测步骤
const nextStep = () => {
  if (activeStep.value < steps.length - 1) {
    activeStep.value++;
  }
};

const prevStep = () => {
  if (activeStep.value > 0) {
    activeStep.value--;
  }
};

// 模拟检测数据
const detectionData = ref({
  signal: {
    score: 78,
    details: [
      {
        name: '噪声一致性',
        score: 82,
        description: '分析视频中的噪声模式是否自然',
      },
      {
        name: '压缩伪影',
        score: 75,
        description: '检测视频压缩中的不自然模式',
      },
      {
        name: '频域分析',
        score: 80,
        description: '分析信号的频率特性是否符合真实视频',
      },
    ],
    timestamp: new Date().toISOString(),
  },
  visual: {
    score: 85,
    details: [
      {
        name: '眼部自然度',
        score: 88,
        description: '眼睛眨眼频率、眼神活力的自然程度',
      },
      {
        name: '口型同步性',
        score: 82,
        description: '说话时口型与声音的同步程度',
      },
      { name: '面部表情', score: 90, description: '表情自然度和多样性' },
      { name: '肤色一致性', score: 81, description: '皮肤色调和光照的一致性' },
    ],
    timestamp: new Date().toISOString(),
  },
});

// 模拟图片路径
const getImagePath = (step) => {
  return step === 0
    ? 'https://img.alicdn.com/imgextra/i3/O1CN01ZDK7Ck1L6XEsQAO61_!!6000000001258-2-tps-352-284.png'
    : 'https://img.alicdn.com/imgextra/i3/O1CN01Mpfb8p1kvxRGYQokz_!!6000000004745-2-tps-1628-1280.png';
};

// 获取当前步骤数据
const currentStepData = computed(() => {
  return activeStep.value === 0
    ? detectionData.value.signal
    : detectionData.value.visual;
});

// 获取进度条状态
const getProgressStatus = (score) => {
  if (score >= 80) return 'success';
  if (score >= 60) return 'warning';
  return 'exception';
};

// 获取数字人概率描述
const digitalHumanProbability = computed(() => {
  // 综合两个阶段的得分
  const avgScore =
    (detectionData.value.signal.score + detectionData.value.visual.score) / 2;
  const realProbability = 100 - avgScore;

  return {
    score: realProbability,
    status: getProgressStatus(avgScore),
    text: realProbability >= 70 ? '疑似数字人' : '可能是真实人物',
    description:
      realProbability >= 70
        ? '检测结果显示此视频很可能使用了数字人技术'
        : '检测结果显示此视频可能展示的是真实人物',
  };
});

// 格式化日期时间
const formatDateTime = (dateStr) => {
  if (!dateStr) return '未知时间';
  const date = new Date(dateStr);
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// 生成随机噪声图
onMounted(() => {
  // 实际应用中可以在此处加载真实检测数据
});
</script>

<template>
  <div class="digital-human-container">
    <!-- 标题和步骤控制按钮放在一起 -->
    <div class="header-controls">
      <h3 class="section-heading">数字人检测分析</h3>
      
      <!-- 步骤按钮移到顶部 -->
      <div class="step-controls">
        <el-button
          :icon="ArrowLeft"
          :disabled="activeStep === 0"
          @click="prevStep"
          size="small"
        >
          上一步
        </el-button>
        <el-button
          type="primary"
          :disabled="activeStep === steps.length - 1"
          @click="nextStep"
          size="small"
        >
          下一步 <el-icon class="el-icon--right"><ArrowRight /></el-icon>
        </el-button>
      </div>
    </div>

    <!-- 步骤条 -->
    <div class="steps-container">
      <el-steps :active="activeStep" finish-status="success" align-center>
        <el-step
          v-for="(step, index) in steps"
          :key="index"
          :title="step.title"
          :description="step.description"
        />
      </el-steps>
    </div>

    <!-- 主要内容区域 -->
    <div class="detection-content">
      <!-- 左侧图片区域 -->
      <div class="image-section">
        <el-card class="image-card">
          <div class="image-title">
            <span v-if="activeStep === 0">
              <el-icon><Camera /></el-icon> 信号层分析图
            </span>
            <span v-else>
              <el-icon><VideoCamera /></el-icon> 视觉感知层分析图
            </span>
          </div>
          <div class="image-container">
            <el-image
              :src="getImagePath(activeStep)"
              fit="contain"
              :preview-src-list="[getImagePath(activeStep)]"
            >
              <template #error>
                <div class="image-error">
                  <el-icon><Picture /></el-icon>
                  <div class="error-text">加载分析图失败</div>
                </div>
              </template>
            </el-image>
          </div>
          <div class="timestamp-info">
            分析时间: {{ formatDateTime(currentStepData.timestamp) }}
          </div>
        </el-card>
      </div>

      <!-- 右侧得分区域 -->
      <div class="score-section">
        <el-card class="score-card">
          <div class="score-title">{{ steps[activeStep].title }}得分</div>

          <div class="main-score-display">
            <el-progress
              type="dashboard"
              :percentage="currentStepData.score"
              :status="getProgressStatus(currentStepData.score)"
              :stroke-width="10"
              :width="150"
            />
            <div class="score-description">
              {{
                currentStepData.score >= 80
                  ? '高真实度'
                  : currentStepData.score >= 60
                    ? '中等真实度'
                    : '低真实度'
              }}
            </div>
          </div>

          <el-divider>详细指标</el-divider>

          <div class="detail-scores">
            <div
              v-for="(detail, index) in currentStepData.details"
              :key="index"
              class="detail-score-item"
            >
              <div class="detail-name-container">
                <div class="detail-name" :title="detail.description">
                  {{ detail.name }}
                </div>
                <el-progress
                  :percentage="detail.score"
                  :status="getProgressStatus(detail.score)"
                  :stroke-width="6"
                  :show-text="true"
                />
              </div>
            </div>
          </div>
        </el-card>

        <!-- 数字人概率卡片 -->
        <el-card class="digital-human-card" v-if="activeStep === 1">
          <div class="digital-human-header">
            <span>数字人可能性分析</span>
            <el-tag
              :type="digitalHumanProbability.score >= 70 ? 'danger' : 'success'"
            >
              {{ digitalHumanProbability.text }}
            </el-tag>
          </div>

          <div class="digital-human-content">
            <el-progress
              type="dashboard"
              :percentage="Math.round(digitalHumanProbability.score)"
              :status="
                digitalHumanProbability.score >= 70 ? 'exception' : 'success'
              "
              :stroke-width="10"
              :width="120"
            />
            <div class="probability-description">
              {{ digitalHumanProbability.description }}
            </div>
          </div>
        </el-card>
      </div>
    </div>

    <!-- 移除了底部的步骤按钮 -->
  </div>
</template>

<style scoped>
.digital-human-container {
  height: 100%;
  overflow: auto;
  padding: 0 4px;
}

/* 新增：标题和控制按钮的布局 */
.header-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.section-heading {
  margin-bottom: 0; /* 修改了margin，因为现在在flex容器中 */
  font-size: 1.125rem;
  font-weight: 500;
}

/* 修改步骤控制按钮样式 */
.step-controls {
  display: flex;
  gap: 0.5rem;
}

.steps-container {
  margin-bottom: 2rem;
  max-width: 90%;
  margin-left: auto;
  margin-right: auto;
}

.detection-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

@media (min-width: 768px) {
  .detection-content {
    flex-direction: row;
  }

  .image-section {
    width: 50%;
  }

  .score-section {
    width: 50%;
  }
}

.image-section,
.score-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.image-card,
.score-card,
.digital-human-card {
  height: 100%;
}

.image-title {
  font-weight: 500;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 6px;
}

.image-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
  background-color: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.timestamp-info {
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: #909399;
  text-align: right;
}

.score-title {
  font-weight: 500;
  margin-bottom: 1.5rem;
  text-align: center;
  font-size: 1.125rem;
}

.main-score-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 1rem;
}

.score-description {
  margin-top: 0.75rem;
  font-size: 1rem;
  font-weight: 500;
}

.detail-scores {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.detail-score-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.detail-name-container {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.detail-name {
  font-weight: 500;
  font-size: 0.875rem;
  color: #606266;
}

.digital-human-card {
  margin-top: 1rem;
}

.digital-human-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  font-weight: 500;
}

.digital-human-content {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.probability-description {
  margin-top: 1rem;
  text-align: center;
  color: #606266;
}

.image-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.error-text {
  margin-top: 8px;
}
</style>