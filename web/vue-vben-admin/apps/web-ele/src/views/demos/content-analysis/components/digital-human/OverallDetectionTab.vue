<script lang="ts" setup>
import {
  ElCard,
  ElTag,
  ElButton,
  ElEmpty,
  ElStatistic,
  ElDivider,
  ElImage,
  ElDescriptions,
  ElDescriptionsItem,
  ElProgress,
  ElIcon,
} from 'element-plus';
import {
  Picture,
  Refresh,
  VideoCamera,
  Warning,
  InfoFilled,
} from '@element-plus/icons-vue';

const props = defineProps({
  isStepAvailable: {
    type: Function,
    required: true,
  },
  getDetectionResult: {
    type: Function,
    required: true,
  },
  detectionData: {
    type: Object,
    default: null,
  },
  getVideoId: {
    type: Function,
    required: true,
  },
  retestSingleStep: {
    type: Function,
    required: true,
  },
  detectionStatus: {
    type: String,
    default: 'not_started',
  },
  // 新增的状态属性
  stepStatus: {
    type: String,
    default: 'not_started',
  },
  isProcessing: {
    type: Boolean,
    default: false,
  },
});
</script>

<template>
  <el-card class="analysis-card">
    <template #header>
      <div class="card-header">
        <div class="header-left">
          <el-icon><Picture /></el-icon>
          <span>整体检测分析</span>
          <el-tag v-if="isStepAvailable('overall')" :type="getDetectionResult('overall').prediction === 'Human' ? 'success' : 'danger'">
            {{ getDetectionResult('overall').prediction === 'Human' ? '真实内容' : 'AI生成内容' }}
          </el-tag>
          <el-tag v-else type="info">未检测</el-tag>
        </div>
        <el-button 
          v-if="isStepAvailable('overall')"
          type="primary" 
          size="small" 
          :icon="Refresh"
          @click="retestSingleStep('overall')"
          :disabled="detectionStatus === 'processing'"
        >
          重新检测
        </el-button>
      </div>
    </template>

    <!-- 未检测状态 -->
    <div v-if="!isStepAvailable('overall')" class="not-detected">
      <el-empty 
        description="未进行整体检测"
        :image-size="100"
      >
        <template #image>
          <el-icon size="100" color="#c0c4cc"><Picture /></el-icon>
        </template>
        <el-button type="primary" @click="retestSingleStep('overall')">
          开始整体检测
        </el-button>
      </el-empty>
    </div>

    <!-- 已检测状态 -->
    <div v-else class="overall-analysis">
      <!-- 主要指标 -->
      <div class="main-metrics">
        <el-statistic
          title="真实度"
          :value="detectionData.overall?.human_probability ? detectionData.overall.human_probability * 100 : 0"
          suffix="%"
          :value-style="{ color: (detectionData.overall?.human_probability || 0) > 0.5 ? '#67c23a' : '#f56c6c' }"
        />
        <el-statistic
          title="AI概率"
          :value="detectionData.overall?.ai_probability ? detectionData.overall.ai_probability * 100 : 0"
          suffix="%"
          :value-style="{ color: (detectionData.overall?.ai_probability || 0) > 0.5 ? '#f56c6c' : '#67c23a' }"
        />
        <el-statistic
          title="置信度"
          :value="detectionData.overall?.confidence ? detectionData.overall.confidence * 100 : 0"
          suffix="%"
          :value-style="{ color: '#409eff' }"
        />
      </div>

      <!-- 整体特征图片展示 -->
      <el-divider>整体特征分析图片</el-divider>
      <div class="detection-images">
        <div class="images-grid">
          <!-- 时空一致性分析图等图片组件... -->
        </div>
      </div>

      <!-- 检测特征 -->
      <el-divider>检测算法特征</el-divider>
      <div class="algorithm-features">
        <div class="feature-grid">
          <div class="feature-card">
            <el-icon size="24" color="#409eff"><VideoCamera /></el-icon>
            <h4>时空一致性检测</h4>
            <p>分析视频帧间的时间连续性和空间一致性，识别生成内容的时序异常</p>
          </div>
          <div class="feature-card">
            <el-icon size="24" color="#67c23a"><Picture /></el-icon>
            <h4>全局纹理分析</h4>
            <p>检测整体画面的纹理特征，识别AI生成算法留下的纹理痕迹</p>
          </div>
          <div class="feature-card">
            <el-icon size="24" color="#e6a23c"><Warning /></el-icon>
            <h4>生成伪影识别</h4>
            <p>检测AI生成模型产生的视觉伪影和不自然的生成痕迹</p>
          </div>
          <div class="feature-card">
            <el-icon size="24" color="#f56c6c"><InfoFilled /></el-icon>
            <h4>多尺度特征融合</h4>
            <p>结合像素级、特征级和语义级的多尺度特征进行综合判断</p>
          </div>
        </div>
      </div>

      <!-- 检测技术详情 -->
      <el-divider>技术详情</el-divider>
      <div class="raw-data">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="检测算法">
            全局特征深度学习模型
          </el-descriptions-item>
          <el-descriptions-item label="分析维度">
            像素级、特征级、语义级
          </el-descriptions-item>
          <el-descriptions-item label="特征类型">
            时空一致性、纹理特征、生成痕迹
          </el-descriptions-item>
          <el-descriptions-item label="检测精度">
            高精度智能识别
          </el-descriptions-item>
          <el-descriptions-item label="模型架构">
            深度卷积神经网络
          </el-descriptions-item>
          <el-descriptions-item label="训练策略">
            对抗性训练增强泛化能力
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 概率分布可视化 -->
      <el-divider>概率分布</el-divider>
      <div class="probability-distribution">
        <div class="probability-bar">
          <div class="bar-label">AI生成概率</div>
          <el-progress
            :percentage="detectionData.overall?.ai_probability ? Math.round(detectionData.overall.ai_probability * 100) : 0"
            status="exception"
            :stroke-width="20"
            show-text
            :format="(percentage) => `${(detectionData.overall?.ai_probability * 100).toFixed(3)}%`"
          />
          <span class="percentage-value">{{ (detectionData.overall?.ai_probability * 100).toFixed(3) }}%</span>
        </div>
        <div class="probability-bar">
          <div class="bar-label">真实内容概率</div>
          <el-progress
            :percentage="detectionData.overall?.human_probability ? Math.round(detectionData.overall.human_probability * 100) : 0"
            status="success"
            :stroke-width="20"
            show-text
            :format="(percentage) => `${(detectionData.overall?.human_probability * 100).toFixed(3)}%`"
          />
          <span class="percentage-value">{{ (detectionData.overall?.human_probability * 100).toFixed(3) }}%</span>
        </div>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
/* 基础卡片样式 */
.analysis-card {
  min-height: 500px;
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 500;
  width: 100%;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
}

/* 未检测状态样式 */
.not-detected {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
}

/* 主要指标 */
.main-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  margin-bottom: 2rem;
  text-align: center;
}

/* 检测图片展示区域 */
.detection-images {
  margin: 1.5rem 0;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 1rem;
}

/* 算法特征网格 */
.algorithm-features {
  margin: 1.5rem 0;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.feature-card {
  padding: 1.5rem;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  text-align: center;
  background: #fafafa;
  transition: all 0.2s ease;
}

.feature-card:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.1);
  background: white;
}

.feature-card h4 {
  margin: 0.75rem 0 0.5rem 0;
  color: #303133;
  font-size: 1rem;
  font-weight: 600;
}

.feature-card p {
  margin: 0;
  color: #606266;
  font-size: 0.875rem;
  line-height: 1.4;
}

/* 概率分布 */
.probability-distribution {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.probability-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.bar-label {
  width: 120px;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-metrics {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .feature-grid {
    grid-template-columns: 1fr;
  }
  
  .probability-bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .bar-label {
    width: auto;
  }
}
</style>
