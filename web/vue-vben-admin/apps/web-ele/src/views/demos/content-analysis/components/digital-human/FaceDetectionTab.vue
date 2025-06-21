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
  Camera,
  Refresh,
  Picture,
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
  formatDateTime: {
    type: Function,
    required: true,
  },
  getProgressStatus: {
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
          <el-icon><Camera /></el-icon>
          <span>面部检测分析</span>
          <el-tag v-if="isStepAvailable('face')" :type="getDetectionResult('face').prediction === 'Human' ? 'success' : 'danger'">
            {{ getDetectionResult('face').prediction === 'Human' ? '真实面部' : 'AI生成面部' }}
          </el-tag>
          <el-tag v-else type="info">未检测</el-tag>
        </div>
        <el-button 
          v-if="isStepAvailable('face')"
          type="primary" 
          size="small" 
          :icon="Refresh"
          @click="retestSingleStep('face')"
          :disabled="detectionStatus === 'processing'"
        >
          重新检测
        </el-button>
      </div>
    </template>

    <!-- 未检测状态 -->
    <div v-if="!isStepAvailable('face')" class="not-detected">
      <el-empty 
        description="未进行面部检测"
        :image-size="100"
      >
        <template #image>
          <el-icon size="100" color="#c0c4cc"><Camera /></el-icon>
        </template>
        <el-button type="primary" @click="retestSingleStep('face')">
          开始面部检测
        </el-button>
      </el-empty>
    </div>

    <!-- 已检测状态 -->
    <div v-else class="face-analysis">
      <!-- 主要指标 -->
      <div class="main-metrics">
        <el-statistic
          title="真实度"
          :value="detectionData.face?.human_probability * 100"
          suffix="%"
          :value-style="{ color: detectionData.face?.human_probability > 0.5 ? '#67c23a' : '#f56c6c' }"
        />
        <el-statistic
          title="AI概率"
          :value="detectionData.face?.ai_probability * 100"
          suffix="%"
          :value-style="{ color: detectionData.face?.ai_probability > 0.5 ? '#f56c6c' : '#67c23a' }"
        />
        <el-statistic
          title="置信度"
          :value="detectionData.face?.confidence * 100"
          suffix="%"
          :value-style="{ color: '#409eff' }"
        />
      </div>

      <!-- 面部检测特征图片展示 -->
      <el-divider>检测特征图片</el-divider>
      <div class="detection-images">
        <div class="images-grid">
          <!-- 面部关键点检测图 -->
          <div class="image-item">
            <div class="image-header">
              <span class="item-name">面部关键点检测</span>
              <el-tag type="primary" size="small">特征分析</el-tag>
            </div>
            <div class="image-placeholder">
              <el-image
                :src="`/api/videos/${getVideoId()}/digital-human/face-keypoints`"
                fit="contain"
                :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/face-keypoints`]"
                :hide-on-click-modal="true"
              >
                <template #error>
                  <div class="image-error">
                    <el-icon size="32"><Picture /></el-icon>
                    <div class="error-text">
                      <div>面部关键点检测图</div>
                      <div class="error-subtitle">暂未生成</div>
                    </div>
                  </div>
                </template>
                <template #placeholder>
                  <div class="image-loading">
                    <el-icon class="loading-icon"><Refresh /></el-icon>
                    <div>加载中...</div>
                  </div>
                </template>
              </el-image>
            </div>
            <div class="image-description">
              基于Xception模型的面部关键点定位和特征提取分析
            </div>
          </div>
          <!-- 纹理特征分析图 -->
          <div class="image-item">
            <div class="image-header">
              <span class="item-name">纹理特征分析</span>
              <el-tag type="warning" size="small">纹理检测</el-tag>
            </div>
            <div class="image-placeholder">
              <el-image
                :src="`/api/videos/${getVideoId()}/digital-human/face-texture`"
                fit="contain"
                :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/face-texture`]"
                :hide-on-click-modal="true"
              >
                <template #error>
                  <div class="image-error">
                    <el-icon size="32"><Picture /></el-icon>
                    <div class="error-text">
                      <div>面部纹理特征图</div>
                      <div class="error-subtitle">暂未生成</div>
                    </div>
                  </div>
                </template>
                <template #placeholder>
                  <div class="image-loading">
                    <el-icon class="loading-icon"><Refresh /></el-icon>
                    <div>加载中...</div>
                  </div>
                </template>
              </el-image>
            </div>
            <div class="image-description">
              面部皮肤纹理和细节的真实性检测分析
            </div>
          </div>
        </div>
      </div>

      <!-- 算法详情 -->
      <el-divider>检测算法详情</el-divider>
      <div class="algorithm-details">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="检测模型">
            基于Xception深度神经网络
          </el-descriptions-item>
          <el-descriptions-item label="训练数据集">
            大规模真实/合成人脸数据集
          </el-descriptions-item>
          <el-descriptions-item label="特征提取">
            多层次面部特征
          </el-descriptions-item>
          <el-descriptions-item label="检测精度">
            高精度伪造面部识别
          </el-descriptions-item>
          <el-descriptions-item label="分析维度">
            关键点、表情、纹理、边缘
          </el-descriptions-item>
          <el-descriptions-item label="检测时间">
            {{ formatDateTime(detectionData.face?.raw_results?.metadata?.timestamp) }}
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 检测结果 -->
      <el-divider>检测结果</el-divider>
      <div class="risk-assessment">
        <el-progress
          type="dashboard"
          :percentage="Math.round(detectionData.face?.human_probability * 100)"
          :status="getProgressStatus(detectionData.face?.human_probability * 100)"
          :width="150"
        />
        <div class="risk-info">
          <h4>{{ detectionData.face?.prediction === 'Human' ? '真实面部' : 'AI生成面部' }}</h4>
          <p>基于Xception模型算法的面部真实性检测结果</p>
          <div class="confidence-info">
            <el-tag :type="detectionData.face?.confidence > 0.7 ? 'success' : 'warning'">
              置信度: {{ (detectionData.face?.confidence * 100).toFixed(1) }}%
            </el-tag>
          </div>
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

.image-item {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  background: #fafafa;
}

.image-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: #f5f7fa;
  border-bottom: 1px solid #dcdfe6;
}

.item-name {
  font-weight: 500;
  color: #303133;
}

.image-placeholder {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
}

.image-error,
.image-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #909399;
  text-align: center;
  gap: 0.5rem;
}

.error-text {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.error-subtitle {
  font-size: 0.75rem;
  color: #c0c4cc;
}

.image-loading .loading-icon {
  animation: spin 2s linear infinite;
}

.image-description {
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  color: #606266;
  line-height: 1.4;
}

/* 算法详情 */
.algorithm-details {
  margin: 1.5rem 0;
}

/* 风险评估 */
.risk-assessment {
  display: flex;
  align-items: center;
  gap: 2rem;
}

.risk-info {
  flex: 1;
}

.risk-info h4 {
  margin: 0 0 0.5rem 0;
  color: #409eff;
}

/* 置信度信息 */
.confidence-info {
  margin-top: 0.75rem;
}

/* 动画 */
@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-metrics {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .images-grid {
    grid-template-columns: 1fr;
  }
  
  .risk-assessment {
    flex-direction: column;
    gap: 1rem;
  }
}
</style>
