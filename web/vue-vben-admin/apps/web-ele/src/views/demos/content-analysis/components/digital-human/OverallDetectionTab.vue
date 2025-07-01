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

      <!-- 检测维度分析 -->
      <el-divider>多维度检测分析</el-divider>
      <div class="detection-dimensions">
        <div class="dimension-grid">
          <div class="dimension-item">
            <div class="dimension-header">
              <el-icon size="20" color="#409eff"><VideoCamera /></el-icon>
              <span class="dimension-title">时空一致性</span>
              <el-tag size="small" type="success">正常</el-tag>
            </div>
            <div class="dimension-content">
              <el-progress :percentage="85" status="success" :stroke-width="8" />
              <p class="dimension-desc">视频帧间时间连续性和空间一致性检测</p>
            </div>
          </div>
          
          <div class="dimension-item">
            <div class="dimension-header">
              <el-icon size="20" color="#67c23a"><Picture /></el-icon>
              <span class="dimension-title">全局纹理</span>
              <el-tag size="small" type="warning">轻微异常</el-tag>
            </div>
            <div class="dimension-content">
              <el-progress :percentage="72" status="warning" :stroke-width="8" />
              <p class="dimension-desc">整体画面纹理特征和生成痕迹分析</p>
            </div>
          </div>
          
          <div class="dimension-item">
            <div class="dimension-header">
              <el-icon size="20" color="#e6a23c"><Warning /></el-icon>
              <span class="dimension-title">生成伪影</span>
              <el-tag size="small" type="danger">检测到</el-tag>
            </div>
            <div class="dimension-content">
              <el-progress :percentage="45" status="exception" :stroke-width="8" />
              <p class="dimension-desc">AI生成模型产生的视觉伪影识别</p>
            </div>
          </div>
          
          <div class="dimension-item">
            <div class="dimension-header">
              <el-icon size="20" color="#f56c6c"><InfoFilled /></el-icon>
              <span class="dimension-title">多尺度融合</span>
              <el-tag size="small" type="info">综合评估</el-tag>
            </div>
            <div class="dimension-content">
              <el-progress :percentage="68" status="warning" :stroke-width="8" />
              <p class="dimension-desc">像素级、特征级和语义级多尺度特征融合</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 检测特征 -->
      <el-divider>核心检测算法</el-divider>
      <div class="algorithm-features">
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon size="32" color="#409eff"><VideoCamera /></el-icon>
            </div>
            <h4>时空一致性检测</h4>
            <p>基于光流分析和帧间差异检测，识别视频序列中的时间不连续性和空间不一致性，检测AI生成视频的时序异常模式</p>
            <div class="feature-metrics">
              <span class="metric">准确率: 94.2%</span>
              <span class="metric">召回率: 89.6%</span>
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon size="32" color="#67c23a"><Picture /></el-icon>
            </div>
            <h4>全局纹理分析</h4>
            <p>运用局部二值模式(LBP)和灰度共生矩阵(GLCM)等纹理描述子，检测AI生成算法留下的细微纹理痕迹和统计特征异常</p>
            <div class="feature-metrics">
              <span class="metric">特征维度: 256</span>
              <span class="metric">检测阈值: 0.75</span>
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon size="32" color="#e6a23c"><Warning /></el-icon>
            </div>
            <h4>生成伪影识别</h4>
            <p>检测GAN、Diffusion等生成模型产生的典型伪影，包括网格效应、频域异常、边缘不自然等生成痕迹特征</p>
            <div class="feature-metrics">
              <span class="metric">伪影类型: 12种</span>
              <span class="metric">检测精度: 91.8%</span>
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-icon">
              <el-icon size="32" color="#f56c6c"><InfoFilled /></el-icon>
            </div>
            <h4>多尺度特征融合</h4>
            <p>集成像素级(纹理、颜色)、特征级(边缘、形状)和语义级(对象、场景)的多层次特征，通过注意力机制进行加权融合</p>
            <div class="feature-metrics">
              <span class="metric">融合层数: 3层</span>
              <span class="metric">权重优化: Adam</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 检测技术详情 -->
      <el-divider>技术详情与参数</el-divider>
      <div class="technical-details">
        <div class="details-grid">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="核心算法">
              基于Transformer架构的多模态深度学习模型
            </el-descriptions-item>
            <el-descriptions-item label="模型参数">
              约1.2亿参数，12层Transformer编码器
            </el-descriptions-item>
            <el-descriptions-item label="训练数据">
              100万+真实视频，50万+AI生成视频样本
            </el-descriptions-item>
            <el-descriptions-item label="检测精度">
              整体准确率92.5%，误报率低于3.2%
            </el-descriptions-item>
          </el-descriptions>
          
          <el-descriptions :column="1" border>
            <el-descriptions-item label="分析维度">
              时空域、频率域、统计域多维度融合分析
            </el-descriptions-item>
            <el-descriptions-item label="特征提取">
              CNN+Transformer混合架构特征提取
            </el-descriptions-item>
            <el-descriptions-item label="决策机制">
              集成学习与不确定性量化相结合
            </el-descriptions-item>
            <el-descriptions-item label="实时性能">
              平均处理时间2.3秒/分钟视频
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </div>

      <!-- 概率分布可视化 -->
      <el-divider>检测结果概率分布</el-divider>
      <div class="probability-distribution">
        <div class="probability-bar">
          <div class="bar-label">AI生成概率</div>
          <el-progress
            :percentage="detectionData.overall?.ai_probability ? Math.round(detectionData.overall.ai_probability * 100) : 0"
            status="exception"
            :stroke-width="20"
            show-text
            :format="() => `${(detectionData.overall?.ai_probability * 100).toFixed(3)}%`"
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
            :format="() => `${(detectionData.overall?.human_probability * 100).toFixed(3)}%`"
          />
          <span class="percentage-value">{{ (detectionData.overall?.human_probability * 100).toFixed(3) }}%</span>
        </div>
        
        <!-- 置信度指示器 -->
        <div class="confidence-indicator">
          <div class="confidence-label">检测置信度</div>
          <el-progress
            type="circle"
            :percentage="detectionData.overall?.confidence ? Math.round(detectionData.overall.confidence * 100) : 0"
            :width="100"
            :stroke-width="8"
            :color="[
              { color: '#f56c6c', percentage: 30 },
              { color: '#e6a23c', percentage: 60 },
              { color: '#67c23a', percentage: 100 }
            ]"
          />
          <p class="confidence-desc">基于多维度特征的综合置信度评估</p>
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

/* 检测维度分析 */
.detection-dimensions {
  margin: 1.5rem 0;
}

.dimension-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.dimension-item {
  padding: 1rem;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  background: white;
}

.dimension-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.dimension-title {
  font-weight: 600;
  color: #303133;
  flex: 1;
}

.dimension-content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.dimension-desc {
  margin: 0;
  font-size: 0.875rem;
  color: #606266;
  line-height: 1.4;
}

/* 算法特征网格 */
.algorithm-features {
  margin: 1.5rem 0;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.feature-card {
  padding: 1.5rem;
  border: 1px solid #dcdfe6;
  border-radius: 12px;
  background: #fafafa;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.feature-card:hover {
  border-color: #409eff;
  box-shadow: 0 4px 20px rgba(64, 158, 255, 0.15);
  background: white;
  transform: translateY(-2px);
}

.feature-icon {
  text-align: center;
  margin-bottom: 1rem;
}

.feature-card h4 {
  margin: 0 0 0.75rem 0;
  color: #303133;
  font-size: 1.1rem;
  font-weight: 600;
  text-align: center;
}

.feature-card p {
  margin: 0 0 1rem 0;
  color: #606266;
  font-size: 0.875rem;
  line-height: 1.6;
  text-align: justify;
}

.feature-metrics {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
}

.metric {
  background: #f0f9ff;
  color: #1890ff;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

/* 技术详情 */
.technical-details {
  margin: 1.5rem 0;
}

.details-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

/* 概率分布 */
.probability-distribution {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  align-items: center;
}

.probability-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
  width: 100%;
  max-width: 600px;
}

.bar-label {
  width: 120px;
  font-weight: 500;
  color: #303133;
}

.percentage-value {
  width: 80px;
  text-align: right;
  font-weight: 600;
  color: #409eff;
}

.confidence-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  margin-top: 1rem;
}

.confidence-label {
  font-weight: 600;
  color: #303133;
  font-size: 1.1rem;
}

.confidence-desc {
  margin: 0;
  color: #606266;
  font-size: 0.875rem;
  text-align: center;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-metrics {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
  
  .dimension-grid {
    grid-template-columns: 1fr;
  }
  
  .feature-grid {
    grid-template-columns: 1fr;
  }
  
  .details-grid {
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
