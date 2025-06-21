<script lang="ts" setup>
import {
  ElCard,
  ElTag,
  ElButton,
  ElEmpty,
  ElStatistic,
  ElDivider,
  ElImage,
  ElTable,
  ElTableColumn,
  ElProgress,
  ElIcon,
} from 'element-plus';
import {
  VideoCamera,
  Refresh,
  Picture,
  CircleCheck,
  CircleClose,
} from '@element-plus/icons-vue';

const props = defineProps({
  isStepAvailable: {
    type: Function,
    required: true,
  },
  bodyDataFixed: {
    type: Object,
    default: null,
  },
  detectionData: {
    type: Object,
    default: null,
  },
  bodyAnalysisDetails: {
    type: Array,
    default: () => [],
  },
  getVideoId: {
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
          <el-icon><VideoCamera /></el-icon>
          <span>躯体检测分析</span>
          <el-tag v-if="isStepAvailable('body')" :type="bodyDataFixed.prediction === 'Human' ? 'success' : 'danger'">
            {{ bodyDataFixed.prediction === 'Human' ? '真实躯体' : 'AI生成躯体' }}
          </el-tag>
          <el-tag v-else type="info">未检测</el-tag>
        </div>
        <el-button 
          v-if="isStepAvailable('body')"
          type="primary" 
          size="small" 
          :icon="Refresh"
          @click="retestSingleStep('body')"
          :disabled="detectionStatus === 'processing'"
        >
          重新检测
        </el-button>
      </div>
    </template>

    <!-- 未检测状态 -->
    <div v-if="!isStepAvailable('body')" class="not-detected">
      <el-empty 
        description="未进行躯体检测"
        :image-size="100"
      >
        <template #image>
          <el-icon size="100" color="#c0c4cc"><VideoCamera /></el-icon>
        </template>
        <el-button type="primary" @click="retestSingleStep('body')">
          开始躯体检测
        </el-button>
      </el-empty>
    </div>

    <!-- 已检测状态 -->
    <div v-else class="body-analysis">
      <!-- 主要指标 -->
      <div class="main-metrics">
        <el-statistic
          title="真实度"
          :value="bodyDataFixed?.human_probability * 100"
          suffix="%"
          :value-style="{ color: bodyDataFixed?.human_probability > 0.5 ? '#67c23a' : '#f56c6c' }"
        />
        <el-statistic
          title="AI概率"
          :value="bodyDataFixed?.ai_probability * 100"
          suffix="%"
          :value-style="{ color: bodyDataFixed?.ai_probability > 0.5 ? '#f56c6c' : '#67c23a' }"
        />
        <el-statistic
          title="置信度"
          :value="bodyDataFixed?.confidence * 100"
          suffix="%"
          :value-style="{ color: '#409eff' }"
        />
      </div>

      <!-- 异常项目图片展示区域 -->
      <el-divider>异常项目图片</el-divider>
      <div class="anomaly-images">
        <div class="images-grid">
          <div 
            v-for="item in bodyAnalysisDetails" 
            :key="item.name"
            class="image-item"
            v-show="item.response !== '是'"
          >
            <div class="image-header">
              <span class="item-name">{{ item.name }}</span>
              <el-tag type="danger" size="small">异常</el-tag>
            </div>
            <div class="image-placeholder">
              <el-image
                :src="`/api/videos/${getVideoId()}/digital-human/anomaly-image/${item.name}`"
                fit="contain"
                :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/anomaly-image/${item.name}`]"
                :hide-on-click-modal="true"
              >
                <template #error>
                  <div class="image-error">
                    <el-icon size="32"><Picture /></el-icon>
                    <div class="error-text">
                      <div>{{ item.name }}异常图片</div>
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
              {{ item.description }}
            </div>
          </div>
        </div>
        
        <!-- 如果没有异常项目 -->
        <div v-if="bodyAnalysisDetails.every(item => item.response === '是')" class="no-anomaly">
          <el-icon size="48" color="#67c23a"><CircleCheck /></el-icon>
          <p>所有检测项目均正常，未发现异常</p>
        </div>
      </div>

      <!-- 检测标准详情 -->
      <el-divider>检测项目详情</el-divider>
      <el-table :data="bodyAnalysisDetails" style="width: 100%">
        <el-table-column prop="name" label="检测项目" width="120" />
        <el-table-column prop="description" label="检测内容" />
        <el-table-column prop="response" label="检测结果" width="100" align="center">
          <template #default="scope">
            <el-tag :type="scope.row.status">
              <el-icon v-if="scope.row.response === '是'"><CircleCheck /></el-icon>
              <el-icon v-else><CircleClose /></el-icon>
              {{ scope.row.response === '是' ? '正常' : '异常' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="weightedScore" label="得分" width="100" align="center">
          <template #default="scope">
            {{ scope.row.weightedScore.toFixed(2) }}
          </template>
        </el-table-column>
      </el-table>

      <!-- 总体评分 -->
      <el-divider>总体评分</el-divider>
      <div class="total-score">
        <el-progress
          type="dashboard"
          :percentage="Math.round(detectionData.body?.raw_results?.total_score * 100)"
          :status="getProgressStatus(detectionData.body?.raw_results?.total_score * 100)"
          :width="150"
        />
        <div class="score-info">
          <h4>综合得分: {{ (detectionData.body?.raw_results?.total_score * 100).toFixed(1) }}%</h4>
          <p>基于智能算法的躯体异常检测评分</p>
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

/* 异常图片展示区域 */
.anomaly-images {
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

.no-anomaly {
  text-align: center;
  padding: 2rem;
  color: #67c23a;
}

.no-anomaly p {
  margin: 0.5rem 0 0;
  color: #606266;
}

/* 总体评分 */
.total-score {
  display: flex;
  align-items: center;
  gap: 2rem;
}

.score-info {
  flex: 1;
}

.score-info h4 {
  margin: 0 0 0.5rem 0;
  color: #409eff;
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
  
  .total-score {
    flex-direction: column;
    gap: 1rem;
  }
}
</style>
