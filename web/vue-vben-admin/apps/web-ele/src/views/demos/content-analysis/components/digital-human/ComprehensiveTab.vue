<script lang="ts" setup>
import { computed } from 'vue';
import {
  ElCard,
  ElDivider,
  ElProgress,
  ElTag,
  ElAlert,
  ElDescriptions,
  ElDescriptionsItem,
  ElButton,
  ElEmpty,
  ElIcon,
} from 'element-plus';
import {
  Camera,
  VideoCamera,
  Picture,
  InfoFilled,
  CircleCheck,
  CircleClose,
} from '@element-plus/icons-vue';

const props = defineProps({
  comprehensiveAnalysis: {
    type: Object,
    default: null,
  },
  detectionData: {
    type: Object,
    default: null,
  },
  isStepAvailable: {
    type: Function,
    required: true,
  },
  getDetectionResult: {
    type: Function,
    required: true,
  },
  bodyDataFixed: {
    type: Object,
    default: null,
  },
  getProgressStatus: {
    type: Function,
    required: true,
  },
  getRiskLevel: {
    type: Function,
    required: true,
  },
  formatDateTime: {
    type: Function,
    required: true,
  },
  handleModuleClick: {
    type: Function,
    required: true,
  },
  jumpToStep: {
    type: Function,
    required: true,
  },
  startDetection: {
    type: Function,
    required: true,
  },
  // 新增的状态属性
  stepStatuses: {
    type: Object,
    default: () => ({})
  },
  isStepProcessing: {
    type: Function,
    required: true,
  },
  isStepFailed: {
    type: Function,
    required: true,
  },
});
</script>

<template>
  <el-card class="analysis-card">
    <template #header>
      <div class="card-header">
        <div class="header-left">
          <el-icon><InfoFilled /></el-icon>
          <span>综合评估结果</span>
          <el-tag v-if="comprehensiveAnalysis" :type="comprehensiveAnalysis.prediction === 'Human' ? 'success' : 'danger'" size="large">
            {{ comprehensiveAnalysis.prediction === 'Human' ? '真实人物' : '疑似数字人' }}
          </el-tag>
        </div>
      </div>
    </template>

    <div class="comprehensive-analysis" v-if="comprehensiveAnalysis">
      <!-- 检测模块状态概览 -->
      <div class="detection-overview">
        <h4>检测模块状态</h4>
        <div class="module-status-grid">
          <div 
            class="status-item clickable" 
            :class="{ 'completed': isStepAvailable('face'), 'not-detected': !isStepAvailable('face') }"
            @click="handleModuleClick('face')"
          >
            <el-icon><Camera /></el-icon>
            <span>面部检测</span>
            <el-tag v-if="isStepAvailable('face')" :type="getDetectionResult('face').prediction === 'Human' ? 'success' : 'danger'" size="small">
              {{ getDetectionResult('face').prediction === 'Human' ? '真实' : 'AI' }}
            </el-tag>
            <el-tag v-else type="info" size="small">点击检测</el-tag>
          </div>
          
          <div 
            class="status-item clickable" 
            :class="{ 'completed': isStepAvailable('body'), 'not-detected': !isStepAvailable('body') }"
            @click="handleModuleClick('body')"
          >
            <el-icon><VideoCamera /></el-icon>
            <span>躯体检测</span>
            <el-tag v-if="isStepAvailable('body')" :type="bodyDataFixed.prediction === 'Human' ? 'success' : 'danger'" size="small">
              {{ bodyDataFixed.prediction === 'Human' ? '真实' : 'AI' }}
            </el-tag>
            <el-tag v-else type="info" size="small">点击检测</el-tag>
          </div>
          
          <div 
            class="status-item clickable" 
            :class="{ 'completed': isStepAvailable('overall'), 'not-detected': !isStepAvailable('overall') }"
            @click="handleModuleClick('overall')"
          >
            <el-icon><Picture /></el-icon>
            <span>整体检测</span>
            <el-tag v-if="isStepAvailable('overall')" :type="getDetectionResult('overall').prediction === 'Human' ? 'success' : 'danger'" size="small">
              {{ getDetectionResult('overall').prediction === 'Human' ? '真实' : 'AI' }}
            </el-tag>
            <el-tag v-else type="info" size="small">点击检测</el-tag>
          </div>
        </div>
      </div>

      <el-divider />

      <!-- 最终结果展示 -->
      <div class="final-result">
        <div class="result-score">
          <el-progress
            type="dashboard"
            :percentage="Math.round(comprehensiveAnalysis.finalScore)"
            :status="getProgressStatus(comprehensiveAnalysis.finalScore)"
            :width="200"
            :stroke-width="15"
          />
          <div class="score-label">真实度评分</div>
        </div>
        
        <div class="result-details">
          <div class="detail-item">
            <span class="label">最终预测:</span>
            <el-tag :type="comprehensiveAnalysis.prediction === 'Human' ? 'success' : 'danger'" size="large">
              {{ comprehensiveAnalysis.prediction === 'Human' ? '真实人物' : '疑似数字人' }}
            </el-tag>
          </div>
          <div class="detail-item">
            <span class="label">置信度:</span>
            <span class="value">{{ comprehensiveAnalysis.confidence.toFixed(1) }}%</span>
          </div>
          <div class="detail-item">
            <span class="label">AI概率:</span>
            <span class="value danger">{{ comprehensiveAnalysis.aiProbability.toFixed(1) }}%</span>
          </div>
          <div class="detail-item">
            <span class="label">检测一致性:</span>
            <el-tag :type="comprehensiveAnalysis.consensus ? 'success' : 'warning'">
              {{ comprehensiveAnalysis.consensus ? '一致' : '分歧' }}
            </el-tag>
          </div>
        </div>
      </div>

      <!-- 各模块得分 -->
      <el-divider>已检测模块得分</el-divider>
      <div class="component-scores">
        <div v-if="comprehensiveAnalysis.scores.face !== undefined" class="score-item">
          <div class="score-header">
            <span>面部检测得分</span>
            <el-button text size="small" @click="jumpToStep('face')">查看详情</el-button>
          </div>
          <el-progress
            :percentage="Math.round(comprehensiveAnalysis.scores.face)"
            :status="getProgressStatus(comprehensiveAnalysis.scores.face)"
          />
        </div>
        
        <div v-if="comprehensiveAnalysis.scores.body !== undefined" class="score-item">
          <div class="score-header">
            <span>躯体检测得分</span>
            <el-button text size="small" @click="jumpToStep('body')">查看详情</el-button>
          </div>
          <el-progress
            :percentage="Math.round(comprehensiveAnalysis.scores.body)"
            :status="getProgressStatus(comprehensiveAnalysis.scores.body)"
          />
        </div>
        
        <div v-if="comprehensiveAnalysis.scores.overall !== undefined" class="score-item">
          <div class="score-header">
            <span>整体检测得分</span>
            <el-button text size="small" @click="jumpToStep('overall')">查看详情</el-button>
          </div>
          <el-progress
            :percentage="Math.round(comprehensiveAnalysis.scores.overall)"
            :status="getProgressStatus(comprehensiveAnalysis.scores.overall)"
          />
        </div>
      </div>

      <!-- 检测一致性 -->
      <el-divider>检测一致性</el-divider>
      <div class="voting-statistics">
        <div class="vote-result">
          <div class="vote-item">
            <el-icon size="24" color="#f56c6c"><CircleClose /></el-icon>
            <span>AI生成: {{ comprehensiveAnalysis.votes.ai }} 个模块</span>
          </div>
          <div class="vote-item">
            <el-icon size="24" color="#67c23a"><CircleCheck /></el-icon>
            <span>真实人物: {{ comprehensiveAnalysis.votes.human }} 个模块</span>
          </div>
        </div>
        
        <div class="consensus-info">
          <el-alert
            :title="comprehensiveAnalysis.availableDetections.length === 1 ? '单一检测结果' : (comprehensiveAnalysis.consensus ? '检测结果一致' : '检测结果存在分歧')"
            :type="comprehensiveAnalysis.consensus ? 'success' : 'warning'"
            :description="comprehensiveAnalysis.availableDetections.length === 1 ? 
              `基于${comprehensiveAnalysis.availableDetections.length}个检测模块的结果，建议增加其他检测维度以提高准确性。` :
              (comprehensiveAnalysis.consensus ? 
                `${comprehensiveAnalysis.availableDetections.length}个检测模块的结果高度一致，增强了判定的可靠性。` : 
                `${comprehensiveAnalysis.availableDetections.length}个检测模块的结果存在分歧，建议进一步人工审核。`)"
            show-icon
            :closable="false"
          />
        </div>
      </div>

      <!-- 检测摘要 -->
      <el-divider>检测摘要</el-divider>
      <div class="detection-summary">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="检测时间">
            {{ formatDateTime(detectionData.completed_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="检测模块">
            {{ comprehensiveAnalysis.availableDetections.join('、') }}
          </el-descriptions-item>
          <el-descriptions-item label="检测数量">
            {{ comprehensiveAnalysis.availableDetections.length }} / 3 个模块
          </el-descriptions-item>
          <el-descriptions-item label="算法类型">
            深度学习 + 智能分析
          </el-descriptions-item>
          <el-descriptions-item label="融合策略">
            {{ comprehensiveAnalysis.availableDetections.length > 1 ? '多模块智能融合' : '单模块检测' }}
          </el-descriptions-item>
          <el-descriptions-item label="风险等级" :span="2">
            <el-tag :type="getRiskLevel(comprehensiveAnalysis.aiProbability / 100).type" size="large">
              {{ getRiskLevel(comprehensiveAnalysis.aiProbability / 100).level }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </div>

    <!-- 如果没有任何检测结果，显示启动检测界面 -->
    <div v-else class="no-detection-results">
      <el-empty description="暂无检测结果">
        <template #image>
          <el-icon size="100" color="#c0c4cc"><InfoFilled /></el-icon>
        </template>
        <el-button type="primary" @click="startDetection">
          开始全面检测
        </el-button>
      </el-empty>
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

/* 检测模块状态概览 */
.detection-overview {
  margin-bottom: 2rem;
}

.detection-overview h4 {
  margin-bottom: 1rem;
  color: #303133;
  font-size: 1.125rem;
  font-weight: 500;
}

.module-status-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1rem;
}

.status-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  border: 2px solid #dcdfe6;
  border-radius: 8px;
  background: #fafafa;
  transition: all 0.2s ease;
  gap: 0.5rem;
}

.status-item.clickable {
  cursor: pointer;
}

.status-item.clickable:hover {
  border-color: #409eff;
  background: #f0f9ff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.2);
}

.status-item.completed {
  border-color: #67c23a;
  background: #f0f9ff;
}

.status-item.not-detected {
  border-color: #c0c4cc;
  background: #f5f7fa;
}

.status-item span {
  font-weight: 500;
  text-align: center;
}

.status-item .el-tag {
  margin-top: 0.25rem;
}

/* 最终结果展示 */
.final-result {
  display: flex;
  gap: 3rem;
  margin-bottom: 2rem;
  align-items: center;
}

.result-score {
  text-align: center;
}

.score-label {
  margin-top: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.result-details {
  flex: 1;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 6px;
}

.label {
  font-weight: 500;
}

.value {
  font-weight: bold;
}

.value.danger {
  color: #f56c6c;
}

/* 组件得分 */
.component-scores {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.score-item {
  background: white;
  padding: 1rem;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
}

.score-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

/* 投票统计 */
.voting-statistics {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.vote-result {
  display: flex;
  gap: 2rem;
  justify-content: center;
}

.vote-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
}

/* 没有检测结果的状态 */
.no-detection-results {
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .module-status-grid {
    grid-template-columns: 1fr;
  }
  
  .final-result {
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .vote-result {
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }
}
</style>
