<script lang="ts" setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';
import axios from 'axios';
import { useRoute } from 'vue-router';
import {
  ElButton,
  ElProgress,
  ElCard,
  ElImage,
  ElTag,
  ElDivider,
  ElSteps,
  ElStep,
  ElIcon,
  ElMessage,
  ElSkeleton,
  ElSkeletonItem,
  ElAlert,
  ElCollapse,
  ElCollapseItem,
  ElTable,
  ElTableColumn,
  ElTooltip,
  ElDescriptions,
  ElDescriptionsItem,
  ElStatistic,
} from 'element-plus';
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  VideoCamera,
  Picture,
  Refresh,
  Warning,
  InfoFilled,
  QuestionFilled,
  CircleCheck,
  CircleClose,
} from '@element-plus/icons-vue';

const props = defineProps({
  videoData: {
    type: Object,
    default: () => ({}),
  },
});

const route = useRoute();
const retestSingleStep = async (stepType) => {
  try {
    const videoId = getVideoId();
    if (!videoId) {
      ElMessage.error('未找到视频ID');
      return;
    }

    detectionStatus.value = 'loading';
    detectionError.value = null;

    console.log(`开始单步重检: ${stepType}`);

    const response = await axios.post(`/api/videos/${videoId}/digital-human/detect`, {
      types: [stepType], // 只检测指定类型
      comprehensive: false, // 单步检测不需要综合评估
    });

    if (response.data.code === 200) {
      detectionStatus.value = 'processing';
      ElMessage.success(`${getStepName(stepType)}重新检测已启动`);
      startPolling();
    } else {
      throw new Error(response.data.message || '启动重检失败');
    }
  } catch (error) {
    console.error('单步重检失败:', error);
    detectionStatus.value = 'completed'; // 恢复到完成状态
    ElMessage.error('重检失败: ' + (error.response?.data?.message || error.message));
  }
};
const getStepName = (stepType) => {
  const names = {
    face: '面部检测',
    body: '躯体检测',
    overall: '整体检测'
  };
  return names[stepType] || stepType;
};
// 修复躯体数据计算
const bodyDataFixed = computed(() => {
  if (!detectionData.value?.body) return null;
  
  const body = detectionData.value.body;
  
  // 修复：如果所有检测项目都正常，应该是高真实度
  if (body.raw_results?.total_score === 1.0) {
    return {
      ...body,
      // 修正显示数据
      human_probability: 1.0, // 100% 真实
      ai_probability: 0.0,    // 0% AI
      confidence: 1.0,        // 100% 置信度
      prediction: 'Human'     // 预测为真实
    };
  }
  
  return body;
});
// 检测步骤 - 扩展为4个步骤
const activeStep = ref(0);
const steps = [
  { title: '面部检测', description: '基于Xception模型的面部伪造检测' },
  { title: '躯体检测', description: '基于多模态LLM的躯体异常检测' },
  { title: '整体检测', description: '全局特征的综合性分析检测' },
  { title: '综合评估', description: '多层次融合决策与最终判定' },
];

// 检测状态管理
const detectionStatus = ref('not_started');
const detectionData = ref(null);
const detectionError = ref(null);
const isPolling = ref(false);
const pollingTimer = ref(null);
const maxRetries = ref(0);
const MAX_RETRIES = 100;

// 权重配置
const DETECTION_WEIGHTS = {
  face: 0.3,
  body: 0.2,
  overall: 0.5
};

// 躯体检测标准
const BODY_CRITERIA = [
  { name: '关节位置', description: '检查关节弯曲角度和相对位置的自然性', weight: 1.5 },
  { name: '手指数量', description: '验证手指数量的正确性和形态自然性', weight: 1.0 },
  { name: '面部表情', description: '分析面部表情的协调性和真实性', weight: 1.2 },
  { name: '身体比例', description: '评估身体各部分比例的合理性', weight: 1.3 },
  { name: '动作流畅度', description: '检查动作的自然性和物理合理性', weight: 1.4 },
];

// 获取视频ID
const getVideoId = () => {
  return route.query.id || props.videoData?.video?.id;
};

// 轮询和状态管理函数
const stopPolling = () => {
  console.log('停止轮询');
  isPolling.value = false;
  maxRetries.value = 0;
  if (pollingTimer.value) {
    clearInterval(pollingTimer.value);
    pollingTimer.value = null;
  }
};

const startPolling = () => {
  if (isPolling.value) return;
  
  console.log('开始轮询状态');
  isPolling.value = true;
  maxRetries.value = 0;
  
  pollingTimer.value = setInterval(() => {
    maxRetries.value++;
    console.log(`轮询检查状态... (${maxRetries.value}/${MAX_RETRIES})`);
    
    if (maxRetries.value > MAX_RETRIES) {
      stopPolling();
      detectionStatus.value = 'failed';
      detectionError.value = '检测超时，请重新尝试';
      ElMessage.error('检测超时，请重新尝试');
      return;
    }
    
    checkDetectionStatus();
  }, 5000);
};

// API调用函数
const checkDetectionStatus = async () => {
  try {
    const videoId = getVideoId();
    if (!videoId) {
      stopPolling();
      return;
    }

    const response = await axios.get(`/api/videos/${videoId}/digital-human/status`);
    
    if (response.data.code === 200) {
      const data = response.data.data;
      
      if (data.status === 'completed') {
        detectionStatus.value = 'completed';
        detectionData.value = data.results;
        stopPolling();
        ElMessage.success('数字人检测完成');
      } else if (data.status === 'processing') {
        if (detectionStatus.value !== 'processing') {
          detectionStatus.value = 'processing';
        }
      } else if (data.status === 'failed') {
        detectionStatus.value = 'failed';
        detectionError.value = data.error_message || '检测失败';
        stopPolling();
        ElMessage.error('检测失败: ' + detectionError.value);
      }
    }
  } catch (error) {
    console.error('查询检测状态失败:', error);
    if (error.response?.status === 404) {
      detectionStatus.value = 'not_started';
      stopPolling();
    }
  }
};

const startDetection = async () => {
  try {
    const videoId = getVideoId();
    if (!videoId) {
      ElMessage.error('未找到视频ID');
      return;
    }

    detectionStatus.value = 'loading';
    detectionError.value = null;

    const response = await axios.post(`/api/videos/${videoId}/digital-human/detect`, {
      types: ['face', 'body', 'overall'],
      comprehensive: true,
    });

    if (response.data.code === 200) {
      detectionStatus.value = 'processing';
      ElMessage.success('数字人检测已启动，正在分析中...');
      startPolling();
    } else {
      throw new Error(response.data.message || '启动检测失败');
    }
  } catch (error) {
    console.error('启动数字人检测失败:', error);
    detectionStatus.value = 'failed';
    detectionError.value = error.response?.data?.message || error.message || '启动检测失败';
    ElMessage.error('启动检测失败: ' + detectionError.value);
  }
};

const restartDetection = () => {
  stopPolling();
  detectionStatus.value = 'not_started';
  detectionData.value = null;
  detectionError.value = null;
  activeStep.value = 0;
};

// 步骤切换
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

// 数据处理和计算
const getProgressStatus = (score) => {
  if (score >= 80) return 'success';
  if (score >= 60) return 'warning';
  return 'exception';
};

const getRiskLevel = (aiProbability) => {
  if (aiProbability >= 0.8) return { level: '高风险', type: 'danger' };
  if (aiProbability >= 0.5) return { level: '中风险', type: 'warning' };
  return { level: '低风险', type: 'success' };
};

// 综合分析计算
const comprehensiveAnalysis = computed(() => {
  if (!detectionData.value) return null;
  
  const data = detectionData.value;
  
  // 计算加权得分
  const scores = {};
  let weightedSum = 0;
  let totalWeight = 0;
  
  if (data.face) {
    scores.face = data.face.human_probability * 100;
    weightedSum += data.face.human_probability * DETECTION_WEIGHTS.face;
    totalWeight += DETECTION_WEIGHTS.face;
  }
  
  if (data.body) {
    scores.body = data.body.human_probability * 100;
    weightedSum += data.body.human_probability * DETECTION_WEIGHTS.body;
    totalWeight += DETECTION_WEIGHTS.body;
  }
  
  if (data.overall) {
    scores.overall = data.overall.human_probability * 100;
    weightedSum += data.overall.human_probability * DETECTION_WEIGHTS.overall;
    totalWeight += DETECTION_WEIGHTS.overall;
  }
  
  const calculatedScore = totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 0;
  const finalScore = data.comprehensive?.human_probability * 100 || calculatedScore;
  
  return {
    scores,
    calculatedScore,
    finalScore,
    aiProbability: data.comprehensive?.ai_probability * 100 || (100 - finalScore),
    confidence: data.comprehensive?.confidence * 100 || 50,
    prediction: data.comprehensive?.prediction || (finalScore >= 50 ? 'Human' : 'AI-Generated'),
    votes: data.comprehensive?.votes || { ai: 0, human: 0 },
    consensus: data.comprehensive?.consensus ?? false,
  };
});

// 面部检测详情
const faceAnalysisDetails = computed(() => {
  if (!detectionData.value?.face?.raw_results) return null;
  
  const raw = detectionData.value.face.raw_results;
  return {
    model: raw.metadata?.model_name || 'Unknown',
    dataset: raw.metadata?.test_datasets?.[0] || 'Unknown',
    accuracy: raw.UADFV?.acc || 0,
    predictionMean: raw.UADFV?.pred_mean || 0,
    predictionStd: raw.UADFV?.pred_std || 0,
    timestamp: raw.metadata?.timestamp || '',
    weightsPath: raw.metadata?.weights_path || '',
  };
});

// 躯体检测详情
const bodyAnalysisDetails = computed(() => {
  if (!detectionData.value?.body?.raw_results?.criteria) return [];
  
  return detectionData.value.body.raw_results.criteria.map(criterion => ({
    name: criterion.name,
    description: BODY_CRITERIA.find(c => c.name === criterion.name)?.description || '详细检测标准',
    response: criterion.response,
    score: criterion.score,
    weight: criterion.weight,
    weightedScore: criterion.weighted_score,
    status: criterion.response === '是' ? 'success' : 'danger',
  }));
});

// 整体检测详情
const overallAnalysisDetails = computed(() => {
  if (!detectionData.value?.overall) return null;
  
  return {
    probability: detectionData.value.overall.raw_results?.prob || 0,
    confidence: detectionData.value.overall.confidence,
    prediction: detectionData.value.overall.prediction,
    aiProbability: detectionData.value.overall.ai_probability,
    humanProbability: detectionData.value.overall.human_probability,
  };
});

// 格式化函数
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

const formatPercentage = (value, decimals = 1) => {
  return (value * 100).toFixed(decimals) + '%';
};

// 生命周期管理
onMounted(() => {
  const videoId = getVideoId();
  if (videoId) {
    setTimeout(() => {
      checkDetectionStatus();
    }, 100);
  }
});

watch(() => route.query.id, (newId, oldId) => {
  if (newId && newId !== oldId) {
    stopPolling();
    restartDetection();
    setTimeout(() => {
      checkDetectionStatus();
    }, 500);
  }
}, { immediate: false });

onUnmounted(() => {
  stopPolling();
});
</script>
<template>
  <div class="digital-human-container">
    <!-- 标题和步骤控制 -->
    <div class="header-controls">
      <h3 class="section-heading">数字人检测分析</h3>
      
      <div class="step-controls" v-if="detectionStatus === 'completed'">
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
    <div class="steps-container" v-if="detectionStatus === 'completed'">
      <el-steps :active="activeStep" finish-status="success" align-center>
        <el-step
          v-for="(step, index) in steps"
          :key="index"
          :title="step.title"
          :description="step.description"
        />
      </el-steps>
    </div>

    <!-- 未开始检测状态 -->
    <div v-if="detectionStatus === 'not_started'" class="detection-prompt">
      <el-card class="prompt-card">
        <div class="prompt-content">
          <el-icon class="prompt-icon" size="48"><VideoCamera /></el-icon>
          <h4>数字人检测</h4>
          <p>启动多层次数字人检测分析系统</p>
          <div class="detection-features">
            <div class="feature-item">
              <el-icon><Camera /></el-icon>
              <span>面部真实性检测</span>
            </div>
            <div class="feature-item">
              <el-icon><VideoCamera /></el-icon>
              <span>躯体异常分析</span>
            </div>
            <div class="feature-item">
              <el-icon><CircleCheck /></el-icon>
              <span>整体特征评估</span>
            </div>
            <div class="feature-item">
              <el-icon><InfoFilled /></el-icon>
              <span>智能融合决策</span>
            </div>
          </div>
          <p class="prompt-desc">
            系统将从面部、躯体、整体三个维度进行深度分析，
            采用智能融合算法输出最终检测结果。预计分析时间：15-20分钟。
          </p>
          <el-button 
            type="primary" 
            size="large" 
            :icon="Play"
            @click="startDetection"
          >
            开始检测
          </el-button>
        </div>
      </el-card>
    </div>

    <!-- 加载中状态 -->
    <div v-else-if="detectionStatus === 'loading'" class="detection-loading">
      <el-card class="loading-card">
        <div class="loading-content">
          <el-icon class="loading-icon" size="48"><Refresh /></el-icon>
          <h4>正在启动检测...</h4>
          <p>系统正在初始化智能检测引擎</p>
        </div>
      </el-card>
    </div>

    <!-- 检测进行中状态 -->
    <div v-else-if="detectionStatus === 'processing'" class="detection-processing">
      <el-card class="processing-card">
        <div class="processing-header">
          <h4>
            <el-icon><VideoCamera /></el-icon>
            数字人检测进行中
          </h4>
          <el-button type="text" @click="checkDetectionStatus" :icon="Refresh">
            刷新状态
          </el-button>
        </div>
        
        <div class="processing-steps">
          <el-steps :active="1" status="process">
            <el-step title="面部检测" description="面部特征分析中" />
            <el-step title="躯体检测" description="躯体异常检测中" />
            <el-step title="整体检测" description="全局特征提取中" />
            <el-step title="综合评估" description="智能融合计算中" />
          </el-steps>
        </div>

        <div class="processing-info">
          <el-alert
            title="深度分析进行中"
            type="info"
            description="系统正在执行多层次检测算法，包括面部真实性检测、躯体异常分析、全局特征评估等，请耐心等待..."
            :closable="false"
            show-icon
          />
        </div>
      </el-card>
    </div>

    <!-- 检测失败状态 -->
    <div v-else-if="detectionStatus === 'failed'" class="detection-failed">
      <el-card class="failed-card">
        <div class="failed-content">
          <el-icon class="failed-icon" size="48"><Warning /></el-icon>
          <h4>检测失败</h4>
          <p class="error-message">{{ detectionError }}</p>
          <div class="failed-actions">
            <el-button type="primary" @click="restartDetection">
              重新检测
            </el-button>
            <el-button @click="checkDetectionStatus">
              刷新状态
            </el-button>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 检测完成状态 -->
    <div v-else-if="detectionStatus === 'completed' && detectionData" class="detection-content">
      
      <!-- 面部检测结果 -->
      <div v-if="activeStep === 0" class="analysis-step">
        <el-card class="analysis-card">
          <template #header>
            <div class="card-header">
              <div class="header-left">
                <el-icon><Camera /></el-icon>
                <span>面部检测分析</span>
                <el-tag v-if="detectionData.face" :type="detectionData.face.prediction === 'Human' ? 'success' : 'danger'">
                  {{ detectionData.face.prediction === 'Human' ? '真实面部' : 'AI生成面部' }}
                </el-tag>
              </div>
              <!-- 单步重检按钮 -->
              <el-button 
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

          <div class="face-analysis">
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
                    基于深度学习模型的面部关键点定位和特征提取分析
                  </div>
                </div>

                <!-- 面部表情分析图 -->
                <div class="image-item">
                  <div class="image-header">
                    <span class="item-name">表情真实性分析</span>
                    <el-tag type="info" size="small">表情检测</el-tag>
                  </div>
                  <div class="image-placeholder">
                    <el-image
                      :src="`/api/videos/${getVideoId()}/digital-human/face-expression`"
                      fit="contain"
                      :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/face-expression`]"
                      :hide-on-click-modal="true"
                    >
                      <template #error>
                        <div class="image-error">
                          <el-icon size="32"><Picture /></el-icon>
                          <div class="error-text">
                            <div>表情真实性分析图</div>
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
                    面部表情的自然性和协调性智能分析结果
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
                  多层次面部特征深度学习
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
                <p>基于深度学习算法的面部真实性检测结果</p>
                <div class="confidence-info">
                  <el-tag :type="detectionData.face?.confidence > 0.7 ? 'success' : 'warning'">
                    置信度: {{ (detectionData.face?.confidence * 100).toFixed(1) }}%
                  </el-tag>
                </div>
              </div>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 躯体检测结果 -->
      <div v-if="activeStep === 1" class="analysis-step">
        <el-card class="analysis-card">
          <template #header>
            <div class="card-header">
              <div class="header-left">
                <el-icon><VideoCamera /></el-icon>
                <span>躯体检测分析</span>
                <el-tag v-if="bodyDataFixed" :type="bodyDataFixed.prediction === 'Human' ? 'success' : 'danger'">
                  {{ bodyDataFixed.prediction === 'Human' ? '真实躯体' : 'AI生成躯体' }}
                </el-tag>
              </div>
              <!-- 单步重检按钮 -->
              <el-button 
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

          <div class="body-analysis">
            <!-- 主要指标 - 使用修复后的数据 -->
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
                <!-- 为每个检测项目预留图片位置 -->
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
                    <!-- 预留图片位置 -->
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
      </div>

      <!-- 整体检测结果 -->
      <!-- 整体检测结果 -->
      <div v-if="activeStep === 2" class="analysis-step">
        <el-card class="analysis-card">
          <template #header>
            <div class="card-header">
              <div class="header-left">
                <el-icon><Picture /></el-icon>
                <span>整体检测分析</span>
                <el-tag v-if="detectionData.overall" :type="detectionData.overall.prediction === 'Human' ? 'success' : 'danger'">
                  {{ detectionData.overall.prediction === 'Human' ? '真实内容' : 'AI生成内容' }}
                </el-tag>
              </div>
              <!-- 单步重检按钮 -->
              <el-button 
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

          <div class="overall-analysis">
            <!-- 主要指标 -->
            <div class="main-metrics">
              <el-statistic
                title="真实度"
                :value="detectionData.overall?.human_probability * 100"
                suffix="%"
                :value-style="{ color: detectionData.overall?.human_probability > 0.5 ? '#67c23a' : '#f56c6c' }"
              />
              <el-statistic
                title="AI概率"
                :value="detectionData.overall?.ai_probability * 100"
                suffix="%"
                :value-style="{ color: detectionData.overall?.ai_probability > 0.5 ? '#f56c6c' : '#67c23a' }"
              />
              <el-statistic
                title="置信度"
                :value="detectionData.overall?.confidence * 100"
                suffix="%"
                :value-style="{ color: '#409eff' }"
              />
            </div>

            <!-- 整体特征图片展示 -->
            <el-divider>整体特征分析图片</el-divider>
            <div class="detection-images">
              <div class="images-grid">
                <!-- 时空一致性分析图 -->
                <div class="image-item">
                  <div class="image-header">
                    <span class="item-name">时空一致性分析</span>
                    <el-tag type="primary" size="small">时序特征</el-tag>
                  </div>
                  <div class="image-placeholder">
                    <el-image
                      :src="`/api/videos/${getVideoId()}/digital-human/temporal-consistency`"
                      fit="contain"
                      :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/temporal-consistency`]"
                      :hide-on-click-modal="true"
                    >
                      <template #error>
                        <div class="image-error">
                          <el-icon size="32"><Picture /></el-icon>
                          <div class="error-text">
                            <div>时空一致性分析图</div>
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
                    视频帧间的时间连续性和空间一致性检测分析
                  </div>
                </div>

                <!-- 全局纹理特征图 -->
                <div class="image-item">
                  <div class="image-header">
                    <span class="item-name">全局纹理特征</span>
                    <el-tag type="info" size="small">纹理分析</el-tag>
                  </div>
                  <div class="image-placeholder">
                    <el-image
                      :src="`/api/videos/${getVideoId()}/digital-human/global-texture`"
                      fit="contain"
                      :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/global-texture`]"
                      :hide-on-click-modal="true"
                    >
                      <template #error>
                        <div class="image-error">
                          <el-icon size="32"><Picture /></el-icon>
                          <div class="error-text">
                            <div>全局纹理特征图</div>
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
                    整体画面的纹理一致性和生成痕迹检测
                  </div>
                </div>

                <!-- 生成痕迹检测图 -->
                <div class="image-item">
                  <div class="image-header">
                    <span class="item-name">生成痕迹检测</span>
                    <el-tag type="warning" size="small">痕迹分析</el-tag>
                  </div>
                  <div class="image-placeholder">
                    <el-image
                      :src="`/api/videos/${getVideoId()}/digital-human/generation-artifacts`"
                      fit="contain"
                      :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/generation-artifacts`]"
                      :hide-on-click-modal="true"
                    >
                      <template #error>
                        <div class="image-error">
                          <el-icon size="32"><Picture /></el-icon>
                          <div class="error-text">
                            <div>生成痕迹检测图</div>
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
                    AI生成算法特有的视觉伪影和生成痕迹识别
                  </div>
                </div>

                <!-- 边缘一致性分析图 -->
                <div class="image-item">
                  <div class="image-header">
                    <span class="item-name">边缘一致性分析</span>
                    <el-tag type="success" size="small">边缘检测</el-tag>
                  </div>
                  <div class="image-placeholder">
                    <el-image
                      :src="`/api/videos/${getVideoId()}/digital-human/edge-consistency`"
                      fit="contain"
                      :preview-src-list="[`/api/videos/${getVideoId()}/digital-human/edge-consistency`]"
                      :hide-on-click-modal="true"
                    >
                      <template #error>
                        <div class="image-error">
                          <el-icon size="32"><Picture /></el-icon>
                          <div class="error-text">
                            <div>边缘一致性分析图</div>
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
                    图像边缘的自然性和一致性深度学习检测
                  </div>
                </div>
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
                  :percentage="Math.round(detectionData.overall?.ai_probability * 100)"
                  status="exception"
                  :stroke-width="20"
                />
              </div>
              <div class="probability-bar">
                <div class="bar-label">真实内容概率</div>
                <el-progress
                  :percentage="Math.round(detectionData.overall?.human_probability * 100)"
                  status="success"
                  :stroke-width="20"
                />
              </div>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 综合评估结果 -->
      <div v-if="activeStep === 3" class="analysis-step">
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
            <el-divider>各模块检测得分</el-divider>
            <div class="component-scores">
              <div class="score-item" v-if="comprehensiveAnalysis.scores.face !== undefined">
                <div class="score-header">
                  <span>面部检测得分</span>
                </div>
                <el-progress
                  :percentage="Math.round(comprehensiveAnalysis.scores.face)"
                  :status="getProgressStatus(comprehensiveAnalysis.scores.face)"
                />
              </div>
              
              <div class="score-item" v-if="comprehensiveAnalysis.scores.body !== undefined">
                <div class="score-header">
                  <span>躯体检测得分</span>
                </div>
                <el-progress
                  :percentage="Math.round(comprehensiveAnalysis.scores.body)"
                  :status="getProgressStatus(comprehensiveAnalysis.scores.body)"
                />
              </div>
              
              <div class="score-item" v-if="comprehensiveAnalysis.scores.overall !== undefined">
                <div class="score-header">
                  <span>整体检测得分</span>
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
                  :title="comprehensiveAnalysis.consensus ? '检测结果一致' : '检测结果存在分歧'"
                  :type="comprehensiveAnalysis.consensus ? 'success' : 'warning'"
                  :description="comprehensiveAnalysis.consensus ? 
                    '多个检测模块的结果高度一致，增强了判定的可靠性。' : 
                    '不同检测模块的结果存在分歧，建议进一步人工审核。'"
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
                  面部、躯体、整体
                </el-descriptions-item>
                <el-descriptions-item label="算法类型">
                  深度学习 + 智能分析
                </el-descriptions-item>
                <el-descriptions-item label="融合策略">
                  多模块智能融合
                </el-descriptions-item>
                <el-descriptions-item label="风险等级" :span="2">
                  <el-tag :type="getRiskLevel(comprehensiveAnalysis.aiProbability / 100).type" size="large">
                    {{ getRiskLevel(comprehensiveAnalysis.aiProbability / 100).level }}
                  </el-tag>
                </el-descriptions-item>
              </el-descriptions>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button @click="restartDetection" :icon="Refresh">
          全部重新检测
        </el-button>
        <el-button type="primary" @click="activeStep = 3">
          查看综合结果
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 检测图片展示区域 */
.detection-images {
  margin: 1.5rem 0;
}

/* 算法详情 */
.algorithm-details {
  margin: 1.5rem 0;
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

/* 置信度信息 */
.confidence-info {
  margin-top: 0.75rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
  
  .algorithm-features {
    margin: 1rem 0;
  }
  
  .feature-card {
    padding: 1rem;
  }
}
/* 基础容器样式 */
.digital-human-container {
  height: 100%;
  overflow: auto;
  padding: 0 4px;
}

/* 标题和控制按钮 */
.header-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.section-heading {
  margin-bottom: 0;
  font-size: 1.125rem;
  font-weight: 500;
}

.step-controls {
  display: flex;
  gap: 0.5rem;
}

/* 步骤条 */
.steps-container {
  margin-bottom: 2rem;
}

/* 提示状态 */
.detection-prompt,
.detection-loading,
.detection-failed {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.prompt-card,
.loading-card,
.failed-card {
  max-width: 600px;
  width: 100%;
}

.prompt-content,
.loading-content,
.failed-content {
  text-align: center;
  padding: 2rem;
}

.prompt-icon,
.loading-icon,
.failed-icon {
  color: #409eff;
  margin-bottom: 1rem;
}

.failed-icon {
  color: #f56c6c;
}

.loading-icon {
  animation: spin 2s linear infinite;
}

/* 检测特性 */
.detection-features {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
  text-align: left;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 6px;
}

.prompt-desc {
  color: #909399;
  margin: 1rem 0 2rem;
  line-height: 1.6;
}

/* 处理中状态 */
.processing-card {
  min-height: 400px;
}

.processing-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.processing-header h4 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
}

.processing-steps {
  margin-bottom: 2rem;
}

.processing-info {
  margin-top: 2rem;
}

/* 失败状态 */
.error-message {
  color: #f56c6c;
  margin: 1rem 0;
}

.failed-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
}

/* 分析步骤 */
.analysis-step {
  margin-bottom: 2rem;
}

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

/* 风险评估 */
.risk-assessment,
.total-score {
  display: flex;
  align-items: center;
  gap: 2rem;
}

.risk-info,
.score-info {
  flex: 1;
}

.risk-info h4,
.score-info h4 {
  margin: 0 0 0.5rem 0;
  color: #409eff;
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

/* 综合分析 */
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

/* 操作按钮 */
.action-buttons {
  margin-top: 2rem;
  text-align: center;
  display: flex;
  gap: 1rem;
  justify-content: center;
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
  
  .final-result {
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .detection-features {
    grid-template-columns: 1fr;
  }
  
  .vote-result {
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }
  
  .images-grid {
    grid-template-columns: 1fr;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
  }
  
  .header-left {
    width: 100%;
  }
}

/* Element Plus 组件样式覆盖 */
:deep(.el-card__body) {
  padding: 20px;
}

:deep(.el-progress) {
  margin-bottom: 0.5rem;
}

:deep(.el-divider) {
  margin: 1.5rem 0;
}

:deep(.el-statistic__content) {
  display: flex;
  flex-direction: column;
  align-items: center;
}

:deep(.el-descriptions__label) {
  font-weight: 500;
}
</style>