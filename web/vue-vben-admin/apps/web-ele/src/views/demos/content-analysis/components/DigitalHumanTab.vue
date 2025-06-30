<script lang="ts" setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue';
import axios from 'axios';
import { useRoute } from 'vue-router';
import {
  ElButton,
  ElIcon,
  ElMessage,
  ElCard,
} from 'element-plus';
import {
  ArrowLeft,
  ArrowRight,
  VideoCamera,
  Refresh,
  Warning,
  VideoPlay,
  Camera,
  CircleCheck,
  InfoFilled,
} from '@element-plus/icons-vue';

// 导入拆分的子组件
import ComprehensiveTab from './digital-human/ComprehensiveTab.vue';
import FaceDetectionTab from './digital-human/FaceDetectionTab.vue';
import BodyDetectionTab from './digital-human/BodyDetectionTab.vue';
import OverallDetectionTab from './digital-human/OverallDetectionTab.vue';

const props = defineProps({
  videoData: {
    type: Object,
    default: () => ({}),
  },
});

const route = useRoute();

// 检测状态管理 - 改进状态逻辑
const detectionStatus = ref('not_started');
const detectionData = ref(null);
const detectionError = ref(null);
const isPolling = ref(false);
const pollingTimer = ref(null);
const maxRetries = ref(0);
const MAX_RETRIES = 100;

// 新增：组件卸载标志
const isUnmounted = ref(false);

// 新增：单独跟踪各个检测步骤的状态
const stepStatuses = ref({
  face: 'not_started',     // not_started, processing, completed, failed
  body: 'not_started',
  overall: 'not_started',
  comprehensive: 'not_started'
});

// 新增：检测进度信息
const detectionProgress = ref({
  current_step: '',
  progress: 0,
  processing_types: [] // 当前正在处理的检测类型
});

// 计算整体检测状态 - 支持部分完成
const overallDetectionStatus = computed(() => {
  // 如果从未开始过任何检测
  if (!detectionData.value && detectionStatus.value === 'not_started') {
    return 'not_started';
  }
  
  // 如果有任何检测在进行中
  const hasProcessing = Object.values(stepStatuses.value).some(status => status === 'processing');
  if (hasProcessing) {
    return 'partial_processing'; // 新状态：部分处理中
  }
  
  // 如果有任何检测完成
  const hasCompleted = Object.values(stepStatuses.value).some(status => status === 'completed');
  if (hasCompleted) {
    return 'partial_completed'; // 新状态：部分完成
  }
  
  // 如果所有检测都失败
  const allFailed = Object.values(stepStatuses.value).every(status => status === 'failed' || status === 'not_started');
  if (allFailed && Object.values(stepStatuses.value).some(status => status === 'failed')) {
    return 'failed';
  }
  
  return detectionStatus.value; // 返回原始状态
});

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

// 计算实际可用的步骤 - 综合评估放在第一位
const availableSteps = computed(() => {
  if (detectionStatus.value !== 'completed' || !detectionData.value) {
    return [
      { title: '综合评估', description: '多层次融合决策与最终判定', type: 'comprehensive' },
      { title: '面部检测', description: '基于Xception模型的面部伪造检测', type: 'face' },
      { title: '躯体检测', description: '基于多模态LLM的躯体异常检测', type: 'body' },
      { title: '整体检测', description: '全局特征的综合性分析检测', type: 'overall' },
    ];
  }

  const steps = [];
  const data = detectionData.value;
  
  // 综合评估始终放在第一位（只要有任何检测结果就显示）
  if (data.face || data.body || data.overall || data.comprehensive) {
    steps.push({ title: '综合评估', description: '多层次融合决策与最终判定', type: 'comprehensive' });
  }
  
  // 添加实际执行的检测步骤
  if (data.face) {
    steps.push({ title: '面部检测', description: '基于Xception模型的面部伪造检测', type: 'face' });
  }
  if (data.body) {
    steps.push({ title: '躯体检测', description: '基于多模态LLM的躯体异常检测', type: 'body' });
  }
  if (data.overall) {
    steps.push({ title: '整体检测', description: '全局特征的综合性分析检测', type: 'overall' });
  }
  
  return steps;
});

// 当前步骤类型 - 默认为综合评估
const currentStepType = computed(() => {
  return availableSteps.value[activeStep.value]?.type || 'comprehensive';
});

// 检测状态检查函数
const isStepAvailable = (stepType) => {
  // 综合评估：只要有任何完成的检测就可以查看
  if (stepType === 'comprehensive') {
    return stepStatuses.value.face === 'completed' || 
           stepStatuses.value.body === 'completed' || 
           stepStatuses.value.overall === 'completed';
  }
  
  // 其他检测：只有完成状态才能查看
  return stepStatuses.value[stepType] === 'completed';
};

// 检测步骤是否正在进行中
const isStepProcessing = (stepType) => {
  return stepStatuses.value[stepType] === 'processing';
};

// 检测步骤是否失败
const isStepFailed = (stepType) => {
  return stepStatuses.value[stepType] === 'failed';
};

// 获取检测结果或默认值
const getDetectionResult = (stepType) => {
  if (!detectionData.value || !detectionData.value[stepType]) {
    return {
      available: false,
      human_probability: 0,
      ai_probability: 0,
      confidence: 0,
      prediction: 'Not Detected',
      raw_results: null
    };
  }
  
  return {
    available: true,
    ...detectionData.value[stepType]
  };
};

// 修复躯体数据计算 - 添加可用性检查
const bodyDataFixed = computed(() => {
  const bodyResult = getDetectionResult('body');
  if (!bodyResult.available) return bodyResult;
  
  const body = detectionData.value.body;
  
  // 修复：如果所有检测项目都正常，应该是高真实度
  if (body.raw_results?.total_score === 1.0) {
    return {
      ...bodyResult,
      human_probability: 1.0,
      ai_probability: 0.0,
      confidence: 1.0,
      prediction: 'Human'
    };
  }
  
  return bodyResult;
});

// 综合分析计算 - 只计算可用的检测结果
const comprehensiveAnalysis = computed(() => {
  if (!detectionData.value) return null;
  
  const data = detectionData.value;
  const scores = {};
  let weightedSum = 0;
  let totalWeight = 0;
  
  // 只计算实际执行的检测
  const availableDetections = [];
  if (data.face) {
    scores.face = data.face.human_probability * 100;
    weightedSum += data.face.human_probability * DETECTION_WEIGHTS.face;
    totalWeight += DETECTION_WEIGHTS.face;
    availableDetections.push('face');
  }
  
  if (data.body) {
    scores.body = data.body.human_probability * 100;
    weightedSum += data.body.human_probability * DETECTION_WEIGHTS.body;
    totalWeight += DETECTION_WEIGHTS.body;
    availableDetections.push('body');
  }
  
  if (data.overall) {
    scores.overall = data.overall.human_probability * 100;
    weightedSum += data.overall.human_probability * DETECTION_WEIGHTS.overall;
    totalWeight += DETECTION_WEIGHTS.overall;
    availableDetections.push('overall');
  }
  
  // 如果只有一个检测结果，直接使用
  if (availableDetections.length === 1) {
    const singleType = availableDetections[0];
    const singleResult = data[singleType];
    return {
      scores,
      calculatedScore: singleResult.human_probability * 100,
      finalScore: singleResult.human_probability * 100,
      aiProbability: singleResult.ai_probability * 100,
      confidence: singleResult.confidence * 100,
      prediction: singleResult.prediction,
      votes: { ai: singleResult.prediction === 'AI-Generated' ? 1 : 0, human: singleResult.prediction === 'Human' ? 1 : 0 },
      consensus: true,
      availableDetections
    };
  }
  
  // 多个检测结果的综合计算
  const calculatedScore = totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 0;
  const finalScore = data.comprehensive?.human_probability * 100 || calculatedScore;
  
  return {
    scores,
    calculatedScore,
    finalScore,
    aiProbability: data.comprehensive?.ai_probability * 100 || (100 - finalScore),
    confidence: data.comprehensive?.confidence * 100 || 50,
    prediction: data.comprehensive?.prediction || (finalScore >= 50 ? 'Human' : 'AI-Generated'),
    votes: data.comprehensive?.votes || { 
      ai: availableDetections.filter(type => data[type].prediction === 'AI-Generated').length,
      human: availableDetections.filter(type => data[type].prediction === 'Human').length
    },
    consensus: data.comprehensive?.consensus ?? true,
    availableDetections
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

// 工具函数
const getVideoId = () => {
  return route.query.id || props.videoData?.video?.id;
};

const getStepName = (stepType) => {
  const names = {
    face: '面部检测',
    body: '躯体检测',
    overall: '整体检测',
    comprehensive: '综合评估'
  };
  return names[stepType] || stepType;
};

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

// API调用函数 - 改进状态更新逻辑
const checkDetectionStatus = async () => {
  try {
    const videoId = getVideoId();
    if (!videoId) {
      console.warn('数字人检测: 视频ID为空，停止轮询');
      stopPolling();
      return;
    }

    console.log(`数字人检测: 检查状态 - 视频ID: ${videoId}`);
    const response = await axios.get(`/api/videos/${videoId}/digital-human/status`);
    
    console.log('数字人检测状态响应:', response.data);
    
    if (response.data.code === 200) {
      const data = response.data.data;
      
      // 更新进度信息
      if (data.progress !== undefined) {
        detectionProgress.value.progress = data.progress;
        detectionProgress.value.current_step = data.current_step || '';
      }
      
      if (data.status === 'completed') {
        console.log('数字人检测完成，获取结果...');
        detectionStatus.value = 'completed';
        
        // 尝试加载完整结果
        await loadExistingDetectionResult();
        
        stopPolling();
        ElMessage.success('数字人检测完成');
      } else if (data.status === 'processing') {
        if (detectionStatus.value !== 'processing') {
          console.log('数字人检测进行中...');
          detectionStatus.value = 'processing';
        }
        
        // 根据当前步骤更新步骤状态
        updateProcessingStepStatus(data.current_step);
        
        // 检查是否有部分结果可以显示
        if (data.progress >= 30) { // 假设30%以上可能有部分结果
          try {
            await loadExistingDetectionResult();
          } catch (partialError) {
            // 部分结果加载失败，继续等待
            console.log('暂时无法获取部分结果，继续等待');
          }
        }
        
      } else if (data.status === 'failed') {
        console.error('数字人检测失败:', data.error_message);
        detectionStatus.value = 'failed';
        detectionError.value = data.error_message || '检测失败';
        stopPolling();
        ElMessage.error('检测失败: ' + detectionError.value);
      }
    } else {
      console.error('API响应错误:', response.data);
    }
  } catch (error) {
    console.error('查询检测状态失败:', error);
    if (error.response?.status === 404) {
      console.log('检测记录不存在，设置为未开始状态');
      detectionStatus.value = 'not_started';
      stopPolling();
    } else {
      console.error('检测状态查询异常:', error.response?.data || error.message);
    }
  }
};

// 新增：更新步骤状态的函数
const updateStepStatuses = (results) => {
  // 如果有 module_statuses 字段，直接使用
  if (results.module_statuses) {
    stepStatuses.value = {
      face: results.module_statuses.face || 'not_started',
      body: results.module_statuses.body || 'not_started',
      overall: results.module_statuses.overall || 'not_started',
      comprehensive: results.module_statuses.comprehensive || 'not_started'
    };
    return;
  }
  
  // 兼容旧版本：根据检测结果推断状态
  stepStatuses.value = {
    face: 'not_started',
    body: 'not_started',
    overall: 'not_started',
    comprehensive: 'not_started'
  };
  
  // 根据结果更新状态
  if (results?.face) {
    stepStatuses.value.face = 'completed';
  }
  if (results?.body) {
    stepStatuses.value.body = 'completed';
  }
  if (results?.overall) {
    stepStatuses.value.overall = 'completed';
  }
  if (results?.comprehensive) {
    stepStatuses.value.comprehensive = 'completed';
  }
};

// 新增：根据当前步骤更新处理状态
const updateProcessingStepStatus = (currentStep) => {
  // 重置处理状态
  Object.keys(stepStatuses.value).forEach(key => {
    if (stepStatuses.value[key] === 'processing') {
      stepStatuses.value[key] = 'not_started';
    }
  });
  
  // 根据当前步骤设置处理状态
  if (currentStep?.includes('face')) {
    stepStatuses.value.face = 'processing';
  } else if (currentStep?.includes('body')) {
    stepStatuses.value.body = 'processing';
  } else if (currentStep?.includes('overall')) {
    stepStatuses.value.overall = 'processing';
  } else if (currentStep?.includes('comprehensive')) {
    stepStatuses.value.comprehensive = 'processing';
  }
};

// 新增：加载已有检测结果的函数 - 支持部分完成
const loadExistingDetectionResult = async () => {
  try {
    const videoId = getVideoId();
    if (!videoId) return;

    const response = await axios.get(`/api/videos/${videoId}/digital-human/result`);
    if (response.data.code === 200) {
      console.log('加载已有数字人检测结果:', response.data.data);
      detectionData.value = response.data.data.detection;
      
      // 根据实际状态设置检测状态
      if (response.data.data.status === 'completed') {
        detectionStatus.value = 'completed';
      } else if (response.data.data.status === 'processing') {
        // 部分完成状态：有些模块完成了，有些还在进行中
        detectionStatus.value = 'partial_completed';
      }
      
      // 更新步骤状态
      updateStepStatuses(response.data.data.detection);
      
      console.log('数字人检测状态已更新，数据:', detectionData.value);
    }
  } catch (error) {
    console.log('没有找到已有的数字人检测结果:', error.response?.status);
    // 400可能表示部分完成但还在处理中，尝试检查状态
    if (error.response?.status === 400) {
      console.log('检测可能在进行中，检查状态...');
      setTimeout(() => {
        if (!isUnmounted.value) {
          checkDetectionStatus();
        }
      }, 100);
    } else if (error.response?.status !== 404) {
      console.error('加载检测结果异常:', error);
    }
  }
};

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

// 新增：防止事件冒泡的处理函数 - 但不要过度阻止
const stopEventPropagation = (event) => {
  if (event) {
    event.stopPropagation();
  }
};

// 新增：发射事件给父组件，通知内部tab变化（仅用于调试）
const emit = defineEmits(['update:activeTab']);

// 定义activeStep变量
const activeStep = ref(0);

// 修改：步骤切换逻辑 - 简化事件处理
const nextStep = () => {
  if (activeStep.value < availableSteps.value.length - 1) {
    activeStep.value++;
  }
};

const prevStep = () => {
  if (activeStep.value > 0) {
    activeStep.value--;
  }
};

const jumpToStep = (stepType) => {
  const stepIndex = availableSteps.value.findIndex(step => step.type === stepType);
  if (stepIndex !== -1) {
    activeStep.value = stepIndex;
  }
};

const handleModuleClick = (stepType) => {
  if (stepType === 'comprehensive') {
    return;
  }
  
  if (isStepAvailable(stepType)) {
    jumpToStep(stepType);
  } else {
    retestSingleStep(stepType);
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

// 修改：watch activeStep变化时通知父组件 - 使用 nextTick 避免冲突
watch(activeStep, async (newStep) => {
  await nextTick();
  const currentType = availableSteps.value[newStep]?.type;
  emit('update:activeTab', currentType);
}, { immediate: false });

// 修改初始化逻辑 - 改进状态判断
onMounted(async () => {
  const videoId = getVideoId();
  console.log('数字人检测组件初始化，视频ID:', videoId);
  
  if (videoId) {
    // 默认显示综合评估页面
    activeStep.value = 0;
    
    try {
      // 优先加载已有结果
      console.log('开始加载已有检测结果...');
      await loadExistingDetectionResult();
      
      console.log('加载结果完成，当前状态:', {
        detectionStatus: detectionStatus.value,
        hasData: !!detectionData.value,
        stepStatuses: stepStatuses.value
      });
      
      // 如果成功加载了数据，确保状态正确设置
      if (detectionData.value) {
        detectionStatus.value = 'completed';
        updateStepStatuses(detectionData.value);
        console.log('数据加载成功，状态已更新为completed');
      } else if (detectionStatus.value === 'not_started') {
        console.log('没有已有结果，检查检测状态...');
        // 延迟检查，避免在组件挂载阶段触发
        setTimeout(() => {
          if (!isUnmounted.value) {
            checkDetectionStatus();
          }
        }, 100);
      }
    } catch (error) {
      console.error('初始化检测状态时出错:', error);
      // 出错时也检查一下状态，但要确保组件未卸载
      setTimeout(() => {
        if (!isUnmounted.value) {
          checkDetectionStatus();
        }
      }, 100);
    }
  }
});

// 修改路由监听器
watch(() => route.query.id, (newId, oldId) => {
  if (newId && newId !== oldId) {
    stopPolling();
    restartDetection();
    // 延迟检查状态，避免在路由切换过程中触发
    setTimeout(() => {
      if (!isUnmounted.value) {
        checkDetectionStatus();
      }
    }, 500);
  }
}, { immediate: false });

// 修改：卸载钩子 - 增强清理逻辑
onUnmounted(() => {
  console.log('数字人检测组件即将卸载，清理资源...');
  isUnmounted.value = true;
  stopPolling();
  
  // 清理可能存在的延时器
  if (pollingTimer.value) {
    clearInterval(pollingTimer.value);
    pollingTimer.value = null;
  }
  
  // 重置所有状态
  detectionStatus.value = 'not_started';
  detectionData.value = null;
  detectionError.value = null;
  maxRetries.value = 0;
  
  // 清理事件监听器
  emit('update:activeTab', null);
});
</script>

<template>
  <!-- 修改：简化事件处理，避免过度阻止 -->
  <div 
    class="digital-human-container" 
    data-component="digital-human"
  >
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
          :disabled="activeStep === availableSteps.length - 1"
          @click="nextStep"
          size="small"
        >
          下一步 <el-icon class="el-icon--right"><ArrowRight /></el-icon>
        </el-button>
      </div>
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
            :icon="VideoPlay"
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
    <div v-else-if="overallDetectionStatus === 'partial_completed' || overallDetectionStatus === 'partial_processing' || detectionStatus === 'completed' || detectionStatus === 'partial_completed'" class="detection-content">
      
      <!-- 如果有检测在进行中，显示进度提示 -->
      <div v-if="overallDetectionStatus === 'partial_processing'" class="processing-banner">
        <el-alert
          title="部分检测进行中"
          :description="`当前步骤: ${detectionProgress.current_step || '未知'} - 进度: ${detectionProgress.progress}%`"
          type="info"
          :closable="false"
          show-icon
        >
          <template #action>
            <el-button size="small" @click="checkDetectionStatus" :icon="Refresh">
              刷新
            </el-button>
          </template>
        </el-alert>
      </div>
      
      <!-- 综合评估结果 - 使用子组件 -->
      <ComprehensiveTab
        v-if="currentStepType === 'comprehensive'"
        :comprehensive-analysis="comprehensiveAnalysis"
        :detection-data="detectionData"
        :is-step-available="isStepAvailable"
        :get-detection-result="getDetectionResult"
        :body-data-fixed="bodyDataFixed"
        :get-progress-status="getProgressStatus"
        :get-risk-level="getRiskLevel"
        :format-date-time="formatDateTime"
        :handle-module-click="handleModuleClick"
        :jump-to-step="jumpToStep"
        :start-detection="startDetection"
        :step-statuses="stepStatuses"
        :is-step-processing="isStepProcessing"
        :is-step-failed="isStepFailed"
      />

      <!-- 面部检测结果 - 使用子组件 -->
      <FaceDetectionTab
        v-else-if="currentStepType === 'face'"
        :is-step-available="isStepAvailable"
        :get-detection-result="getDetectionResult"
        :detection-data="detectionData"
        :get-video-id="getVideoId"
        :format-date-time="formatDateTime"
        :get-progress-status="getProgressStatus"
        :retest-single-step="retestSingleStep"
        :detection-status="detectionStatus"
        :step-status="stepStatuses.face"
        :is-processing="isStepProcessing('face')"
      />

      <!-- 躯体检测结果 - 使用子组件 -->
      <BodyDetectionTab
        v-else-if="currentStepType === 'body'"
        :is-step-available="isStepAvailable"
        :body-data-fixed="bodyDataFixed"
        :detection-data="detectionData"
        :body-analysis-details="bodyAnalysisDetails"
        :get-video-id="getVideoId"
        :get-progress-status="getProgressStatus"
        :retest-single-step="retestSingleStep"
        :detection-status="detectionStatus"
        :step-status="stepStatuses.body"
        :is-processing="isStepProcessing('body')"
      />

      <!-- 整体检测结果 - 使用子组件 -->
      <OverallDetectionTab
        v-else-if="currentStepType === 'overall'"
        :is-step-available="isStepAvailable"
        :get-detection-result="getDetectionResult"
        :detection-data="detectionData"
        :get-video-id="getVideoId"
        :retest-single-step="retestSingleStep"
        :detection-status="detectionStatus"
        :step-status="stepStatuses.overall"
        :is-processing="isStepProcessing('overall')"
      />

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button 
          @click="restartDetection" 
          :icon="Refresh"
        >
          全部重新检测
        </el-button>
        <el-button 
          type="primary" 
          @click="activeStep = 0" 
          v-if="currentStepType !== 'comprehensive'"
        >
          返回综合评估
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 新增：组件容器样式，确保事件隔离但不过度阻止 */
.digital-human-container {
  position: relative;
  z-index: 1;
  height: 100%;
  overflow: auto;
  padding: 0 4px;
}

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

/* 状态提示样式 */
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
  .header-controls {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
  
  .detection-features {
    grid-template-columns: 1fr;
  }
}
</style>