<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
// 导入所需的 Element Plus 组件
import {
  ElMessage,
  ElContainer,
  ElAside,
  ElMain,
  ElInput,
  ElButton,
  ElTag,
  ElProgress,
  ElCollapse,
  ElCollapseItem,
  ElAlert,
  ElEmpty,
  ElImage,
  ElIcon,
  ElTimeline,
  ElTimelineItem,
  ElCard,
  ElTooltip,
  ElDivider,
  ElResult,
  ElSteps,
  ElStep,
} from 'element-plus';

import {
  Picture,
  Timer,
  Download,
  Refresh,
  Document,
  VideoCamera,
  DataAnalysis,
  ChatLineSquare,
  ArrowRight,
  ArrowLeft,
  Search,
  Loading,
  CircleCheckFilled,
  CircleCloseFilled,
} from '@element-plus/icons-vue';

const route = useRoute();
const router = useRouter();
const videoId = ref(route.query.id);
const awemeId = ref(route.query.awemeId);
const profileId = ref(route.query.profileId);

const loading = ref(true);
const refreshing = ref(false);
const loadingVideos = ref(false);
const videoData = ref(null);
const videoList = ref([]);
const searchQuery = ref('');
const activeCollapse = ref(['logs']);
// 左侧边栏是否收起
const collapsed = ref(false);

// 切换左侧边栏收起状态
const toggleSidebar = () => {
  collapsed.value = !collapsed.value;
};

// 过滤视频列表
const filteredVideos = computed(() => {
  if (!searchQuery.value) return videoList.value;

  const query = searchQuery.value.toLowerCase();
  return videoList.value.filter(
    (video) => video.desc && video.desc.toLowerCase().includes(query),
  );
});

// 是否禁用分析按钮
const isAnalyzeDisabled = computed(() => {
  if (!videoData.value) return true;
  return (
    videoData.value.status === 'processing' ||
    videoData.value.status === 'completed'
  );
});
const formatTaskLogs = (task) => {
  if (!task || !task.logs || task.logs.length === 0) {
    return '暂无日志数据';
  }
  
  // 按时间正序排序
  const sortedLogs = [...task.logs].sort((a, b) => 
    new Date(a.created_at) - new Date(b.created_at)
  );
  
  // 格式化日志内容
  return sortedLogs.map(log => 
    `[${log.level}] ${formatDateTime(log.created_at)} - ${log.message}`
  ).join('\n');
};

// 获取分析按钮文本
const getAnalyzeButtonText = () => {
  if (!videoData.value) return '分析视频';

  switch (videoData.value.status) {
    case 'processing':
      return '分析中...';
    case 'completed':
      return '已分析';
    case 'failed':
      return '重新分析';
    case 'not_downloaded':
      return '下载并分析';
    default:
      return '开始分析';
  }
};

// 获取状态文本
const getStatusText = (status) => {
  const texts = {
    pending: '等待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '处理失败',
    not_downloaded: '未下载',
  };

  return texts[status] || '未知状态';
};

// 获取进度条状态
const getProgressStatus = (status) => {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'exception';
  if (status === 'not_downloaded') return 'info';
  return '';
};

// 获取进度标签类型
const getProgressTagType = (status) => {
  const types = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
    not_downloaded: 'info',
  };

  return types[status] || 'info';
};

// 获取任务标签类型
const getTagType = (status) => {
  const types = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
  };

  return types[status] || 'info';
};

// 获取时间线项类型
const getTimelineItemType = (status) => {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  return 'primary';
};

// 自动刷新计时器
let refreshTimer = null;

// 加载用户视频列表
const loadUserVideos = async () => {
  if (!profileId.value) {
    console.warn('缺少用户ID参数');
    return;
  }

  try {
    loadingVideos.value = true;

    // 先检查 profileId 是否需要转换（是否为 sec_user_id 格式）
    if (profileId.value.startsWith('MS4') || profileId.value.length > 20) {
      console.log('需要转换的 sec_user_id:', profileId.value);

      // 调用 API 获取真实用户 ID
      const userResponse = await axios.get(
        `/api/account/by-secuid/${profileId.value}`,
      );
      if (userResponse.data.code === 200 && userResponse.data.data) {
        // 更新为真实的用户 ID
        profileId.value = userResponse.data.data.id;
        console.log('已将 sec_user_id 转换为真实 ID:', profileId.value);
      } else {
        throw new Error('无法获取用户信息');
      }
    }

    // 使用正确的用户 ID 获取视频列表
    const response = await axios.get(`/api/account/${profileId.value}/videos`, {
      params: { page: 1, per_page: 100 },
    });

    if (response.data.code === 200) {
      videoList.value = response.data.data.videos || [];
      console.log(`成功加载 ${videoList.value.length} 个视频`);
    } else {
      throw new Error(response.data.message || '获取视频列表失败');
    }
  } catch (error) {
    console.error('加载视频列表失败:', error);
    ElMessage.error(`加载视频列表失败: ${error.message || '未知错误'}`);
  } finally {
    loadingVideos.value = false;
  }
};

// 选择视频
const selectVideo = (video) => {
  // 如果选择的是当前视频，不做任何操作
  if (video.aweme_id === awemeId.value) return;

  // 更新路由
  router.push({
    name: 'VideoProcessingDetails',
    query: {
      awemeId: video.aweme_id,
      id: video.video_file_id,
      profileId: profileId.value,
    },
  });

  // 重新加载数据
  awemeId.value = video.aweme_id;
  videoId.value = video.video_file_id;
  fetchProcessingDetails();
};

// 获取视频处理详情
const fetchProcessingDetails = async () => {
  if (!awemeId.value) {
    ElMessage.error('缺少视频ID参数');
    return;
  }

  try {
    loading.value = true;

    const response = await axios.get(
      `/api/account/videos/${awemeId.value}/processing-details`,
    );

    if (response.data.code === 200) {
      videoData.value = response.data.data;

      // 如果处理还在进行中，设置自动刷新
      if (
        videoData.value.status === 'processing' ||
        videoData.value.status === 'pending'
      ) {
        setupAutoRefresh();
      } else {
        clearAutoRefresh();
      }

      // 默认选择最新的非完成任务，或最后一个任务
      if (videoData.value.tasks && videoData.value.tasks.length > 0) {
        const processingIndex = videoData.value.tasks.findIndex(
          (t) => t.status === 'processing',
        );
        if (processingIndex >= 0) {
          selectStep(processingIndex);
        } else {
          selectStep(videoData.value.tasks.length - 1);
        }
      } else if (videoData.value.source_type === 'download') {
        selectStep(-1);
      }
    } else {
      throw new Error(response.data.message || '获取处理详情失败');
    }
  } catch (error) {
    console.error('获取视频处理详情失败:', error);
    ElMessage.error(`获取处理详情失败: ${error.message || '未知错误'}`);
  } finally {
    loading.value = false;
  }
};

// 分析视频
const analyzeVideo = async () => {
  if (!awemeId.value || isAnalyzeDisabled.value) return;

  try {
    refreshing.value = true;

    const response = await axios.post(
      `/api/account/videos/${awemeId.value}/analyze`,
    );

    if (response.data.code === 200) {
      ElMessage.success('分析任务已启动');

      // 更新数据
      await fetchProcessingDetails();
    } else {
      throw new Error(response.data.message || '启动分析失败');
    }
  } catch (error) {
    console.error('分析视频失败:', error);
    ElMessage.error(`分析失败: ${error.message || '未知错误'}`);
  } finally {
    refreshing.value = false;
  }
};

// 刷新数据
const refreshData = async () => {
  if (refreshing.value) return;

  refreshing.value = true;
  await fetchProcessingDetails();
  refreshing.value = false;
};

// 设置自动刷新
const setupAutoRefresh = () => {
  clearAutoRefresh(); // 先清除已有的定时器
  refreshTimer = setInterval(() => {
    refreshData();
  }, 5000); // 每5秒刷新一次
};

// 清除自动刷新
const clearAutoRefresh = () => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
};

// 返回上一页
const goBack = () => {
  router.go(-1);
};

// 查看分析报告
const viewAnalysisReport = () => {
  if (videoData.value && videoData.value.video_id) {
    router.push(
      `/main/analysis-records/details?id=${videoData.value.video_id}`,
    );
  } else {
    ElMessage.warning('无法查看分析报告，缺少视频文件ID');
  }
};

// 获取任务图标
const getTaskIcon = (taskType) => {
  const icons = {
    transcription: Document, // 视频转录
    extract: DataAnalysis, // 信息提取
    summary: ChatLineSquare, // 生成摘要
    assessment: DataAnalysis, // 内容评估
    classify: ChatLineSquare, // 风险分类
    report: Document, // 威胁报告
    default: Document,
  };

  return icons[taskType] || icons.default;
};

// 获取任务名称
const getTaskName = (taskType) => {
  const names = {
    transcription: '视频转录',
    extract: '信息提取',
    summary: '生成摘要',
    assessment: '内容评估',
    classify: '风险分类',
    report: '威胁报告',
    default: '处理任务',
  };

  return names[taskType] || names.default;
};

// 新增变量，用于步骤条
const activeStep = ref(0);
const selectedTaskIndex = ref(-2); // -2:未选择 -1:下载步骤 0及以上:任务索引

// 计算得到当前选中的任务
const selectedTask = computed(() => {
  if (
    selectedTaskIndex.value >= 0 &&
    videoData.value &&
    videoData.value.tasks
  ) {
    return videoData.value.tasks[selectedTaskIndex.value];
  }
  return null;
});

// 计算是否显示下载详情
const showDownloadDetails = computed(() => {
  return (
    selectedTaskIndex.value === -1 &&
    videoData.value &&
    videoData.value.source_type === 'download'
  );
});

// 选择步骤
const selectStep = (index) => {
  selectedTaskIndex.value = index;

  // 设置activeStep
  if (index === -1) {
    activeStep.value = 0;
  } else {
    // 如果有下载步骤，需要+1
    activeStep.value =
      videoData.value && videoData.value.source_type === 'download'
        ? index + 1
        : index;
  }
};

// 获取步骤状态
const getStepStatus = (status) => {
  switch (status) {
    case 'completed':
      return 'success';
    case 'processing':
      return 'process';
    case 'failed':
      return 'error';
    case 'pending':
      return 'wait';
    default:
      return 'wait';
  }
};

// 获取步骤描述
const getStepDescription = (task) => {
  switch (task.status) {
    case 'completed':
      return `已完成 (${formatDateTime(task.completed_at)})`;
    case 'processing':
      return `处理中 (${task.progress}%)`;
    case 'failed':
      return '处理失败';
    case 'pending':
      return '等待处理';
    default:
      return '';
  }
};

// 获取任务日志 - 使用固定占位符
const getTaskLog = (task) => {
  return '这是任务日志内容的占位文本。后续将由实际处理日志替代。';
};

// 获取下载日志 - 使用固定占位符
const getDownloadLog = () => {
  return '这是下载日志内容的占位文本。后续将由实际下载日志替代。';
};

// 格式化日期
const formatDate = (dateString) => {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
};

// 格式化日期时间
const formatDateTime = (dateString) => {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return `${formatDate(dateString)} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
};

// 组件挂载时获取数据
onMounted(() => {
  // 1. 尝试从URL查询参数获取profileId
  if (route.query.profileId) {
    profileId.value = route.query.profileId;
    console.log('从URL获取到用户ID:', profileId.value);
  }
  // 2. 尝试从localStorage获取
  else if (localStorage.getItem('lastProfileId')) {
    profileId.value = localStorage.getItem('lastProfileId');
    console.log('从localStorage获取到用户ID:', profileId.value);
  }
  // 3. 尝试从referrer URL中解析userId
  else {
    const referrer = document.referrer;
    if (referrer && referrer.includes('userId=')) {
      const urlObj = new URL(referrer);
      const userId = urlObj.searchParams.get('userId');
      if (userId) {
        profileId.value = userId;
        localStorage.setItem('lastProfileId', userId);
        console.log('从referrer获取到用户ID:', profileId.value);
      }
    }
  }

  // 加载视频处理详情
  fetchProcessingDetails();

  // 尝试加载用户视频列表
  if (profileId.value) {
    loadUserVideos();
  } else {
    console.warn('无法获取用户ID，将不加载视频列表');
  }
});

// 组件卸载前清除定时器
onBeforeUnmount(() => {
  clearAutoRefresh();
});
</script>

<template>
  <div class="video-process-container">
    <el-container v-loading="loading">
      <!-- 左侧：视频列表 -->
      <el-aside :width="collapsed ? '60px' : '200px'" class="video-list-aside">
        <!-- 收起/展开按钮 -->
        <div class="sidebar-toggle" @click="toggleSidebar">
          <el-icon v-if="collapsed">
            <ArrowRight />
          </el-icon>
          <el-icon v-else>
            <ArrowLeft />
          </el-icon>
        </div>

        <!-- 边栏内容 - 当展开时显示 -->
        <template v-if="!collapsed">
          <div class="aside-header">
            <el-input
              v-model="searchQuery"
              placeholder="搜索视频"
              clearable
              prefix-icon="Search"
              size="small"
            />
          </div>
          <div class="video-list-container">
            <!-- 添加视频列表内容 -->
            <div v-if="loadingVideos" class="list-loading">
              <el-icon class="rotating"><Loading /></el-icon>
              <span>加载视频列表...</span>
            </div>
            <el-empty
              v-else-if="videoList.length === 0"
              description="暂无视频"
            />
            <div v-else class="video-list">
              <div
                v-for="video in filteredVideos"
                :key="video.aweme_id"
                class="video-list-item"
                :class="{ active: video.aweme_id === awemeId }"
                @click="selectVideo(video)"
              >
                <div class="video-list-cover">
                  <el-image :src="video.cover_url" fit="cover">
                    <template #error>
                      <div class="cover-placeholder">
                        <el-icon><Picture /></el-icon>
                      </div>
                    </template>
                  </el-image>
                  <div
                    v-if="video.analysis_status === 'completed'"
                    class="video-status-badge completed"
                  >
                    <el-icon><CircleCheckFilled /></el-icon>
                  </div>
                  <div
                    v-else-if="video.analysis_status === 'processing'"
                    class="video-status-badge processing"
                  >
                    <el-icon><Loading /></el-icon>
                  </div>
                </div>
                <div class="video-list-info">
                  <div class="video-title">{{ video.desc || '无标题' }}</div>
                  <div class="video-meta">
                    <el-icon><Timer /></el-icon>
                    {{ formatDate(video.create_time) }}
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="aside-footer">
            <el-button
              size="small"
              @click="loadUserVideos"
              :loading="loadingVideos"
            >
              <el-icon><Refresh /></el-icon> 刷新列表
            </el-button>
          </div>
        </template>

        <!-- 收起状态下只显示图标列表 -->
        <div v-else class="collapsed-video-list">
          <el-tooltip content="返回" placement="right">
            <div class="collapsed-action-icon" @click="goBack">
              <el-icon><ArrowLeft /></el-icon>
            </div>
          </el-tooltip>
          <el-tooltip content="刷新列表" placement="right">
            <div
              class="collapsed-action-icon"
              @click="loadUserVideos"
              :class="{ loading: loadingVideos }"
            >
              <el-icon><Refresh /></el-icon>
            </div>
          </el-tooltip>

          <el-divider></el-divider>

          <!-- 简化的视频列表 -->
          <div v-if="videoList.length > 0" class="collapsed-videos">
            <el-tooltip
              v-for="video in videoList"
              :key="video.aweme_id"
              :content="video.desc || '无标题'"
              placement="right"
            >
              <div
                class="collapsed-video-item"
                :class="{ active: video.aweme_id === awemeId }"
                @click="selectVideo(video)"
              >
                <el-image :src="video.cover_url" fit="cover">
                  <template #error>
                    <el-icon><Picture /></el-icon>
                  </template>
                </el-image>
              </div>
            </el-tooltip>
          </div>
        </div>
      </el-aside>

      <!-- 右侧：处理详情 -->
      <el-main class="process-detail-main">
        <!-- 未下载状态显示结果页 -->
        <template
          v-if="!loading && videoData && videoData.status === 'not_downloaded'"
        >
          <div class="not-downloaded-container">
            <el-result
              icon="warning"
              title="视频尚未下载"
              sub-title="该视频需要先从平台下载到服务器才能进行分析"
            >
              <template #extra>
                <el-button
                  type="primary"
                  @click="analyzeVideo"
                  :loading="refreshing"
                >
                  下载并分析视频
                </el-button>
                <el-button @click="goBack">返回</el-button>
              </template>
            </el-result>
          </div>
        </template>
        <!-- 已下载状态显示详细处理信息 -->
        <template v-else-if="!loading && videoData">
          <!-- 视频基本信息 -->
          <div class="video-info-header">
            <div class="video-cover">
              <el-image
                :src="videoData.cover_url"
                fit="cover"
                style="width: 120px; height: 160px; border-radius: 4px"
              >
                <template #error>
                  <div class="image-placeholder">
                    <el-icon><Picture /></el-icon>
                  </div>
                </template>
              </el-image>
            </div>

            <div class="video-info-content">
              <h2 class="video-title">{{ videoData.desc || '无标题' }}</h2>
              <div class="video-meta">
                <el-tag
                  size="small"
                  :type="
                    videoData.source_type === 'upload' ? 'success' : 'primary'
                  "
                >
                  {{
                    videoData.source_type === 'upload' ? '上传视频' : '抖音视频'
                  }}
                </el-tag>
                <span v-if="videoData.download_time" class="time-info">
                  <el-icon><Timer /></el-icon>
                  {{ formatDate(videoData.download_time) }}
                </span>
              </div>

              <!-- 总体进度指示器 (环形) -->
              <div class="overall-progress-container">
                <el-progress
                  type="dashboard"
                  :percentage="Math.round(videoData.progress)"
                  :status="getProgressStatus(videoData.status)"
                  :stroke-width="12"
                  :width="120"
                >
                  <template #default="{ percentage }">
                    <div class="central-progress">
                      <span class="percentage-value">{{ percentage }}%</span>
                      <span class="percentage-label">{{
                        getStatusText(videoData.status)
                      }}</span>
                    </div>
                  </template>
                </el-progress>

                <div class="progress-info">
                  <div class="progress-status">
                    当前状态：
                    <el-tag :type="getProgressTagType(videoData.status)">
                      {{ getStatusText(videoData.status) }}
                    </el-tag>
                  </div>
                  <div
                    v-if="videoData.status === 'completed'"
                    class="progress-time"
                  >
                    <span class="time-label">完成于:</span>
                    <span class="time-value">{{
                      formatDateTime(videoData.completed_time)
                    }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 处理阶段步骤条 -->
          <div class="processing-steps">
            <h3>处理阶段</h3>

            <!-- 步骤条 -->
            <el-steps
              :active="activeStep"
              finish-status="success"
              :process-status="
                videoData.status === 'processing' ? 'process' : 'finish'
              "
              align-center
            >
              <!-- 下载步骤(如果是从URL下载视频) -->
              <el-step
                v-if="videoData.source_type === 'download'"
                title="下载视频"
                :status="'success'"
                :icon="Download"
                description="视频已从抖音成功下载到服务器"
                @click="selectStep(-1)"
              />

              <!-- 动态处理步骤 -->
              <el-step
                v-for="(task, index) in videoData.tasks"
                :key="task.id"
                :title="getTaskName(task.task_type)"
                :status="getStepStatus(task.status)"
                :icon="getTaskIcon(task.task_type)"
                :description="getStepDescription(task)"
                @click="selectStep(index)"
              />
            </el-steps>

            <!-- 步骤详情 -->
            <div class="step-details" v-if="selectedTask">
              <el-card class="step-detail-card" :class="selectedTask.status">
                <template #header>
                  <div class="step-card-header">
                    <h4>{{ getTaskName(selectedTask.task_type) }} 详情</h4>
                    <el-tag
                      size="small"
                      :type="getTagType(selectedTask.status)"
                    >
                      {{ getStatusText(selectedTask.status) }}
                    </el-tag>
                  </div>
                </template>

                <div class="task-progress">
                  <el-progress
                    :percentage="selectedTask.progress"
                    :status="getProgressStatus(selectedTask.status)"
                    :stroke-width="8"
                    :format="(percent) => `${percent}%`"
                  />
                </div>

                <div class="task-timeline">
                  <div v-if="selectedTask.started_at" class="timeline-row">
                    <span class="timeline-label">开始时间:</span>
                    <span class="timeline-value">{{
                      formatDateTime(selectedTask.started_at)
                    }}</span>
                  </div>
                  <div v-if="selectedTask.completed_at" class="timeline-row">
                    <span class="timeline-label">完成时间:</span>
                    <span class="timeline-value">{{
                      formatDateTime(selectedTask.completed_at)
                    }}</span>
                  </div>
                </div>

                <div v-if="selectedTask.error" class="task-error">
                  <el-alert
                    type="error"
                    :title="selectedTask.error"
                    :closable="false"
                    show-icon
                  />
                </div>

                <!-- 步骤日志 -->
                <div class="step-log">
                  <h5>处理日志</h5>
                  <el-input
                    type="textarea"
                    :rows="8"
                    placeholder="暂无日志数据"
                    readonly
                    :model-value="formatTaskLogs(selectedTask)"
                    class="log-textarea"
                  ></el-input>
                </div>
              </el-card>
            </div>

            <!-- 下载步骤详情 -->
            <div class="step-details" v-else-if="showDownloadDetails">
              <el-card class="step-detail-card completed">
                <template #header>
                  <div class="step-card-header">
                    <h4>下载视频 详情</h4>
                    <el-tag size="small" type="success">已完成</el-tag>
                  </div>
                </template>

                <p>视频已从抖音成功下载到服务器</p>

                <div class="task-timeline">
                  <div class="timeline-row">
                    <span class="timeline-label">下载时间:</span>
                    <span class="timeline-value">{{
                      formatDateTime(videoData.download_time)
                    }}</span>
                  </div>
                </div>

                <!-- 下载日志 -->
                <div class="step-log">
                  <h5>下载日志</h5>
                  <el-input
                    type="textarea"
                    :rows="8"
                    placeholder="暂无日志数据"
                    readonly
                    :model-value="getDownloadLog()"
                    class="log-textarea"
                  ></el-input>
                </div>
              </el-card>
            </div>

            <div class="step-actions">
              <el-button @click="refreshData" :loading="refreshing">
                <el-icon><Refresh /></el-icon> 刷新数据
              </el-button>
              <el-button
                v-if="videoData.status === 'completed'"
                type="success"
                @click="viewAnalysisReport"
              >
                查看分析报告
              </el-button>
              <el-button
                type="primary"
                @click="analyzeVideo"
                :disabled="isAnalyzeDisabled"
              >
                {{ getAnalyzeButtonText() }}
              </el-button>
            </div>
          </div>
        </template>

        <!-- 无数据情况显示空状态 -->
        <template v-else-if="!loading">
          <div class="empty-state">
            <el-empty description="暂无视频处理数据">
              <el-button @click="goBack">返回</el-button>
            </el-empty>
          </div>
        </template>
      </el-main>
    </el-container>
  </div>
</template>

<style scoped>
/* 容器样式 */
.video-process-container {
  height: calc(100vh - 100px);
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  display: flex;
}

/* 左侧边栏基础样式 */
.video-list-aside {
  display: flex;
  flex-direction: column;
  border-right: 1px solid #e4e7ed;
  background-color: #f5f7fa;
  height: 100%;
  overflow: hidden;
  position: relative;
  transition: width 0.3s;
}

.sidebar-toggle {
  position: absolute;
  top: 50%;
  right: -6px;
  width: 24px;
  height: 24px;
  background-color: #409eff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 10;
  transform: translateY(-50%);
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.sidebar-toggle:hover {
  background-color: #66b1ff;
}

/* 收起状态样式 */
.collapsed-video-list {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 20px;
  width: 100%;
  overflow-y: auto;
}

.collapsed-action-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-radius: 4px;
  margin-bottom: 5px;
  font-size: 18px;
  color: #606266;
}

.collapsed-action-icon:hover {
  background-color: #e9f5ff;
  color: #409eff;
}

.collapsed-videos {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 10px 0;
  gap: 10px;
}

.collapsed-video-item {
  width: 40px;
  height: 40px;
  border-radius: 4px;
  overflow: hidden;
  cursor: pointer;
  position: relative;
}

.collapsed-video-item.active {
  border: 2px solid #409eff;
}

/* 视频列表布局 */
.aside-header,
.aside-footer {
  padding: 12px;
  border-bottom: 1px solid #e4e7ed;
}

.video-list-container {
  flex: 1;
  overflow-y: auto;
}

.list-loading {
  height: 100px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #909399;
}

.rotating {
  animation: rotate 2s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.video-list-item {
  display: flex;
  gap: 8px;
  padding: 8px;
  cursor: pointer;
  border-bottom: 1px solid #ebeef5;
  transition: all 0.3s ease;
}

.video-list-item:hover {
  background-color: #ecf5ff;
}

.video-list-item.active {
  background-color: #ecf5ff;
  border-right: 3px solid #409eff;
}

.video-list-cover {
  width: 50px;
  height: 70px;
  flex-shrink: 0;
  position: relative;
  border-radius: 4px;
  overflow: hidden;
}

.video-status-badge {
  position: absolute;
  top: 0;
  right: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.video-status-badge.completed {
  background-color: #67c23a;
  color: white;
}

.video-status-badge.processing {
  background-color: #e6a23c;
  color: white;
}

.video-list-info {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.video-list-info .video-title {
  font-size: 11px;
  max-height: 30px;
  line-height: 15px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  margin-bottom: 3px;
}
.video-list-info .video-meta {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 10px;
  color: #909399;
  margin-top: 2px;
}

.video-list-info .video-meta .el-icon {
  font-size: 10px;
}
/* 右侧主内容区样式 */
.process-detail-main {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
  background-color: #f5f7fa;
}

/* 未下载状态容器样式 */
.not-downloaded-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin: 20px;
  padding: 30px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 空状态样式 */
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80%;
}

/* 视频信息卡片 */
.video-info-header {
  display: flex;
  gap: 20px;
  margin-bottom: 30px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.video-title {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 20px;
}

.time-info {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #909399;
}

/* 进度指示器样式 */
.overall-progress-container {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-top: 15px;
}

.central-progress {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.percentage-value {
  font-size: 24px;
  font-weight: bold;
}

.percentage-label {
  margin-top: 5px;
  font-size: 12px;
  color: #606266;
}

/* 步骤条样式 */
.processing-steps {
  margin: 30px 0;
  padding: 20px;
  border-radius: 8px;
  background-color: white;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.processing-steps h3 {
  margin-top: 0;
  margin-bottom: 25px;
  font-size: 18px;
  color: #303133;
  font-weight: normal;
  padding-bottom: 15px;
  border-bottom: 1px solid #ebeef5;
}

:deep(.el-step__title),
:deep(.el-step__icon-inner) {
  cursor: pointer;
}

:deep(.el-step__title):hover,
:deep(.el-step__icon):hover {
  color: #409eff;
}

/* 步骤详情卡片 */
.step-details {
  margin-top: 30px;
}

.step-detail-card {
  border-radius: 8px;
}

.step-detail-card.completed {
  border-left: 4px solid #67c23a;
}

.step-detail-card.processing {
  border-left: 4px solid #e6a23c;
}

.step-detail-card.failed {
  border-left: 4px solid #f56c6c;
}

.step-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.step-card-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 500;
}

.task-timeline {
  background-color: #f8f9fa;
  padding: 8px 12px;
  border-radius: 4px;
  margin: 10px 0;
  font-size: 13px;
}

.timeline-row {
  display: flex;
  margin-bottom: 4px;
}

.timeline-label {
  width: 80px;
  color: #606266;
}

.step-log {
  margin-top: 20px;
}

.step-log h5 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 14px;
  color: #606266;
}

.log-textarea {
  font-family: 'Courier New', Courier, monospace;
  font-size: 12px;
}

/* 底部操作按钮 */
.bottom-actions {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 30px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 图片占位符 */
.image-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  background-color: #f5f7fa;
  color: #909399;
  border-radius: 4px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .video-info-header {
    flex-direction: column;
  }

  .overall-progress-container {
    flex-direction: column;
  }

  .video-list-aside {
    width: 100% !important;
    max-height: 300px;
  }
}
.step-actions {
  display: flex;
  justify-content: center; /* 确保按钮居中 */
  gap: 15px;
  margin-top: 30px; /* 增加与上方步骤详情的间距 */
  padding-top: 20px; /* 内部填充，进一步增加间距 */
  border-top: 1px solid #ebeef5; /* 添加上边框作为分隔 */
}
</style>
