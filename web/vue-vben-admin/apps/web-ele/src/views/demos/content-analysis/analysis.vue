<script lang="ts" setup>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import {
  ElMessage,
  ElMenu,
  ElMenuItem,
  ElCard,
  ElIcon,
  ElTag,
  ElResult,
  ElButton,
  ElLoading,
} from 'element-plus';
import {
  Close,
  VideoPlay,
  Expand,
  Fold,
  Document,
  VideoCamera,
  Warning,
  TrendCharts,
} from '@element-plus/icons-vue';

// 创建markdown-it实例
const md = new MarkdownIt({
  html: true,
  breaks: true,
  linkify: true,
  typographer: true,
});

// 导入拆分出的标签页组件
import SummaryTab from './components/SummaryTab.vue';
import SubtitlesTab from './components/SubtitlesTab.vue';
import DigitalHumanTab from './components/DigitalHumanTab.vue';
import ProcessTab from './components/ProcessTab.vue';
import FactCheckTab from './components/FactCheckTab.vue';
import ThreatReportTab from './components/ThreatReportTab.vue';

const router = useRouter();
const route = useRoute();
const activeTab = ref('summary');

// 新增：视频面板收起状态
const isVideoCollapsed = ref(false);
const videoDuration = ref(0);
const videoPlayer = ref(null);

// 数据状态
const loading = ref(true);
const videoData = ref(null);
const videoSrc = ref('');
const subtitlesData = ref({ chunks: [], text: '' });
const summary = ref('');
const assessmentData = ref({});
const summaryLoading = ref(false);
const factCheckLoading = ref(false);
const factCheckData = ref(null);
const factCheckError = ref(null);
const factCheckNotFound = ref(false);
const reportLoading = ref(false);
const reportData = ref(null);
const reportError = ref(null);

// 添加数字人检测状态跟踪
const digitalHumanData = ref(null);
const digitalHumanLoading = ref(false);

// 数字人检测状态
const digitalHumanStatus = computed(() => {
  if (digitalHumanLoading.value) {
    return { text: '检测中', color: 'warning' };
  }
  if (!digitalHumanData.value) {
    return { text: '未检测', color: 'info' };
  }
  
  const hasResults = digitalHumanData.value.face || digitalHumanData.value.body || digitalHumanData.value.overall;
  if (hasResults) {
    const isAI = digitalHumanData.value.comprehensive?.prediction === 'AI-Generated';
    return { 
      text: isAI ? 'AI生成' : '真实内容', 
      color: isAI ? 'danger' : 'success' 
    };
  }
  
  return { text: '未完成', color: 'info' };
});

// 事实核查状态
const factCheckStatus = computed(() => {
  if (factCheckLoading.value) {
    return { text: '核查中', color: 'warning' };
  }
  if (factCheckError.value) {
    return { text: '核查失败', color: 'danger' };
  }
  if (!factCheckData.value) {
    return { text: '未核查', color: 'info' };
  }
  
  if (factCheckData.value.status === 'processing') {
    return { text: '处理中', color: 'warning' };
  }
  if (factCheckData.value.status === 'completed') {
    const hasIssues = factCheckData.value.results?.some(item => 
      item.verification_result?.credibility === 'low' || 
      item.verification_result?.accuracy === 'questionable'
    );
    return { 
      text: hasIssues ? '发现问题' : '核查通过', 
      color: hasIssues ? 'danger' : 'success' 
    };
  }
  
  return { text: '未完成', color: 'info' };
});

// 威胁报告状态
const threatReportStatus = computed(() => {
  if (reportLoading.value) {
    return { text: '生成中', color: 'warning' };
  }
  if (reportError.value) {
    return { text: '生成失败', color: 'danger' };
  }
  if (!reportData.value) {
    return { text: '未生成', color: 'info' };
  }
  
  if (reportData.value.risk_level) {
    const level = reportData.value.risk_level;
    return {
      text: level === 'high' ? '高风险' : level === 'medium' ? '中风险' : '低风险',
      color: level === 'high' ? 'danger' : level === 'medium' ? 'warning' : 'success'
    };
  }
  
  return { text: '已完成', color: 'success' };
});

// 新增：切换视频面板显示/隐藏
const toggleVideoPanel = () => {
  isVideoCollapsed.value = !isVideoCollapsed.value;
};

// 新增：视频加载完成事件
const onVideoLoaded = () => {
  if (videoPlayer.value) {
    videoDuration.value = videoPlayer.value.duration;
  }
};

// 新增：格式化时长
const formatDuration = (seconds) => {
  if (!seconds) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// 跳转到评估理由详情页
const goToReasoning = (itemKey) => {
  const videoId = route.query.id;
  router.push({
    name: 'AssessmentReason',
    query: {
      id: videoId,
      item: itemKey,
    },
  });
};

// 重新生成摘要
const regenerateSummary = async () => {
  try {
    const videoId = route.query.id;
    if (!videoId) {
      ElMessage.error('未提供视频ID');
      return;
    }

    summaryLoading.value = true;
    ElMessage.info('开始重新生成摘要...');

    const response = await axios.post(`/api/summary/video/${videoId}`, {
      force: true,
    });

    if (response.data.code === 0) {
      await loadVideoData();
      ElMessage.success('摘要已重新生成');
    } else {
      throw new Error(response.data.message || '生成失败');
    }
  } catch (error) {
    console.error('重新生成摘要失败:', error);
    ElMessage.error('重新生成摘要失败: ' + (error.message || '未知错误'));
  } finally {
    summaryLoading.value = false;
  }
};

// 复制字幕文本
const copySubtitleText = () => {
  if (subtitlesData.value && subtitlesData.value.text) {
    navigator.clipboard
      .writeText(subtitlesData.value.text)
      .then(() => {
        ElMessage.success('文本已复制到剪贴板');
      })
      .catch(() => {
        ElMessage.error('复制失败，请手动复制');
      });
  } else {
    ElMessage.warning('没有可复制的文本');
  }
};

// 加载数字人检测数据
const loadDigitalHumanData = async () => {
  try {
    digitalHumanLoading.value = true;
    const videoId = route.query.id;
    if (!videoId) return;

    const response = await axios.get(`/api/videos/${videoId}/digital-human/result`);
    if (response.data.code === 200) {
      digitalHumanData.value = response.data.data;
    }
  } catch (error) {
    console.error('加载数字人检测数据失败:', error);
  } finally {
    digitalHumanLoading.value = false;
  }
};

// 加载事实核查数据
const loadFactCheckData = async () => {
  try {
    factCheckLoading.value = true;
    factCheckError.value = null;

    const videoId = route.query.id;
    if (!videoId) {
      throw new Error('未提供视频ID');
    }

    const response = await axios.get(`/api/videos/${videoId}/factcheck/result`);

    if (response.data.code === 200) {
      factCheckData.value = response.data.data;

      // 如果状态是processing，设置定时器轮询
      if (factCheckData.value.status === 'processing') {
        setTimeout(() => loadFactCheckData(), 5000);
      }
    } else {
      throw new Error(response.data.message || '获取事实核查结果失败');
    }
  } catch (error) {
    console.error('加载事实核查数据失败:', error);
    factCheckError.value = error.message || '加载事实核查数据失败';
  } finally {
    factCheckLoading.value = false;
  }
};

// 生成事实核查结果
const generateFactCheck = async () => {
  try {
    factCheckLoading.value = true;
    factCheckError.value = null;
    factCheckNotFound.value = false;

    const videoId = route.query.id;
    if (!videoId) {
      throw new Error('未提供视频ID');
    }

    ElMessage.info('正在启动事实核查，这可能需要几分钟时间...');

    const response = await axios.post(`/api/videos/${videoId}/factcheck`);

    if (response.data.code === 200) {
      if (response.data.data && response.data.data.fact_check_result) {
        factCheckData.value = response.data.data.fact_check_result;
        ElMessage.success('事实核查已启动，请等待结果');
        setTimeout(() => loadFactCheckData(), 5000);
      } else {
        factCheckData.value = response.data.data;
        ElMessage.success('事实核查已完成');
      }
    } else {
      throw new Error(response.data.message || '事实核查请求失败');
    }
  } catch (error) {
    console.error('生成事实核查失败:', error);
    factCheckError.value = error.message || '生成事实核查失败';
    ElMessage.error('生成事实核查失败: ' + error.message);
  } finally {
    factCheckLoading.value = false;
  }
};

// 重新生成分析报告
const regenerateReport = async () => {
  try {
    reportLoading.value = true;
    reportError.value = null;

    const videoId = route.query.id;
    const classifyResponse = await axios.post(
      `/api/videos/${videoId}/classify-risk`,
    );

    if (classifyResponse.data.code !== 200) {
      throw new Error(classifyResponse.data.message || '风险评估失败');
    }

    const reportResponse = await axios.post(
      `/api/videos/${videoId}/generate-report`,
    );

    if (reportResponse.data.code === 200) {
      await loadVideoData();
      ElMessage.success('报告已重新生成');
    } else {
      throw new Error(reportResponse.data.message || '生成报告失败');
    }
  } catch (error) {
    console.error('生成分析报告失败:', error);
    reportError.value = error.message || '生成分析报告失败';
    ElMessage.error('生成分析报告失败: ' + error.message);
  } finally {
    reportLoading.value = false;
  }
};

// 导出报告
const exportReport = () => {
  if (!reportData.value) return;

  const reportText = `# ${videoData.value.video.title} 分析报告\n\n`;
  const blob = new Blob([reportText + reportData.value.report], {
    type: 'text/markdown',
  });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `分析报告_${new Date().toISOString().split('T')[0]}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

// 加载独立报告数据
const loadReportDataOnly = async (videoId) => {
  try {
    reportLoading.value = true;
    reportError.value = null;

    const response = await axios.get(`/api/videos/${videoId}/report`);

    if (response.data.code === 200) {
      reportData.value = response.data.data;
    } else {
      throw new Error(response.data.message || '获取报告失败');
    }
  } catch (error) {
    console.error('加载报告数据失败:', error);
    reportError.value = error.message || '加载报告数据失败';
  } finally {
    reportLoading.value = false;
  }
};

// 处理标签页切换
const handleTabChange = (key) => {
  activeTab.value = key;

  // 按需加载数据
  if (key === 'threat' && !reportData.value && route.query.id) {
    loadReportDataOnly(route.query.id);
  } else if (key === 'factcheck' && !factCheckData.value && route.query.id) {
    loadFactCheckData();
  } else if (key === 'digitalhuman' && !digitalHumanData.value && route.query.id) {
    loadDigitalHumanData();
  }
};

// 点击统计项跳转到对应tab
const clickStatItem = (tabKey) => {
  handleTabChange(tabKey);
};

// 加载视频数据
const loadVideoData = async () => {
  try {
    loading.value = true;
    const videoId = route.query.id;

    if (!videoId) {
      ElMessage.error('未提供视频ID');
      return;
    }

    const response = await axios.get(`/api/videos/${videoId}/analysis`);
    videoData.value = response.data.data;

    // 设置视频源
    videoSrc.value = videoData.value.video.url;

    // 设置字幕数据
    if (videoData.value.transcript) {
      subtitlesData.value = videoData.value.transcript;
    }

    // 解析Markdown摘要
    if (videoData.value.analysis && videoData.value.analysis.summary) {
      summary.value = md.render(videoData.value.analysis.summary);
    }

    // 保存评估数据
    if (videoData.value.analysis && videoData.value.analysis.assessments) {
      assessmentData.value = videoData.value.analysis.assessments;
    } else {
      assessmentData.value = {};
    }

    // 提取报告信息
    if (videoData.value.analysis) {
      reportData.value = {
        report: videoData.value.analysis.report,
        risk_level: videoData.value.analysis.risk?.level,
        risk_probability: videoData.value.analysis.risk?.probability,
        timestamp: videoData.value.analysis.timestamp,
        scores: {
          background_sufficiency:
            videoData.value.analysis.assessments?.p1?.score,
          background_accuracy: videoData.value.analysis.assessments?.p2?.score,
          content_completeness: videoData.value.analysis.assessments?.p3?.score,
          intention_legitimacy: videoData.value.analysis.assessments?.p4?.score,
          publisher_credibility:
            videoData.value.analysis.assessments?.p5?.score,
          emotional_neutrality: videoData.value.analysis.assessments?.p6?.score,
          behavior_autonomy: videoData.value.analysis.assessments?.p7?.score,
          information_consistency:
            videoData.value.analysis.assessments?.p8?.score,
        },
      };
    }

    loading.value = false;
  } catch (error) {
    console.error('加载视频数据失败:', error);
    ElMessage.error('加载视频数据失败');
    loading.value = false;
  }
};

// 页面加载时获取数据
onMounted(() => {
  loadVideoData();
});
</script>

<template>
  <div class="analysis-page">
    <!-- 如果有子路由被激活，显示子路由内容 -->
    <router-view v-if="$route.path.includes('/reason')" />

    <!-- 主要内容区域 -->
    <div v-else>
      <!-- 页面头部 -->
      <div class="page-header">
        <div class="header-left">
          <h2 class="page-title">视频内容分析</h2>
          <div class="breadcrumb" v-if="videoData?.video?.title">
            <span class="breadcrumb-item">{{ videoData.video.title }}</span>
          </div>
        </div>
        <div class="header-controls">
          <el-button
            :icon="isVideoCollapsed ? Expand : Fold"
            @click="toggleVideoPanel"
            type="primary"
            :text="true"
            size="small"
          >
            {{ isVideoCollapsed ? '展开视频' : '收起视频' }}
          </el-button>
        </div>
      </div>

      <!-- 内容容器 -->
      <div
        class="content-container"
        :class="{ 'video-collapsed': isVideoCollapsed }"
      >
        <!-- 左侧视频面板 -->
        <transition name="slide-fade">
          <div v-show="!isVideoCollapsed" class="video-panel">
            <el-card class="video-card" shadow="hover">
              <!-- 收起按钮 -->
              <template #header>
                <div class="card-header">
                  <span class="card-title">
                    <el-icon><VideoPlay /></el-icon>
                    视频播放
                  </span>
                  <el-button
                    :icon="Close"
                    @click="toggleVideoPanel"
                    type="danger"
                    :text="true"
                    circle
                    size="small"
                  />
                </div>
              </template>

              <!-- 视频播放器 -->
              <div class="video-container">
                <video
                  ref="videoPlayer"
                  controls
                  :src="videoSrc"
                  class="video-player"
                  @loadedmetadata="onVideoLoaded"
                >
                  您的浏览器不支持视频播放
                </video>
              </div>

              <!-- 视频信息 -->
              <div class="video-info">
                <h3 class="video-title">
                  {{ videoData?.video?.title || '未知标题' }}
                </h3>
                <div class="video-meta">
                  <el-tag type="info" size="small" v-if="videoDuration">
                    <el-icon><VideoPlay /></el-icon>
                    {{ formatDuration(videoDuration) }}
                  </el-tag>
                  <el-tag
                    type="success"
                    size="small"
                    v-if="videoData?.video?.platform"
                  >
                    {{ videoData.video.platform }}
                  </el-tag>
                </div>
                <div class="video-tags" v-if="videoData?.video?.tags?.length">
                  <el-tag
                    v-for="tag in videoData.video.tags.slice(0, 3)"
                    :key="tag"
                    size="small"
                    class="tag-item"
                  >
                    {{ tag }}
                  </el-tag>
                  <span
                    v-if="videoData.video.tags.length > 3"
                    class="more-tags"
                  >
                    +{{ videoData.video.tags.length - 3 }}
                  </span>
                </div>
                
                <el-divider>分析概览</el-divider>

                <div class="video-analysis-overview">
                  <!-- 风险等级指示器 -->
                  <div class="risk-indicator" v-if="reportData?.risk_level">
                    <div class="risk-header">
                      <span class="risk-label">风险等级</span>
                      <el-tag
                        :type="
                          reportData.risk_level === 'high'
                            ? 'danger'
                            : reportData.risk_level === 'medium'
                              ? 'warning'
                              : 'success'
                        "
                        size="small"
                      >
                        {{
                          reportData.risk_level === 'high'
                            ? '高风险'
                            : reportData.risk_level === 'medium'
                              ? '中风险'
                              : '低风险'
                        }}
                      </el-tag>
                    </div>
                    <el-progress
                      :percentage="
                        Math.round((reportData.risk_probability || 0) * 100)
                      "
                      :status="
                        reportData.risk_level === 'high'
                          ? 'exception'
                          : reportData.risk_level === 'medium'
                            ? 'warning'
                            : 'success'
                      "
                      :stroke-width="6"
                    />
                  </div>

                  <!-- 快速统计 -->
                  <div class="quick-stats">
                    <div class="stat-row">
                      <div 
                        class="stat-item clickable" 
                        :data-status="'info'"
                        @click="clickStatItem('subtitles')"
                        title="点击查看字幕列表"
                      >
                        <el-icon><Document /></el-icon>
                        <div class="stat-content">
                          <div class="stat-value">
                            {{ subtitlesData?.chunks?.length || 0 }}
                          </div>
                          <div class="stat-label">字幕段落</div>
                        </div>
                      </div>

                      <div 
                        class="stat-item clickable" 
                        :data-status="digitalHumanStatus.color"
                        @click="clickStatItem('digitalhuman')"
                        title="点击查看数字人检测"
                      >
                        <el-icon><VideoCamera /></el-icon>
                        <div class="stat-content">
                          <div class="stat-value">
                            {{ digitalHumanStatus.text }}
                          </div>
                          <div class="stat-label">数字人检测</div>
                        </div>
                      </div>
                    </div>

                    <div class="stat-row">
                      <div 
                        class="stat-item clickable" 
                        :data-status="factCheckStatus.color"
                        @click="clickStatItem('factcheck')"
                        title="点击查看事实核查"
                      >
                        <el-icon><Warning /></el-icon>
                        <div class="stat-content">
                          <div class="stat-value">
                            {{ factCheckStatus.text }}
                          </div>
                          <div class="stat-label">事实核查</div>
                        </div>
                      </div>

                      <div 
                        class="stat-item clickable" 
                        :data-status="threatReportStatus.color"
                        @click="clickStatItem('threat')"
                        title="点击查看威胁报告"
                      >
                        <el-icon><TrendCharts /></el-icon>
                        <div class="stat-content">
                          <div class="stat-value">
                            {{ threatReportStatus.text }}
                          </div>
                          <div class="stat-label">威胁报告</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </el-card>
          </div>
        </transition>

        <!-- 右侧分析面板 -->
        <div class="analysis-panel" :class="{ 'full-width': isVideoCollapsed }">
          <el-card class="analysis-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span class="card-title">分析结果</span>
                <el-button
                  v-if="isVideoCollapsed"
                  :icon="VideoPlay"
                  @click="toggleVideoPanel"
                  type="primary"
                  :text="true"
                  size="small"
                >
                  查看视频
                </el-button>
              </div>
            </template>

            <div class="card-content">
              <!-- 顶部导航菜单 -->
              <el-menu
                :default-active="activeTab"
                class="analysis-tabs"
                mode="horizontal"
                @select="handleTabChange"
              >
                <el-menu-item index="summary">总结摘要</el-menu-item>
                <el-menu-item index="subtitles">字幕列表</el-menu-item>
                <el-menu-item index="digitalhuman">数字人检测</el-menu-item>
                <el-menu-item index="process">分析过程</el-menu-item>
                <el-menu-item index="factcheck">事实核查</el-menu-item>
                <el-menu-item index="threat">威胁报告</el-menu-item>
              </el-menu>

              <!-- 内容区域，可滚动 -->
              <div class="content-area">
                <!-- 总结摘要内容 -->
                <SummaryTab
                  v-if="activeTab === 'summary'"
                  :summary="summary"
                  :loading="summaryLoading"
                  :video-title="videoData?.video?.title"
                  @regenerate="regenerateSummary"
                />

                <!-- 字幕列表内容 -->
                <SubtitlesTab
                  v-else-if="activeTab === 'subtitles'"
                  :subtitles-data="subtitlesData"
                  @copy-text="copySubtitleText"
                />

                <!-- 数字人检测内容 -->
                <DigitalHumanTab
                  v-else-if="activeTab === 'digitalhuman'"
                  :video-data="videoData"
                />

                <!-- 分析过程内容 -->
                <ProcessTab
                  v-else-if="activeTab === 'process'"
                  :assessment-data="assessmentData"
                  @view-reasoning="goToReasoning"
                />

                <!-- 事实核查内容 -->
                <FactCheckTab
                  v-else-if="activeTab === 'factcheck'"
                  :fact-check-data="factCheckData"
                  :loading="factCheckLoading"
                  :error="factCheckError"
                  :not-found="factCheckNotFound"
                  @load-data="loadFactCheckData"
                  @generate-check="generateFactCheck"
                />

                <!-- 威胁报告内容 -->
                <ThreatReportTab
                  v-else-if="activeTab === 'threat'"
                  :report-data="reportData"
                  :loading="reportLoading"
                  :error="reportError"
                  :video-title="videoData?.video?.title"
                  @regenerate="regenerateReport"
                  @export="exportReport"
                />
              </div>
            </div>
          </el-card>
        </div>
      </div>

      <!-- 悬浮切换按钮（视频收起时显示） -->
      <transition name="fade">
        <div v-show="isVideoCollapsed" class="floating-toggle">
          <el-button
            :icon="VideoPlay"
            @click="toggleVideoPanel"
            type="primary"
            circle
            size="large"
          />
        </div>
      </transition>
    </div>
  </div>
</template>

<style scoped>
/* 新增：视频分析概览样式 */
.video-analysis-overview {
  margin-top: 16px;
}

.risk-indicator {
  margin-bottom: 20px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.risk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.risk-label {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
}

.quick-stats {
  margin-bottom: 16px;
}

.stat-row {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.stat-row:last-child {
  margin-bottom: 0;
}

.stat-item {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  background: #fafafa;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
  transition: all 0.2s ease;
  position: relative;
}

.stat-item.clickable {
  cursor: pointer;
}

.stat-item.clickable:hover {
  background: #f0f9eb;
  border-color: #67c23a;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(103, 194, 58, 0.2);
}

.stat-item.clickable:active {
  transform: translateY(0);
}

/* 根据状态添加不同的边框色 */
.stat-item[data-status="success"] {
  border-left: 3px solid #67c23a;
}

.stat-item[data-status="warning"] {
  border-left: 3px solid #e6a23c;
}

.stat-item[data-status="danger"] {
  border-left: 3px solid #f56c6c;
}

.stat-item[data-status="info"] {
  border-left: 3px solid #909399;
}

.stat-item .el-icon {
  color: #409eff;
  font-size: 16px;
}

.stat-content {
  flex: 1;
  min-width: 0;
}

.stat-value {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  line-height: 1.2;
}

.stat-label {
  font-size: 11px;
  color: #909399;
  line-height: 1.2;
}

/* 分割线样式调整 */
.video-info :deep(.el-divider) {
  margin: 16px 0 12px 0;
  border-top: 1px solid #f0f0f0;
}

.video-info :deep(.el-divider__text) {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
  background: white;
  padding: 0 12px;
}

/* 响应式调整 */
@media (max-width: 480px) {
  .stat-row {
    flex-direction: column;
    gap: 8px;
  }
}

.analysis-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 16px 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

.breadcrumb {
  font-size: 14px;
  color: #8492a6;
}

.breadcrumb-item {
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
}

.content-container {
  display: flex;
  gap: 20px;
  min-height: calc(100vh - 120px);
  transition: all 0.3s ease;
}

.content-container.video-collapsed {
  gap: 0;
}

.video-panel {
  flex: 0 0 400px;
  transition: all 0.3s ease;
}

.analysis-panel {
  flex: 1;
  transition: all 0.3s ease;
}

.analysis-panel.full-width {
  flex: 1;
  max-width: 100%;
}

.video-card,
.analysis-card {
  height: 100%;
  border-radius: 12px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-weight: 600;
  color: #2c3e50;
  display: flex;
  align-items: center;
  gap: 8px;
}

.video-container {
  margin-bottom: 16px;
  border-radius: 8px;
  overflow: hidden;
  background: #000;
}

.video-player {
  width: 100%;
  height: auto;
  max-height: 300px;
  object-fit: contain;
}

.video-info {
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.video-title {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 12px 0;
  line-height: 1.4;
}

.video-meta {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  align-items: center;
}

.video-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}

.tag-item {
  border-radius: 12px;
}

.more-tags {
  font-size: 12px;
  color: #909399;
  background: #f4f4f5;
  padding: 2px 6px;
  border-radius: 10px;
}

.card-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.analysis-tabs {
  border-bottom: 1px solid #e4e7ed;
  margin-bottom: 20px;
}

.content-area {
  flex: 1;
  overflow: auto;
  padding-right: 4px;
}

/* 悬浮按钮 */
.floating-toggle {
  position: fixed;
  bottom: 30px;
  left: 30px;
  z-index: 1000;
}

/* 动画效果 */
.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: all 0.3s ease;
}

.slide-fade-enter-from {
  opacity: 0;
  transform: translateX(-100%);
}

.slide-fade-leave-to {
  opacity: 0;
  transform: translateX(-100%);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 卡片内容样式修复 */
:deep(.el-card__body) {
  height: 100%;
  padding: 20px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* 菜单样式 */
:deep(.el-menu-item) {
  height: 48px;
  line-height: 48px;
  border-bottom: none;
}

:deep(.el-menu--horizontal > .el-menu-item.is-active) {
  border-bottom: 2px solid #409eff;
  font-weight: 500;
  color: #409eff;
}

:deep(.el-menu--horizontal) {
  border-bottom: 1px solid #e4e7ed;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .analysis-page {
    padding: 10px;
  }

  .content-container {
    flex-direction: column;
    gap: 10px;
  }

  .video-panel {
    flex: none;
  }

  .page-header {
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
  }

  .header-controls {
    align-self: flex-end;
  }
}

@media (max-width: 480px) {
  .page-title {
    font-size: 20px;
  }

  .breadcrumb-item {
    max-width: 200px;
  }
}
</style>