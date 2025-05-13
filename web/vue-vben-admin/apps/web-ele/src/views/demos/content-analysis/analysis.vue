<script lang="ts" setup>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';
import { Refresh, Download, CopyDocument } from '@element-plus/icons-vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import {
  ElButton,
  ElIcon,
  ElTag,
  ElTooltip,
  ElTable,
  ElTableColumn,
  ElImage,
  ElMessage,
  ElCard,
  ElInput,
  ElPagination,
  ElMenu,
  ElMenuItem, // 添加菜单相关组件
  ElInfiniteScroll,
  ElScrollbar,
  ElProgress,
  ElResult,
  ElMessageBox,
} from 'element-plus';
// 定义评估项的语义映射
// 创建markdown-it实例
const md = new MarkdownIt({
  html: true, // 启用HTML标签
  breaks: true, // 将换行符转换为<br>
  linkify: true, // 自动将URL转换为链接
  typographer: true, // 启用一些语言中性的替换+引号美化
});

const assessmentNames = {
  p1: '背景信息充分性',
  p2: '背景信息准确性',
  p3: '内容完整性',
  p4: '意图正当性',
  p5: '发布者信誉',
  p6: '情感中立性',
  p7: '行为自主性',
  p8: '信息一致性',
};

// 添加到script部分
const router = useRouter(); // 别忘了导入useRouter

// 添加跳转到评估理由详情页的方法
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
// 添加数据加载状态
const loading = ref(true);
const videoData = ref(null);
const videoSrc = ref('');
const subtitlesData = ref({ chunks: [], text: '' });
const route = useRoute();
const summary = ref(''); // 存储解析后的摘要HTML
const assessmentData = ref({}); // 新增：专门存储评估数据
// 添加重新生成摘要函数
const summaryLoading = ref(false);
const regenerateSummary = async () => {
  try {
    const videoId = route.query.id;
    if (!videoId) {
      ElMessage.error('未提供视频ID');
      return;
    }

    summaryLoading.value = true;
    ElMessage.info('开始重新生成摘要...');

    // 调用后端重新生成摘要的API
    const response = await axios.post(`/api/summary/video/${videoId}`, {
      force: true, // 强制重新生成
    });

    if (response.data.code === 0) {
      // 重新获取视频数据以更新摘要
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
// 添加评估数据可用性检查的计算属性
const hasAssessments = computed(() => {
  return assessmentData.value && Object.keys(assessmentData.value).length > 0;
});
const formatDate = (date) => {
  if (!date) return '未知时间';
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};
// 添加格式化评估项的计算属性
const assessmentItems = computed(() => {
  if (!hasAssessments.value) return [];

  return Object.entries(assessmentData.value)
    .filter(
      ([_, item]) => item && item.score !== null && item.score !== undefined,
    )
    .map(([key, item]) => ({
      key,
      name: assessmentNames[key] || key,
      score: item.score,
      reasoning: item.reasoning,
    }));
});
// 修改loadVideoData函数，从分析接口获取所有数据
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

    // 保存评估数据到专门的变量
    if (videoData.value.analysis && videoData.value.analysis.assessments) {
      assessmentData.value = videoData.value.analysis.assessments;
    } else {
      assessmentData.value = {};
    }

    // 从分析数据中直接提取报告信息
    if (videoData.value.analysis) {
      reportData.value = {
        report: videoData.value.analysis.report,
        risk_level: videoData.value.analysis.risk?.level,
        risk_probability: videoData.value.analysis.risk?.probability,
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
const regenerateReport = async () => {
  try {
    reportLoading.value = true;
    reportError.value = null;

    // 先调用风险分类API
    const videoId = route.query.id as string;
    const classifyResponse = await axios.post(
      `/api/videos/${videoId}/classify-risk`,
    );

    if (classifyResponse.data.code !== 200) {
      throw new Error(classifyResponse.data.message || '风险评估失败');
    }

    // 生成新报告
    const reportResponse = await axios.post(
      `/api/videos/${videoId}/generate-report`,
    );

    if (reportResponse.data.code === 200) {
      // 重新加载所有数据
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
  URL.revokeObjectURL(url); // 清理创建的对象URL
};
// 修改菜单选择处理函数
const handleTabChange = (key: string) => {
  activeTab.value = key;

  // 当切换到威胁报告标签时，如果没有报告数据则加载
  if (key === 'threat' && !reportData.value && route.query.id) {
    loadReportDataOnly(route.query.id as string);
  }
};
// 页面加载时获取数据
onMounted(() => {
  loadVideoData();
});
// 添加导航菜单激活状态
const activeTab = ref('summary');
// 添加缺失的时间戳格式化函数
const formatTimestamp = (seconds: number | undefined): string => {
  if (seconds === undefined) return '00:00';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};
const reportLoading = ref(false);
const reportData = ref(null);
const reportError = ref(null);
const riskLevelInfo = computed(() => {
  if (!reportData.value || !reportData.value.risk_level)
    return { class: 'info', color: '#909399', text: '未评估' };

  const level = reportData.value.risk_level.toLowerCase();
  switch (level) {
    case 'low':
      return { class: 'success', color: '#67C23A', text: '低风险' };
    case 'medium':
      return { class: 'warning', color: '#E6A23C', text: '中等风险' };
    case 'high':
      return { class: 'danger', color: '#F56C6C', text: '高风险' };
    default:
      return { class: 'info', color: '#909399', text: '未评估' };
  }
});
// 根据评分获取进度条颜色
const getScoreColor = (score: number): string => {
  if (score >= 0.8) return '#67C23A'; // 绿色
  if (score >= 0.5) return '#E6A23C'; // 橙色
  return '#F56C6C'; // 红色
};

// 格式化评分值（保留1位小数）
const formatScore = (score: number): string => {
  // 检查score是否为数字，包括0
  return typeof score === 'number' ? score.toFixed(1) : 'N/A';
};

// 添加复制功能
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
</script>

<template>
  <!-- 加载状态 -->
  <div
    v-if="loading"
    style="
      display: flex;
      height: 100%;
      align-items: center;
      justify-content: center;
    "
  >
    <div style="text-align: center">
      <el-icon
        class="loading-icon"
        style="font-size: 2.25rem; margin-bottom: 1rem"
      >
        <i class="el-icon-loading" />
      </el-icon>
      <div>加载数据中...</div>
    </div>
  </div>

  <!-- 如果有子路由被激活，显示子路由内容 -->
  <router-view v-else-if="$route.path.includes('/reason')" />

  <!-- 视频分析内容，仅在数据加载后显示 -->
  <div v-else class="content-container">
    <!-- 左侧卡片 - 占35%且高度100% -->
    <el-card class="side-card">
      <div class="card-content">
        <div class="video-container">
          <video controls :src="videoSrc" style="max-height: 100%"></video>
        </div>
        <!-- 添加视频标题和标签 -->
        <div class="video-info">
          <h3 class="video-title">{{ videoData.video.title }}</h3>
          <div class="video-tags">
            <el-tag
              v-for="tag in videoData.video.tags"
              :key="tag"
              size="small"
              >{{ tag }}</el-tag
            >
          </div>
        </div>
      </div>
    </el-card>

    <!-- 右侧卡片 - 占65%且高度100% -->
    <el-card class="main-card">
      <div class="card-content">
        <!-- 顶部导航菜单 -->
        <el-menu
          :default-active="activeTab"
          class="analysis-tabs border-0"
          mode="horizontal"
          @select="handleTabChange"
        >
          <el-menu-item index="summary">总结摘要</el-menu-item>
          <el-menu-item index="subtitles">字幕列表</el-menu-item>
          <el-menu-item index="process">分析过程</el-menu-item>
          <el-menu-item index="threat">威胁报告</el-menu-item>
        </el-menu>

        <!-- 内容区域，可滚动 -->
        <div class="content-area">
          <!-- 总结摘要内容 -->
          <div v-if="activeTab === 'summary'">
            <!-- 使用v-html渲染Markdown转换后的HTML -->
            <div v-if="summary" class="markdown-body" v-html="summary"></div>
            <p v-else class="no-content">暂无摘要内容</p>

            <!-- 添加重新生成按钮 -->
            <div class="action-button-container">
              <el-button
                type="primary"
                :loading="summaryLoading"
                @click="regenerateSummary"
                size="small"
                icon="Refresh"
              >
                重新生成摘要
              </el-button>
            </div>
          </div>

          <!-- 字幕列表内容 -->
          <div
            v-else-if="activeTab === 'subtitles'"
            class="subtitles-container"
          >
            <!-- 整体布局容器 -->
            <div class="subtitles-layout">
              <!-- 完整文本区域 -->
              <div class="section-header">
                <h4 class="section-title">完整文本:</h4>
                <el-button
                  size="small"
                  type="primary"
                  @click="copySubtitleText"
                  :icon="CopyDocument"
                  text
                >
                  复制文本
                </el-button>
              </div>
              <div class="text-preview-container">
                <el-scrollbar height="75px">
                  <p class="text-preview-content">
                    {{ subtitlesData.text }}
                  </p>
                </el-scrollbar>
              </div>

              <!-- 字幕列表区域 -->
              <div class="subtitles-list-container">
                <div class="section-header">
                  <h4 class="section-title">字幕时间轴:</h4>
                  <span class="subtitle-count">
                    共 {{ subtitlesData.chunks.length }} 个片段
                  </span>
                </div>
                <el-scrollbar height="65vh" class="subtitle-scrollbar">
                  <div style="padding: 0.25rem">
                    <div
                      v-for="(chunk, index) in subtitlesData.chunks"
                      :key="index"
                      class="subtitle-chunk"
                    >
                      <div class="subtitle-timestamp">
                        {{ formatTimestamp(chunk.timestamp[0]) }} -
                        {{ formatTimestamp(chunk.timestamp[1]) }}
                      </div>
                      <div class="subtitle-text">{{ chunk.text }}</div>
                    </div>
                  </div>
                </el-scrollbar>
              </div>
            </div>
          </div>

          <!-- 分析过程内容 -->
          <div v-else-if="activeTab === 'process'">
            <h3 class="section-heading">视频分析过程</h3>

            <!-- 使用hasAssessments和assessmentItems计算属性 -->
            <div v-if="hasAssessments" class="assessment-list">
              <div
                v-for="item in assessmentItems"
                :key="item.key"
                class="assessment-item"
              >
                <div class="assessment-header">
                  <div class="assessment-title">
                    {{ item.name }} ({{ item.key }})
                  </div>
                  <div
                    class="assessment-score"
                    :style="{ color: getScoreColor(item.score) }"
                  >
                    {{ formatScore(item.score) }}
                  </div>
                </div>

                <el-progress
                  :percentage="item.score * 100"
                  :color="getScoreColor(item.score)"
                  :stroke-width="10"
                  :show-text="false"
                />

                <!-- 修改这里，添加点击事件和鼠标悬停样式 -->
                <div
                  v-if="item.reasoning"
                  class="reasoning-link"
                  @click="goToReasoning(item.key)"
                >
                  点击查看详细评估理由
                </div>
                <div v-else class="no-reasoning">无评估理由</div>
              </div>
            </div>

            <!-- 没有评估数据时显示提示 -->
            <div v-else class="empty-state">
              <div class="emoji-placeholder">📊</div>
              <div>暂无分析数据</div>
            </div>
          </div>

          <!-- 威胁报告内容 -->
          <div v-else-if="activeTab === 'threat'">
            <div class="threat-report-header">
              <h3 class="section-heading">内容威胁分析报告</h3>
              <div class="report-timestamp">
                生成时间: {{ formatDate(reportData?.timestamp || new Date()) }}
              </div>
            </div>

            <!-- 加载状态 -->
            <div v-if="reportLoading" class="loading-container">
              <el-skeleton :rows="10" animated />
            </div>

            <!-- 错误状态 -->
            <el-result
              v-else-if="reportError"
              icon="error"
              :title="reportError"
              sub-title="无法获取分析报告数据"
            >
              <template #extra>
                <el-button type="primary" @click="loadAnalysisReport"
                  >重试</el-button
                >
              </template>
            </el-result>

            <!-- 报告数据显示 -->
            <div v-else-if="reportData" class="analysis-report">
              <!-- 风险等级信息 -->
              <el-card
                class="risk-info-card"
                :class="`border-${riskLevelInfo.class}`"
              >
                <div class="risk-info-header">
                  <div class="risk-level-container">
                    <el-tag
                      :type="riskLevelInfo.class"
                      size="large"
                      effect="dark"
                      class="risk-level-tag"
                    >
                      {{ riskLevelInfo.text }}
                    </el-tag>
                    <div class="risk-probability">
                      风险概率:
                      <span :style="{ color: riskLevelInfo.color }">
                        {{ (reportData.risk_probability * 100).toFixed(1) }}%
                      </span>
                    </div>
                  </div>
                  <div class="action-buttons">
                    <!-- 重新生成按钮 -->
                    <el-button
                      type="primary"
                      @click="regenerateReport"
                      :icon="Refresh"
                      size="small"
                    >
                      重新生成
                    </el-button>
                    <!-- 添加导出按钮 -->
                    <el-button
                      type="success"
                      @click="exportReport"
                      :icon="Download"
                      size="small"
                      class="export-button"
                    >
                      导出报告
                    </el-button>
                  </div>
                </div>
              </el-card>

              <!-- 分析报告内容 -->
              <el-card class="report-content">
                <div class="report-container">
                  <div
                    class="markdown-body"
                    v-html="md.render(reportData.report)"
                  ></div>
                </div>
              </el-card>
              <!-- 评分摘要 -->
            </div>
            <!-- 没有报告时显示 -->
            <div v-else>
              <el-result icon="info" title="暂无分析报告">
                <template #sub-title>
                  <p>系统尚未对此视频生成分析报告，点击下方按钮生成。</p>
                </template>
                <template #extra>
                  <el-button type="primary" @click="loadAnalysisReport">
                    生成分析报告
                  </el-button>
                </template>
              </el-result>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>
<style scoped>
/* 主容器布局 */
.content-container {
  display: flex;
  height: calc(100vh - 120px); /* 固定高度，减去头部和可能的边距 */
  flex-direction: column;
  gap: 1rem;
  padding: 1rem; /* 增加容器内边距 */
  box-sizing: border-box; /* 确保内边距不会增加容器实际尺寸 */
  overflow: hidden; /* 防止外部滚动 */
}

@media (min-width: 768px) {
  .content-container {
    flex-direction: row;
  }
}

/* 卡片样式 */
.side-card,
.main-card {
  height: 100%; /* 确保两边卡片高度一致 */
  display: flex;
  flex-direction: column;
  overflow: hidden; /* 防止卡片自身溢出 */
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.1),
    0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

@media (min-width: 768px) {
  .side-card {
    width: 35%;
  }

  .main-card {
    width: 65%;
  }
}

.main-card {
  height: 100%;
  width: 100%;
  overflow: hidden;
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.1),
    0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.card-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* 视频区域样式 */
.video-container {
  overflow: hidden;
  border-radius: 0.5rem;
  flex: 1; /* 让视频容器占据可用空间 */
  display: flex;
  flex-direction: column;
}

.video-info {
  margin-top: 1rem;
  padding-left: 0.25rem;
  padding-right: 0.25rem;
}

.video-title {
  font-size: 1.125rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.video-tags {
  margin-top: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

/* 内容区域 */
.content-area {
  flex: 1;
  overflow: auto;
  padding: 1rem;
}

/* 加载图标 */
.loading-icon {
  animation: spin 2s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 无内容提示 */
.no-content {
  color: #6b7280;
}

/* 操作按钮区 */
.action-button-container {
  margin-top: 1rem;
  display: flex;
  justify-content: flex-end;
}

/* 字幕部分 */
.subtitles-container {
  height: 100%;
}

.subtitles-layout {
  display: flex;
  height: calc(100% - 2rem);
  flex-direction: column;
}

.section-header {
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.section-title {
  font-weight: 500;
}

.subtitle-count {
  font-size: 0.75rem;
  color: #6b7280;
}

.text-preview-container {
  margin-bottom: 1rem;
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
  background-color: #f9fafb;
  padding: 1rem;
  height: 120px;
}

.text-preview-content {
  line-height: 1.625;
  color: #374151;
}

.subtitles-list-container {
  display: flex;
  flex: 1;
  flex-direction: column;
}

/* 字幕列表固定高度 */
.subtitle-scrollbar {
  height: calc(65vh - 200px) !important; /* 使用固定计算值而非百分比 */
  border: 1px solid #f3f4f6;
  border-radius: 0.25rem;
}

.subtitle-chunk {
  margin: 0.75rem;
  border-radius: 0.25rem;
  background-color: #f9fafb;
  padding: 0.75rem;
  transition: background-color 0.2s;
}

.subtitle-chunk:hover {
  background-color: #f3f4f6;
}

.subtitle-timestamp {
  margin-bottom: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.subtitle-text {
  color: #1f2937;
}
/* 修复卡片内容区溢出问题 */
:deep(.el-card__body) {
  height: 100%;
  padding: 15px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
/* 分析过程 */
.section-heading {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.assessment-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.assessment-item {
  border-left-width: 4px;
  border-left-color: #3b82f6;
  border-left-style: solid;
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  padding-left: 1rem;
}

.assessment-header {
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.assessment-title {
  font-weight: 500;
}

.assessment-score {
  font-size: 1.125rem;
  font-weight: 700;
}

.reasoning-link {
  margin-top: 0.5rem;
  cursor: pointer;
  color: #4b5563;
}

.reasoning-link:hover {
  color: #3b82f6;
}

.no-reasoning {
  margin-top: 0.5rem;
  color: #4b5563;
}

.empty-state {
  padding-top: 2rem;
  padding-bottom: 2rem;
  text-align: center;
  color: #6b7280;
}

.emoji-placeholder {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

/* 威胁报告 */
.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 3rem;
  padding-bottom: 3rem;
}

.risk-info-card {
  margin-bottom: 1rem;
  border-top-width: 4px;
}

.risk-info-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.risk-level-container {
  display: flex;
  align-items: center;
}

.risk-level-tag {
  margin-right: 0.75rem;
}

.risk-probability {
  font-size: 1.125rem;
  font-weight: 500;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
}

.export-button {
  margin-left: 0.5rem;
}

/* 自定义菜单样式 */
:deep(.el-menu-item) {
  height: 48px;
  line-height: 48px;
}

:deep(.el-menu--horizontal > .el-menu-item.is-active) {
  border-bottom: 2px solid #409eff;
  font-weight: 500;
}

/* 添加Markdown样式 */
:deep(.markdown-body) {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  color: #24292e;
  word-break: break-word;
}

:deep(
  .markdown-body h1,
  .markdown-body h2,
  .markdown-body h3,
  .markdown-body h4
) {
  margin-top: 24px;
  margin-bottom: 16px;
  font-weight: 600;
  line-height: 1.25;
}

:deep(.markdown-body h1) {
  font-size: 2em;
}
:deep(.markdown-body h2) {
  font-size: 1.5em;
  padding-bottom: 0.3em;
  border-bottom: 1px solid #eaecef;
}
:deep(.markdown-body h3) {
  font-size: 1.25em;
}
:deep(.markdown-body p) {
  margin-bottom: 16px;
}
:deep(.markdown-body ul, .markdown-body ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}
:deep(.markdown-body li) {
  margin-bottom: 0.25em;
}
:deep(.markdown-body pre) {
  padding: 16px;
  overflow: auto;
  font-size: 85%;
  line-height: 1.45;
  background-color: #f6f8fa;
  border-radius: 3px;
}
:deep(.markdown-body code) {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(27, 31, 35, 0.05);
  border-radius: 3px;
}

.report-content {
  margin-bottom: 1rem;
}

.border-success {
  border-top-color: #67c23a;
}

.border-warning {
  border-top-color: #e6a23c;
}

.border-danger {
  border-top-color: #f56c6c;
}

.border-info {
  border-top-color: #909399;
}

/* 增强markdown样式，特别是对报告中的重要标记 */
:deep(.markdown-body p) {
  line-height: 1.8;
}

:deep(.markdown-body strong) {
  color: #f56c6c;
  font-weight: 600;
}

:deep(.markdown-body h2) {
  margin-top: 1.5rem;
  font-size: 1.3rem;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3rem;
}

/* 突出显示带有▲符号的内容 */
:deep(.markdown-body p:has(> ▲)) {
  background-color: rgba(253, 246, 236, 0.6);
  padding: 0.5rem;
  border-radius: 4px;
  border-left: 3px solid #e6a23c;
}
/* 报告容器样式 */
.report-container {
  padding: 10px 5px;
}

/* 报告内容增强样式 */
:deep(.markdown-body) {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  line-height: 1.8;
  color: #333;
  word-break: break-word;
}

/* 标题样式增强 */
:deep(.markdown-body h2) {
  margin-top: 28px;
  font-size: 20px;
  font-weight: 600;
  border-bottom: 2px solid #409eff;
  padding-bottom: 8px;
  color: #303133;
}

:deep(.markdown-body h3) {
  margin-top: 24px;
  font-size: 17px;
  font-weight: 600;
  color: #409eff;
  background-color: #ecf5ff;
  padding: 8px 12px;
  border-radius: 4px;
}

/* 风险警告突出显示 */
:deep(.markdown-body p:has(> ▲)) {
  background-color: #fef0f0;
  padding: 12px 16px;
  border-radius: 6px;
  border-left: 4px solid #f56c6c;
  margin-bottom: 20px;
}

/* 突出显示风险标记 */
:deep(.markdown-body p ▲) {
  color: #f56c6c;
  font-weight: bold;
  margin-right: 4px;
}

/* 增强列表样式 */
:deep(.markdown-body ol) {
  padding-left: 22px;
  margin-bottom: 20px;
}

:deep(.markdown-body ol li) {
  margin-bottom: 10px;
  padding-left: 6px;
}

/* 突出显示粗体文本 */
:deep(.markdown-body strong) {
  color: #e6a23c;
  font-weight: bold;
  background-color: rgba(255, 229, 100, 0.3);
  padding: 0 4px;
  border-radius: 3px;
}

/* 突出显示风险类别 */
:deep(.markdown-body p strong:first-of-type) {
  display: inline-block;
  margin-right: 5px;
}

/* 突出显示评分数据 */
:deep(.markdown-body p span.score) {
  font-weight: bold;
}

:deep(.markdown-body p span.score-high) {
  color: #67c23a;
}

:deep(.markdown-body p span.score-medium) {
  color: #e6a23c;
}

:deep(.markdown-body p span.score-low) {
  color: #f56c6c;
}

/* 增强代码块样式 */
:deep(.markdown-body code) {
  color: #476582;
  background-color: rgba(27, 31, 35, 0.05);
  padding: 2px 5px;
  border-radius: 3px;
}

/* 表格样式增强 */
:deep(.markdown-body table) {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}

:deep(.markdown-body table th) {
  background: #f2f6fc;
  padding: 12px;
  border: 1px solid #ebeef5;
}

:deep(.markdown-body table td) {
  padding: 12px;
  border: 1px solid #ebeef5;
}

/* 结论部分特殊样式 */
:deep(.markdown-body > p:first-child) {
  font-size: 16px;
  background-color: #fef0f0;
  padding: 15px;
  border-radius: 6px;
  border-left: 5px solid #f56c6c;
  font-weight: 500;
  margin-bottom: 25px;
}
.threat-report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.report-timestamp {
  font-size: 14px;
  color: #909399;
  font-style: italic;
}
</style>
