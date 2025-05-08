<script lang="ts" setup>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';
import { Refresh } from '@element-plus/icons-vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import { CopyDocument } from '@element-plus/icons-vue'; // 添加复制图标

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
const components = {
  Refresh,
};
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
  return score ? score.toFixed(1) : 'N/A';
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
  <div v-if="loading" class="flex h-full items-center justify-center">
    <div class="text-center">
      <el-icon class="is-loading mb-4 text-4xl"
        ><i class="el-icon-loading"
      /></el-icon>
      <div>加载数据中...</div>
    </div>
  </div>
  <!-- 如果有子路由被激活，显示子路由内容 -->
  <router-view v-else-if="$route.path.includes('/reason')" />

  <!-- 视频分析内容，仅在数据加载后显示 -->
  <div v-else class="flex h-full gap-4">
    <!-- 左侧卡片 - 占35%且高度100% -->
    <el-card class="h-full w-[35%] overflow-hidden shadow-md">
      <div class="flex h-full flex-col p-2">
        <div class="video-container overflow-hidden rounded-lg">
          <video controls :src="videoSrc" class="w-full flex-1"></video>
        </div>
        <!-- 添加视频标题和标签 -->
        <div class="mt-4 px-1">
          <h3 class="text-lg font-medium">{{ videoData.video.title }}</h3>
          <div class="mt-2 flex flex-wrap gap-1">
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
    <el-card class="h-full w-[65%] overflow-hidden shadow-md">
      <div class="flex h-full flex-col">
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
        <div class="flex-1 overflow-auto p-4">
          <!-- 总结摘要内容 -->
          <div v-if="activeTab === 'summary'">
            <!-- 使用v-html渲染Markdown转换后的HTML -->
            <div v-if="summary" class="markdown-body" v-html="summary"></div>
            <p v-else class="text-gray-500">暂无摘要内容</p>

            <!-- 添加重新生成按钮 -->
            <div class="mt-4 flex justify-end">
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
          <div v-else-if="activeTab === 'subtitles'" class="h-full">
            <!-- 整体布局容器 -->
            <div class="flex h-[calc(100%-2rem)] flex-col">
              <!-- 完整文本区域 -->
              <div class="mb-2 flex items-center justify-between">
                <h4 class="font-medium">完整文本:</h4>
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
              <div
                class="mb-4 rounded-lg border border-gray-200 bg-gray-50 p-4"
                style="height: 120px"
              >
                <el-scrollbar height="75px">
                  <p class="leading-relaxed text-gray-700">
                    {{ subtitlesData.text }}
                  </p>
                </el-scrollbar>
              </div>

              <!-- 字幕列表区域 -->
              <div class="flex flex-1 flex-col">
                <div class="mb-2 flex items-center justify-between">
                  <h4 class="font-medium">字幕时间轴:</h4>
                  <span class="text-xs text-gray-500"
                    >共 {{ subtitlesData.chunks.length }} 个片段</span
                  >
                </div>
                <el-scrollbar
                  height="65vh"
                  class="subtitle-scrollbar rounded border border-gray-100"
                >
                  <div class="p-1">
                    <div
                      v-for="(chunk, index) in subtitlesData.chunks"
                      :key="index"
                      class="m-3 rounded bg-gray-50 p-3 transition-colors hover:bg-gray-100"
                    >
                      <div class="mb-1 text-xs text-gray-500">
                        {{ formatTimestamp(chunk.timestamp[0]) }} -
                        {{ formatTimestamp(chunk.timestamp[1]) }}
                      </div>
                      <div class="text-gray-800">{{ chunk.text }}</div>
                    </div>
                  </div>
                </el-scrollbar>
              </div>
            </div>
          </div>

          <!-- 分析过程内容 -->
          <div v-else-if="activeTab === 'process'">
            <h3 class="mb-4 text-lg font-medium">视频分析过程</h3>

            <!-- 使用hasAssessments和assessmentItems计算属性 -->
            <div v-if="hasAssessments" class="space-y-4">
              <div
                v-for="item in assessmentItems"
                :key="item.key"
                class="border-l-4 border-blue-500 py-2 pl-4"
              >
                <div class="mb-2 flex items-center justify-between">
                  <div class="font-medium">
                    {{ item.name }} ({{ item.key }})
                  </div>
                  <div
                    class="text-lg font-bold"
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
                  class="mt-2 cursor-pointer text-gray-600 hover:text-blue-500"
                  @click="goToReasoning(item.key)"
                >
                  点击查看详细评估理由
                </div>
                <div v-else class="mt-2 text-gray-600">无评估理由</div>
              </div>
            </div>

            <!-- 没有评估数据时显示提示 -->
            <div v-else class="py-8 text-center text-gray-500">
              <div class="mb-2 text-2xl">📊</div>
              <div>暂无分析数据</div>
            </div>
          </div>

          <!-- 威胁报告内容 -->
          <div v-else-if="activeTab === 'threat'">
            <h3 class="mb-4 text-lg font-medium">内容威胁分析报告</h3>

            <!-- 加载状态 -->
            <div
              v-if="reportLoading"
              class="flex items-center justify-center py-12"
            >
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
                class="mb-4 border-t-4"
                :class="`border-${riskLevelInfo.class}`"
              >
                <div class="flex items-center justify-between">
                  <div class="flex items-center">
                    <el-tag
                      :type="riskLevelInfo.class"
                      size="large"
                      effect="dark"
                      class="mr-3"
                    >
                      {{ riskLevelInfo.text }}
                    </el-tag>
                    <div class="text-lg font-medium">
                      风险概率:
                      <span :style="{ color: riskLevelInfo.color }"
                        >{{
                          (reportData.risk_probability * 100).toFixed(1)
                        }}%</span
                      >
                    </div>
                  </div>
                  <div>
                    <!-- 重新生成按钮 -->
                    <el-button
                      type="primary"
                      @click="regenerateReport"
                      :icon="Refresh"
                      size="small"
                    >
                      重新生成
                    </el-button>
                  </div>
                </div>
              </el-card>

              <!-- 分析报告内容 -->
              <el-card class="report-content">
                <div
                  class="markdown-body"
                  v-html="md.render(reportData.report)"
                ></div>
              </el-card>

              <!-- 评分摘要 -->
              <el-card class="mt-4">
                <template #header>
                  <div class="font-medium">评估指标摘要</div>
                </template>
                <div class="grid grid-cols-2 gap-4">
                  <div
                    v-for="(score, key) in reportData.scores"
                    :key="key"
                    class="score-item"
                  >
                    <div class="mb-1 flex items-center justify-between">
                      <div class="text-sm">
                        {{ assessmentNames[key.replace('_', '')] || key }}
                      </div>
                      <div
                        :style="{ color: getScoreColor(score) }"
                        class="font-bold"
                      >
                        {{ score ? score.toFixed(1) : 'N/A' }}
                      </div>
                    </div>
                    <el-progress
                      :percentage="score * 100"
                      :color="getScoreColor(score)"
                      :stroke-width="8"
                      :show-text="false"
                    />
                  </div>
                </div>
              </el-card>
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
</style>
