<script lang="ts" setup>
import { ref, onMounted,computed } from 'vue';
import axios from 'axios';
import { useRoute } from 'vue-router';
import { marked } from 'marked'; // 使用命名导入

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
} from 'element-plus';
// 定义评估项的语义映射
const assessmentNames = {
  p1: '背景信息充分性',
  p2: '背景信息准确性',
  p3: '内容完整性',
  p4: '不当意图',
  p5: '发布者历史',
  p6: '情感煽动性',
  p7: '诱导行为',
  p8: '信息一致性',
};
// 添加数据加载状态
const loading = ref(true);
const videoData = ref(null);
const videoSrc = ref('');
const subtitlesData = ref({ chunks: [], text: '' });
const route = useRoute();
const summary = ref(''); // 存储解析后的摘要HTML
const assessmentData = ref({}); // 新增：专门存储评估数据
// 添加评估数据可用性检查的计算属性
const hasAssessments = computed(() => {
  return assessmentData.value && Object.keys(assessmentData.value).length > 0;
});

// 添加格式化评估项的计算属性
const assessmentItems = computed(() => {
  if (!hasAssessments.value) return [];
  
  return Object.entries(assessmentData.value)
    .filter(([_, item]) => item && item.score !== null && item.score !== undefined)
    .map(([key, item]) => ({
      key,
      name: assessmentNames[key] || key,
      score: item.score,
      reasoning: item.reasoning
    }));
});
// 根据ID加载视频数据
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
      summary.value = marked(videoData.value.analysis.summary);
    }

    // 保存评估数据到专门的变量
    if (videoData.value.analysis && videoData.value.analysis.assessments) {
      assessmentData.value = videoData.value.analysis.assessments;
      console.log('加载了评估数据:', assessmentData.value);
    } else {
      console.warn('没有找到评估数据');
      assessmentData.value = {};
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
// 添加导航菜单激活状态
const activeTab = ref('summary');
// 添加缺失的时间戳格式化函数
const formatTimestamp = (seconds: number | undefined): string => {
  if (seconds === undefined) return '00:00';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};
// 菜单选择处理函数
const handleTabChange = (key: string) => {
  activeTab.value = key;
};
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

  <!-- 视频分析内容，仅在数据加载后显示 -->
  <div v-else class="flex h-full gap-4">
    <!-- 左侧卡片 - 占35%且高度100% -->
    <el-card class="h-full w-[35%] overflow-hidden shadow-md">
      <div class="flex h-full flex-col">
        <video controls :src="videoSrc" class="w-full flex-1"></video>
        <!-- 添加视频标题和标签 -->
        <div class="mt-4 p-2">
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
          </div>

          <!-- 字幕列表内容 -->
          <div v-else-if="activeTab === 'subtitles'" class="h-full">
            <!-- 整体布局容器 -->
            <div class="flex h-[calc(100%-2rem)] flex-col">
              <!-- 完整文本区域 -->
              <h4 class="font-medium">完整文本:</h4>
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

                <div class="mt-2 text-gray-600">
                  {{ item.reasoning ? '点击查看详细评估理由' : '无评估理由' }}
                </div>
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

            <!-- 根据风险等级展示不同状态 -->
            <div
              v-if="videoData.video.riskLevel === 'safe'"
              class="mb-4 rounded-lg bg-green-50 p-4"
            >
              <div class="flex items-center">
                <el-tag type="success" class="mr-2">安全</el-tag>
                <span class="font-medium">未检测到明显威胁</span>
              </div>
              <p class="mt-2 text-gray-600">
                此视频内容未发现违规或敏感内容，可以安全发布。
              </p>
            </div>

            <div
              v-else-if="videoData.video.riskLevel === 'warning'"
              class="mb-4 rounded-lg bg-yellow-50 p-4"
            >
              <div class="flex items-center">
                <el-tag type="warning" class="mr-2">警告</el-tag>
                <span class="font-medium">检测到潜在风险</span>
              </div>
              <p class="mt-2 text-gray-600">
                此视频可能含有敏感内容或误导信息，建议谨慎发布。
              </p>
            </div>

            <div
              v-else-if="videoData.video.riskLevel === 'danger'"
              class="mb-4 rounded-lg bg-red-50 p-4"
            >
              <div class="flex items-center">
                <el-tag type="danger" class="mr-2">危险</el-tag>
                <span class="font-medium">检测到高风险内容</span>
              </div>
              <p class="mt-2 text-gray-600">此视频含有违规内容，不建议发布。</p>
            </div>

            <div v-else class="mb-4 rounded-lg bg-gray-50 p-4">
              <div class="flex items-center">
                <el-tag type="info" class="mr-2">处理中</el-tag>
                <span class="font-medium">风险评估进行中</span>
              </div>
              <p class="mt-2 text-gray-600">
                系统正在评估此视频的风险等级，请稍后查看。
              </p>
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
</style>
