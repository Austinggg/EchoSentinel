<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import MarkdownIt from 'markdown-it';

import {
  ElButton,
  ElCard,
  ElAlert,
  ElIcon,
  ElProgress,
  ElDivider,
  ElSkeleton,
  ElSkeletonItem,
  ElMessage,
  ElBreadcrumb,
  ElBreadcrumbItem,
} from 'element-plus';
import { ArrowLeft, InfoFilled } from '@element-plus/icons-vue';

const route = useRoute();
const router = useRouter();
const loading = ref(true);
const videoId = ref('');
const itemKey = ref('');
const reasoningData = ref(null);
const videoTitle = ref('');

// 创建markdown-it实例
const md = new MarkdownIt({
  html: true,
  breaks: true,
  linkify: true,
  typographer: true,
});

// 高亮功能相关
const enableHighlight = ref(true);
const userHighlights = ref<string[]>([]);

// 定义关键词列表（根据不同评估项）
const getKeywords = (itemKey: string): string[] => {
  const keywordMap = {
    p1: ['背景信息', '充分', '完整', '详细', '具体', '全面'],
    p2: ['准确', '客观', '事实', '证据', '数据', '研究', '科学', '误导', '夸大', '偏见'],
    p3: ['完整性', '遗漏', '缺失', '表达', '逻辑'],
    p4: ['意图', '目的', '动机', '正当', '合理'],
    p5: ['信誉', '可信', '权威', '专业', '历史'],
    p6: ['中立', '情感', '煽动', '极端', '偏激', '客观'],
    p7: ['自主', '诱导', '强迫', '选择', '自由'],
    p8: ['一致', '矛盾', '冲突', '统一', '协调']
  };
  return keywordMap[itemKey] || [];
};

// 高亮关键词的函数
const highlightKeywords = (text: string, keywords: string[]): string => {
  if (!enableHighlight.value || !keywords.length) return text;
  
  let result = text;
  keywords.forEach(keyword => {
    const regex = new RegExp(`(${keyword})`, 'gi');
    result = result.replace(regex, '<mark class="keyword-highlight">$1</mark>');
  });
  
  return result;
};

// 高亮用户选择的文本
const highlightUserSelection = (text: string): string => {
  if (!userHighlights.value.length) return text;
  
  let result = text;
  userHighlights.value.forEach(highlight => {
    const regex = new RegExp(`(${highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    result = result.replace(regex, '<mark class="user-highlight">$1</mark>');
  });
  
  return result;
};

// 处理文本选择
const handleTextSelection = () => {
  const selection = window.getSelection();
  if (selection && selection.toString().trim()) {
    const selectedText = selection.toString().trim();
    if (!userHighlights.value.includes(selectedText)) {
      userHighlights.value.push(selectedText);
      ElMessage.success(`已高亮: ${selectedText}`);
    }
    selection.removeAllRanges();
  }
};

// 清除所有用户高亮
const clearUserHighlights = () => {
  userHighlights.value = [];
  ElMessage.info('已清除所有用户高亮');
};

// 切换高亮功能
const toggleHighlight = () => {
  enableHighlight.value = !enableHighlight.value;
  // 重新渲染内容
  if (reasoningData.value && reasoningData.value.reasoning) {
    renderReasoningContent();
  }
};

// 渲染推理内容
const renderReasoningContent = () => {
  if (!reasoningData.value || !reasoningData.value.reasoning) return;
  
  let content = reasoningData.value.reasoning;
  
  // 如果没有markdown格式，添加标题和段落结构
  if (!content.includes('#')) {
    content = `## 评估理由\n\n${content}`;
  }
  
  // 先转换markdown
  let renderedContent = md.render(content);
  
  // 应用关键词高亮
  if (enableHighlight.value) {
    const keywords = getKeywords(itemKey.value);
    renderedContent = highlightKeywords(renderedContent, keywords);
  }
  
  // 应用用户高亮
  renderedContent = highlightUserSelection(renderedContent);
  
  reasoningData.value.renderedReasoning = renderedContent;
};

// 定义评估项的语义映射
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

// 根据评分获取进度条颜色
const getScoreColor = (score: number): string => {
  if (score >= 0.8) return '#67C23A'; // 绿色
  if (score >= 0.5) return '#E6A23C'; // 橙色
  return '#F56C6C'; // 红色
};

// 格式化评分值（保留1位小数）
const formatScore = (score: number): string => {
  return score !== null && score !== undefined ? score.toFixed(1) : 'N/A';
};

// 返回上一页
const goBack = () => {
  router.back();
};

// 加载详细评估理由数据
const loadReasoningData = async () => {
  try {
    loading.value = true;
    videoId.value = route.query.id as string;
    itemKey.value = route.query.item as string;

    if (!videoId.value || !itemKey.value) {
      ElMessage.error('参数不完整，无法加载评估理由');
      loading.value = false;
      return;
    }

    // 获取视频分析数据
    const response = await axios.get(`/api/videos/${videoId.value}/analysis`);
    const data = response.data.data;
    
    if (!data || !data.video) {
      ElMessage.error('无法加载视频数据');
      loading.value = false;
      return;
    }

    // 设置视频标题
    videoTitle.value = data.video.title || '未知视频';

    // 获取特定评估项的理由
    if (data.analysis && data.analysis.assessments && data.analysis.assessments[itemKey.value]) {
      const assessmentItem = data.analysis.assessments[itemKey.value];
      
      reasoningData.value = {
        key: itemKey.value,
        name: assessmentNames[itemKey.value] || itemKey.value,
        score: assessmentItem.score,
        reasoning: assessmentItem.reasoning,
        renderedReasoning: '',
      };
      
      // 渲染内容
      renderReasoningContent();
    } else {
      ElMessage.warning(`未找到评估项 ${itemKey.value} 的理由数据`);
    }

    loading.value = false;
  } catch (error) {
    console.error('加载评估理由失败:', error);
    ElMessage.error('加载评估理由数据失败');
    loading.value = false;
  }
};

// 重新评估
const reassessItem = async () => {
  try {
    if (!videoId.value || !itemKey.value) return;

    ElMessage.info(`正在重新评估 ${assessmentNames[itemKey.value]}...`);
    const response = await axios.post(`/api/videos/${videoId.value}/assess/${itemKey.value}`, {
      force: true
    });

    if (response.data.code === 0) {
      ElMessage.success('评估完成，正在刷新数据');
      // 重新加载数据
      await loadReasoningData();
    } else {
      throw new Error(response.data.message || '评估失败');
    }
  } catch (error) {
    console.error('重新评估失败:', error);
    ElMessage.error('重新评估失败: ' + (error.message || '未知错误'));
  }
};

// 页面加载时获取数据
onMounted(() => {
  loadReasoningData();
});
</script>

<template>
  <div class="reason-page">
    <!-- 面包屑导航 -->
    <div class="mb-4 flex items-center justify-between">
      <el-button @click="goBack" type="default" size="small">
        <el-icon class="mr-1"><ArrowLeft /></el-icon>
        返回
      </el-button>
      
      <!-- 高亮控制按钮 -->
      <div class="flex items-center gap-2">
        <el-button 
          :type="enableHighlight ? 'primary' : 'default'" 
          size="small" 
          @click="toggleHighlight"
        >
          {{ enableHighlight ? '关闭高亮' : '开启高亮' }}
        </el-button>
        <el-button 
          v-if="userHighlights.length > 0"
          type="warning" 
          size="small" 
          @click="clearUserHighlights"
        >
          清除标记 ({{ userHighlights.length }})
        </el-button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="flex h-full items-center justify-center py-12">
      <el-skeleton :rows="10" animated />
    </div>

    <!-- 内容区域 -->
    <el-card v-else-if="reasoningData" class="mb-4">
      <template #header>
        <div class="flex items-center justify-between">
          <div>
            <h2 class="m-0 text-lg font-medium">
              {{ reasoningData.name }} 评估详情
            </h2>
            <p class="mt-1 text-sm text-gray-500">
              视频: {{ videoTitle }}
            </p>
          </div>
          <div class="flex items-center">
            <span class="mr-4">评分: 
              <span 
                class="text-xl font-bold" 
                :style="{ color: getScoreColor(reasoningData.score) }"
              >
                {{ formatScore(reasoningData.score) }}
              </span>
            </span>
            <el-button type="primary" size="small" @click="reassessItem">
              重新评估
            </el-button>
          </div>
        </div>
      </template>

      <!-- 进度条 -->
      <div class="mb-6">
        <el-progress 
          :percentage="reasoningData.score * 100" 
          :color="getScoreColor(reasoningData.score)"
          :format="() => formatScore(reasoningData.score)"
          :stroke-width="14"
        />
        <div class="mt-2 flex justify-between text-sm">
          <span class="text-red-500">风险高</span>
          <span class="text-yellow-500">中等</span>
          <span class="text-green-500">安全</span>
        </div>
      </div>

      <!-- 评估理由说明 -->
      <el-alert
        type="info"
        :closable="false"
        class="mb-4"
      >
        <template #title>
          <div class="flex items-center">
            <el-icon class="mr-1"><InfoFilled /></el-icon>
            <span>评估说明</span>
          </div>
        </template>
        <div class="text-sm">
          <p v-if="itemKey === 'p1'">
            <strong>背景信息充分性</strong> 评估内容是否提供了足够的背景信息，帮助用户全面了解主题。
            分数越高表示背景信息越充分。
          </p>
          <p v-else-if="itemKey === 'p2'">
            <strong>背景信息准确性</strong> 评估内容提供的背景信息是否准确，分数越高表示信息准确度越高。
          </p>
          <p v-else-if="itemKey === 'p3'">
            <strong>内容完整性</strong> 评估内容表达是否完整，没有重要信息的遗漏。分数越高表示内容越完整。
          </p>
          <p v-else-if="itemKey === 'p4'">
            <strong>意图正当性</strong> 评估内容的意图是否正当，分数越高表示意图越正当。
          </p>
          <p v-else-if="itemKey === 'p5'">
            <strong>发布者信誉</strong> 评估发布者的历史信誉，分数越高表示发布者信誉越好。
          </p>
          <p v-else-if="itemKey === 'p6'">
            <strong>情感中立性</strong> 评估内容的情感表达是否中立，分数越高表示情感越中立、越不煽动。
          </p>
          <p v-else-if="itemKey === 'p7'">
            <strong>行为自主性</strong> 评估内容是否尊重用户自主选择，分数越高表示越不诱导行为。
          </p>
          <p v-else-if="itemKey === 'p8'">
            <strong>信息一致性</strong> 评估内容中的各个陈述是否保持一致性，分数越高表示一致性越好。
          </p>
        </div>
      </el-alert>

      <!-- 高亮使用说明 -->
      <el-alert
        v-if="enableHighlight"
        type="success"
        :closable="false"
        class="mb-4"
        show-icon
      >
        <template #title>
          <span>文本高亮已启用</span>
        </template>
        <div class="text-sm">
          <p class="mb-2">
            <span class="keyword-highlight px-1">关键词</span> 会自动高亮显示
          </p>
          <p>选中文本可以手动标记重点内容</p>
        </div>
      </el-alert>

      <el-divider>详细评估理由</el-divider>
      
      <!-- 使用v-html渲染Markdown转换后的HTML -->
      <div 
        v-if="reasoningData.renderedReasoning" 
        class="markdown-body reasoning-content"
        v-html="reasoningData.renderedReasoning"
        @mouseup="handleTextSelection"
      ></div>
      
      <!-- 如果没有理由内容 -->
      <div v-else class="py-6 text-center text-gray-500">
        <p>未提供详细评估理由</p>
      </div>
    </el-card>
    
    <!-- 没有数据 -->
    <el-card v-else class="text-center py-8">
      <el-icon class="text-gray-400 text-3xl mb-2">
        <i class="el-icon-warning"></i>
      </el-icon>
      <p class="text-gray-500">未找到评估理由数据</p>
      <el-button class="mt-4" @click="goBack" type="primary">
        返回视频分析
      </el-button>
    </el-card>
  </div>
</template>

<style scoped>
.reason-page {
  padding: 16px;
}

/* 高亮样式 */
:deep(.keyword-highlight) {
  background: linear-gradient(120deg, #a8e6cf 0%, #dcedc8 100%);
  color: #2e7d32;
  font-weight: 500;
  padding: 1px 3px;
  border-radius: 3px;
  box-shadow: 0 1px 3px rgba(46, 125, 50, 0.2);
}

:deep(.user-highlight) {
  background: linear-gradient(120deg, #ffecb3 0%, #fff3c4 100%);
  color: #e65100;
  font-weight: 500;
  padding: 1px 3px;
  border-radius: 3px;
  box-shadow: 0 1px 3px rgba(230, 81, 0, 0.2);
  cursor: pointer;
}

:deep(.user-highlight:hover) {
  background: linear-gradient(120deg, #ffe082 0%, #ffecb3 100%);
}

/* Markdown样式增强 */
:deep(.markdown-body) {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  color: #24292e;
  word-break: break-word;
  user-select: text;
}

:deep(.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4) {
  margin-top: 24px;
  margin-bottom: 16px;
  font-weight: 600;
  line-height: 1.25;
  color: #1976d2;
}

:deep(.markdown-body h1) { 
  font-size: 2em;
  border-bottom: 2px solid #1976d2;
  padding-bottom: 0.3em;
}
:deep(.markdown-body h2) {
  font-size: 1.5em;
  padding-bottom: 0.3em;
  border-bottom: 1px solid #eaecef;
}
:deep(.markdown-body h3) { 
  font-size: 1.25em;
  color: #1565c0;
}

:deep(.markdown-body p) { 
  margin-bottom: 16px;
  text-align: justify;
}

:deep(.markdown-body ul, .markdown-body ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}

:deep(.markdown-body li) { 
  margin-bottom: 0.5em;
  line-height: 1.6;
}

:deep(.markdown-body pre) {
  padding: 16px;
  overflow: auto;
  font-size: 85%;
  line-height: 1.45;
  background-color: #f6f8fa;
  border-radius: 6px;
  border-left: 4px solid #1976d2;
}

:deep(.markdown-body code) {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(27, 31, 35, 0.05);
  border-radius: 3px;
  font-weight: 500;
}

:deep(.markdown-body strong) {
  font-weight: 600;
  color: #d32f2f;
}

:deep(.markdown-body em) {
  font-style: italic;
  color: #1976d2;
}

.reasoning-content {
  padding: 20px;
  background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
  border-radius: 8px;
  border-left: 4px solid #1890ff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  position: relative;
}

.reasoning-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, #1890ff, #52c41a, #faad14);
  border-radius: 8px 8px 0 0;
}

/* 选择文本时的样式 */
.reasoning-content::selection {
  background: rgba(24, 144, 255, 0.2);
}

.reasoning-content :deep(*::selection) {
  background: rgba(24, 144, 255, 0.2);
}
</style>