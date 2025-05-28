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
  ElIcon, // 添加这个
  ElTag, // 添加这个
  ElResult, // 添加这个
} from 'element-plus';
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
import DigitalHumanTab from './components/DigitalHumanTab.vue'; // 添加这一行
import ProcessTab from './components/ProcessTab.vue';
import FactCheckTab from './components/FactCheckTab.vue';
import ThreatReportTab from './components/ThreatReportTab.vue';

const router = useRouter();
const route = useRoute();
const activeTab = ref('summary');

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
  }
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
  <!-- 加载状态 -->
  <div v-if="loading" class="loading-container">
    <div class="loading-content">
      <el-icon class="loading-icon">
        <i class="el-icon-loading" />
      </el-icon>
      <div>加载数据中...</div>
    </div>
  </div>

  <!-- 如果有子路由被激活，显示子路由内容 -->
  <router-view v-else-if="$route.path.includes('/reason')" />

  <!-- 视频分析内容，仅在数据加载后显示 -->
  <div v-else class="content-container">
    <!-- 左侧卡片 - 视频播放区域 -->
    <el-card class="side-card">
      <div class="card-content">
        <div class="video-container">
          <video controls :src="videoSrc" style="max-height: 100%"></video>
        </div>
        <!-- 视频标题和标签 -->
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

    <!-- 右侧卡片 - 分析内容区域 -->
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
</template>

<style scoped>
/* 加载状态 */
.loading-container {
  display: flex;
  height: 100%;
  align-items: center;
  justify-content: center;
}

.loading-content {
  text-align: center;
}

.loading-icon {
  font-size: 2.25rem;
  margin-bottom: 1rem;
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

/* 主容器布局 */
.content-container {
  display: flex;
  height: calc(100vh);
  flex-direction: column;
  gap: 1rem;
  padding: 1rem;
  box-sizing: border-box;
  overflow: hidden;
}

@media (min-width: 768px) {
  .content-container {
    flex-direction: row;
  }
}

/* 卡片样式 */
.side-card,
.main-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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
  flex: 1;
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

/* 修复卡片内容区溢出问题 */
:deep(.el-card__body) {
  height: 100%;
  padding: 15px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
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
</style>
