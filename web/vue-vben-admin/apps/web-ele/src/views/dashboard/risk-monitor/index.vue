<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import {
  ElCard,
  ElTag,
  ElAvatar,
  ElTooltip,
  ElPopover,
  ElEmpty,
  ElSkeleton,
  ElSkeletonItem,
  ElButton, // 新增：缺少的按钮组件
  ElTabs,
  ElTabPane,
  ElIcon,
  ElRow,
  ElCol,
  ElLoading,
  ElMessage,
  ElStatistic,
  ElImage,
} from 'element-plus';
import { useTransition } from '@vueuse/core';

import {
  Warning,
  VideoPlay,
  User,
  InfoFilled,
  Calendar,
  Link,
  Refresh,
  DataAnalysis,
  CaretTop,
  CaretBottom,
  TrendCharts,
  WarningFilled,
  Ship,
  Monitor,
  Picture,
  DArrowRight, // 添加 Picture 图标
} from '@element-plus/icons-vue';

const videoCount = ref(0);
const falsehoodCount = ref(0);
const userCount = ref(0);
const digitalCount = ref(0);

// 正确使用 useTransition
const transitionVideoCount = useTransition(videoCount);
const transitionFalsehoodCount = useTransition(falsehoodCount);
const transitionUserCount = useTransition(userCount);
const transitionDigitalCount = useTransition(digitalCount);

// 获取趋势数据（这里是模拟数据）
const getTrendInfo = (type) => {
  const trends = {
    videos: { isUp: true, percent: '24%' },
    falsehoods: { isUp: false, percent: '12%' },
    users: { isUp: true, percent: '8%' },
    digital: { isUp: true, percent: '32%' },
  };
  return trends[type] || { isUp: true, percent: '0%' };
};

// 在数据加载完成后应用动画
const applyTransitions = () => {
  if (monitorData.value) {
    videoCount.value = monitorData.value.high_risk_videos?.length || 0;
    falsehoodCount.value = monitorData.value.falsehoods?.length || 0;
    userCount.value = monitorData.value.high_risk_users?.length || 0;
    digitalCount.value = monitorData.value.digital_human_users?.length || 0;
  }
};

const router = useRouter();
const loading = ref(true);
const activeTab = ref('videos'); // 默认显示风险视频标签页
const monitorData = ref({
  high_risk_videos: [],
  falsehoods: [],
  high_risk_users: [],
  digital_human_users: [],
});
const getRiskClass = (probability) => {
  if (!probability && probability !== 0) return 'risk-low';
  if (probability >= 0.7) return 'risk-high';
  if (probability >= 0.4) return 'risk-medium';
  return 'risk-low';
};
// 加载风险监控数据
const loadRiskMonitorData = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/risk-monitor');
    if (
      response.data &&
      (response.data.code === 200 || response.data.code === 0)
    ) {
      monitorData.value = response.data.data;
    } else {
      console.error(
        '无法获取风险监控数据:',
        response.data?.message || '未知错误',
      );
    }
  } catch (error) {
    console.error('获取风险监控数据失败:', error);
  } finally {
    loading.value = false;
    applyTransitions();
  }
};

// 格式化日期时间
const formatDateTime = (dateStr) => {
  if (!dateStr) return '未知时间';
  const date = new Date(dateStr);
  return date
    .toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
    .replace(/\//g, '-');
};

// 格式化平台名称
const formatPlatform = (platform) => {
  if (!platform) return '上传';
  switch (platform.toLowerCase()) {
    case 'douyin':
      return '抖音';
    case 'tiktok':
      return 'TikTok';
    case 'bilibili':
      return 'B站';
    case 'upload':
      return '上传';
    default:
      return platform;
  }
};
const getPlatformIcon = (platform) => {
  if (!platform) return VideoPlay;
  switch (platform.toLowerCase()) {
    case 'douyin':
      return DArrowRight; // 使用抖音风格图标
    case 'tiktok':
      return DArrowRight; // 使用TikTok风格图标
    case 'bilibili':
      return VideoPlay; // 使用B站风格图标
    default:
      return VideoPlay;
  }
};

// 格式化数字人概率
const formatProbability = (probability) => {
  if (probability === undefined || probability === null) return '未知';
  return `${Math.round(probability * 100)}%`;
};

// 查看视频详情
const viewVideoDetail = (videoId) => {
  router.push(`/demos/content-analysis/analysis?id=${videoId}`);
};

// 查看用户详情
const viewUserProfile = (platform, userId) => {
  router.push(`/main/user-profile?platform=${platform}&userId=${userId}`);
};

onMounted(() => {
  loadRiskMonitorData();
});

setTimeout(() => {
  applyTransitions();
}, 100); // 给渲染一点时间
</script>

<template>
  <div class="risk-monitor-container p-4">
    <div class="dashboard-header">
      <h2 class="dashboard-title">
        <el-icon class="dashboard-icon"><Warning /></el-icon>
        风险监控中心
      </h2>
      <el-button type="primary" size="small" @click="loadRiskMonitorData">
        <el-icon><Refresh /></el-icon> 刷新数据
      </el-button>
    </div>
    <!-- 顶部风险概览卡片 - 使用 ElStatistic 组件 -->
    <el-card class="overview-card">
      <el-row class="overview-row" :gutter="20">
        <!-- 高风险视频 -->
        <el-col :span="6" class="overview-col">
          <div
            class="overview-icon"
            style="background-color: rgba(245, 108, 108, 0.1)"
          >
            <el-icon size="24" color="#F56C6C"><VideoPlay /></el-icon>
          </div>
          <div class="overview-content">
            <div class="overview-title">高风险视频</div>
            <el-statistic
              :value="transitionVideoCount"
              :precision="0"
              value-style="color: var(--el-color-danger); font-size: 24px; font-weight: bold;"
              class="stat-block"
            >
            </el-statistic>
            <div class="overview-trend">
              <span>较上周</span>
              <span
                :class="getTrendInfo('videos').isUp ? 'trend-up' : 'trend-down'"
              >
                {{ getTrendInfo('videos').percent }}
                <el-icon
                  ><component
                    :is="getTrendInfo('videos').isUp ? CaretTop : CaretBottom"
                /></el-icon>
              </span>
            </div>
          </div>
        </el-col>

        <!-- 不实信息 -->
        <el-col :span="6" class="overview-col">
          <div
            class="overview-icon"
            style="background-color: rgba(230, 162, 60, 0.1)"
          >
            <el-icon size="24" color="#E6A23C"><InfoFilled /></el-icon>
          </div>
          <div class="overview-content">
            <div class="overview-title">不实信息</div>
            <el-statistic
              :value="transitionFalsehoodCount"
              :precision="0"
              value-style="color: var(--el-color-warning); font-size: 24px; font-weight: bold;"
              class="stat-block"
            >
            </el-statistic>
            <div class="overview-trend">
              <span>较上周</span>
              <span
                :class="
                  getTrendInfo('falsehoods').isUp ? 'trend-up' : 'trend-down'
                "
              >
                {{ getTrendInfo('falsehoods').percent }}
                <el-icon
                  ><component
                    :is="
                      getTrendInfo('falsehoods').isUp ? CaretTop : CaretBottom
                    "
                /></el-icon>
              </span>
            </div>
          </div>
        </el-col>

        <!-- 高风险用户 -->
        <el-col :span="6" class="overview-col">
          <div
            class="overview-icon"
            style="background-color: rgba(64, 158, 255, 0.1)"
          >
            <el-icon size="24" color="#409EFF"><User /></el-icon>
          </div>
          <div class="overview-content">
            <div class="overview-title">高风险用户</div>
            <el-statistic
              :value="transitionUserCount"
              :precision="0"
              value-style="color: var(--el-color-primary); font-size: 24px; font-weight: bold;"
              class="stat-block"
            >
            </el-statistic>
            <div class="overview-trend">
              <span>较上周</span>
              <span
                :class="getTrendInfo('users').isUp ? 'trend-up' : 'trend-down'"
              >
                {{ getTrendInfo('users').percent }}
                <el-icon
                  ><component
                    :is="getTrendInfo('users').isUp ? CaretTop : CaretBottom"
                /></el-icon>
              </span>
            </div>
          </div>
        </el-col>

        <!-- 数字人用户 -->
        <el-col :span="6" class="overview-col">
          <div
            class="overview-icon"
            style="background-color: rgba(103, 194, 58, 0.1)"
          >
            <el-icon size="24" color="#67C23A"><Monitor /></el-icon>
          </div>
          <div class="overview-content">
            <div class="overview-title">数字人用户</div>
            <el-statistic
              :value="transitionDigitalCount"
              :precision="0"
              value-style="color: var(--el-color-success); font-size: 24px; font-weight: bold;"
              class="stat-block"
            >
            </el-statistic>
            <div class="overview-trend">
              <span>较上周</span>
              <span
                :class="
                  getTrendInfo('digital').isUp ? 'trend-up' : 'trend-down'
                "
              >
                {{ getTrendInfo('digital').percent }}
                <el-icon
                  ><component
                    :is="getTrendInfo('digital').isUp ? CaretTop : CaretBottom"
                /></el-icon>
              </span>
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="3" animated />
      <el-skeleton :rows="3" animated class="mt-6" />
    </div>

    <div v-else class="dashboard-content">
      <!-- 使用标签页组织内容 -->
      <el-tabs v-model="activeTab" type="card" class="risk-monitor-tabs">
        <!-- 风险视频标签页 -->
        <el-tab-pane label="风险视频" name="videos">
          <div class="tab-content">
            <div class="section-title">
              <el-icon><VideoPlay /></el-icon> 高风险视频监控
            </div>

            <div
              v-if="
                monitorData.high_risk_videos &&
                monitorData.high_risk_videos.length > 0
              "
              class="risk-videos-grid"
            >
              <el-card
                v-for="video in monitorData.high_risk_videos"
                :key="video.id"
                class="video-card"
                shadow="hover"
                @click="viewVideoDetail(video.id)"
              >
                <div class="video-thumbnail">
                  <el-image
                    :src="video.cover_url || ''"
                    fit="cover"
                    class="video-cover-image"
                    :preview-src-list="video.cover_url ? [video.cover_url] : []"
                  >
                    <template #error>
                      <div class="video-placeholder">
                        <el-icon><Picture /></el-icon>
                        <div class="error-text">加载失败</div>
                      </div>
                    </template>
                  </el-image>
                  <div class="platform-badge">
                    {{ formatPlatform(video.platform) }}
                  </div>
                </div>
                <div class="video-info">
                  <div class="video-title" :title="video.filename">
                    {{ video.filename }}
                  </div>
                  <div class="video-meta">
                    <div class="upload-time">
                      <el-icon><Calendar /></el-icon>
                      {{ formatDateTime(video.upload_time).split(' ')[0] }}
                    </div>
                    <div
                      class="risk-meter"
                      :class="getRiskClass(video.digital_human_probability)"
                    >
                      数字人概率:
                      {{ formatProbability(video.digital_human_probability) }}
                    </div>
                  </div>
                </div>
              </el-card>
            </div>
            <el-empty v-else description="暂无高风险视频数据" />
          </div>
        </el-tab-pane>

        <!-- 事实核查标签页 -->
        <el-tab-pane label="事实核查" name="factcheck">
          <div class="tab-content">
            <div class="section-title">
              <el-icon><InfoFilled /></el-icon> 事实核查不实信息
            </div>

            <div
              v-if="monitorData.falsehoods && monitorData.falsehoods.length > 0"
              class="falsehood-timeline"
            >
              <div
                v-for="(item, index) in monitorData.falsehoods"
                :key="index"
                class="falsehood-item"
                @click="viewVideoDetail(item.video_id)"
              >
                <div class="timeline-dot"></div>
                <div class="falsehood-content">
                  <div class="falsehood-header">
                    <el-tag type="danger" effect="dark" size="small"
                      >不实信息</el-tag
                    >
                    <div class="check-time">
                      {{ formatDateTime(item.check_time) }}
                    </div>
                  </div>
                  <div class="falsehood-claim">{{ item.claim }}</div>
                  <div class="falsehood-footer">
                    <div class="source-video">
                      <el-icon><VideoPlay /></el-icon>
                      来源: {{ item.video_name }}
                    </div>
                    <el-button size="small" type="danger" text
                      >查看详情</el-button
                    >
                  </div>
                  <div class="fact-conclusion">
                    <div class="conclusion-label">核查结论:</div>
                    <div class="conclusion-content">{{ item.conclusion }}</div>
                  </div>
                </div>
              </div>
            </div>
            <el-empty v-else description="暂无事实核查不实信息" />
          </div>
        </el-tab-pane>

        <!-- 风险用户标签页 -->
        <el-tab-pane label="风险用户" name="users">
          <div class="tab-content">
            <div class="section-title">
              <el-icon><User /></el-icon> 高风险用户监控
            </div>

            <div
              v-if="
                monitorData.high_risk_users &&
                monitorData.high_risk_users.length > 0
              "
              class="user-cards-container"
            >
              <el-card
                v-for="user in monitorData.high_risk_users"
                :key="user.platform_user_id"
                class="user-risk-card"
                shadow="hover"
                @click="viewUserProfile(user.platform, user.platform_user_id)"
              >
                <div class="user-avatar">
                  <el-avatar :size="50" :src="user.avatar" fit="cover">
                    {{ user.nickname?.charAt(0) || '?' }}
                  </el-avatar>
                  <div class="platform-icon">
                    <el-tag size="small">{{
                      formatPlatform(user.platform)
                    }}</el-tag>
                  </div>
                </div>
                <div class="user-info">
                  <div class="user-name">{{ user.nickname }}</div>
                  <div class="risk-gauge-container">
                    <div class="risk-gauge-track">
                      <div
                        class="risk-gauge-fill"
                        :style="{
                          width: `${(user.digital_human_probability || 0) * 100}%`,
                        }"
                        :class="getRiskClass(user.digital_human_probability)"
                      ></div>
                    </div>
                    <div class="risk-gauge-label">
                      数字人概率:
                      <span
                        :class="getRiskClass(user.digital_human_probability)"
                      >
                        {{ formatProbability(user.digital_human_probability) }}
                      </span>
                    </div>
                  </div>
                </div>
              </el-card>
            </div>
            <el-empty v-else description="暂无高风险用户数据" />

            <div class="section-title mt-6">
              <el-icon><Link /></el-icon> 最近识别数字人用户
            </div>

            <div
              v-if="
                monitorData.digital_human_users &&
                monitorData.digital_human_users.length > 0
              "
              class="digital-human-container"
            >
              <div
                v-for="(user, index) in monitorData.digital_human_users"
                :key="index"
                class="digital-human-item"
                @click="viewUserProfile(user.platform, user.platform_user_id)"
              >
                <el-avatar :size="40" :src="user.avatar">{{
                  user.nickname?.charAt(0) || '?'
                }}</el-avatar>
                <div class="dh-user-info">
                  <div class="dh-username">{{ user.nickname }}</div>
                  <div class="dh-meta">
                    <el-tag size="small" type="danger">{{
                      formatProbability(user.digital_human_probability)
                    }}</el-tag>
                    <div class="dh-platform">
                      {{ formatPlatform(user.platform) }}
                    </div>
                    <div class="dh-time">
                      {{ formatDateTime(user.completed_at).split(' ')[0] }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <el-empty v-else description="暂无数字人用户数据" />
          </div>
        </el-tab-pane>

        <!-- 可以添加统计分析标签页 -->
        <el-tab-pane label="统计分析" name="analytics">
          <div class="tab-content">
            <div class="coming-soon-container">
              <el-empty description="统计分析功能即将上线，敬请期待">
                <template #image>
                  <el-icon class="coming-soon-icon"><DataAnalysis /></el-icon>
                </template>
              </el-empty>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<style scoped>
.risk-monitor-container {
  min-height: 100vh;
  background-color: var(--el-bg-color-page);
  color: var(--el-text-color-primary);
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.dashboard-title {
  display: flex;
  align-items: center;
  font-size: 22px;
  font-weight: 600;
  margin: 0;
}

.dashboard-icon {
  font-size: 24px;
  margin-right: 12px;
  color: var(--el-color-danger);
}

.dashboard-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.section-title {
  display: flex;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
  margin: 12px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.section-title .el-icon {
  margin-right: 8px;
  font-size: 20px;
}


.video-card {
  cursor: pointer;
  transition: transform 0.2s;
  overflow: hidden;
  height: 100%;
}

.video-card:hover {
  transform: translateY(-4px);
}

.video-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.05);
  color: var(--el-color-primary);
}

.video-placeholder .el-icon {
  font-size: 36px;
  opacity: 0.7;
}

.platform-badge {
  position: absolute;
  top: 8px;
  left: 8px;
  padding: 2px 8px;
  background-color: rgba(0, 0, 0, 0.6);
  color: white;
  border-radius: 4px;
  font-size: 12px;
}

.video-info {
  padding: 12px 0 4px;
}

.video-title {
  font-weight: 600;
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.video-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.upload-time {
  display: flex;
  align-items: center;
  gap: 4px;
}

.risk-meter {
  font-weight: 500;
}

.risk-high {
  color: var(--el-color-danger);
}

.risk-medium {
  color: var(--el-color-warning);
}

.risk-low {
  color: var(--el-color-success);
}

/* 事实核查时间线 */
.falsehood-timeline {
  position: relative;
  margin-left: 16px;
  padding-left: 20px;
  border-left: 2px dashed var(--el-border-color);
}

.falsehood-item {
  position: relative;
  margin-bottom: 20px;
  cursor: pointer;
  transition: transform 0.15s;
}

.falsehood-item:hover {
  transform: translateX(4px);
}

.timeline-dot {
  position: absolute;
  left: -29px;
  top: 12px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background-color: var(--el-color-danger);
  border: 3px solid var(--el-bg-color);
}

.falsehood-content {
  background-color: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
  border-left: 4px solid var(--el-color-danger);
  border-radius: 4px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.falsehood-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.check-time {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.falsehood-claim {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 12px;
  color: var(--el-text-color-primary);
}

.falsehood-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 13px;
}

.source-video {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--el-text-color-secondary);
}

.fact-conclusion {
  margin-top: 12px;
  font-size: 14px;
  background-color: rgba(var(--el-color-danger-rgb), 0.05);
  padding: 10px;
  border-radius: 4px;
}

.conclusion-label {
  font-weight: 600;
  margin-bottom: 6px;
  color: var(--el-color-danger);
}

.conclusion-content {
  color: var(--el-text-color-regular);
  line-height: 1.5;
}

/* 用户风险区域 */
.user-cards-container {
  display: flex;
  overflow-x: auto;
  gap: 16px;
  padding-bottom: 8px;
}

.user-risk-card {
  min-width: 240px;
  max-width: 300px;
  cursor: pointer;
  transition: transform 0.2s;
}

.user-risk-card:hover {
  transform: translateY(-4px);
}

.user-avatar {
  position: relative;
  display: flex;
  justify-content: center;
  padding-bottom: 16px;
}

.platform-icon {
  position: absolute;
  bottom: 0;
  right: 80px;
}

.user-info {
  text-align: center;
  margin-top: 8px;
}

.user-name {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 12px;
}

.risk-gauge-container {
  margin-top: 12px;
}

.risk-gauge-track {
  height: 6px;
  background-color: var(--el-fill-color-lighter);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
}

.risk-gauge-fill {
  height: 100%;
  border-radius: 3px;
}

.risk-gauge-label {
  text-align: right;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

/* 数字人用户列表 */
.digital-human-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}

.digital-human-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 6px;
  background-color: var(--el-bg-color);
  border: 1px solid var(--el-border-color-light);
  cursor: pointer;
  transition: all 0.2s;
}

.digital-human-item:hover {
  border-color: var(--el-color-danger-light-5);
  box-shadow: 0 2px 12px rgba(var(--el-color-danger-rgb), 0.1);
  transform: translateY(-2px);
}

.dh-user-info {
  flex: 1;
  min-width: 0;
}

.dh-username {
  font-weight: 500;
  margin-bottom: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dh-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.dh-platform {
  border-left: 1px solid var(--el-border-color-lighter);
  padding-left: 8px;
}
.mb-4 {
  margin-bottom: 1rem;
}

.risk-monitor-tabs {
  background-color: var(--el-bg-color);
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.tab-content {
  padding: 16px;
}

.coming-soon-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

.coming-soon-icon {
  font-size: 64px;
  color: var(--el-color-primary);
}

/* 覆盖 el-tabs 默认样式 */
:deep(.el-tabs__item) {
  font-size: 16px;
  padding: 0 20px;
  height: 48px;
  line-height: 48px;
}

:deep(.el-tabs__header) {
  margin-bottom: 0;
}

.trend-up {
  color: var(--el-color-success);
  display: inline-flex;
  align-items: center;
  gap: 2px;
}

.trend-down {
  color: var(--el-color-danger);
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
/* 更紧凑的一行风险概览样式 */
.overview-card {
  margin-bottom: 24px;
}

.overview-row {
  display: flex;
  align-items: center;
}

.overview-col {
  display: flex;
  align-items: center;
  padding: 12px 0;
  position: relative;
}

.overview-col:not(:last-child)::after {
  content: '';
  position: absolute;
  right: 0;
  top: 15%;
  height: 70%;
  width: 1px;
  background-color: var(--el-border-color-lighter);
}

.overview-icon {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
}

.overview-content {
  flex: 1;
}

.overview-title {
  font-size: 14px;
  color: var(--el-text-color-secondary);
  margin-bottom: 8px;
}

.overview-value {
  font-size: 24px;
  font-weight: bold;
  line-height: 1;
  margin-bottom: 8px;
}

.overview-value.danger {
  color: var(--el-color-danger);
}
.overview-value.warning {
  color: var(--el-color-warning);
}
.overview-value.primary {
  color: var(--el-color-primary);
}
.overview-value.success {
  color: var(--el-color-success);
}

.overview-trend {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

/* 保留原有的趋势样式 */
.trend-up,
.trend-down {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}

.trend-up {
  color: var(--el-color-success);
}

.trend-down {
  color: var(--el-color-danger);
}

/* 移动端响应式处理 */
@media (max-width: 768px) {
  .overview-col {
    padding: 12px;
  }

  .overview-col:not(:last-child)::after {
    display: none;
  }

  .overview-icon {
    width: 36px;
    height: 36px;
    margin-right: 8px;
  }

  .overview-value {
    font-size: 18px;
  }
}
.stat-block {
  margin-bottom: 8px;
}

:deep(.el-statistic__content) {
  margin: 0;
  padding: 0;
}

:deep(.el-statistic__head) {
  display: none; /* 隐藏标题，我们已经有自己的标题了 */
}

.video-card:hover .video-cover-image {
  transform: scale(1.05);
}
/* 修改视频卡片网格以适应竖屏视频 */
.risk-videos-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); /* 减小宽度 */
  gap: 16px;
}

/* 修改视频缩略图为竖屏比例 */
.video-thumbnail {
  position: relative;
  height: 240px; /* 增加高度 */
  width: 100%;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
}

.video-cover-image {
  height: 100%;
  width: 100%;
  object-fit: cover;
  transition: transform 0.3s;
}
</style>
