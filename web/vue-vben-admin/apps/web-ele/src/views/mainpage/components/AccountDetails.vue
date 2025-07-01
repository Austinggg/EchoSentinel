<script lang="ts" setup>
import { defineProps, defineEmits } from 'vue';
import {
  ElCard,
  ElAvatar,
  ElButton,
  ElIcon,
  ElTooltip,
  ElDivider,
} from 'element-plus';
import {
  Share,
  Refresh,
  CircleCheckFilled,
  Location,
} from '@element-plus/icons-vue';

// 导入统计图表组件
import StatsCharts from './StatsCharts.vue';

// Props
const props = defineProps<{
  accountInfo: any;
  platformName: string;
  loading: boolean;
}>();

// Emits
const emit = defineEmits<{
  refresh: [];
}>();

// 格式化数字(显示为1.2k, 3.5w等)
const formatNumber = (num) => {
  if (!num) return '0';
  if (num >= 10000) {
    return (num / 10000).toFixed(1) + 'w';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'k';
  }
  return num;
};

// 打开抖音主页
const openDouyinProfile = () => {
  if (props.accountInfo?.sec_uid) {
    const url = `https://www.douyin.com/user/${props.accountInfo.sec_uid}`;
    window.open(url, '_blank');
  }
};

// 刷新账号信息
const handleRefresh = () => {
  emit('refresh');
};
</script>

<template>
  <el-card class="user-card">
    <template #header>
      <div class="card-header">
        <span class="card-header-title">账号详情</span>
        <div class="card-header-actions">
          <!-- 访问抖音主页按钮 -->
          <el-button
            type="primary"
            size="small"
            plain
            @click="openDouyinProfile"
            v-if="accountInfo && accountInfo.sec_uid"
          >
            <el-icon><Share /></el-icon>
            访问抖音主页
          </el-button>
          <el-button
            type="primary"
            size="small"
            plain
            @click="handleRefresh"
            :loading="loading"
          >
            <el-icon><Refresh /></el-icon>
            刷新信息
          </el-button>
        </div>
      </div>
    </template>

    <!-- 用户信息布局 -->
    <div class="user-info-container">
      <!-- 左侧：用户头像 -->
      <div class="user-avatar-section">
        <div class="account-avatar-container">
          <el-avatar
            :size="100"
            :src="accountInfo.avatar_medium || accountInfo.avatar"
            class="account-avatar"
          />
          <div v-if="accountInfo.custom_verify" class="verified-badge">
            <el-tooltip :content="accountInfo.custom_verify">
              <el-icon><CircleCheckFilled /></el-icon>
            </el-tooltip>
          </div>
        </div>
      </div>

      <!-- 右侧：基本用户信息 -->
      <div class="user-details-section">
        <div class="account-header">
          <h2 class="account-name">{{ accountInfo.nickname }}</h2>
          <div class="account-id">
            ID: {{ accountInfo.sec_uid?.substring(0, 20) }}...
          </div>
        </div>

        <div class="account-stats">
          <div class="stat-item">
            <div class="stat-value">
              {{ formatNumber(accountInfo.following_count) }}
            </div>
            <div class="stat-label">关注</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">
              {{ formatNumber(accountInfo.follower_count) }}
            </div>
            <div class="stat-label">粉丝</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">
              {{ formatNumber(accountInfo.total_favorited || 0) }}
            </div>
            <div class="stat-label">获赞</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">
              {{ formatNumber(accountInfo.aweme_count) }}
            </div>
            <div class="stat-label">作品</div>
          </div>
        </div>

        <!-- 用户地理位置 -->
        <div v-if="accountInfo.ip_location" class="account-location">
          <el-icon><Location /></el-icon> {{ accountInfo.ip_location }}
        </div>

        <!-- 用户简介 -->
        <div v-if="accountInfo.signature" class="account-bio">
          {{ accountInfo.signature }}
        </div>
      </div>
    </div>

    <!-- 统计图表区域 -->
    <div class="analysis-section">
      <el-divider content-position="center">内容分析概览</el-divider>
      <StatsCharts :account-info="accountInfo" />
    </div>
  </el-card>
</template>

<style scoped>
/* 用户卡片样式 */
.user-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header-title {
  font-size: 18px;
  font-weight: 500;
}

.card-header-actions {
  display: flex;
  gap: 10px;
}

/* 用户信息布局 */
.user-info-container {
  display: flex;
  gap: 30px;
  margin-bottom: 30px;
}

.user-avatar-section {
  flex-shrink: 0;
}

.user-details-section {
  flex: 1;
}

/* 用户头像和认证徽章 */
.account-avatar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  margin-bottom: 16px;
}

.account-avatar {
  border: 3px solid #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  margin-bottom: 12px;
}

.verified-badge {
  position: absolute;
  bottom: 65px;
  right: calc(50% - 50px);
  background: #409eff;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  border: 2px solid white;
}

/* 账号信息 */
.account-header {
  margin-bottom: 15px;
}

.account-name {
  font-size: 22px;
  margin: 0 0 4px 0;
}

.account-id {
  color: #909399;
  font-size: 14px;
}

.account-stats {
  display: flex;
  gap: 30px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.stat-item {
  padding: 15px 25px;
  text-align: center;
  transition: all 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-5px);
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #606266;
}

.account-location {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #606266;
  margin: 10px 0;
  font-size: 14px;
}

.account-bio {
  margin: 16px 0;
  line-height: 1.5;
  color: #606266;
  word-break: break-word;
}

/* 分析概览区域 */
.analysis-section {
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .user-info-container {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .account-stats {
    justify-content: center;
  }

  .account-location,
  .account-bio {
    justify-content: center;
  }

  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .card-header-actions {
    width: 100%;
    justify-content: space-around;
  }
}
</style>
