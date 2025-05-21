<script lang="ts" setup>
import { ref, onMounted, computed } from 'vue';
import { useRoute } from 'vue-router';
import axios from 'axios';
import {
  ElCard,
  ElAvatar,
  ElDescriptions,
  ElDescriptionsItem,
  ElDivider,
  ElProgress,
  ElTag,
  ElImage,
  ElButton,
  ElTable,
  ElTableColumn,
  ElSkeleton,
  ElSkeletonItem,
  ElEmpty,
  ElAlert,
} from 'element-plus';

import { Location, User, Share, Warning } from '@element-plus/icons-vue';
import ClusterScatter from '../demos/userAnalyse/ClusterScatter.vue';

const route = useRoute();
const platform = computed(() => route.query.platform as string);
const userId = computed(() => route.query.userId as string);

// 数据状态
const loading = ref(true);
const error = ref('');
const userProfile = ref(null);
const analyticsLoading = ref(false);
const rankInfo = ref({ lossValue: 0, anomalyScore: 0 });
const similarUsers = ref([]);
const similarClusters = ref([]);

// 修改loadUserProfile函数
const loadUserProfile = async () => {
  try {
    loading.value = true;
    const response = await axios.post('/api/userAnalyse/getProfile', {
      sec_uid: userId.value,
    });
    
    // 修改这里，接受code为0或200的情况
    if (response.data && (response.data.code === 200 || response.data.code === 0)) {
      userProfile.value = response.data.data;
      // 加载完用户信息后加载异常分析
      await loadRankAnalysis();
    } else {
      throw new Error(response.data?.message || '获取用户信息失败');
    }
  } catch (err) {
    console.error('加载用户信息失败:', err);
    error.value = '获取用户信息失败，请确认该用户已添加到系统';
  } finally {
    loading.value = false;
  }
};

// 加载用户异常分析数据
const loadRankAnalysis = async () => {
  try {
    analyticsLoading.value = true;
    const response = await axios.post('/api/userAnalyse/getRank', {
      sec_uid: userId.value,
    });

    if (response.data && (response.data.code === 200 || response.data.code === 0)) {
      rankInfo.value = response.data.data;
      // 继续加载相似用户和集群数据
      await Promise.all([loadSimilarUsers(), loadSimilarClusters()]);
    }
  } catch (err) {
    console.error('加载异常分析失败:', err);
  } finally {
    analyticsLoading.value = false;
  }
};

// 加载相似用户
const loadSimilarUsers = async () => {
  try {
    const response = await axios.post('/api/userAnalyse/similarUser', {
      sec_uid: userId.value,
    });

    if (response.data && (response.data.code === 200 || response.data.code === 0)) {
      similarUsers.value = response.data.data.similarUser;
    }
  } catch (err) {
    console.error('加载相似用户失败:', err);
  }
};

// 加载相似集群
const loadSimilarClusters = async () => {
  try {
    const response = await axios.post('/api/userAnalyse/similarCluster', {
      sec_uid: userId.value,
    });

    if (response.data && (response.data.code === 200 || response.data.code === 0)) {
      similarClusters.value = response.data.data.similarCluster;
    }
  } catch (err) {
    console.error('加载相似集群失败:', err);
  }
};

// 异常分数颜色
const colors = [
  { color: '#67C23A', percentage: 30 },
  { color: '#E6A23C', percentage: 70 },
  { color: '#F56C6C', percentage: 100 },
];

// 格式化性别
const formatGender = (gender) => {
  if (!gender) return '未知';
  switch (gender) {
    case '1':
    case '男':
      return '男';
    case '2':
    case '女':
      return '女';
    default:
      return '未知';
  }
};

// 格式化数字
const formatNumber = (num) => {
  if (!num && num !== 0) return '0';
  
  num = parseInt(num);
  if (num >= 10000 * 10000) {
    return (num / (10000 * 10000)).toFixed(1) + '亿';
  } else if (num >= 10000) {
    return (num / 10000).toFixed(1) + 'w';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'k';
  }
  return num.toString();
};

// 打开抖音链接
const openDouyinProfile = () => {
  if (userProfile.value?.sec_uid) {
    const url = `https://www.douyin.com/user/${userProfile.value.sec_uid}`;
    window.open(url, '_blank');
  }
};

// 用于跟踪ClusterScatter组件引用
const clusterScatterRef = ref(null);

// 标记用户位置
const showUserPosition = () => {
  if (clusterScatterRef.value) {
    clusterScatterRef.value.markPoint(userId.value);
  }
};

// 重置散点图
const resetClusterView = () => {
  if (clusterScatterRef.value) {
    clusterScatterRef.value.reDraw();
  }
};

// 打开相似用户链接
const openSimilarUserProfile = (row) => {
  if (row.sec_uid) {
    const url = `https://www.douyin.com/user/${row.sec_uid}`;
    window.open(url, '_blank');
  }
};

onMounted(() => {
  loadUserProfile();
});
</script>

<template>
  <div class="user-profile-container">
    <el-card v-if="error" class="mb-5">
      <el-alert :title="error" type="error" show-icon />
    </el-card>

    <!-- 用户基本信息卡片 -->
    <el-card v-loading="loading" class="mb-5">
      <template #header>
        <div class="card-header">
          <div class="card-title">用户基本信息</div>
          <div class="card-actions">
            <el-button 
              type="primary" 
              size="small" 
              plain 
              @click="openDouyinProfile"
              v-if="userProfile && userProfile.sec_uid"
            >
              <el-icon><Share /></el-icon>
              访问抖音主页
            </el-button>
          </div>
        </div>
      </template>

      <div class="profile-content" v-if="userProfile">
        <div class="profile-header">
          <el-avatar 
            :size="80" 
            :src="userProfile.avatar_medium" 
            class="profile-avatar"
          >
            {{ userProfile.nickname?.charAt(0) || '?' }}
          </el-avatar>
          <div class="profile-info">
            <h2 class="profile-name">{{ userProfile.nickname }}</h2>
            <div class="profile-details">
              <span class="profile-id">ID: {{ userProfile.sec_uid }}</span>
              <span v-if="userProfile.ip_location" class="profile-location">
                <el-icon><Location /></el-icon>
                IP属地: {{ userProfile.ip_location }}
              </span>
            </div>
            <div class="profile-signature" v-if="userProfile.signature">
              {{ userProfile.signature }}
            </div>
          </div>
        </div>

        <el-divider />

        <el-descriptions :column="3" border>
          <el-descriptions-item label="性别">{{ formatGender(userProfile.gender) }}</el-descriptions-item>
          <el-descriptions-item label="用户年龄">{{ userProfile.user_age || '未知' }}</el-descriptions-item>
          <el-descriptions-item label="位置">
            {{ userProfile.country }} {{ userProfile.province }} {{ userProfile.city }}
          </el-descriptions-item>
          <el-descriptions-item label="作品数">{{ formatNumber(userProfile.aweme_count) }}</el-descriptions-item>
          <el-descriptions-item label="获赞数">{{ formatNumber(userProfile.total_favorited) }}</el-descriptions-item>
          <el-descriptions-item label="喜欢数">{{ formatNumber(userProfile.favoriting_count) }}</el-descriptions-item>
          <el-descriptions-item label="粉丝数">{{ formatNumber(userProfile.follower_count) }}</el-descriptions-item>
          <el-descriptions-item label="关注数">{{ formatNumber(userProfile.following_count) }}</el-descriptions-item>
          <el-descriptions-item label="账号特性">
            <el-tag v-if="userProfile.is_star" type="success" class="ml-2">明星</el-tag>
            <el-tag v-if="userProfile.is_gov_media_vip" type="info" class="ml-2">政媒</el-tag>
            <el-tag v-if="userProfile.is_mix_user" type="warning" class="ml-2">混合账号</el-tag>
            <el-tag v-if="userProfile.is_series_user" type="primary" class="ml-2">系列账号</el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <div class="profile-covers" v-if="userProfile.covers && userProfile.covers.length">
          <h3>用户封面</h3>
          <div class="covers-container">
            <div v-for="(url, index) in userProfile.covers" :key="index" class="cover-item">
              <el-image 
                :src="url" 
                fit="cover"
                :preview-src-list="userProfile.covers"
                :initial-index="index"
              />
            </div>
          </div>
        </div>
      </div>
      
      <el-skeleton v-else :loading="loading" animated>
        <template #template>
          <div class="skeleton-content">
            <div class="skeleton-header">
              <el-skeleton-item variant="circle" style="width: 80px; height: 80px;" />
              <div class="skeleton-info">
                <el-skeleton-item variant="h3" style="width: 200px;" />
                <el-skeleton-item variant="text" style="width: 240px;" />
                <el-skeleton-item variant="text" style="width: 300px;" />
              </div>
            </div>
            <el-skeleton-item variant="p" style="width: 100%;" />
            <el-skeleton-item variant="p" style="width: 100%;" />
            <el-skeleton-item variant="p" style="width: 100%;" />
          </div>
        </template>
      </el-skeleton>
    </el-card>

    <!-- 用户分析卡片 -->
    <el-card v-if="userProfile" class="mb-5" v-loading="analyticsLoading">
      <template #header>
        <div class="card-header">
          <div class="card-title">用户异常分析</div>
        </div>
      </template>

      <div class="analysis-content">
        <div class="analysis-metrics">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="本次样本重构误差">{{ rankInfo.lossValue }}</el-descriptions-item>
            <el-descriptions-item label="异常分数说明">
              重构损失越大，异常分数越大，越有可能是异常用户
            </el-descriptions-item>
          </el-descriptions>
        </div>
        
        <div class="analysis-gauge">
          <el-progress
            type="dashboard"
            :percentage="rankInfo.anomalyScore"
            :color="colors"
          >
            <template #default="{ percentage }">
              <span class="percentage-value">{{ percentage }}%</span>
              <span class="percentage-label">异常分数</span>
            </template>
          </el-progress>
        </div>
      </div>

      <div v-if="rankInfo.anomalyScore >= 70" class="mt-4">
        <el-alert
          title="高风险用户"
          type="error"
          description="该用户异常分数较高，可能存在风险行为，建议进行人工审核"
          show-icon
          :closable="false"
        />
      </div>
      <div v-else-if="rankInfo.anomalyScore >= 50" class="mt-4">
        <el-alert
          title="中风险用户"
          type="warning"
          description="该用户异常分数中等，建议关注其内容变化"
          show-icon
          :closable="false"
        />
      </div>
    </el-card>

    <!-- 用户集群展示 -->
    <el-card v-if="userProfile" class="mb-5">
      <template #header>
        <div class="card-header">
          <div class="card-title">用户集群展示</div>
          <div class="card-actions">
            <el-button type="primary" size="small" @click="showUserPosition">
              显示用户位置
            </el-button>
            <el-button type="info" size="small" @click="resetClusterView">
              重置视图
            </el-button>
          </div>
        </div>
      </template>

      <ClusterScatter 
        ref="clusterScatterRef" 
        :sec-uid="userId" 
        style="width: 100%; height: 400px;"
      />
    </el-card>

    <!-- 相似用户 -->
    <el-card v-if="userProfile && similarUsers.length > 0" class="mb-5">
      <template #header>
        <div class="card-header">
          <div class="card-title">相似用户</div>
        </div>
      </template>

      <el-table 
        :data="similarUsers" 
        style="width: 100%" 
        @row-click="openSimilarUserProfile"
      >
        <el-table-column label="头像" width="80">
          <template #default="scope">
            <el-avatar :src="scope.row.avatar_medium">
              {{ scope.row.nickname?.charAt(0) || '?' }}
            </el-avatar>
          </template>
        </el-table-column>
        <el-table-column prop="nickname" label="昵称" />
        <el-table-column prop="similarity" label="相似度">
          <template #default="scope">
            <el-progress 
              :percentage="scope.row.similarity * 100" 
              :format="(val) => val.toFixed(1) + '%'" 
              :stroke-width="10"
              :color="'#409EFF'"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="scope">
            <el-button 
              type="primary" 
              link
              @click.stop="openSimilarUserProfile(scope.row)"
            >
              查看用户
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 相似集群 -->
    <el-card v-if="userProfile && similarClusters.length > 0" class="mb-5">
      <template #header>
        <div class="card-header">
          <div class="card-title">相似集群</div>
        </div>
      </template>

      <div class="similar-clusters">
        <div 
          v-for="(cluster, index) in similarClusters" 
          :key="index"
          class="cluster-item"
        >
          <div class="cluster-header">
            <h4>集群 {{ cluster.cluster_id }}</h4>
          </div>
          <div class="avatar-group">
            <el-avatar 
              v-for="(avatar, i) in cluster.avatar_list" 
              :key="i"
              :src="avatar"
              :size="40"
              class="cluster-avatar"
            />
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.user-profile-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.card-actions {
  display: flex;
  gap: 10px;
}

/* 用户基础信息样式 */
.profile-content {
  padding: 10px 0;
}

.profile-header {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.profile-avatar {
  flex-shrink: 0;
}

.profile-info {
  flex: 1;
}

.profile-name {
  margin: 0 0 10px 0;
  font-size: 22px;
}

.profile-details {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-bottom: 10px;
  color: #606266;
}

.profile-location {
  display: flex;
  align-items: center;
  gap: 5px;
}

.profile-signature {
  margin-top: 10px;
  font-style: italic;
  color: #606266;
  line-height: 1.5;
}

.profile-covers {
  margin-top: 20px;
}

.covers-container {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
}

.cover-item {
  width: 120px;
  height: 120px;
  border-radius: 8px;
  overflow: hidden;
}

.cover-item :deep(.el-image) {
  width: 100%;
  height: 100%;
}

/* 分析内容样式 */
.analysis-content {
  display: flex;
  flex-wrap: wrap;
  gap: 30px;
}

.analysis-metrics {
  flex: 2;
}

.analysis-gauge {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}

.percentage-value {
  display: block;
  font-size: 28px;
  font-weight: bold;
}

.percentage-label {
  display: block;
  margin-top: 5px;
  font-size: 14px;
  color: #606266;
}

/* 相似集群样式 */
.similar-clusters {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.cluster-item {
  border: 1px solid #EBEEF5;
  border-radius: 8px;
  padding: 15px;
  width: calc(33.33% - 20px);
  min-width: 250px;
}

.cluster-header {
  margin-bottom: 10px;
}

.cluster-header h4 {
  margin: 0;
  color: #303133;
}

.avatar-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.cluster-avatar {
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 骨架屏样式 */
.skeleton-content {
  padding: 20px 0;
}

.skeleton-header {
  display: flex;
  gap: 20px;
  margin-bottom: 30px;
}

.skeleton-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .analysis-content {
    flex-direction: column;
  }
  
  .cluster-item {
    width: 100%;
  }
}
</style>