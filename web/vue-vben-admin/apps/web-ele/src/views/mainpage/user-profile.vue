<script lang="ts" setup>
import type {
  RankInfo,
  UserCluster,
  UserFullProfile,
} from '../demos/userAnalyse/types';

import { computed, onMounted, reactive, ref, watch } from 'vue'; // 添加watch
import { useRoute, useRouter } from 'vue-router'; // 添加useRouter
import axios from 'axios'; // 添加这行
import {
  Location,
  Share,
  ArrowLeft,
  ArrowRight,
  UserFilled,
  Search,
  Fold,     // 添加收起图标
  Expand,   // 添加展开图标
} from '@element-plus/icons-vue'; // 添加缺失的图标
import {
  ElAlert,
  ElAvatar,
  ElButton,
  ElCard,
  ElDescriptions,
  ElDescriptionsItem,
  ElDivider,
  ElIcon,
  ElImage,
  ElInput, // 添加这个
  ElMessage, // 添加这个
  ElPagination, // 添加这个
  ElProgress,
  ElSkeleton,
  ElSkeletonItem,
  ElTable,
  ElTableColumn,
  ElTag,
  ElEmpty, // 添加这个
} from 'element-plus';

import { requestClient } from '#/api/request';

const route = useRoute();
const router = useRouter(); // 添加router
const userId = computed(() => route.query.userId as string);
const platform = computed(() => route.query.platform as string); // 获取平台

// 数据状态
const loading = ref(true);
const error = ref('');
const userProfile = ref<UserFullProfile>();
const analyticsLoading = ref(false);
const rankInfo = reactive<RankInfo>({ lossValue: 0, anomalyScore: 0 });
const similarUsers = ref([]);
const similarClusters = ref<UserCluster[]>([]);

// 添加用户列表相关状态
const userListVisible = ref(true); // 侧边栏可见性
const userList = ref([]); // 用户列表数据
const userListLoading = ref(false); // 用户列表加载状态
const searchUserKeyword = ref(''); // 用户搜索关键词
const currentPage = ref(1); // 用户列表当前页
const pageSize = ref(10); // 用户列表每页数量
const totalUsers = ref(0); // 用户总数
const initialLoad = ref(true); // 添加这行

// 过滤后的用户列表
const filteredUserList = computed(() => {
  if (!searchUserKeyword.value) return userList.value;

  const keyword = searchUserKeyword.value.toLowerCase();
  return userList.value.filter(
    (user) => user.nickname && user.nickname.toLowerCase().includes(keyword),
  );
});

// 加载用户列表
const loadUserList = async () => {
  try {
    userListLoading.value = true;

    const params = {
      page: currentPage.value,
      per_page: pageSize.value,
      sort_by: 'created_at',
      sort_order: 'desc',
      status: 'completed',
    };

    if (platform.value) {
      params.platform = platform.value;
    }

    const response = await axios.get('/api/analysis/tasks', { params });

    if (response.data.code === 200) {
      userList.value = response.data.data.tasks || [];
      totalUsers.value = response.data.data.total || 0;

      // 添加这段逻辑：如果是初始加载且没有传递userId，默认选择第一个用户
      if (initialLoad.value && !userId.value && userList.value.length > 0) {
        const firstUser = userList.value[0];
        await router.replace({
          path: '/main/user-profile',
          query: {
            platform: firstUser.platform,
            userId: firstUser.platform_user_id,
          },
        });
        initialLoad.value = false;
        return; // 等待路由更新后再加载用户画像
      }

      initialLoad.value = false;
    } else {
      throw new Error(response.data.message || '获取用户列表失败');
    }
  } catch (err) {
    console.error('加载用户列表失败:', err);
    userList.value = [];
    totalUsers.value = 0;
    initialLoad.value = false;
  } finally {
    userListLoading.value = false;
  }
};
// 切换用户
const switchUser = (newUser) => {
  if (newUser.platform_user_id === userId.value) return;

  router.push({
    path: '/main/user-profile',
    query: {
      platform: newUser.platform,
      userId: newUser.platform_user_id,
    },
  });
};

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val;
  loadUserList();
};

// 切换侧边栏可见性
const toggleUserList = () => {
  userListVisible.value = !userListVisible.value;
};

// 监听URL变化，重新加载用户画像
watch(
  () => route.query.userId,
  (newId) => {
    if (newId && !initialLoad.value) {
      // 添加 !initialLoad.value 条件
      loading.value = true;
      loadUserProfile();
    }
  },
);

// 修改 loadUserProfile 函数
const loadUserProfile = async () => {
  // 添加这个检查：如果没有userId，不执行加载
  if (!userId.value) {
    loading.value = false;
    return;
  }

  try {
    loading.value = true;
    analyticsLoading.value = true;

    const data = await requestClient.post<UserFullProfile>(
      '/userAnalyse/getProfile',
      {
        sec_uid: userId.value,
      },
    );
    userProfile.value = data;
    loading.value = false;

    if (userProfile.value.sec_uid) {
      await loadRankAnalysis();
      await loadSimilarUsers();
      await loadSimilarClusters();
    }
  } catch (error) {
    console.error('加载用户画像失败:', error);
    error.value = '加载用户画像失败';
    loading.value = false;
    analyticsLoading.value = false;
  }
};
// 加载用户异常分析数据
const loadRankAnalysis = async () => {
  try {
    const data = await requestClient.post<RankInfo>('/userAnalyse/getRank', {
      sec_uid: userId.value,
    });
    rankInfo.lossValue = data.lossValue;
    rankInfo.anomalyScore = data.anomalyScore;
    analyticsLoading.value = false;
  } catch (error) {
    console.error('加载用户分析数据失败:', error);
    // 设置默认值，避免页面崩溃
    rankInfo.lossValue = 0;
    rankInfo.anomalyScore = 0;
    analyticsLoading.value = false;
    ElMessage.error('用户分析数据加载失败');
  }
};

// 加载相似用户
const loadSimilarUsers = async () => {
  const response = await requestClient.post('/userAnalyse/similarUser', {
    sec_uid: userId.value,
  });
  similarUsers.value = response.similarUser;
};

// 加载相似集群
const loadSimilarClusters = async () => {
  const response = await requestClient.post('/userAnalyse/similarCluster', {
    sec_uid: userId.value,
  });
  similarClusters.value = response.similarCluster;
};

// 异常分数颜色
const colors = [
  { color: '#67C23A', percentage: 30 },
  { color: '#E6A23C', percentage: 50 },
  { color: '#F56C6C', percentage: 70 },
];

// 格式化性别
const formatGender = (gender: any) => {
  if (!gender) return '未知';
  switch (gender) {
    case '1':
    case '男': {
      return '男';
    }
    case '2':
    case '女': {
      return '女';
    }
    default: {
      return '未知';
    }
  }
};

// 格式化数字
const formatNumber = (num: any) => {
  if (!num && num !== 0) return '0';

  num = Number.parseInt(num);
  if (num >= 10_000 * 10_000) {
    return `${(num / (10_000 * 10_000)).toFixed(1)}亿`;
  } else if (num >= 10_000) {
    return `${(num / 10_000).toFixed(1)}w`;
  } else if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}k`;
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

// 打开相似用户链接
const openSimilarUserProfile = (row: { sec_uid: any }) => {
  if (row.sec_uid) {
    const url = `https://www.douyin.com/user/${row.sec_uid}`;
    window.open(url, '_blank');
  }
};
// 点击集群中的用户头像
const handleClusterAvatarClick = (
  cluster_index: number,
  avatar_index: number,
) => {
  const uid = similarClusters.value[cluster_index]?.sec_uids[avatar_index];
  const url = `https://www.douyin.com/user/${uid}`;
  window.open(url, '_blank');
};
// 修改 onMounted
onMounted(async () => {
  // 先加载用户列表
  await loadUserList();

  // 如果有userId且不是初始加载，再加载用户画像
  if (userId.value && !initialLoad.value) {
    await loadUserProfile();
  }
});
</script>

<template>
  <div class="user-profile-page">
    <!-- 用户列表侧边栏 -->
    <div class="user-list-sidebar" :class="{ collapsed: !userListVisible }">
      <!-- 侧边栏内容保持不变 -->
      <div class="sidebar-header">
        <div class="header-title">
          <h3>用户列表</h3>
          <div class="sidebar-toggle" @click="toggleUserList">
            <ElIcon>
              <Fold v-if="userListVisible" />
              <Expand v-else />
            </ElIcon>
          </div>
        </div>
        <ElInput
          v-model="searchUserKeyword"
          placeholder="搜索用户..."
          clearable
          size="small"
        >
          <template #prefix>
            <ElIcon><Search /></ElIcon>
          </template>
        </ElInput>
      </div>
      <!-- ... 侧边栏其他内容保持不变 ... -->
      <div class="sidebar-content" v-loading="userListLoading">
        <ElEmpty
          v-if="filteredUserList.length === 0"
          description="暂无用户数据"
        />
        <div v-else class="user-list">
          <div
            v-for="(user, index) in filteredUserList"
            :key="index"
            class="user-list-item"
            :class="{ active: user.platform_user_id === userId }"
            @click="switchUser(user)"
          >
            <ElAvatar :size="36" :src="user.avatar">
              {{ user.nickname?.charAt(0) || '?' }}
            </ElAvatar>
            <div class="user-list-info">
              <div class="user-list-name">{{ user.nickname }}</div>
              <div class="user-list-platform">
                <ElTag size="small">{{ user.platform }}</ElTag>
                <ElTag
                  v-if="user.digital_human_probability >= 0.7"
                  size="small"
                  type="danger"
                >
                  数字人
                </ElTag>
                <ElTag
                  v-else-if="
                    user.digital_human_probability !== undefined &&
                    user.digital_human_probability < 0.7
                  "
                  size="small"
                  type="success"
                >
                  真实人
                </ElTag>
              </div>
            </div>
          </div>
        </div>
        <div class="sidebar-pagination">
          <ElPagination
            v-if="totalUsers > pageSize"
            :current-page="currentPage"
            :page-size="pageSize"
            :total="totalUsers"
            layout="prev, pager, next"
            small
            @current-change="handleCurrentChange"
          />
        </div>
      </div>
    </div>
    
    <!-- 内容区域 - 修复滚动问题 -->
    <div class="user-profile-container">
      <div v-if="!userId && !userListLoading" class="no-user-selected">
        <ElCard>
          <div class="no-user-content">
            <ElIcon size="64" color="#c0c4cc">
              <UserFilled />
            </ElIcon>
            <h3>请选择用户</h3>
            <p>从左侧列表中选择一个用户查看详细信息</p>
          </div>
        </ElCard>
      </div>
      
      <!-- 关键修改：添加 profile-content-wrapper 容器 -->
      <div v-else class="profile-content-wrapper">
        <ElCard v-if="error" class="mb-5">
          <ElAlert :title="error" type="error" show-icon />
        </ElCard>

        <!-- 用户基本信息卡片 -->
        <ElCard v-loading="loading" class="mb-5">
          <template #header>
            <div class="card-header">
              <div class="card-title">用户基本信息</div>
              <div class="card-actions">
                <ElButton
                  type="primary"
                  size="small"
                  plain
                  @click="openDouyinProfile"
                  v-if="userProfile && userProfile.sec_uid"
                >
                  <ElIcon><Share /></ElIcon>
                  访问抖音主页
                </ElButton>
              </div>
            </div>
          </template>

          <div class="profile-content" v-if="userProfile">
            <div class="profile-header">
              <ElAvatar
                :size="80"
                :src="userProfile.avatar_medium"
                class="profile-avatar"
              >
                {{ userProfile.nickname?.charAt(0) || '?' }}
              </ElAvatar>
              <div class="profile-info">
                <h2 class="profile-name">{{ userProfile.nickname }}</h2>
                <div class="profile-details">
                  <span class="profile-id">ID: {{ userProfile.sec_uid }}</span>
                  <span v-if="userProfile.ip_location" class="profile-location">
                    <ElIcon><Location /></ElIcon>
                    IP属地: {{ userProfile.ip_location }}
                  </span>
                </div>
                <div class="profile-signature" v-if="userProfile.signature">
                  {{ userProfile.signature }}
                </div>
              </div>
            </div>

            <ElDivider />

            <ElDescriptions :column="3" border>
              <ElDescriptionsItem label="性别">
                {{ formatGender(userProfile.gender) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="用户年龄">
                {{ userProfile.user_age || '未知' }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="位置">
                {{ userProfile.country }} {{ userProfile.province }}
                {{ userProfile.city }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="作品数">
                {{ formatNumber(userProfile.aweme_count) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="获赞数">
                {{ formatNumber(userProfile.total_favorited) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="喜欢数">
                {{ formatNumber(userProfile.favoriting_count) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="粉丝数">
                {{ formatNumber(userProfile.follower_count) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="关注数">
                {{ formatNumber(userProfile.following_count) }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="账号特性">
                <ElTag v-if="userProfile.is_star" type="success" class="ml-2">
                  明星
                </ElTag>
                <ElTag
                  v-if="userProfile.is_gov_media_vip"
                  type="info"
                  class="ml-2"
                >
                  政媒
                </ElTag>
                <ElTag
                  v-if="userProfile.is_mix_user"
                  type="warning"
                  class="ml-2"
                >
                  混合账号
                </ElTag>
                <ElTag
                  v-if="userProfile.is_series_user"
                  type="primary"
                  class="ml-2"
                >
                  系列账号
                </ElTag>
              </ElDescriptionsItem>
            </ElDescriptions>

            <div
              class="profile-covers"
              v-if="userProfile.covers && userProfile.covers.length > 0"
            >
              <h3>用户封面</h3>
              <div class="covers-container">
                <div
                  v-for="(url, index) in userProfile.covers"
                  :key="index"
                  class="cover-item"
                >
                  <ElImage
                    :src="url"
                    fit="cover"
                    :preview-src-list="userProfile.covers"
                    :initial-index="index"
                  />
                </div>
              </div>
            </div>
          </div>

          <ElSkeleton v-else :loading="loading" animated>
            <template #template>
              <div class="skeleton-content">
                <div class="skeleton-header">
                  <ElSkeletonItem
                    variant="circle"
                    style="width: 80px; height: 80px"
                  />
                  <div class="skeleton-info">
                    <ElSkeletonItem variant="h3" style="width: 200px" />
                    <ElSkeletonItem variant="text" style="width: 240px" />
                    <ElSkeletonItem variant="text" style="width: 300px" />
                  </div>
                </div>
                <ElSkeletonItem variant="p" style="width: 100%" />
                <ElSkeletonItem variant="p" style="width: 100%" />
                <ElSkeletonItem variant="p" style="width: 100%" />
              </div>
            </template>
          </ElSkeleton>
        </ElCard>

        <!-- 用户分析卡片 -->
        <ElCard v-if="userProfile" class="mb-5" v-loading="analyticsLoading">
          <template #header>
            <div class="card-header">
              <div class="card-title">用户异常分析</div>
            </div>
          </template>

          <div class="analysis-content">
            <div class="analysis-metrics">
              <ElDescriptions :column="1" border>
                <ElDescriptionsItem label="本次样本重构误差">
                  {{ rankInfo.lossValue }}
                </ElDescriptionsItem>
                <ElDescriptionsItem label="异常分数说明">
                  重构损失越大，异常分数越大，越有可能是异常用户
                </ElDescriptionsItem>
              </ElDescriptions>
            </div>

            <div class="analysis-gauge">
              <ElProgress
                type="dashboard"
                :percentage="rankInfo.anomalyScore"
                :color="colors"
              >
                <template #default="{ percentage }">
                  <span class="percentage-value">{{ percentage }}%</span>
                  <span class="percentage-label">异常分数</span>
                </template>
              </ElProgress>
            </div>
          </div>

          <div v-if="rankInfo.anomalyScore >= 70" class="mt-4">
            <ElAlert
              title="高风险用户"
              type="error"
              description="该用户异常分数较高，可能存在风险行为，建议进行人工审核"
              show-icon
              :closable="false"
            />
          </div>
          <div v-else-if="rankInfo.anomalyScore >= 50" class="mt-4">
            <ElAlert
              title="中风险用户"
              type="warning"
              description="该用户异常分数中等，建议关注其内容变化"
              show-icon
              :closable="false"
            />
          </div>
        </ElCard>

        <!-- 相似用户 -->
        <ElCard v-if="userProfile && similarUsers.length > 0" class="mb-5">
          <template #header>
            <div class="card-header">
              <div class="card-title">相似用户</div>
            </div>
          </template>

          <ElTable
            :data="similarUsers"
            style="width: 100%"
            @row-click="openSimilarUserProfile"
          >
            <ElTableColumn label="头像" width="80">
              <template #default="scope">
                <ElAvatar :src="scope.row.avatar_medium">
                  {{ scope.row.nickname?.charAt(0) || '?' }}
                </ElAvatar>
              </template>
            </ElTableColumn>
            <ElTableColumn prop="nickname" label="昵称" />
            <ElTableColumn prop="similarity" label="相似度">
              <template #default="scope">
                <ElProgress
                  :percentage="scope.row.similarity * 100"
                  :format="(val) => `${val.toFixed(1)}%`"
                  :stroke-width="10"
                  color="#409EFF"
                />
              </template>
            </ElTableColumn>
            <ElTableColumn label="操作" width="120">
              <template #default="scope">
                <ElButton
                  type="primary"
                  link
                  @click.stop="openSimilarUserProfile(scope.row)"
                >
                  查看用户
                </ElButton>
              </template>
            </ElTableColumn>
          </ElTable>
        </ElCard>

        <!-- 相似集群 -->
        <ElCard v-if="userProfile && similarClusters.length > 0" class="mb-5">
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
                <ElAvatar
                  v-for="(avatar, i) in cluster.avatar_list"
                  @click="() => handleClusterAvatarClick(index, i)"
                  :key="i"
                  :src="avatar"
                  :size="40"
                  class="cluster-avatar"
                />
              </div>
            </div>
          </div>
        </ElCard>
      </div>
    </div>
  </div>
</template>

<style scoped>
.user-profile-page {
  display: flex;
  position: relative;
  height: 100vh; /* 固定页面高度 */
  width: 100%;
  overflow: hidden;
}

/* 侧边栏样式 */
.user-list-sidebar {
  width: 300px;
  border-right: 1px solid #e6e6e6;
  background-color: #f8f8f8;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
  overflow: hidden;
  position: relative;
  height: 100%;
  flex-shrink: 0;
}

.user-list-sidebar.collapsed {
  width: 0;
  transform: translateX(-300px);
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #e6e6e6;
  background-color: #ffffff;
}

/* 标题栏样式 */
.header-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.header-title h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
  font-weight: 600;
}

/* 切换按钮新样式 */
.sidebar-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  background-color: transparent;
}

.sidebar-toggle:hover {
  background-color: #f2f6fc;
}

.sidebar-toggle .el-icon {
  font-size: 16px;
  color: #606266;
}

.sidebar-toggle:hover .el-icon {
  color: #409eff;
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.user-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.user-list-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;
}

.user-list-item:hover {
  background-color: #eaeaea;
  border-color: #d1d5db;
}

.user-list-item.active {
  background-color: #ecf5ff;
  border-left: 3px solid #409eff;
  border-color: #409eff;
}

.user-list-info {
  flex: 1;
  min-width: 0;
}

.user-list-name {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #303133;
}

.user-list-platform {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.sidebar-pagination {
  display: flex;
  justify-content: center;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e6e6e6;
}

/* 内容区域 - 修改为固定高度和滚动 */
.user-profile-container {
  flex: 1;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 可滚动的内容包装器 */
.profile-content-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

/* 无用户选择时的样式 */
.no-user-selected {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  padding: 20px;
}

.no-user-content {
  text-align: center;
  color: #909399;
  padding: 40px;
}

.no-user-content h3 {
  margin: 16px 0 8px 0;
  color: #606266;
  font-size: 18px;
  font-weight: 500;
}

.no-user-content p {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

/* 卡片样式 */
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
  color: #303133;
  font-weight: 600;
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
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.profile-covers {
  margin-top: 20px;
}

.profile-covers h3 {
  margin: 0 0 10px 0;
  font-size: 16px;
  color: #303133;
  font-weight: 600;
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
  border: 1px solid #e6e6e6;
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
  align-items: flex-start;
}

.analysis-metrics {
  flex: 2;
  min-width: 300px;
}

.analysis-gauge {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  min-width: 200px;
}

.percentage-value {
  display: block;
  font-size: 28px;
  font-weight: bold;
  color: #303133;
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
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 15px;
  width: calc(33.33% - 20px);
  min-width: 250px;
  background-color: #ffffff;
  transition: all 0.2s ease;
}

.cluster-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.1);
}

.cluster-header {
  margin-bottom: 10px;
}

.cluster-header h4 {
  margin: 0;
  color: #303133;
  font-size: 16px;
  font-weight: 600;
}

.avatar-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.cluster-avatar {
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: all 0.2s ease;
}

.cluster-avatar:hover {
  transform: scale(1.1);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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

/* 表格样式增强 */
.el-table {
  border-radius: 8px;
  overflow: hidden;
}

.el-table .el-table__row:hover > td {
  background-color: #f5f7fa;
}

/* 标签样式增强 */
.el-tag {
  margin-right: 4px;
  margin-bottom: 4px;
}

/* 进度条样式增强 */
.el-progress {
  margin: 8px 0;
}

/* 右侧内容区域滚动条样式 */
.profile-content-wrapper::-webkit-scrollbar {
  width: 8px;
}

.profile-content-wrapper::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.profile-content-wrapper::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

.profile-content-wrapper::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 左侧侧边栏滚动条样式 */
.sidebar-content::-webkit-scrollbar {
  width: 6px;
}

.sidebar-content::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.sidebar-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.sidebar-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 响应式调整 */
@media (max-width: 1200px) {
  .cluster-item {
    width: calc(50% - 20px);
  }
}

@media (max-width: 768px) {
  .user-list-sidebar {
    width: 280px;
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    z-index: 100;
  }
  
  .user-list-sidebar.collapsed {
    transform: translateX(-280px);
  }
  
  .profile-content-wrapper {
    padding: 10px;
  }
  
  .analysis-content {
    flex-direction: column;
    gap: 20px;
  }
  
  .analysis-metrics,
  .analysis-gauge {
    flex: none;
    min-width: auto;
  }

  .cluster-item {
    width: 100%;
    min-width: auto;
  }
  
  .profile-header {
    flex-direction: column;
    text-align: center;
    gap: 15px;
  }
  
  .profile-details {
    justify-content: center;
  }
  
  .covers-container {
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .user-list-sidebar {
    width: 260px;
  }
  
  .user-list-sidebar.collapsed {
    transform: translateX(-260px);
  }
  
  .profile-content-wrapper {
    padding: 8px;
  }
  
  .card-header {
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
  }
  
  .profile-name {
    font-size: 18px;
  }
  
  .cover-item {
    width: 100px;
    height: 100px;
  }
}

/* 加载状态样式 */
.el-loading-mask {
  border-radius: 8px;
}

/* 空状态样式 */
.el-empty {
  padding: 40px 20px;
}

.el-empty .el-empty__description {
  color: #909399;
  font-size: 14px;
}
</style>
