<script lang="ts" setup>
import { ref, onMounted, computed, onBeforeUnmount } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import { ElAlert, ElMessage } from 'element-plus';

// 导入子组件
import AccountDetails from './components/AccountDetails.vue';
import VideoList from './components/VideoList.vue';

const route = useRoute();
const router = useRouter();

// 获取路由参数
const platform = computed(() => route.query.platform as string);
const userId = computed(() => route.query.userId as string);

// 状态变量
const loading = ref(false);
const error = ref('');
const accountInfo = ref(null);

// 使用计算属性来确定平台名称
const platformName = computed(() => {
  switch (platform.value) {
    case 'douyin':
      return '抖音';
    case 'tiktok':
      return 'TikTok';
    case 'bilibili':
      return 'Bilibili';
    default:
      return platform.value;
  }
});

// 加载用户信息
const loadUserInfo = async () => {
  try {
    loading.value = true;

    // 尝试从数据库获取用户信息
    const dbResponse = await axios.get(
      `/api/account/by-secuid/${userId.value}`,
    );

    if (dbResponse.data.code === 200 && dbResponse.data.data) {
      // 数据库中已有用户数据
      accountInfo.value = dbResponse.data.data;
      console.log('从数据库加载用户信息成功:', accountInfo.value);
      return;
    }

    // 如果数据库没有，尝试从抖音API获取
    if (platform.value === 'douyin') {
      const response = await axios.get(
        `/api/douyin/web/handler_user_profile?sec_user_id=${userId.value}`,
      );

      if (response.data.code === 200) {
        accountInfo.value = response.data.data.user;
        console.log('从抖音API加载用户信息成功:', accountInfo.value);
      } else {
        throw new Error(response.data.message || '获取用户信息失败');
      }
    }
  } catch (err) {
    console.error('加载用户信息失败:', err);
    error.value = '获取用户信息失败';
  } finally {
    loading.value = false;
  }
};

// 刷新账号信息
const handleAccountRefresh = () => {
  loadUserInfo();
};

// 初始加载
onMounted(() => {
  if (!platform.value || !userId.value) {
    error.value = '缺少必要的参数';
    return;
  }
  loadUserInfo();
});
</script>

<template>
  <div
    v-if="$route.path === '/main/analysis-tasks/user-content'"
    class="user-content-container"
  >
    <!-- 错误提示 -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      show-icon
      :closable="false"
      class="error-alert"
    />

    <!-- 账号详情组件 -->
    <AccountDetails
      v-if="accountInfo"
      :account-info="accountInfo"
      :platform-name="platformName"
      :loading="loading"
      @refresh="handleAccountRefresh"
    />

    <!-- 视频列表组件 -->
    <VideoList
      v-if="accountInfo"
      :account-info="accountInfo"
      :platform="platform.value"
    />
  </div>
  <router-view v-else />
</template>

<style scoped>
.user-content-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.error-alert {
  margin-bottom: 20px;
}

@media (max-width: 768px) {
  .user-content-container {
    padding: 10px;
  }
}
</style>
