<script lang="ts" setup>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router'; // 添加这行
import axios from 'axios';
import {
  ElAlert,
  ElAvatar,
  ElButton,
  ElCard,
  ElCol,
  ElDivider,
  ElEmpty,
  ElIcon,
  ElImage,
  ElInput,
  ElMessage,
  ElNotification,
  ElRow,
  ElSegmented,
  ElSpace,
  ElTabPane,
  ElTabs,
  ElTag,
  ElTooltip,
  ElLoading
} from 'element-plus';

import {
  UserFilled,
  CircleCheckFilled,
  Link as LinkIcon,
} from '@element-plus/icons-vue';

// 平台定义
const platforms = [
  {
    id: 'douyin',
    name: '抖音',
    icon: '/icons/tiktok.png',
    placeholder: '请输入抖音用户主页链接',
    example: 'https://www.douyin.com/user/MS4wLjABAAAAWo0PSqzO1iCb7vAvcNm1xuf7jefYxgQb5ajsoLiiGBE',
    urlPattern: /douyin\.com\/user\/([^?]+)/,
    likeLabel: '获赞',
  },
  {
    id: 'tiktok',
    name: 'TikTok',
    icon: '/icons/tiktok.png',
    placeholder: '请输入TikTok用户主页链接',
    example: 'https://www.tiktok.com/@username',
    urlPattern: /tiktok\.com\/@([^?]+)/,
    likeLabel: '喜欢',
  },
  {
    id: 'bilibili',
    name: 'Bilibili',
    icon: '/icons/bilibili.png',
    placeholder: '请输入Bilibili用户主页链接',
    example: 'https://space.bilibili.com/12345678',
    urlPattern: /space\.bilibili\.com\/(\d+)/,
    likeLabel: '点赞',
  },
];

// 当前状态
const activePlatform = ref('douyin');
const userUrl = ref('');
const accountInfo = ref<any>(null);
const parsing = ref(false);

// 计算属性 - 当前平台信息
const activePlatformInfo = computed(() => {
  return platforms.find((p) => p.id === activePlatform.value) || platforms[0];
});

// 处理平台切换
const handlePlatformChange = (platform: string) => {
  userUrl.value = '';
  accountInfo.value = null;
};

// 解析用户URL
// 解析用户URL并通过API获取数据
const parseError = ref('');

const parseUserUrl = async () => {
  if (!userUrl.value) {
    ElMessage.warning('请输入用户URL');
    return;
  }

  parsing.value = true;

  try {
    // 从URL中提取用户ID
    const match = userUrl.value.match(activePlatformInfo.value.urlPattern);
    const userId = match ? match[1] : null;

    if (!userId) {
      throw new Error(`无法从URL中解析${activePlatformInfo.value.name}用户ID`);
    }

    // 根据平台选择不同的API
    if (activePlatform.value === 'douyin') {
      // 调用抖音用户API
      const response = await axios.get(
        `/api/douyin/web/handler_user_profile?sec_user_id=${userId}`,
      );

      if (response.data.code !== 200) {
        throw new Error(
          'API请求失败: ' + (response.data.status_msg || '未知错误'),
        );
      }

      // 处理API返回的用户数据
      const userData = response.data.data.user;
      accountInfo.value = convertDouyinUserData(userData, userId);
    } else {
      // 其他平台暂时使用模拟数据
      accountInfo.value = getMockAccountInfo(activePlatform.value, userId);
    }

    ElMessage.success('账号解析成功');
  } catch (error) {
    console.error('解析用户URL失败:', error);
    ElMessage.error(error.message || '解析失败，请检查URL格式是否正确');
  } finally {
    parsing.value = false;
  }
};
// 从用户数据中提取标签
const generateUserTags = (userData) => {
  const tags = [];
  
  // 根据认证状态添加标签
  if (userData.custom_verify) {
    tags.push('已认证');
  }
  
  if (userData.is_gov_media_vip) {
    tags.push('政务媒体');
  }
  
  if (userData.is_star) {
    tags.push('明星');
  }
  
  // 根据粉丝数量添加标签
  const followerCount = userData.follower_count || 0;
  if (followerCount >= 10000000) {
    tags.push('超级大V');
  } else if (followerCount >= 1000000) {
    tags.push('大V用户');
  } else if (followerCount >= 100000) {
    tags.push('知名用户');
  }
  
  // 根据作品数量添加标签
  const awemeCount = userData.aweme_count || 0;
  if (awemeCount >= 1000) {
    tags.push('高产创作者');
  } else if (awemeCount >= 100) {
    tags.push('活跃创作者');
  }
  
  // 根据获赞数添加标签
  const totalFavorited = userData.total_favorited || 0;
  if (totalFavorited >= 10000000) {
    tags.push('人气爆款');
  } else if (totalFavorited >= 1000000) {
    tags.push('高人气');
  }
  
  // 根据地理位置添加标签
  if (userData.ip_location) {
    const location = userData.ip_location.replace('IP属地：', '');
    if (location) {
      tags.push(location);
    }
  }
  
  // 根据年龄添加标签
  const userAge = userData.user_age;
  if (userAge > 0) {
    if (userAge < 25) {
      tags.push('年轻用户');
    } else if (userAge < 35) {
      tags.push('青年用户');
    } else {
      tags.push('成熟用户');
    }
  }
  
  return tags;
};
// 将抖音API返回的用户数据转换为统一格式
const convertDouyinUserData = (userData, userId) => {
  return {
    nickname: userData.nickname || '未知用户名',
    uniqueId: userId,
    sec_uid: userData.sec_uid || userId, // 添加 sec_uid
    avatar: userData.avatar_larger?.url_list?.[0] || '',
    signature: userData.signature || '',
    verified: !!userData.custom_verify || userData.verification_type > 0,
    followingCount: userData.following_count || 0,
    followerCount: userData.follower_count || 0,
    likeCount: userData.total_favorited || 0,
    awemeCount: userData.aweme_count || 0,
    
    // 添加更多字段映射
    gender: userData.gender || 0, // 0: 未知, 1: 男, 2: 女
    city: userData.city || '',
    province: userData.province || '',
    country: userData.country || '',
    district: userData.district || '',
    favoriting_count: userData.favoriting_count || 0,
    user_age: userData.user_age || -1,
    ip_location: userData.ip_location || '',
    
    // 认证相关
    custom_verify: userData.custom_verify || '',
    enterprise_verify_reason: userData.enterprise_verify_reason || '',
    verification_type: userData.verification_type || 0,
    
    // 布尔值字段
    show_favorite_list: userData.show_favorite_list || false,
    is_gov_media_vip: userData.is_gov_media_vip || false,
    is_mix_user: userData.is_mix_user || false,
    is_star: userData.is_star || false,
    is_series_user: userData.is_series_user || false,
    
    // 其他头像链接
    avatar_medium: userData.avatar_medium?.url_list?.[0] || '',
    avatar_thumb: userData.avatar_thumb?.url_list?.[0] || '',
    
    tags: generateUserTags(userData),
    location: userData.ip_location || '',
    userAge: userData.user_age || -1,
  };
};

// 从用户数据中提取标签
const analyzeAccount = async () => {
  if (!accountInfo.value) return;
  
  const loading = ElLoading.service({
    lock: true,
    text: '正在添加账号到分析系统...',
    background: 'rgba(0, 0, 0, 0.7)',
  });
  
  try {
    // 提取完整的用户数据
    const userData = {
      platform: activePlatform.value,
      platform_user_id: accountInfo.value.uniqueId,
      sec_uid: accountInfo.value.sec_uid || accountInfo.value.uniqueId,
      nickname: accountInfo.value.nickname,
      signature: accountInfo.value.signature || '',
      
      // 头像相关
      avatar: accountInfo.value.avatar || '',
      avatar_medium: accountInfo.value.avatar_medium || accountInfo.value.avatar || '',
      
      // 基础信息
      gender: accountInfo.value.gender || 0,
      city: accountInfo.value.city || '',
      province: accountInfo.value.province || '',
      country: accountInfo.value.country || '',
      district: accountInfo.value.district || '',
      
      // 统计数据
      aweme_count: accountInfo.value.awemeCount || accountInfo.value.aweme_count || 0,
      follower_count: accountInfo.value.followerCount || accountInfo.value.follower_count || 0,
      following_count: accountInfo.value.followingCount || accountInfo.value.following_count || 0,
      total_favorited: accountInfo.value.likeCount || accountInfo.value.total_favorited || 0,
      favoriting_count: accountInfo.value.favoriting_count || 0,
      
      // 其他信息
      user_age: accountInfo.value.userAge || accountInfo.value.user_age || -1,
      ip_location: accountInfo.value.ip_location || accountInfo.value.location || '',
      
      // 认证信息
      custom_verify: accountInfo.value.custom_verify || '',
      enterprise_verify_reason: accountInfo.value.enterprise_verify_reason || '',
      verification_type: accountInfo.value.verification_type || 0,
      
      // 布尔值
      show_favorite_list: accountInfo.value.show_favorite_list || false,
      is_gov_media_vip: accountInfo.value.is_gov_media_vip || false,
      is_mix_user: accountInfo.value.is_mix_user || false,
      is_star: accountInfo.value.is_star || false,
      is_series_user: accountInfo.value.is_series_user || false,
    };
    
    console.log('提交的完整用户数据:', userData);
    
    // 发送请求到后端API添加用户
    const response = await axios.post('/api/account/add', userData);
    
    if (response.data.code === 200) {
      // 获取返回的用户ID
      const userId = response.data.data.user_id;
      const taskId = response.data.data.task_id;
      const isExisting = response.data.data.existing;
      
      if (isExisting) {
        // 账号已存在
        ElMessage.warning(response.data.message);
      } else {
        // 新添加的账号，获取视频数据
        loading.setText('正在获取账号视频数据...');
        
        try {
          const videoResponse = await axios.post(`/api/account/${userId}/fetch_videos`, {
            max_videos: 30 // 最多获取30个视频
          });
          
          if (videoResponse.data.code === 200) {
            ElMessage.success(`账号添加成功，已获取${videoResponse.data.data.videos_added}个视频`);
          } else {
            console.warn('获取视频数据部分失败:', videoResponse.data.message);
            ElMessage.warning('账号添加成功，但视频数据获取不完整');
          }
        } catch (videoError) {
          console.error('获取视频数据失败:', videoError);
          ElMessage.warning('账号添加成功，但视频数据获取失败');
        }
      }
      
      // 延迟1秒后跳转到分析任务页面
      setTimeout(() => {
        router.push({
          path: '/main/analysis-tasks',
          query: {
            highlight: taskId, // 高亮显示新添加的任务
            new: !isExisting ? 'true' : 'false'
          }
        });
      }, 1500);
      
    } else {
      throw new Error(response.data.message || '添加账号失败');
    }
  } catch (error) {
    console.error('添加账号失败:', error);
    ElMessage.error(error.message || '操作失败，请稍后重试');
  } finally {
    loading.close();
  }
};

const router = useRouter();
// 查看内容列表
const viewContentList = () => {
  if (!accountInfo.value) return;
  
  // 构建内容列表页面的URL，将用户ID作为参数传递
  const contentListUrl = `/main/user-content?platform=${activePlatform.value}&userId=${accountInfo.value.uniqueId}`;
  
  // 使用 router 进行页面导航
  router.push(contentListUrl);
};

// 格式化数字
const formatNumber = (num: number) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (num >= 10000) {
    return (num / 10000).toFixed(1) + 'W';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num;
};

// 模拟获取账号信息 (实际应用中应由API提供)
const getMockAccountInfo = (platform: string, userId: string) => {
  // 根据平台返回模拟数据
  const mockData = {
    douyin: {
      nickname: '抖音测试账号',
      uniqueId: userId,
      avatar:
        'https://p16.tiktokcdn.com/musically-maliva-obj/1663771803664390~c5_720x720.jpeg',
      signature: '这是一个抖音测试账号的简介，展示各种短视频创作。',
      verified: true,
      followingCount: 235,
      followerCount: 1250000,
      likeCount: 9800000,
      tags: ['创作者', '舞蹈', '生活'],
    },
    kuaishou: {
      nickname: '快手测试用户',
      uniqueId: userId,
      avatar:
        'https://tx2.a.kwimgs.com/uhead/AB/2022/07/24/14/BMjAyMjA3MjQxNDU5MDlfMTEwNzE4MTk5Ml8yX2hkNzY5X2E5Ng==_s.jpg',
      signature: '快手账号简介测试，分享精彩生活瞬间。',
      verified: false,
      followingCount: 127,
      followerCount: 53600,
      likeCount: 325000,
      tags: ['搞笑', '日常', '美食'],
    },
    tiktok: {
      nickname: 'TikTok User',
      uniqueId: userId,
      avatar:
        'https://p16.tiktokcdn.com/musically-maliva-obj/1663771803664390~c5_720x720.jpeg',
      signature:
        'TikTok account sharing daily life and travel videos around the world.',
      verified: true,
      followingCount: 342,
      followerCount: 2650000,
      likeCount: 15700000,
      tags: ['Travel', 'Creator', 'Lifestyle'],
    },
    bilibili: {
      nickname: 'B站用户',
      uniqueId: userId,
      avatar:
        'https://i2.hdslb.com/bfs/face/d79637d472c90f45b2476871a637bd4051058ea2.jpg',
      signature: 'B站UP主，专注动漫、游戏视频创作。',
      verified: true,
      followingCount: 182,
      followerCount: 375000,
      likeCount: 4250000,
      tags: ['UP主', '游戏', '动漫'],
    },
  };

  return mockData[platform as keyof typeof mockData] || mockData.douyin;
};
</script>

<template>
  <div class="account-analysis-container">
    <el-card class="main-card">
      <!-- 平台切换标签 -->
      <div class="platform-tabs">
        <el-tabs
          v-model="activePlatform"
          @tab-change="handlePlatformChange"
          type="border-card"
        >
          <el-tab-pane
            v-for="platform in platforms"
            :key="platform.id"
            :label="platform.name"
            :name="platform.id"
          >
            <template #label>
              <div class="tab-label">
                <el-image :src="platform.icon" class="platform-icon"></el-image>
                <span>{{ platform.name }}</span>
              </div>
            </template>
          </el-tab-pane>
        </el-tabs>
      </div>

      <!-- URL输入区域 -->
      <div class="url-input-section">
        <h3>添加{{ activePlatformInfo.name }}账号</h3>
        <p class="subtitle">
          输入{{ activePlatformInfo.name }}用户页面URL添加到分析系统
        </p>

        <el-input
          v-model="userUrl"
          :placeholder="activePlatformInfo.placeholder"
          class="url-input"
          clearable
          :disabled="parsing"
          @keyup.enter="parseUserUrl"
        >
          <template #prefix>
            <el-icon><LinkIcon /></el-icon>
          </template>
          <template #append>
            <el-button type="primary" @click="parseUserUrl" :loading="parsing">
              {{ parsing ? '解析中...' : '解析账号' }}
            </el-button>
          </template>
        </el-input>

        <!-- 添加更明确的错误提示区域 -->
        <div v-if="parseError" class="parse-error">
          <el-alert :title="parseError" type="error" show-icon />
        </div>

        <div class="tips">
          <el-alert
            :title="`${activePlatformInfo.name}用户链接格式示例: ${activePlatformInfo.example}`"
            type="info"
            show-icon
            :closable="false"
          />
        </div>
      </div>

      <!-- 账户预览区域 -->
      <div v-if="accountInfo" class="account-preview-section">
        <el-divider content-position="center">账号预览</el-divider>

        <el-row :gutter="20">
          <el-col :xs="24" :sm="8" :md="6" class="account-avatar-col">
            <div class="account-avatar-container">
              <el-avatar
                :size="100"
                :src="accountInfo.avatar"
                class="account-avatar"
              />
              <div v-if="accountInfo.verified" class="verified-badge">
                <el-tooltip content="已认证账号">
                  <el-icon><CircleCheckFilled /></el-icon>
                </el-tooltip>
              </div>
            </div>
          </el-col>

          <el-col :xs="24" :sm="16" :md="18" class="account-info-col">
            <div class="account-header">
              <h2 class="account-name">{{ accountInfo.nickname }}</h2>
              <div class="account-id">ID: {{ accountInfo.uniqueId }}</div>
            </div>

            <div class="account-stats">
              <div class="stat-item">
                <div class="stat-value">
                  {{ formatNumber(accountInfo.followingCount) }}
                </div>
                <div class="stat-label">关注</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">
                  {{ formatNumber(accountInfo.followerCount) }}
                </div>
                <div class="stat-label">粉丝</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">
                  {{ formatNumber(accountInfo.likeCount) }}
                </div>
                <div class="stat-label">{{ activePlatformInfo.likeLabel }}</div>
              </div>
              <!-- 添加作品数统计 -->
              <div
                class="stat-item"
                v-if="accountInfo.awemeCount !== undefined"
              >
                <div class="stat-value">
                  {{ formatNumber(accountInfo.awemeCount) }}
                </div>
                <div class="stat-label">作品</div>
              </div>
            </div>

            <!-- 添加地理位置信息 -->
            <div v-if="accountInfo.location" class="account-location">
              {{ accountInfo.location }}
            </div>

            <div v-if="accountInfo.signature" class="account-bio">
              {{ accountInfo.signature }}
            </div>

            <div class="account-tags">
              <el-tag
                v-for="(tag, index) in accountInfo.tags"
                :key="index"
                size="small"
                class="account-tag"
              >
                {{ tag }}
              </el-tag>
            </div>
          </el-col>
        </el-row>

        <!-- 账号内容分析按钮 -->
        <div class="action-buttons">
          <el-button type="primary" @click="analyzeAccount"
            >添加并开始分析</el-button
          >
          <el-button @click="viewContentList">查看内容列表</el-button>
        </div>
      </div>

      <!-- 未解析数据时的空状态 -->
      <el-empty
        v-else
        description="请输入用户URL进行解析"
        class="account-empty-state"
      >
        <template #image>
          <el-icon style="font-size: 60px"><UserFilled /></el-icon>
        </template>
      </el-empty>
    </el-card>
  </div>
</template>

<style scoped>
.account-analysis-container {
  max-width: 1000px;
  margin: 20px auto;
  padding: 0 20px;
}

.main-card {
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 平台标签样式 */
.platform-tabs {
  margin-bottom: 24px;
}

.platform-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}

.tab-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.platform-icon {
  width: 20px;
  height: 20px;
}

/* URL输入区域样式 */
.url-input-section {
  margin-bottom: 24px;
}

.url-input-section h3 {
  font-size: 20px;
  margin-bottom: 8px;
  color: #303133;
}

.subtitle {
  color: #606266;
  margin-bottom: 16px;
  font-size: 14px;
}

.url-input {
  margin-bottom: 12px;
}

.tips {
  margin-top: 12px;
}

/* 账户预览区样式 */
.account-preview-section {
  padding: 16px 0;
}

.account-avatar-container {
  display: flex;
  justify-content: center;
  position: relative;
  margin-bottom: 16px;
}

.account-avatar {
  border: 3px solid #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.verified-badge {
  position: absolute;
  bottom: 0;
  right: 0;
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

.account-header {
  margin-bottom: 16px;
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
  gap: 24px;
  margin-bottom: 16px;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
}

.stat-label {
  font-size: 13px;
  color: #909399;
}

.account-bio {
  margin-bottom: 16px;
  line-height: 1.5;
  color: #606266;
}

.account-tags {
  margin-bottom: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.action-buttons {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 16px;
}

.account-empty-state {
  padding: 40px 0;
}

@media (max-width: 768px) {
  .account-avatar-col,
  .account-info-col {
    text-align: center;
  }

  .account-stats {
    justify-content: center;
  }
}
</style>
