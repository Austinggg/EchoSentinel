<script lang="ts" setup>
import { ref, onMounted, computed, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import {
  ElAlert,
  ElAvatar,
  ElButton,
  ElCard,
  ElEmpty,
  ElIcon,
  ElImage,
  ElLoading,
  ElMessage,
  ElPagination,
  ElTable,
  ElTableColumn,
  ElTag,
  ElTooltip,
  ElInput,
  ElSelect,
  ElOption,
  ElMessageBox,
} from 'element-plus';
import {
  ArrowLeft,
  VideoPlay,
  Picture,
  Share,
  Star,
  Search,
  SortDown,
  Refresh,
  Timer,
  WarningFilled,
  InfoFilled,
  CircleCheckFilled,
  CircleCloseFilled,
  Delete,
} from '@element-plus/icons-vue';

const route = useRoute();
const router = useRouter();

// 获取路由参数
const platform = computed(() => route.query.platform as string);
const userId = computed(() => route.query.userId as string);

// 状态变量
const loading = ref(false);
const error = ref('');
const contentList = ref([]);
const totalItems = ref(0);
const accountInfo = ref(null);
const fetchingVideos = ref(false);

// 表格多选相关
const multipleSelection = ref([]);
const multipleTableRef = ref();

// 分页相关
const currentPage = ref(1);
const pageSize = ref(10);

// 搜索和过滤
const searchText = ref('');
const sortField = ref('create_time');
const sortOrder = ref('desc');

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

// 返回按钮处理函数
const goBack = () => {
  router.go(-1);
};

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

// 格式化日期
const formatDate = (timestamp) => {
  if (!timestamp) return '-';
  const date = new Date(typeof timestamp === 'number' ? timestamp * 1000 : timestamp);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
};

// 加载用户信息
const loadUserInfo = async () => {
  try {
    loading.value = true;
    
    // 尝试从数据库获取用户信息
    const dbResponse = await axios.get(`/api/account/by-secuid/${userId.value}`);
    
    if (dbResponse.data.code === 200 && dbResponse.data.data) {
      // 数据库中已有用户数据
      accountInfo.value = dbResponse.data.data;
      console.log('从数据库加载用户信息成功:', accountInfo.value);
      
      // 有了用户ID，加载其视频列表
      loadVideosFromDB();
      return;
    }
    
    // 如果数据库没有，尝试从抖音API获取
    if (platform.value === 'douyin') {
      const response = await axios.get(`/api/douyin/web/handler_user_profile?sec_user_id=${userId.value}`);
      
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

// 从数据库加载视频
const loadVideosFromDB = async () => {
  if (!accountInfo.value?.id) {
    console.log('无法加载视频：缺少用户ID');
    return;
  }
  
  try {
    loading.value = true;
    console.log('正在从数据库加载视频列表...');
    
    // 构建API请求参数
    const params = {
      page: currentPage.value,
      per_page: pageSize.value,
      sort_by: sortField.value,
      sort_order: sortOrder.value,
      search: searchText.value || undefined
    };
    
    // 从后端API获取视频列表
    const response = await axios.get(`/api/account/${accountInfo.value.id}/videos`, { params });
    
    if (response.data.code === 200) {
      contentList.value = response.data.data.videos || [];
      totalItems.value = response.data.data.total || 0;
      console.log('视频列表加载成功，共', totalItems.value, '条记录');
    } else {
      throw new Error(response.data.message || '获取视频列表失败');
    }
  } catch (err) {
    console.error('加载视频列表失败:', err);
    error.value = err.message || '获取视频列表失败';
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
};

// 获取最新视频
const fetchLatestVideos = async () => {
  if (!accountInfo.value?.id) {
    ElMessage.warning('无法获取视频：缺少用户ID');
    return;
  }
  
  try {
    fetchingVideos.value = true;
    
    // 显示加载中提示
    const loading = ElLoading.service({
      lock: true,
      text: '正在获取最新视频...',
      background: 'rgba(0, 0, 0, 0.7)',
    });
    
    // 调用后端API获取最新视频
    const response = await axios.post(`/api/account/${accountInfo.value.id}/fetch_videos`, {
      max_videos: 30 // 最多获取30个视频
    });
    
    if (response.data.code === 200) {
      const videosAdded = response.data.data.videos_added;
      ElMessage.success(`成功获取${videosAdded}个视频`);
      
      // 重新加载视频列表
      await loadVideosFromDB();
    } else {
      throw new Error(response.data.message || '获取视频失败');
    }
  } catch (err) {
    console.error('获取最新视频失败:', err);
    ElMessage.error(err.message || '获取视频失败，请稍后重试');
  } finally {
    fetchingVideos.value = false;
    if (ElLoading.service) {
      ElLoading.service().close();
    }
  }
};

// 处理搜索
const handleSearch = () => {
  currentPage.value = 1; // 重置到第一页
  loadVideosFromDB();
};

// 处理排序变化
const handleSortChange = (column) => {
  if (column.prop) {
    sortField.value = column.prop;
    sortOrder.value = column.order === 'ascending' ? 'asc' : 'desc';
    loadVideosFromDB();
  }
};

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val;
  loadVideosFromDB();
};

// 处理每页大小变化
const handleSizeChange = (val) => {
  pageSize.value = val;
  currentPage.value = 1; // 重置到第一页
  loadVideosFromDB();
};

// 处理表格行点击
const handleRowClick = (row) => {
  // 这里可以实现点击行查看视频详情
  console.log('查看视频详情:', row);
  ElMessage.info(`查看视频: ${row.aweme_id}`);
};

// 处理多选变化
const handleSelectionChange = (val) => {
  multipleSelection.value = val;
  console.log('已选择视频:', multipleSelection.value.length);
};

// 清除选择
const clearSelection = () => {
  if (multipleTableRef.value) {
    multipleTableRef.value.clearSelection();
  }
};

// 根据分享URL生成短链接
const getShortShareUrl = (url) => {
  if (!url) return '-';
  try {
    const urlObj = new URL(url);
    return urlObj.hostname + '/...';
  } catch (e) {
    return url.substring(0, 20) + '...';
  }
};

// 监听搜索关键词变化
watch(searchText, (value) => {
  if (!value) {
    handleSearch(); // 当搜索框清空时，自动重新加载
  }
});

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
  <div class="user-content-container">
    <!-- 返回按钮和标题 -->
    <div class="page-header">
      <el-button type="text" @click="goBack">
        <el-icon><ArrowLeft /></el-icon> 返回
      </el-button>
      <h2 class="page-title">{{ platformName }}用户内容</h2>
      
      <!-- 添加搜索框 -->
      <div class="search-box">
        <el-input
          v-model="searchText"
          placeholder="搜索标题或标签"
          class="search-input"
          clearable
          @keyup.enter="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        <el-button type="primary" @click="handleSearch">搜索</el-button>
      </div>
    </div>

    <!-- 错误提示 -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      show-icon
      :closable="false"
      class="error-alert"
    />

    <!-- 用户信息卡片 -->
    <el-card v-if="accountInfo" class="user-card">
      <div class="user-info-container">
        <div class="user-info">
          <el-avatar :size="60" :src="accountInfo.avatar" />
          <div class="user-details">
            <h3>{{ accountInfo.nickname }}</h3>
            <div class="user-stats">
              <span>{{ formatNumber(accountInfo.following_count) }} 关注</span>
              <span class="divider">|</span>
              <span>{{ formatNumber(accountInfo.follower_count) }} 粉丝</span>
              <span class="divider">|</span>
              <span>{{ formatNumber(accountInfo.aweme_count) }} 作品</span>
            </div>
            <div class="user-signature" v-if="accountInfo.signature">
              {{ accountInfo.signature }}
            </div>
          </div>
        </div>
        
        <!-- 添加刷新按钮 -->
        <div class="user-actions">
          <el-button 
            type="primary"
            :loading="fetchingVideos"
            @click="fetchLatestVideos"
            size="small"
          >
            <el-icon><Refresh /></el-icon>
            获取最新视频
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 视频列表表格 -->
    <el-card class="content-table-card">
      <div class="table-header">
        <div class="table-title">
          <h3>发布视频列表</h3>
          <span class="video-count">共 {{ totalItems }} 条内容</span>
        </div>
        
        <!-- 表格操作区域 -->
        <div class="table-operations" v-if="multipleSelection.length > 0">
          <span class="selected-count">已选择 {{ multipleSelection.length }} 项</span>
          <el-button size="small" @click="clearSelection">清除选择</el-button>
        </div>
      </div>
      
      <!-- 数据表格 -->
      <el-table
        ref="multipleTableRef"
        :data="contentList"
        border
        stripe
        style="width: 100%"
        v-loading="loading"
        @selection-change="handleSelectionChange"
        @sort-change="handleSortChange"
        @row-click="handleRowClick"
      >
        <el-table-column type="selection" width="55" />
        
        <!-- 封面列 -->
        <el-table-column label="封面" width="100" align="center">
          <template #default="{ row }">
            <el-image
              :src="row.cover_url"
              fit="cover"
              style="width: 70px; height: 90px; border-radius: 4px"
              :preview-src-list="[row.cover_url]"
            >
              <template #error>
                <div class="image-placeholder">
                  <el-icon><Picture /></el-icon>
                </div>
              </template>
            </el-image>
          </template>
        </el-table-column>
        
        <!-- 标题/描述列 -->
        <el-table-column label="标题" prop="desc" min-width="240">
          <template #default="{ row }">
            <div class="video-title-cell">
              <el-tooltip :content="row.desc" placement="top" effect="light">
                <div class="multiline-text video-title">{{ row.desc || '无标题' }}</div>
              </el-tooltip>
              
              <!-- 标签 -->
              <div class="video-tags" v-if="row.tags && row.tags.length > 0">
                <el-tag
                  v-for="(tag, index) in row.tags.slice(0, 3)"
                  :key="index"
                  size="small"
                  effect="plain"
                  class="video-tag"
                >
                  {{ tag }}
                </el-tag>
                <el-tag size="small" effect="plain" v-if="row.tags.length > 3">
                  +{{ row.tags.length - 3 }}
                </el-tag>
              </div>
            </div>
          </template>
        </el-table-column>
        
        <!-- 发布时间列 -->
        <el-table-column 
          label="发布时间" 
          prop="create_time" 
          width="120" 
          sortable="custom"
        >
          <template #default="{ row }">
            <div class="time-cell">
              <el-icon><Timer /></el-icon>
              <span>{{ formatDate(row.create_time) }}</span>
            </div>
          </template>
        </el-table-column>
        
        <!-- 点赞数列 -->
        <el-table-column 
          label="点赞数" 
          prop="digg_count" 
          width="100" 
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><Star /></el-icon>
              <span>{{ formatNumber(row.statistics?.digg_count || row.digg_count) }}</span>
            </div>
          </template>
        </el-table-column>
        
        <!-- 评论数列 -->
        <el-table-column 
          label="评论数" 
          prop="comment_count" 
          width="100" 
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><Chat /></el-icon>
              <span>{{ formatNumber(row.statistics?.comment_count || row.comment_count) }}</span>
            </div>
          </template>
        </el-table-column>
        
        <!-- 分享数列 -->
        <el-table-column 
          label="分享数" 
          prop="share_count" 
          width="100" 
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><Share /></el-icon>
              <span>{{ formatNumber(row.statistics?.share_count || row.share_count) }}</span>
            </div>
          </template>
        </el-table-column>
        
        <!-- 分享链接列 -->
        <el-table-column label="分享链接" width="150">
          <template #default="{ row }">
            <el-tooltip :content="row.share_url" placement="top" effect="light">
              <a :href="row.share_url" target="_blank" class="share-link">
                {{ getShortShareUrl(row.share_url) }}
              </a>
            </el-tooltip>
          </template>
        </el-table-column>
        
        <!-- 操作列 -->
        <el-table-column label="操作" width="150" fixed="right" align="center">
          <template #default="{ row }">
            <el-button 
              size="small" 
              type="primary" 
              link
              @click.stop="handleRowClick(row)"
            >
              查看详情
            </el-button>
            <el-button 
              size="small" 
              type="primary" 
              link
              @click.stop="$router.push(`/main/content-analysis?id=${row.aweme_id}`)"
            >
              内容分析
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <!-- 分页组件 -->
      <div class="pagination-container" v-if="totalItems > 0">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="totalItems"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
      
      <!-- 空状态 -->
      <el-empty 
        v-if="contentList.length === 0 && !loading" 
        description="暂无视频数据"
      >
        <template #default>
          <div class="empty-action">
            <p>此账号尚未收集视频数据</p>
            <el-button type="primary" @click="fetchLatestVideos">
              <el-icon><Refresh /></el-icon>
              获取视频数据
            </el-button>
          </div>
        </template>
      </el-empty>
    </el-card>
  </div>
</template>

<style scoped>
.user-content-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}

.page-title {
  margin: 0 0 0 10px;
  font-size: 24px;
  color: #303133;
}

.search-box {
  display: flex;
  gap: 10px;
}

.search-input {
  width: 220px;
}

.error-alert {
  margin-bottom: 20px;
}

.user-card {
  margin-bottom: 20px;
}

.user-info-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.user-info {
  display: flex;
  align-items: center;
}

.user-details {
  margin-left: 15px;
}

.user-details h3 {
  margin: 0 0 5px 0;
  font-size: 18px;
}

.user-stats {
  color: #606266;
  font-size: 14px;
}

.user-signature {
  margin-top: 8px;
  color: #606266;
  font-size: 13px;
  max-width: 400px;
}

.divider {
  margin: 0 8px;
  color: #dcdfe6;
}

.content-table-card {
  margin-bottom: 20px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.table-title {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.table-title h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.video-count {
  color: #909399;
  font-size: 14px;
}

.table-operations {
  display: flex;
  align-items: center;
  gap: 10px;
}

.selected-count {
  background-color: #f0f9eb;
  color: #67c23a;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 13px;
}

.user-actions {
  display: flex;
  gap: 10px;
}

/* 视频标题与标签样式 */
.video-title-cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.multiline-text {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.5;
}

.video-title {
  font-weight: 500;
  color: #303133;
}

.video-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.video-tag {
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 数据单元格样式 */
.stat-cell, .time-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
}

.time-cell {
  justify-content: flex-start;
}

.image-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 70px;
  height: 90px;
  background-color: #f5f7fa;
  color: #909399;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.share-link {
  color: #409eff;
  text-decoration: none;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty-action {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #909399;
}

/* 表格行点击效果 */
:deep(.el-table__row) {
  cursor: pointer;
}

:deep(.el-table__row:hover) {
  background-color: rgba(64, 158, 255, 0.08) !important;
}
</style>