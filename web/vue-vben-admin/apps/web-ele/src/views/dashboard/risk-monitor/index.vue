<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { ElCard, ElTable, ElTableColumn, ElTag, ElProgress, ElAvatar, ElTooltip, ElPopover, ElEmpty, ElSkeleton, ElSkeletonItem } from 'element-plus';
import {
  Warning, VideoPlay, User, InfoFilled, Calendar, Link
} from '@element-plus/icons-vue';

const router = useRouter();
const loading = ref(true);
const monitorData = ref({
  high_risk_videos: [],
  falsehoods: [],
  high_risk_users: [],
  digital_human_users: []
});

// 加载风险监控数据
const loadRiskMonitorData = async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/analytics/risk-monitor');
    if (response.data && (response.data.code === 200 || response.data.code === 0)) {
      monitorData.value = response.data.data;
    } else {
      console.error('无法获取风险监控数据:', response.data?.message || '未知错误');
    }
  } catch (error) {
    console.error('获取风险监控数据失败:', error);
  } finally {
    loading.value = false;
  }
};

// 格式化日期时间
const formatDateTime = (dateStr) => {
  if (!dateStr) return '未知时间';
  const date = new Date(dateStr);
  return date.toLocaleDateString('zh-CN', { 
    year: 'numeric', 
    month: '2-digit', 
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  }).replace(/\//g, '-');
};

// 格式化平台名称
const formatPlatform = (platform) => {
  if (!platform) return '上传';
  switch(platform.toLowerCase()) {
    case 'douyin': return '抖音';
    case 'tiktok': return 'TikTok';
    case 'bilibili': return 'B站';
    case 'upload': return '上传';
    default: return platform;
  }
};

// 获取数字人概率颜色
const getDigitalHumanColor = (probability) => {
  if (probability >= 0.7) return '#F56C6C'; // 高风险红色
  if (probability >= 0.4) return '#E6A23C'; // 中风险黄色
  return '#67C23A'; // 低风险绿色
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
</script>

<template>
  <div class="p-4 risk-monitor-container">
    <h2 class="text-xl font-bold mb-5 flex items-center">
      <el-icon class="mr-2"><Warning /></el-icon>
      风险监控中心
    </h2>
    
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="3" animated />
      <el-skeleton :rows="3" animated class="mt-6" />
    </div>
    
    <div v-else>
      <!-- 第一行：高风险视频和虚假信息核查 -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 mb-4">
        <!-- 高风险视频卡片 -->
        <el-card class="high-risk-videos">
          <template #header>
            <div class="card-header">
              <span class="flex items-center">
                <el-icon class="mr-2"><VideoPlay /></el-icon>
                最近高风险视频
              </span>
            </div>
          </template>
          
          <div v-if="monitorData.high_risk_videos && monitorData.high_risk_videos.length > 0">
            <el-table :data="monitorData.high_risk_videos" style="width: 100%" size="small" max-height="400">
              <el-table-column label="平台" prop="platform" width="80">
                <template #default="{ row }">
                  <el-tag>{{ formatPlatform(row.platform) }}</el-tag>
                </template>
              </el-table-column>
              
              <el-table-column label="视频标题" prop="filename" min-width="200">
                <template #default="{ row }">
                  <el-tooltip :content="row.filename" placement="top" :show-after="500">
                    <div class="truncate" style="max-width: 200px;">{{ row.filename }}</div>
                  </el-tooltip>
                </template>
              </el-table-column>
              
              <el-table-column label="数字人概率" width="120">
                <template #default="{ row }">
                  <el-progress
                    :percentage="(row.digital_human_probability || 0) * 100"
                    :color="getDigitalHumanColor(row.digital_human_probability)"
                    :stroke-width="8"
                    :show-text="true"
                    :format="() => formatProbability(row.digital_human_probability)"
                  />
                </template>
              </el-table-column>
              
              <el-table-column label="操作" width="100" fixed="right">
                <template #default="{ row }">
                  <el-button 
                    type="primary" 
                    link 
                    @click="viewVideoDetail(row.id)"
                  >
                    查看详情
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="暂无高风险视频数据" />
        </el-card>
        
        <!-- 虚假信息核查卡片 -->
        <el-card class="falsehood-checks">
          <template #header>
            <div class="card-header">
              <span class="flex items-center">
                <el-icon class="mr-2"><InfoFilled /></el-icon>
                最近事实核查不实信息
              </span>
            </div>
          </template>
          
          <div v-if="monitorData.falsehoods && monitorData.falsehoods.length > 0">
            <el-table :data="monitorData.falsehoods" style="width: 100%" size="small" max-height="400">
              <el-table-column label="不实断言" prop="claim" min-width="200">
                <template #default="{ row }">
                  <el-popover
                    placement="right"
                    :width="300"
                    trigger="hover"
                    :content="row.conclusion || '无详细解释'"
                  >
                    <template #reference>
                      <div class="truncate" style="max-width: 200px; cursor: pointer;">
                        {{ row.claim }}
                      </div>
                    </template>
                  </el-popover>
                </template>
              </el-table-column>
              
              <el-table-column label="来源视频" prop="video_name" width="150">
                <template #default="{ row }">
                  <el-tooltip :content="row.video_name" placement="top" :show-after="500">
                    <div class="truncate" style="max-width: 150px;">{{ row.video_name }}</div>
                  </el-tooltip>
                </template>
              </el-table-column>
              
              <el-table-column label="核查时间" prop="check_time" width="120">
                <template #default="{ row }">
                  <el-tooltip :content="formatDateTime(row.check_time)" placement="top">
                    <span>{{ formatDateTime(row.check_time).split(' ')[0] }}</span>
                  </el-tooltip>
                </template>
              </el-table-column>
              
              <el-table-column label="操作" width="100" fixed="right">
                <template #default="{ row }">
                  <el-button 
                    type="primary" 
                    link 
                    @click="viewVideoDetail(row.video_id)"
                  >
                    查看详情
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="暂无事实核查不实信息" />
        </el-card>
      </div>
      
      <!-- 第二行：用户风险排行和疑似数字人用户 -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <!-- 高风险用户排行卡片 -->
        <el-card class="high-risk-users">
          <template #header>
            <div class="card-header">
              <span class="flex items-center">
                <el-icon class="mr-2"><User /></el-icon>
                高风险用户排行
              </span>
            </div>
          </template>
          
          <div v-if="monitorData.high_risk_users && monitorData.high_risk_users.length > 0">
            <el-table :data="monitorData.high_risk_users" style="width: 100%" size="small" max-height="400">
              <el-table-column label="用户" min-width="180">
                <template #default="{ row }">
                  <div class="flex items-center">
                    <el-avatar :size="30" :src="row.avatar">
                      {{ row.nickname?.charAt(0) || '?' }}
                    </el-avatar>
                    <div class="ml-2 truncate" style="max-width: 120px;">
                      <el-tooltip :content="row.nickname" placement="top" :show-after="500">
                        <span>{{ row.nickname }}</span>
                      </el-tooltip>
                    </div>
                  </div>
                </template>
              </el-table-column>
              
              <el-table-column label="平台" prop="platform" width="80">
                <template #default="{ row }">
                  <el-tag>{{ formatPlatform(row.platform) }}</el-tag>
                </template>
              </el-table-column>
              
              <el-table-column label="数字人概率" width="150">
                <template #default="{ row }">
                  <el-progress
                    :percentage="(row.digital_human_probability || 0) * 100"
                    :color="getDigitalHumanColor(row.digital_human_probability)"
                    :stroke-width="8"
                    :show-text="true"
                    :format="() => formatProbability(row.digital_human_probability)"
                  />
                </template>
              </el-table-column>
              
              <el-table-column label="操作" width="100" fixed="right">
                <template #default="{ row }">
                  <el-button 
                    type="primary" 
                    link 
                    @click="viewUserProfile(row.platform, row.platform_user_id)"
                  >
                    查看画像
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="暂无高风险用户数据" />
        </el-card>
        
        <!-- 疑似数字人用户卡片 -->
        <el-card class="digital-human-users">
          <template #header>
            <div class="card-header">
              <span class="flex items-center">
                <el-icon class="mr-2"><Link /></el-icon>
                最近识别数字人用户
              </span>
            </div>
          </template>
          
          <div v-if="monitorData.digital_human_users && monitorData.digital_human_users.length > 0">
            <el-table :data="monitorData.digital_human_users" style="width: 100%" size="small" max-height="400">
              <el-table-column label="用户" min-width="180">
                <template #default="{ row }">
                  <div class="flex items-center">
                    <el-avatar :size="30" :src="row.avatar">
                      {{ row.nickname?.charAt(0) || '?' }}
                    </el-avatar>
                    <div class="ml-2 truncate" style="max-width: 120px;">
                      <el-tooltip :content="row.nickname" placement="top" :show-after="500">
                        <span>{{ row.nickname }}</span>
                      </el-tooltip>
                    </div>
                  </div>
                </template>
              </el-table-column>
              
              <el-table-column label="平台" prop="platform" width="80">
                <template #default="{ row }">
                  <el-tag>{{ formatPlatform(row.platform) }}</el-tag>
                </template>
              </el-table-column>
              
              <el-table-column label="数字人概率" width="150">
                <template #default="{ row }">
                  <el-progress
                    :percentage="(row.digital_human_probability || 0) * 100"
                    :color="getDigitalHumanColor(row.digital_human_probability)"
                    :stroke-width="8"
                    :show-text="true"
                    :format="() => formatProbability(row.digital_human_probability)"
                  />
                </template>
              </el-table-column>
              
              <el-table-column label="识别时间" prop="completed_at" width="120">
                <template #default="{ row }">
                  <el-tooltip :content="formatDateTime(row.completed_at)" placement="top">
                    <span>{{ formatDateTime(row.completed_at).split(' ')[0] }}</span>
                  </el-tooltip>
                </template>
              </el-table-column>
              
              <el-table-column label="操作" width="100" fixed="right">
                <template #default="{ row }">
                  <el-button 
                    type="primary" 
                    link 
                    @click="viewUserProfile(row.platform, row.platform_user_id)"
                  >
                    查看画像
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="暂无数字人用户数据" />
        </el-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.risk-monitor-container {
  min-height: 100%;
  background-color: var(--el-bg-color);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
  font-size: 16px;
}

.loading-container {
  margin-top: 20px;
}

.truncate {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>