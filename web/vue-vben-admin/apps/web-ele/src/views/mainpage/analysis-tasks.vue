<script lang="ts" setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';

import {
  Calendar,
  CircleCheck,
  CircleClose,
  Loading,
  Refresh,
  Search,
  View,
  User,          
  Document      
} from '@element-plus/icons-vue';
import axios from 'axios';
import {
  ElAlert,
  ElAvatar,
  ElButton,
  ElCard,
  ElDatePicker,
  ElEmpty,
  ElIcon,
  ElInput,
  ElMessage,
  ElOption,
  ElPagination,
  ElProgress,
  ElSelect,
  ElSlider,
  ElTable,
  ElTableColumn,
  ElTag,
  ElTooltip,
} from 'element-plus';

const router = useRouter();

// 状态变量
const loading = ref(false);
const error = ref('');
const taskList = ref([]);
const totalItems = ref(0);

// 分页相关
const currentPage = ref(1);
const pageSize = ref(10);

// 搜索和过滤
const searchText = ref('');
const platformFilter = ref('');
const statusFilter = ref('');
const dateRange = ref([]);
const sortField = ref('created_at');
const sortOrder = ref('desc');

const probabilityRange = ref([0, 1]); // 默认范围0-1

// 加载分析任务列表
const loadTasks = async () => {
  try {
    loading.value = true;

    // 构建查询参数
    const params = {
      page: currentPage.value,
      per_page: pageSize.value,
      search: searchText.value || undefined,
      platform: platformFilter.value || undefined,
      status: statusFilter.value || undefined,
      min_probability: probabilityRange.value[0],
      max_probability: probabilityRange.value[1],
      sort_by: sortField.value,
      sort_order: sortOrder.value,
    };

    // 如果有日期范围，添加到参数
    if (dateRange.value && dateRange.value.length === 2) {
      params.start_date = dateRange.value[0].toISOString().split('T')[0];
      params.end_date = dateRange.value[1].toISOString().split('T')[0];
    }

    // 发送API请求
    const response = await axios.get('/api/analysis/tasks', { params });

    if (response.data.code === 200) {
      taskList.value = response.data.data.tasks || [];
      totalItems.value = response.data.data.total || 0;
    } else {
      throw new Error(response.data.message || '获取任务列表失败');
    }
  } catch (error_) {
    console.error('加载任务列表失败:', error_);
    error.value = error_.message || '获取任务列表失败';
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
};
// 获取数字人评估结果颜色
const getDigitalHumanColor = (probability) => {
  if (probability === null || probability === undefined) return '#909399'; // 灰色 - 未知
  if (probability >= 0.7) return '#F56C6C'; // 红色 - 高概率
  if (probability >= 0.4) return '#E6A23C'; // 黄色 - 中概率
  return '#67C23A'; // 绿色 - 低概率
};
// 格式化数字人概率显示
const formatProbability = (probability) => {
  if (probability === null || probability === undefined) return '未知';
  return `${(probability * 100).toFixed(1)}%`;
};
// 格式化日期
const formatDate = (timestamp) => {
  if (!timestamp) return '-';

  try {
    const date = new Date(timestamp);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  } catch {
    return timestamp;
  }
};

// 获取平台显示名称
const getPlatformName = (platform) => {
  const platformMap = {
    douyin: '抖音',
    tiktok: 'TikTok',
    bilibili: 'Bilibili',
  };
  return platformMap[platform] || platform;
};

// 获取状态标签类型
const getStatusType = (status) => {
  switch (status) {
    case 'completed': {
      return 'success';
    }
    case 'failed': {
      return 'danger';
    }
    case 'pending': {
      return 'info';
    }
    case 'processing': {
      return 'primary';
    }
    default: {
      return 'info';
    }
  }
};

// 获取状态显示名称
const getStatusName = (status) => {
  const statusMap = {
    pending: '等待中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  };
  return statusMap[status] || status;
};

// 获取风险等级标签类型
const getRiskLevelType = (level) => {
  switch (level) {
    case 'high': {
      return 'danger';
    }
    case 'low': {
      return 'success';
    }
    case 'medium': {
      return 'warning';
    }
    default: {
      return 'info';
    }
  }
};
const probabilityMarks = {
  0: '0%',
  0.3: '30%',
  0.5: '50%',
  0.7: {
    style: {
      color: '#F56C6C',
      fontWeight: 'bold',
    },
    label: '70%',
  },
  1: '100%',
};
// 获取风险等级显示名称
const getRiskLevelName = (level) => {
  const levelMap = {
    high: '高风险',
    medium: '中风险',
    low: '低风险',
  };
  return levelMap[level] || '未知';
};

// 处理搜索
const handleSearch = () => {
  currentPage.value = 1; // 重置到第一页
  loadTasks();
};

// 处理排序变化
const handleSortChange = (column) => {
  if (column.prop) {
    sortField.value = column.prop;
    sortOrder.value = column.order === 'ascending' ? 'asc' : 'desc';
    loadTasks();
  }
};

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val;
  loadTasks();
};

// 处理每页大小变化
const handleSizeChange = (val) => {
  pageSize.value = val;
  currentPage.value = 1; // 重置到第一页
  loadTasks();
};

// 重置筛选条件
const resetFilters = () => {
  searchText.value = '';
  platformFilter.value = '';
  statusFilter.value = '';
  dateRange.value = [];
  probabilityRange.value = [0, 1]; // 重置概率范围
  currentPage.value = 1;
  loadTasks();
};
// 查看用户画像
const viewUserProfile = (row) => {
  router.push(
    `/main/user-profile?platform=${row.platform}&userId=${row.platform_user_id}`,
  );
};
// 查看用户内容
const viewUserContent = (row) => {
  // 修改为子路由路径
  router.push(
    `/main/analysis-tasks/user-content?platform=${row.platform}&userId=${row.platform_user_id}`,
  );
};

// 刷新数据
const refreshData = () => {
  loadTasks();
};

// 初始化
onMounted(() => {
  loadTasks();
});
</script>

<template>
  <div>
    <!-- 只有在访问父路由首页时显示分析任务列表 -->
    <div
      v-if="$route.path === '/main/analysis-tasks'"
      class="analysis-tasks-container"
    >
      <ElCard class="filter-card">
        <div class="filter-header">
          <h2 class="page-title">用户分析任务</h2>
          <div class="filter-actions">
            <ElButton type="primary" plain @click="refreshData">
              <ElIcon><Refresh /></ElIcon> 刷新
            </ElButton>
          </div>
        </div>

        <!-- 筛选条件 -->
        <div class="filters">
          <ElInput
            v-model="searchText"
            placeholder="搜索用户名称"
            class="filter-item"
            clearable
            @keyup.enter="handleSearch"
          >
            <template #prefix>
              <ElIcon><Search /></ElIcon>
            </template>
          </ElInput>

          <ElSelect
            v-model="platformFilter"
            placeholder="平台"
            clearable
            class="filter-item"
          >
            <ElOption label="抖音" value="douyin" />
            <ElOption label="TikTok" value="tiktok" />
            <ElOption label="Bilibili" value="bilibili" />
          </ElSelect>

          <ElSelect
            v-model="statusFilter"
            placeholder="状态"
            clearable
            class="filter-item"
          >
            <ElOption label="等待中" value="pending" />
            <ElOption label="处理中" value="processing" />
            <ElOption label="已完成" value="completed" />
            <ElOption label="失败" value="failed" />
          </ElSelect>

          <ElDatePicker
            v-model="dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            class="filter-item date-range"
          />
          <div class="filter-item probability-slider">
            <div class="slider-header">
              <span class="slider-label">数字人概率:</span>
              <span class="slider-value">{{ probabilityRange[0] * 100 }}% -
                {{ probabilityRange[1] * 100 }}%</span>
            </div>
            <ElSlider
              v-model="probabilityRange"
              range
              :step="0.1"
              :min="0"
              :max="1"
              :marks="probabilityMarks"
              show-stops
              :format-tooltip="(val) => `${(val * 100).toFixed(0)}%`"
            />
          </div>
          <div class="filter-buttons">
            <ElButton type="primary" @click="handleSearch">查询</ElButton>
            <ElButton @click="resetFilters">重置</ElButton>
          </div>
        </div>
      </ElCard>

      <!-- 错误提示 -->
      <ElAlert
        v-if="error"
        :title="error"
        type="error"
        show-icon
        :closable="false"
        class="error-alert"
      />

      <!-- 任务列表表格 -->
      <ElCard class="task-table-card">
        <ElTable
          :data="taskList"
          border
          stripe
          style="width: 100%"
          v-loading="loading"
          @sort-change="handleSortChange"
        >
          <!-- 序号列 -->
          <ElTableColumn type="index" width="50" label="#" />

          <!-- 平台列 -->
          <ElTableColumn label="平台" prop="platform" width="100">
            <template #default="{ row }">
              <ElTag size="small">{{ getPlatformName(row.platform) }}</ElTag>
            </template>
          </ElTableColumn>

          <!-- 用户信息列 -->
          <ElTableColumn label="用户信息" min-width="200">
            <template #default="{ row }">
              <div class="user-info-cell">
                <ElAvatar :size="40" :src="row.avatar">
                  {{ row.nickname?.charAt(0) || '?' }}
                </ElAvatar>
                <div class="user-detail">
                  <div class="user-nickname">{{ row.nickname }}</div>
                  <div class="user-id">ID: {{ row.platform_user_id }}</div>
                </div>
              </div>
            </template>
          </ElTableColumn>

          <!-- 状态列 -->
          <ElTableColumn
            label="状态"
            prop="status"
            width="120"
            sortable="custom"
          >
            <template #default="{ row }">
              <ElTooltip
                :content="row.error"
                placement="top"
                :disabled="row.status !== 'failed'"
              >
                <div class="status-cell">
                  <ElTag :type="getStatusType(row.status)">
                    <ElIcon v-if="row.status === 'processing'">
                      <Loading />
                    </ElIcon>
                    <ElIcon v-if="row.status === 'completed'">
                      <CircleCheck />
                    </ElIcon>
                    <ElIcon v-if="row.status === 'failed'">
                      <CircleClose />
                    </ElIcon>
                    {{ getStatusName(row.status) }}
                  </ElTag>
                  <div class="progress-text" v-if="row.status === 'processing'">
                    {{ Math.floor(row.progress) }}%
                  </div>
                </div>
              </ElTooltip>
            </template>
          </ElTableColumn>
          <!-- 添加数字人概率列 -->
          <ElTableColumn
            label="数字人概率"
            prop="digital_human_probability"
            width="120"
            sortable="custom"
          >
            <template #default="{ row }">
              <ElProgress
                :percentage="(row.digital_human_probability || 0) * 100"
                :color="getDigitalHumanColor(row.digital_human_probability)"
                :stroke-width="8"
                :show-text="true"
                :format="() => formatProbability(row.digital_human_probability)"
              />
            </template>
          </ElTableColumn>
          <!-- 风险等级列 -->
          <ElTableColumn
            label="风险等级"
            prop="risk_level"
            width="120"
            sortable="custom"
          >
            <template #default="{ row }">
              <ElTag
                v-if="row.risk_level"
                :type="getRiskLevelType(row.risk_level)"
              >
                {{ getRiskLevelName(row.risk_level) }}
              </ElTag>
              <span v-else>-</span>
            </template>
          </ElTableColumn>

          <!-- 分析类型列 -->
          <ElTableColumn label="分析类型" prop="analysis_type" width="120">
            <template #default="{ row }">
              <ElTag size="small" effect="plain">
                {{ row.analysis_type }}
              </ElTag>
            </template>
          </ElTableColumn>

          <!-- 创建时间列 -->
          <ElTableColumn
            label="创建时间"
            prop="created_at"
            width="180"
            sortable="custom"
          >
            <template #default="{ row }">
              <div class="time-cell">
                <ElIcon><Calendar /></ElIcon>
                <span>{{ formatDate(row.created_at) }}</span>
              </div>
            </template>
          </ElTableColumn>

          <!-- 完成时间列 -->
          <ElTableColumn
            label="完成时间"
            prop="completed_at"
            width="180"
            sortable="custom"
          >
            <template #default="{ row }">
              <span>{{
                row.completed_at ? formatDate(row.completed_at) : '-'
              }}</span>
            </template>
          </ElTableColumn>

          <!-- 操作列 -->
          <ElTableColumn label="操作" width="200" fixed="right">
            <template #default="{ row }">
              <div class="action-buttons">
                <ElButton type="primary" link @click="viewUserContent(row)">
                  <ElIcon><View /></ElIcon> 查看内容
                </ElButton>
          
                <ElButton type="info" link @click="viewUserProfile(row)">
                  <ElIcon><User /></ElIcon> 用户画像
                </ElButton>
          
                <ElButton
                  v-if="row.status === 'completed'"
                  type="success"
                  link
                  @click="router.push(`/main/analysis-report?task_id=${row.id}`)"
                >
                  <ElIcon><Document /></ElIcon> 查看报告
                </ElButton>
              </div>
            </template>
          </ElTableColumn>
        </ElTable>

        <!-- 分页组件 -->
        <div class="pagination-container" v-if="totalItems > 0">
          <ElPagination
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
        <ElEmpty
          v-if="taskList.length === 0 && !loading"
          description="暂无分析任务"
        >
          <ElButton type="primary" @click="router.push('/main/add-account')">
            添加账号
          </ElButton>
        </ElEmpty>
      </ElCard>
    </div>

    <!-- 当访问子路由时渲染子路由组件 -->
    <router-view v-else />
  </div>
</template>

<style scoped>
.analysis-tasks-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-title {
  margin: 0;
  font-size: 24px;
  color: #303133;
}

.filter-actions {
  display: flex;
  gap: 10px;
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

.filter-item {
  width: 200px;
}

.date-range {
  width: 320px;
}

.filter-buttons {
  margin-left: auto;
  display: flex;
  gap: 10px;
}

.error-alert {
  margin-bottom: 20px;
}

.task-table-card {
  margin-bottom: 20px;
}

.user-info-cell {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-detail {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.user-nickname {
  font-weight: 500;
  color: #303133;
}

.user-id {
  font-size: 12px;
  color: #909399;
}

.status-cell {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-text {
  color: #409eff;
  font-size: 12px;
}

.time-cell {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #606266;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

@media (max-width: 768px) {
  .filter-item,
  .date-range {
    width: 100%;
  }

  .filter-buttons {
    margin-left: 0;
    margin-top: 12px;
    width: 100%;
    justify-content: space-between;
  }
}
/* 添加数字人概率滑块样式 */
.probability-slider {
  width: 100%;
  max-width: 320px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.slider-label {
  font-size: 14px;
  color: #606266;
}
/* 改进数字人概率滑块样式 */
.probability-slider {
  width: 100%;
  max-width: 400px; /* 增加宽度以更好显示标记 */
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.slider-label {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

.slider-value {
  font-size: 13px;
  color: #409eff;
  font-weight: 500;
}

/* 覆盖Element Plus默认样式 */
.probability-slider :deep(.el-slider__stop) {
  width: 6px;
  height: 6px;
}

/* 重要阈值点高亮 */
.probability-slider :deep(.el-slider__marks-text) {
  font-size: 12px;
  padding-top: 4px;
}
/* 操作按钮样式 */
.action-buttons {
  display: flex;
  flex-direction: column;
  align-items: center; /* 居中对齐 */
  gap: 5px;
}

.action-buttons .el-button {
  padding: 4px 0;
  margin-left: 0;
}

.action-buttons .el-button + .el-button {
  margin-left: 0;
}
</style>
