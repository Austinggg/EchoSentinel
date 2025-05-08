<script lang="ts" setup>
import { ref, computed, onMounted } from 'vue';
import {
  ElButton,
  ElIcon,
  ElTag,
  ElTooltip,
  ElTable,
  ElTableColumn,
  ElImage,
  ElMessage,
  ElCard,
  ElInput,
  ElPagination,
  ElBadge,
  ElMessageBox,
} from 'element-plus';
import {
  Timer,
  WarningFilled,
  InfoFilled,
  CircleCheckFilled,
  CircleCloseFilled,
  Delete,
  Search,
} from '@element-plus/icons-vue';
import { getVideoList, deleteVideo, batchDeleteVideos } from '#/api/videoApi';
import type { VideoAnalysisRecord } from '#/api/videoApi';
import { useRouter, RouterView } from 'vue-router';
import axios from 'axios';
// 定义分析结果数据类型
interface AnalysisRecord {
  id: string;
  title: string;
  cover: string;
  summary: string;
  threatLevel: VideoAnalysisRecord['threatLevel']; // 修改为使用VideoAnalysisRecord的威胁等级
  createTime: string;
}
const loading = ref(false);
const analysisData = ref<VideoAnalysisRecord[]>([]);
const total = ref(0);
// 添加分页相关状态
const currentPage = ref(1);
const pageSize = ref(6);
// 添加搜索相关功能
const search = ref('');
const activeFilter = ref<VideoAnalysisRecord['threatLevel'] | null>(null);
// 加载数据方法

async function loadData() {
  loading.value = true;
  try {
    const response = await getVideoList();

    analysisData.value = response.items;
    total.value = response.total;

    console.log('加载数据成功:', analysisData.value);
  } catch (error) {
    ElMessage.error('加载视频分析数据失败');
    console.error(error);
  } finally {
    loading.value = false;
  }
}

// 处理批量删除
async function handleBatchDelete() {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('请先选择要删除的记录');
    return;
  }

  const ids = multipleSelection.value.map((item) => item.id);
  try {
    await batchDeleteVideos(ids);
    ElMessage.success('批量删除成功');
    loadData(); // 重新加载数据
    multipleSelection.value = []; // 清空选择
  } catch (error) {
    ElMessage.error('批量删除失败');
    console.error(error);
  }
}

// 添加筛选状态

// 统计各威胁等级的数量
const threatLevelCounts = computed(() => {
  const counts = {
    processing: 0,
    low: 0,
    medium: 0,
    high: 0,
  };

  analysisData.value.forEach((item) => {
    counts[item.threatLevel]++;
  });

  return counts;
});
// 筛选函数
const filterByThreatLevel = (level: AnalysisRecord['threatLevel']) => {
  if (activeFilter.value === level) {
    // 取消筛选
    activeFilter.value = null;
  } else {
    // 设置筛选
    activeFilter.value = level;
  }

  // 重置到第一页
  currentPage.value = 1;

  // 清除选择
  if (multipleTableRef.value) {
    multipleTableRef.value.clearSelection();
  }
};

// 修改搜索筛选逻辑，加入威胁等级筛选
const searchFilteredData = computed(() =>
  analysisData.value.filter(
    (data) =>
      // 搜索条件筛选
      (!search.value ||
        data.title.toLowerCase().includes(search.value.toLowerCase()) ||
        data.summary.toLowerCase().includes(search.value.toLowerCase())) &&
      // 威胁等级筛选
      (activeFilter.value === null || data.threatLevel === activeFilter.value),
  ),
);

// 计算总条目数量
const totalItems = computed(() => searchFilteredData.value.length);

// 分页过滤的数据
const filteredAnalysisData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value;
  const end = start + pageSize.value;
  return searchFilteredData.value.slice(start, end);
});

// 处理分页变化
const handleCurrentChange = (val: number) => {
  currentPage.value = val;
  // 页码变化时重新从API加载数据
  loadData();

  // 改变页码时清除选择
  if (multipleTableRef.value) {
    multipleTableRef.value.clearSelection();
  }
};

// 添加多选相关的引用和状态
const multipleTableRef = ref<InstanceType<typeof ElTable>>();
const multipleSelection = ref<AnalysisRecord[]>([]);

// 处理选择变化
const handleSelectionChange = (val: AnalysisRecord[]) => {
  multipleSelection.value = val;
  console.log('已选择项:', multipleSelection.value);
};

// 切换选择状态
const toggleSelection = (rows?: AnalysisRecord[]) => {
  if (rows) {
    rows.forEach((row) => {
      multipleTableRef.value!.toggleRowSelection(row, undefined);
    });
  } else {
    multipleTableRef.value!.clearSelection();
  }
};

// 根据威胁等级设置行类名
const tableRowClassName = ({ row }: { row: AnalysisRecord }) => {
  switch (row.threatLevel) {
    case 'high':
      return 'danger-row';
    case 'medium':
      return 'warning-row';
    case 'low':
      return 'success-row'; // 修改为 success-row，显示绿色
    case 'processing':
      return 'info-row'; // 处理中状态使用灰色
    default:
      return '';
  }
};

// 编辑和删除的处理函数
const handleEdit = (id: string) => {
  ElMessage.info(`编辑记录：${id}`);
  // 这里可以添加编辑逻辑或跳转到编辑页面
};
// 添加确认视频删除的函数
const handleDelete = async (videoId) => {
  try {
    // 显示确认删除对话框
    await ElMessageBox.confirm(
      '确定要删除此视频及其所有关联数据吗？此操作不可撤销！',
      '删除确认',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning',
        closeOnClickModal: false,
      },
    );

    // 开始删除操作
    loading.value = true;

    // 调用API删除视频及其关联数据
    const response = await axios.delete(`/api/videos/${videoId}/all`);

    if (response.data.code === 200) {
      // 从表格数据中移除已删除项
      analysisData.value = analysisData.value.filter(
        (item) => item.id !== videoId,
      );

      // 刷新筛选后的表格数据
      filteredAnalysisData.value = filteredAnalysisData.value.filter(
        (item) => item.id !== videoId,
      );

      ElMessage.success('视频及其关联数据已成功删除');
    } else {
      throw new Error(response.data.message || '删除操作失败');
    }
  } catch (error) {
    if (error === 'cancel') {
      // 用户取消删除，不做处理
      return;
    }

    console.error('删除视频失败:', error);
    ElMessage.error(`删除失败: ${error.message || '未知错误'}`);
  } finally {
    loading.value = false;
  }
};

// 修改威胁等级相关函数
const getThreatLevelType = (level: AnalysisRecord['threatLevel']) => {
  const typeMap = {
    processing: 'info',
    low: 'success',
    medium: 'warning',
    high: 'danger',
  } as const;
  return typeMap[level];
};

const getThreatLevelIcon = (level: AnalysisRecord['threatLevel']) => {
  const iconMap = {
    processing: InfoFilled,
    low: CircleCheckFilled,
    medium: WarningFilled,
    high: CircleCloseFilled,
  };
  return iconMap[level];
};

const getThreatLevelText = (level: AnalysisRecord['threatLevel']) => {
  const textMap = {
    processing: '处理中',
    low: '低风险',
    medium: '中风险',
    high: '高风险',
  };
  return textMap[level];
};

// 初始加载
onMounted(() => {
  loadData();
});

const router = useRouter();

// 处理表格行点击事件
const handleRowClick = (row) => {
  // 跳转到分析页面，并传递id参数
  router.push({
    name: 'contentAnalysis',
    query: { id: row.id },
  });
};
</script>

<template>
  <div>
    <!-- 只有在访问父路由首页时显示分析记录列表 -->
    <div v-if="$route.path === '/demos/analysis-records'">
      <!-- 现有的分析记录列表内容 -->
      <el-card class="card" shadow="always">
        <!-- 表格顶部操作区 -->
        <div class="table-operations">
          <!-- 搜索框 -->
          <el-input
            v-model="search"
            placeholder="搜索标题或摘要"
            class="search-input"
            clearable
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <!-- 已选择计数 -->
          <div>
            <span v-if="multipleSelection.length > 0" class="selected-count">
              已选择 {{ multipleSelection.length }} 项
            </span>
          </div>
          <!-- 右侧操作区 -->
          <div class="right-operations">
            <!-- 徽章区域 -->
            <div class="badge-group">
              <!-- 高风险徽章 -->
              <el-badge
                :value="threatLevelCounts.high"
                class="badge-item"
                type="danger"
              >
                <el-button
                  size="small"
                  :type="activeFilter === 'high' ? 'danger' : ''"
                  @click="filterByThreatLevel('high')"
                >
                  高风险
                </el-button>
              </el-badge>

              <!-- 中风险徽章 -->
              <el-badge
                :value="threatLevelCounts.medium"
                class="badge-item"
                type="warning"
              >
                <el-button
                  size="small"
                  :type="activeFilter === 'medium' ? 'warning' : ''"
                  @click="filterByThreatLevel('medium')"
                >
                  中风险
                </el-button>
              </el-badge>

              <!-- 低风险徽章 -->
              <el-badge
                :value="threatLevelCounts.low"
                class="badge-item"
                type="success"
              >
                <el-button
                  size="small"
                  :type="activeFilter === 'low' ? 'success' : ''"
                  @click="filterByThreatLevel('low')"
                >
                  低风险
                </el-button>
              </el-badge>

              <!-- 处理中徽章 -->
              <el-badge
                :value="threatLevelCounts.processing"
                class="badge-item"
                type="info"
              >
                <el-button
                  size="small"
                  :type="activeFilter === 'processing' ? 'info' : ''"
                  @click="filterByThreatLevel('processing')"
                >
                  处理中
                </el-button>
              </el-badge>
            </div>

            <!-- 操作按钮区域 - 两个按钮都只在有选择时显示 -->
            <el-button
              v-if="multipleSelection.length > 0"
              type="danger"
              @click="handleBatchDelete"
              size="small"
            >
              <el-icon><Delete /></el-icon>
              批量删除
            </el-button>
            <el-button
              v-if="multipleSelection.length > 0"
              @click="toggleSelection()"
              size="small"
            >
              清除选择
            </el-button>
          </div>
        </div>
        <!-- 表格主体 -->

        <el-table
          ref="multipleTableRef"
          :data="filteredAnalysisData"
          :row-class-name="tableRowClassName"
          style="width: 100%; height: 100%"
          border
          @selection-change="handleSelectionChange"
        >
          <!-- 多选列 -->
          <el-table-column type="selection" width="55" />
          <!-- 封面列 -->
          <el-table-column label="封面" min-width="120" align="center">
            <template #default="{ row }">
              <el-image
                :src="row.cover"
                fit="cover"
                style="
                  width: 90px;
                  height: 140px;
                  border-radius: 4px;
                  object-position: center;
                "
                :preview="false"
              />
            </template>
          </el-table-column>

          <!-- 标题列 -->
          <el-table-column label="标题" min-width="180" align="left">
            <template #default="{ row }">
              <el-tooltip :content="row.title" placement="top" effect="light">
                <div class="title-cell" @click="handleRowClick(row)">
                  <span class="video-title multiline-text">{{
                    row.title
                  }}</span>
                </div>
              </el-tooltip>
            </template>
          </el-table-column>
          <!-- 摘要列 -->
          <el-table-column label="摘要" min-width="300">
            <template #default="{ row }">
              <el-tooltip :content="row.summary" placement="top" effect="light">
                <div
                  class="summary-cell multiline-text"
                  @click="handleRowClick(row)"
                >
                  {{ row.summary }}
                </div>
              </el-tooltip>
            </template>
          </el-table-column>

          <!-- 威胁等级列 -->
          <el-table-column label="威胁等级" min-width="120" align="center">
            <template #default="{ row }">
              <el-tag :type="getThreatLevelType(row.threatLevel)" effect="dark">
                <el-icon class="mr-1">
                  <component :is="getThreatLevelIcon(row.threatLevel)" />
                </el-icon>
                {{ getThreatLevelText(row.threatLevel) }}
              </el-tag>
            </template>
          </el-table-column>

          <!-- 创建时间列 -->
          <el-table-column label="创建时间" min-width="120" align="center">
            <template #default="{ row }">
              <span class="flex items-center">
                <el-icon class="mr-1"><Timer /></el-icon>
                {{ row.createTime }}
              </span>
            </template>
          </el-table-column>

          <!-- 操作列 -->
          <el-table-column
            label="操作"
            min-width="150"
            align="center"
            fixed="right"
          >
            <template #default="{ row }">
              <div class="space-x-2">
                <el-button
                  size="small"
                  type="primary"
                  @click="handleEdit(row.id)"
                >
                  编辑
                </el-button>
                <el-button
                  size="small"
                  type="danger"
                  @click="handleDelete(row.id)"
                >
                  删除
                </el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>
        <!-- 空数据提示 -->
        <div v-if="filteredAnalysisData.length === 0" class="empty-data">
          <p>没有找到匹配的记录</p>
        </div>
        <!-- 分页组件 -->
        <div class="pagination-container" v-if="searchFilteredData.length > 0">
          <el-pagination
            v-model:current-page="currentPage"
            :page-size="pageSize"
            :small="true"
            background
            layout="prev, pager, next, jumper"
            :total="totalItems"
            @current-change="handleCurrentChange"
          />
        </div>
      </el-card>
    </div>
    <router-view v-else />
  </div>
</template>

<style scoped>
/* 标题列样式 */
.multiline-text {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2; /* 显示3行 */
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.5;
}
/* 摘要样式 */
.summary-cell {
  padding: 4px 0;
  font-size: 14px;
  color: #606266;
  -webkit-line-clamp: 3; /* 摘要显示4行 */
}
/* 移除旧的单行样式 */
.truncate {
  max-width: 280px;
}
/* 标题样式改为多行 */
.title-cell {
  padding: 6px 0;
  max-width: 100%;
}

.video-title {
  font-weight: 600;
  font-size: 15px;
  color: #303133;
  position: relative;
  padding-left: 8px;
  line-height: 1.4;
}

.video-title::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  height: 14px;
  width: 3px;
  background-color: #409eff;
  border-radius: 2px;
}
/* 悬停效果 */
:deep(.el-table__row:hover .video-title) {
  color: #409eff;
}

.pagination-container {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}
.card {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  width: 100%;
}

.table-operations {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.left-operations {
  display: flex;
  align-items: center;
  gap: 12px; /* 控制搜索框和已选计数的间距 */
}
/* 修改后 */
.right-operations {
  display: flex;
  flex-wrap: wrap;
  gap: 16px; /* 增加整体间距 */
  align-items: center;
}

.badge-group {
  display: flex;
  gap: 12px; /* 增加徽章之间的间距 */
  margin-right: 24px; /* 增加徽章组与操作按钮的间距 */
}

.badge-item {
  margin-right: 4px; /* 为每个徽章添加额外间距 */
}
.search-input {
  width: 240px;
}
.empty-data {
  text-align: center;
  padding: 40px 0;
  color: #909399;
  font-size: 14px;
}
.selected-count {
  font-size: 14px;
  color: #606266;
  background-color: #f0f9eb;
  padding: 4px 8px;
  border-radius: 4px;
}

.truncate {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
  max-width: 280px;
}

.mr-1 {
  margin-right: 4px;
}

.space-x-2 > * + * {
  margin-left: 8px;
}

.flex {
  display: flex;
}

.items-center {
  align-items: center;
}

:deep(.el-table .danger-row) {
  background-color: rgba(245, 108, 108, 0.1);
}

:deep(.el-table .warning-row) {
  background-color: rgba(230, 162, 60, 0.1);
}

:deep(.el-table .info-row) {
  background-color: white; /* 将处理中状态改为白底 */
  box-shadow: inset 0 0 0 1px rgba(144, 147, 153, 0.2); /* 添加浅灰色边框，便于识别 */
}

:deep(.el-table .success-row) {
  background-color: rgba(103, 194, 58, 0.1);
}
/* 悬停效果 - 修改为灰色背景 */
:deep(.el-table__row:hover) {
  background-color: rgba(144, 147, 153, 0.1) !important; /* 悬停改为浅灰色 */
  box-shadow: none !important; /* 移除边框阴影 */
}

/* 添加鼠标指针样式，提示行可点击 */
:deep(.el-table__row) {
  cursor: pointer;
}
</style>
