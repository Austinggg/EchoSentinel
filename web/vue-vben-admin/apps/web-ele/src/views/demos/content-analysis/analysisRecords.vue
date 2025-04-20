<script lang="ts" setup>
import { ref, computed } from 'vue';
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

// 定义分析结果数据类型
interface AnalysisRecord {
  id: string;
  title: string;
  cover: string;
  summary: string;
  threatLevel: 'low' | 'medium' | 'high' | 'processing'; // 将'safe'改为'processing'
  createTime: string;
}

// 示例数据
const analysisData = ref<AnalysisRecord[]>([
  {
    id: '001',
    title: '游戏直播片段',
    cover: 'https://picsum.photos/id/111/200/120',
    summary: '主要内容为游戏直播，无敏感内容，适合大部分年龄段观看。',
    threatLevel: 'low',
    createTime: '2025-04-15',
  },
  {
    id: '002',
    title: '政治评论视频',
    cover: 'https://picsum.photos/id/222/200/120',
    summary: '包含部分政治敏感言论，建议审核后再发布。',
    threatLevel: 'medium',
    createTime: '2025-04-14',
  },
  {
    id: '003',
    title: '科普教育内容',
    cover: 'https://picsum.photos/id/333/200/120',
    summary: '科学教育内容，知识准确，适合传播。',
    threatLevel: 'low',
    createTime: '2025-04-12',
  },
  {
    id: '004',
    title: '广告营销视频',
    cover: 'https://picsum.photos/id/444/200/120',
    summary: '存在虚假宣传内容，建议修改后再发布。',
    threatLevel: 'high',
    createTime: '2025-04-10',
  },
  {
    id: '005',
    title: '音乐MV',
    cover: 'https://picsum.photos/id/555/200/120',
    summary: '音乐内容健康，无不良信息。',
    threatLevel: 'low',
    createTime: '2025-04-08',
  },
  {
    id: '006',
    title: '健康健身视频',
    cover: 'https://picsum.photos/id/666/200/120',
    summary: '健身指导视频，部分动作可能存在安全隐患，建议添加警示。',
    threatLevel: 'processing',
    createTime: '2025-04-05',
  },
  {
    id: '007',
    title: '美食烹饪教程',
    cover: 'https://picsum.photos/id/777/200/120',
    summary: '烹饪内容健康，步骤清晰，适合传播。',
    threatLevel: 'low',
    createTime: '2025-04-02',
  },
  {
    id: '008',
    title: '心理健康讲座',
    cover: 'https://picsum.photos/id/888/200/120',
    summary: '涉及部分心理疾病内容，建议添加专业免责声明。',
    threatLevel: 'medium',
    createTime: '2025-03-28',
  },
]);
// 添加筛选状态
const activeFilter = ref<AnalysisRecord['threatLevel'] | null>(null);

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

// 添加分页相关状态
const currentPage = ref(1);
const pageSize = ref(6);

// 添加搜索相关功能
const search = ref('');

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
// 批量删除选中项
const handleBatchDelete = () => {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('请先选择要删除的记录');
    return;
  }

  const ids = multipleSelection.value.map((item) => item.id);
  ElMessage.warning(`批量删除记录：${ids.join(', ')}`);
  // 这里可以添加实际的批量删除逻辑
};

// 根据威胁等级设置行类名
const tableRowClassName = ({ row }: { row: AnalysisRecord }) => {
  switch (row.threatLevel) {
    case 'high':
      return 'danger-row';
    case 'medium':
      return 'warning-row';
    case 'low':
      return 'info-row';
    case 'safe':
      return 'success-row';
    default:
      return '';
  }
};

// 编辑和删除的处理函数
const handleEdit = (id: string) => {
  ElMessage.info(`编辑记录：${id}`);
  // 这里可以添加编辑逻辑或跳转到编辑页面
};

const handleDelete = (id: string) => {
  ElMessage.warning(`删除记录：${id}`);
  // 这里可以添加删除确认逻辑
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
</script>

<template>
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
      stripe
      @selection-change="handleSelectionChange"
    >
      <!-- 多选列 -->
      <el-table-column type="selection" width="55" />
      <!-- 封面列 -->
      <el-table-column label="封面" min-width="200" align="center">
        <template #default="{ row }">
          <el-image
            :src="row.cover"
            fit="cover"
            style="width: 180px; height: 100px; border-radius: 4px"
            :preview="false"
          />
        </template>
      </el-table-column>

      <!-- 标题列 -->
      <el-table-column prop="title" label="标题" min-width="150" />

      <!-- 摘要列 -->
      <el-table-column label="摘要" min-width="300">
        <template #default="{ row }">
          <el-tooltip :content="row.summary" placement="top" effect="light">
            <span class="truncate">{{ row.summary }}</span>
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
            <el-button size="small" type="primary" @click="handleEdit(row.id)">
              编辑
            </el-button>
            <el-button size="small" type="danger" @click="handleDelete(row.id)">
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
</template>

<style scoped>
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

/* 确保表格行着色正确 */
:deep(.danger-row) {
  background-color: rgba(245, 108, 108, 0.1);
}

:deep(.warning-row) {
  background-color: rgba(230, 162, 60, 0.1);
}

:deep(.info-row) {
  background-color: rgba(144, 147, 153, 0.1);
}

:deep(.success-row) {
  background-color: rgba(103, 194, 58, 0.1);
}

/* 鼠标悬停效果 */
:deep(.el-table__row:hover) {
  background-color: rgba(0, 0, 0, 0.05) !important;
}
</style>
