<script lang="ts" setup>
import { computed } from 'vue';
import { Refresh, Download } from '@element-plus/icons-vue';
import { ElButton, ElCard, ElResult, ElSkeleton } from 'element-plus';
import MarkdownIt from 'markdown-it';

// 创建markdown-it实例
const md = new MarkdownIt({
  html: true, // 启用HTML标签
  breaks: true, // 将换行符转换为<br>
  linkify: true, // 自动将URL转换为链接
  typographer: true, // 启用一些语言中性的替换+引号美化
});

// 定义组件接收的props
const props = defineProps({
  reportData: {
    type: Object,
    default: () => null
  },
  loading: {
    type: Boolean,
    default: false
  },
  error: {
    type: String,
    default: null
  },
  videoTitle: {
    type: String,
    default: ''
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['regenerate', 'export']);

// 重新生成报告
const regenerateReport = () => {
  emit('regenerate');
};

// 导出报告
const exportReport = () => {
  emit('export');
};

// 格式化日期
const formatDate = (date) => {
  if (!date) return '未知时间';
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

// 获取风险等级信息
const riskLevelInfo = computed(() => {
  if (!props.reportData || !props.reportData.risk_level)
    return { class: 'info', color: '#909399', text: '未评估' };

  const level = props.reportData.risk_level.toLowerCase();
  switch (level) {
    case 'low':
      return { class: 'success', color: '#67C23A', text: '低风险' };
    case 'medium':
      return { class: 'warning', color: '#E6A23C', text: '中等风险' };
    case 'high':
      return { class: 'danger', color: '#F56C6C', text: '高风险' };
    default:
      return { class: 'info', color: '#909399', text: '未评估' };
  }
});

// 根据评分获取进度条颜色
const getScoreColor = (score) => {
  if (score >= 0.8) return '#67C23A'; // 绿色
  if (score >= 0.5) return '#E6A23C'; // 橙色
  return '#F56C6C'; // 红色
};

// 格式化评分值（保留1位小数）
const formatScore = (score) => {
  return typeof score === 'number' ? score.toFixed(1) : 'N/A';
};
</script>

<template>
  <div class="threat-report">
    <div class="threat-report-header">
      <h3 class="section-heading">内容威胁分析报告</h3>
      <div class="report-timestamp">
        生成时间: {{ formatDate(reportData?.timestamp || new Date()) }}
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <!-- 错误状态 -->
    <el-result
      v-else-if="error"
      icon="error"
      :title="error"
      sub-title="无法获取分析报告数据"
    >
      <template #extra>
        <el-button type="primary" @click="regenerateReport">重试</el-button>
      </template>
    </el-result>

    <!-- 报告数据显示 -->
    <div v-else-if="reportData" class="analysis-report">
      <!-- 风险等级信息 -->
      <el-card
        class="risk-info-card"
        :class="`border-${riskLevelInfo.class}`"
      >
        <div class="risk-info-header">
          <div class="risk-level-container">
            <el-tag
              :type="riskLevelInfo.class"
              size="large"
              effect="dark"
              class="risk-level-tag"
            >
              {{ riskLevelInfo.text }}
            </el-tag>
            <div class="risk-probability">
              风险概率:
              <span :style="{ color: riskLevelInfo.color }">
                {{ (reportData.risk_probability * 100).toFixed(1) }}%
              </span>
            </div>
          </div>
          <div class="action-buttons">
            <!-- 重新生成按钮 -->
            <el-button
              type="primary"
              @click="regenerateReport"
              :icon="Refresh"
              size="small"
            >
              重新生成
            </el-button>
            <!-- 添加导出按钮 -->
            <el-button
              type="success"
              @click="exportReport"
              :icon="Download"
              size="small"
              class="export-button"
            >
              导出报告
            </el-button>
          </div>
        </div>
      </el-card>

      <!-- 分析报告内容 -->
      <el-card class="report-content">
        <div class="report-container">
          <div
            class="markdown-body"
            v-html="md.render(reportData.report)"
          ></div>
        </div>
      </el-card>
    </div>

    <!-- 没有报告时显示 -->
    <div v-else>
      <el-result icon="info" title="暂无分析报告">
        <template #sub-title>
          <p>系统尚未对此视频生成分析报告，点击下方按钮生成。</p>
        </template>
        <template #extra>
          <el-button type="primary" @click="regenerateReport">
            生成分析报告
          </el-button>
        </template>
      </el-result>
    </div>
  </div>
</template>

<style scoped>
.threat-report {
  height: 100%;
  overflow: auto;
}

.threat-report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-heading {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.report-timestamp {
  font-size: 14px;
  color: #909399;
  font-style: italic;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 3rem;
  padding-bottom: 3rem;
}

.risk-info-card {
  margin-bottom: 1rem;
  border-top-width: 4px;
}

.risk-info-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.risk-level-container {
  display: flex;
  align-items: center;
}

.risk-level-tag {
  margin-right: 0.75rem;
}

.risk-probability {
  font-size: 1.125rem;
  font-weight: 500;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
}

.export-button {
  margin-left: 0.5rem;
}

.report-content {
  margin-bottom: 1rem;
}

.report-container {
  padding: 10px 5px;
}

.border-success {
  border-top-color: #67c23a;
}

.border-warning {
  border-top-color: #e6a23c;
}

.border-danger {
  border-top-color: #f56c6c;
}

.border-info {
  border-top-color: #909399;
}

/* Markdown样式 */
:deep(.markdown-body) {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  line-height: 1.8;
  color: #333;
  word-break: break-word;
}

/* 标题样式增强 */
:deep(.markdown-body h2) {
  margin-top: 28px;
  font-size: 20px;
  font-weight: 600;
  border-bottom: 2px solid #409eff;
  padding-bottom: 8px;
  color: #303133;
}

:deep(.markdown-body h3) {
  margin-top: 24px;
  font-size: 17px;
  font-weight: 600;
  color: #409eff;
  background-color: #ecf5ff;
  padding: 8px 12px;
  border-radius: 4px;
}

/* 风险警告突出显示 */
:deep(.markdown-body p:has(> ▲)) {
  background-color: #fef0f0;
  padding: 12px 16px;
  border-radius: 6px;
  border-left: 4px solid #f56c6c;
  margin-bottom: 20px;
}

/* 突出显示风险标记 */
:deep(.markdown-body p ▲) {
  color: #f56c6c;
  font-weight: bold;
  margin-right: 4px;
}

/* 增强列表样式 */
:deep(.markdown-body ol) {
  padding-left: 22px;
  margin-bottom: 20px;
}

:deep(.markdown-body ol li) {
  margin-bottom: 10px;
  padding-left: 6px;
}

/* 突出显示粗体文本 */
:deep(.markdown-body strong) {
  color: #e6a23c;
  font-weight: bold;
  background-color: rgba(255, 229, 100, 0.3);
  padding: 0 4px;
  border-radius: 3px;
}

/* 突出显示风险类别 */
:deep(.markdown-body p strong:first-of-type) {
  display: inline-block;
  margin-right: 5px;
}

/* 增强代码块样式 */
:deep(.markdown-body code) {
  color: #476582;
  background-color: rgba(27, 31, 35, 0.05);
  padding: 2px 5px;
  border-radius: 3px;
}

/* 表格样式增强 */
:deep(.markdown-body table) {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}

:deep(.markdown-body table th) {
  background: #f2f6fc;
  padding: 12px;
  border: 1px solid #ebeef5;
}

:deep(.markdown-body table td) {
  padding: 12px;
  border: 1px solid #ebeef5;
}

/* 结论部分特殊样式 */
:deep(.markdown-body > p:first-child) {
  font-size: 16px;
  background-color: #fef0f0;
  padding: 15px;
  border-radius: 6px;
  border-left: 5px solid #f56c6c;
  font-weight: 500;
  margin-bottom: 25px;
}
</style>