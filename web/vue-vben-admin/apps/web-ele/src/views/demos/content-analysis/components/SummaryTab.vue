<script lang="ts" setup>
import { Refresh } from '@element-plus/icons-vue';
import { ElButton } from 'element-plus';

// 定义组件接收的props
const props = defineProps({
  summary: {
    type: String,
    default: ''
  },
  loading: {
    type: Boolean,
    default: false
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['regenerate']);

// 重新生成摘要
const regenerateSummary = () => {
  emit('regenerate');
};
</script>

<template>
  <div class="summary-container">
    <!-- 使用v-html渲染Markdown转换后的HTML -->
    <div v-if="summary" class="markdown-body" v-html="summary"></div>
    <p v-else class="no-content">暂无摘要内容</p>

    <!-- 重新生成按钮 -->
    <div class="action-button-container">
      <el-button
        type="primary"
        :loading="loading"
        @click="regenerateSummary"
        size="small"
        :icon="Refresh"
      >
        重新生成摘要
      </el-button>
    </div>
  </div>
</template>

<style scoped>
.summary-container {
  padding: 0 4px;
}

.no-content {
  color: #6b7280;
  padding: 20px 0;
  text-align: center;
  font-style: italic;
}

.action-button-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

/* Markdown样式需要在父组件中定义为全局样式，使用deep选择器 */
/* 这里只添加一些额外的样式增强 */
:deep(.markdown-body) {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  line-height: 1.8;
  color: #333;
  word-break: break-word;
}

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

:deep(.markdown-body p) {
  margin-bottom: 16px;
  line-height: 1.8;
}

:deep(.markdown-body ul, .markdown-body ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}

:deep(.markdown-body li) {
  margin-bottom: 8px;
}

/* 强调重要内容 */
:deep(.markdown-body strong) {
  color: #e6a23c;
  font-weight: bold;
  background-color: rgba(255, 229, 100, 0.3);
  padding: 0 4px;
  border-radius: 3px;
}

/* 摘要总结段落的特殊样式 */
:deep(.markdown-body > p:first-child) {
  font-size: 16px;
  background-color: #f0f9eb;
  padding: 15px;
  border-radius: 6px;
  border-left: 5px solid #67c23a;
  font-weight: 500;
  margin-bottom: 25px;
}
</style>