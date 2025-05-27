<script lang="ts" setup>
import { computed } from 'vue';
import { Refresh, InfoFilled } from '@element-plus/icons-vue';
import { 
  ElButton, 
  ElCard, 
  ElTag, 
  ElSkeleton, 
  ElResult,
  ElCollapse,
  ElCollapseItem
} from 'element-plus';
import MarkdownIt from 'markdown-it';

// 创建markdown-it实例
const md = new MarkdownIt({
  html: true,
  breaks: true,
  linkify: true,
  typographer: true,
});

// 定义组件接收的props
const props = defineProps({
  factCheckData: {
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
  notFound: {
    type: Boolean,
    default: false
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['load-data', 'generate-check']);

// 重新加载数据
const loadFactCheckData = () => {
  emit('load-data');
};

// 生成事实核查
const generateFactCheck = () => {
  emit('generate-check');
};

// 格式化日期时间
const formatDate = (date) => {
  if (!date) return '未知时间';
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

// 获取事实核查状态标签的样式
const factCheckStatusInfo = computed(() => {
  if (!props.factCheckData) return { class: 'info', text: '未核查' };

  switch (props.factCheckData.status) {
    case 'completed':
      return { class: 'success', text: '已完成' };
    case 'processing':
      return { class: 'warning', text: '进行中' };
    case 'failed':
      return { class: 'danger', text: '失败' };
    default:
      return { class: 'info', text: '未核查' };
  }
});
</script>

<template>
  <div class="factcheck-container">
    <div class="factcheck-header">
      <h3 class="section-heading">视频事实核查</h3>
      <div v-if="factCheckData?.timestamp" class="factcheck-timestamp">
        核查时间: {{ formatDate(factCheckData.timestamp) }}
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
      sub-title="无法获取事实核查数据"
    >
      <template #extra>
        <el-button type="primary" @click="loadFactCheckData">重试</el-button>
        <el-button type="success" @click="generateFactCheck">重新核查</el-button>
      </template>
    </el-result>

    <!-- 正在处理状态 -->
    <el-result
      v-else-if="factCheckData?.status === 'processing'"
      icon="info"
      title="事实核查正在进行中"
      sub-title="这可能需要几分钟时间，请稍后刷新"
    >
      <template #extra>
        <el-button type="primary" @click="loadFactCheckData">刷新</el-button>
      </template>
    </el-result>

    <!-- 尚未进行核查的状态 -->
    <el-result
      v-else-if="notFound"
      icon="info"
      title="暂无事实核查结果"
      sub-title="该视频尚未进行事实核查，点击下方按钮开始核查。"
    >
      <template #extra>
        <el-button type="primary" @click="generateFactCheck">开始事实核查</el-button>
      </template>
    </el-result>

    <!-- 事实核查结果展示 -->
    <div v-else-if="factCheckData?.worth_checking" class="factcheck-result">
      <!-- 核查状态卡片 -->
      <el-card class="status-card" :class="`border-${factCheckStatusInfo.class}`">
        <div class="status-header">
          <div class="status-info">
            <el-tag
              :type="factCheckStatusInfo.class"
              size="large"
              effect="dark"
              class="status-tag"
            >
              {{ factCheckStatusInfo.text }}
            </el-tag>
            <span class="worth-checking-label">值得核查</span>
          </div>
          <div class="action-buttons">
            <el-button
              type="primary"
              @click="generateFactCheck"
              :icon="Refresh"
              size="small"
            >
              重新核查
            </el-button>
          </div>
        </div>
        <div class="reason-text">
          {{ factCheckData.reason }}
        </div>
      </el-card>

      <!-- 断言列表 -->
      <div v-if="factCheckData.claims && factCheckData.claims.length > 0">
        <h4 class="claims-heading">
          共发现 {{ factCheckData.claims.length }} 条需要核查的断言：
        </h4>

        <!-- 核查结果统计信息 -->
        <div v-if="factCheckData.search_summary" class="summary-stats">
          <div class="stat-item" style="color: #67c23a">
            <div class="stat-value">
              {{ factCheckData.search_summary.true_claims }}
            </div>
            <div class="stat-label">属实</div>
          </div>
          <div class="stat-item" style="color: #f56c6c">
            <div class="stat-value">
              {{ factCheckData.search_summary.false_claims }}
            </div>
            <div class="stat-label">不实</div>
          </div>
          <div class="stat-item" style="color: #909399">
            <div class="stat-value">
              {{ factCheckData.search_summary.uncertain_claims }}
            </div>
            <div class="stat-label">未确定</div>
          </div>
        </div>

        <!-- 断言和核查结果列表 -->
        <div class="claims-list">
          <el-card
            v-for="(result, index) in factCheckData.fact_check_results"
            :key="index"
            class="claim-card"
            :class="{
              'claim-true': result.is_true === '是',
              'claim-false': result.is_true === '否',
              'claim-uncertain': result.is_true !== '是' && result.is_true !== '否',
            }"
          >
            <div class="claim-header">
              <el-tag
                :type="
                  result.is_true === '是'
                    ? 'success'
                    : result.is_true === '否'
                      ? 'danger'
                      : 'info'
                "
                effect="dark"
                size="small"
                class="claim-tag"
              >
                {{
                  result.is_true === '是'
                    ? '属实'
                    : result.is_true === '否'
                      ? '不实'
                      : '未确定'
                }}
              </el-tag>
              <div class="claim-text">{{ result.claim }}</div>
            </div>

            <div class="claim-body">
              <div class="conclusion-text markdown-body">
                <strong>核查结论：</strong>
                <div v-html="md.render(result.conclusion)"></div>
              </div>

              <!-- 搜索详情折叠面板 -->
              <el-collapse v-if="result.search_details">
                <el-collapse-item title="查看搜索详情">
                  <div class="search-details">
                    <div class="search-info">
                      <span class="search-label">搜索关键词：</span>
                      <span class="search-value">{{
                        result.search_details.keywords
                      }}</span>
                    </div>

                    <div class="search-info">
                      <span class="search-label">搜索用时：</span>
                      <span class="search-value">{{
                        result.search_duration?.toFixed(2)
                      }} 秒</span>
                    </div>

                    <!-- 相关搜索结果列表 -->
                    <div
                      v-if="result.search_details.top_results?.length"
                      class="search-results"
                    >
                      <div class="results-heading">相关结果：</div>
                      <div
                        v-for="(searchResult, sIdx) in result.search_details.top_results"
                        :key="sIdx"
                        class="search-result-item"
                      >
                        <div class="result-title">
                          <strong>{{ searchResult.title }}</strong>
                        </div>
                        <div class="result-snippet">
                          {{ searchResult.snippet }}
                        </div>
                        <a
                          :href="searchResult.url"
                          target="_blank"
                          class="result-url"
                          >{{ searchResult.url }}</a
                        >
                      </div>
                    </div>
                  </div>
                </el-collapse-item>
              </el-collapse>
            </div>
          </el-card>
        </div>
      </div>
    </div>

    <!-- 不值得核查状态 -->
    <div
      v-else-if="factCheckData && factCheckData.status === 'completed'"
      class="not-worth-checking"
    >
      <el-card>
        <div class="not-worth-header">
          <el-icon><InfoFilled /></el-icon>
          <span>该视频内容不需要进行事实核查</span>
        </div>
        <div class="reason-text">
          {{
            factCheckData.reason ||
            '该内容没有包含需要核查的重要事实断言。'
          }}
        </div>
        <el-button
          type="primary"
          @click="generateFactCheck"
          size="small"
          class="retry-button"
        >
          重新尝试核查
        </el-button>
      </el-card>
    </div>

    <!-- 没有事实核查数据时的初始状态 -->
    <div v-else>
      <el-result
        icon="info"
        title="暂无事实核查结果"
        sub-title="系统尚未对此视频进行事实核查，点击下方按钮开始核查。"
      >
        <template #extra>
          <el-button type="primary" @click="generateFactCheck">
            开始事实核查
          </el-button>
        </template>
      </el-result>
    </div>
  </div>
</template>

<style scoped>
/* 事实核查样式 */
.factcheck-container {
  height: 100%;
  overflow: auto;
}

.factcheck-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.factcheck-timestamp {
  font-size: 14px;
  color: #909399;
  font-style: italic;
}

.section-heading {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 3rem;
  padding-bottom: 3rem;
}

.status-card {
  margin-bottom: 16px;
  border-top-width: 4px;
}

.status-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.status-info {
  display: flex;
  align-items: center;
}

.status-tag {
  margin-right: 12px;
}

.worth-checking-label {
  background-color: #f0f9eb;
  color: #67c23a;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 500;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
}

.reason-text {
  color: #606266;
  background-color: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  margin-top: 8px;
  font-style: italic;
}

.claims-heading {
  font-size: 16px;
  margin: 24px 0 16px;
}

.summary-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 6px;
}

.stat-item {
  flex: 1;
  text-align: center;
  padding: 12px;
  border-radius: 4px;
  background-color: white;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
}

.stat-label {
  margin-top: 4px;
  font-size: 14px;
}

.claims-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 24px;
}

.claim-card {
  border-left-width: 4px;
  border-left-style: solid;
}

.claim-true {
  border-left-color: #67c23a;
}

.claim-false {
  border-left-color: #f56c6c;
}

.claim-uncertain {
  border-left-color: #909399;
}

.claim-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
  gap: 8px;
}

.claim-text {
  font-weight: 500;
  line-height: 1.5;
}

.claim-body {
  margin-top: 8px;
}

.conclusion-text {
  margin-bottom: 16px;
  line-height: 1.6;
  background-color: #f8f9fa;
  padding: 12px;
  border-radius: 4px;
}

.search-details {
  padding: 8px 0;
}

.search-info {
  margin-bottom: 8px;
}

.search-label {
  font-weight: 500;
  color: #606266;
  margin-right: 8px;
}

.search-value {
  color: #303133;
}

.search-results {
  margin-top: 16px;
}

.results-heading {
  font-weight: 500;
  margin-bottom: 8px;
}

.search-result-item {
  padding: 12px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  margin-bottom: 8px;
  background-color: white;
}

.result-title {
  margin-bottom: 6px;
  color: #303133;
}

.result-snippet {
  font-size: 14px;
  color: #606266;
  margin-bottom: 6px;
}

.result-url {
  font-size: 12px;
  color: #909399;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.not-worth-checking {
  max-width: 800px;
  margin: 0 auto;
}

.not-worth-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 12px;
}

.retry-button {
  margin-top: 16px;
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

:deep(.markdown-body p) {
  margin-bottom: 16px;
}

:deep(.markdown-body strong) {
  color: #409eff;
  font-weight: 600;
}

:deep(.markdown-body ul, .markdown-body ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}

:deep(.markdown-body li) {
  margin-bottom: 8px;
}

:deep(.markdown-body code) {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(27, 31, 35, 0.05);
  border-radius: 3px;
}
</style>