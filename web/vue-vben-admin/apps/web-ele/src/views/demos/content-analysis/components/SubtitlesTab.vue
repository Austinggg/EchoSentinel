<script lang="ts" setup>
import { CopyDocument } from '@element-plus/icons-vue';
import { ElButton, ElScrollbar } from 'element-plus';

// 定义组件接收的props
const props = defineProps({
  subtitlesData: {
    type: Object,
    required: true,
    default: () => ({ chunks: [], text: '' })
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['copy-text']);

// 复制文本函数，调用父组件的方法
const copySubtitleText = () => {
  emit('copy-text');
};

// 格式化时间戳的方法
const formatTimestamp = (seconds) => {
  if (seconds === undefined) return '00:00';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};
</script>

<template>
  <div class="subtitles-container">
    <!-- 整体布局容器 -->
    <div class="subtitles-layout">
      <!-- 完整文本区域 -->
      <div class="section-header">
        <h4 class="section-title">完整文本:</h4>
        <el-button
          size="small"
          type="primary"
          @click="copySubtitleText"
          :icon="CopyDocument"
          text
        >
          复制文本
        </el-button>
      </div>
      <div class="text-preview-container">
        <el-scrollbar height="75px">
          <p class="text-preview-content">
            {{ subtitlesData.text }}
          </p>
        </el-scrollbar>
      </div>

      <!-- 字幕列表区域 -->
      <div class="subtitles-list-container">
        <div class="section-header">
          <h4 class="section-title">字幕时间轴:</h4>
          <span class="subtitle-count">
            共 {{ subtitlesData.chunks.length }} 个片段
          </span>
        </div>
        <el-scrollbar height="65vh" class="subtitle-scrollbar">
          <div style="padding: 0.25rem">
            <div
              v-for="(chunk, index) in subtitlesData.chunks"
              :key="index"
              class="subtitle-chunk"
            >
              <div class="subtitle-timestamp">
                {{ formatTimestamp(chunk.timestamp[0]) }} -
                {{ formatTimestamp(chunk.timestamp[1]) }}
              </div>
              <div class="subtitle-text">{{ chunk.text }}</div>
            </div>
          </div>
        </el-scrollbar>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 复制原组件中相关的样式 */
.subtitles-container {
  height: 100%;
}

.subtitles-layout {
  display: flex;
  height: calc(100% - 2rem);
  flex-direction: column;
}

.section-header {
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.section-title {
  font-weight: 500;
}

.subtitle-count {
  font-size: 0.75rem;
  color: #6b7280;
}

.text-preview-container {
  margin-bottom: 1rem;
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
  background-color: #f9fafb;
  padding: 1rem;
  height: 120px;
}

.text-preview-content {
  line-height: 1.625;
  color: #374151;
}

.subtitles-list-container {
  display: flex;
  flex: 1;
  flex-direction: column;
}

.subtitle-scrollbar {
  height: calc(65vh - 200px) !important;
  border: 1px solid #f3f4f6;
  border-radius: 0.25rem;
}

.subtitle-chunk {
  margin: 0.75rem;
  border-radius: 0.25rem;
  background-color: #f9fafb;
  padding: 0.75rem;
  transition: background-color 0.2s;
}

.subtitle-chunk:hover {
  background-color: #f3f4f6;
}

.subtitle-timestamp {
  margin-bottom: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

.subtitle-text {
  color: #1f2937;
}
</style>