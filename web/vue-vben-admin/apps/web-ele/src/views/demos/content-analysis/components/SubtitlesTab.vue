<script lang="ts" setup>
import { CopyDocument, VideoPlay, Edit } from '@element-plus/icons-vue';
import { ElButton, ElScrollbar, ElMessage } from 'element-plus';
import { computed, ref, watch, nextTick } from 'vue';

// 定义组件接收的props
const props = defineProps({
  subtitlesData: {
    type: Object,
    required: true,
    default: () => ({ chunks: [], text: '' })
  },
  currentTime: {
    type: Number,
    default: 0
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['copy-text', 'seek-to-time']);

// 字幕列表容器引用
const subtitleScrollbar = ref(null);
const activeChunkRef = ref(null);
const chunkRefs = ref([]);

// 设置字幕项的引用
const setChunkRef = (el, index) => {
  if (el) {
    chunkRefs.value[index] = el;
  }
};

// 新增：处理后的文本状态
const processedText = ref('');
const isProcessingText = ref(false);

// 检测文本是否缺少标点符号
const needsPunctuation = computed(() => {
  const text = props.subtitlesData?.text || '';
  if (!text) return false;
  
  // 检查是否包含常见的中文标点符号
  const punctuationRegex = /[。！？；：，、]/;
  const hasPunctuation = punctuationRegex.test(text);
  
  // 如果文本长度超过20字符且没有标点，则认为需要添加标点
  return text.length > 20 && !hasPunctuation;
});

// 显示的文本（优先显示处理后的文本）
const displayText = computed(() => {
  return processedText.value || props.subtitlesData?.text || '';
});

// 添加标点符号的函数
const addPunctuation = (text) => {
  if (!text) return text;
  
  // 基本的标点添加规则
  let processedText = text
    // 在句子结尾添加句号（检测到语气词或完整意思）
    .replace(/([好不好|对不对|是不是|知道吗|明白吗|懂吗])$/g, '$1？')
    .replace(/([了|啊|呀|吧|呢|哦|哈])$/g, '$1。')
    // 在连词前添加逗号
    .replace(/(但是|不过|然后|接着|还有|而且|所以|因此|因为|由于)/g, '，$1')
    // 在转折处添加逗号
    .replace(/([的是|就是|问题是|事情是])([^，。！？])/g, '$1，$2')
    // 在列举中添加顿号
    .replace(/([和|与|还有|以及])([^，。！？、])/g, '$1、$2')
    // 在句子中间的停顿处添加逗号
    .replace(/([A-Za-z0-9\u4e00-\u9fa5]{8,}?)([不要|尽量|每个|身体|人家])/g, '$1，$2')
    // 确保句子结尾有标点
    .replace(/([^。！？；：，、])$/g, '$1。');
  
  return processedText;
};

// 智能添加标点符号
const smartAddPunctuation = async () => {
  try {
    isProcessingText.value = true;
    const originalText = props.subtitlesData?.text || '';
    
    if (!originalText) {
      ElMessage.warning('没有文本可以处理');
      return;
    }
    
    // 使用本地规则添加标点
    const processed = addPunctuation(originalText);
    processedText.value = processed;
    
    ElMessage.success('标点符号已添加');
  } catch (error) {
    console.error('添加标点失败:', error);
    ElMessage.error('添加标点失败');
  } finally {
    isProcessingText.value = false;
  }
};

// 重置到原始文本
const resetText = () => {
  processedText.value = '';
  ElMessage.info('已重置为原始文本');
};

// 复制文本函数（复制处理后的文本）
const copySubtitleText = () => {
  const textToCopy = displayText.value;
  if (textToCopy) {
    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        ElMessage.success('文本已复制到剪贴板');
      })
      .catch(() => {
        // 如果复制失败，也通知父组件
        emit('copy-text');
      });
  } else {
    ElMessage.warning('没有可复制的文本');
  }
};

// 跳转到指定时间
const seekToTime = (startTime) => {
  emit('seek-to-time', startTime);
};

// 格式化时间戳的方法
const formatTimestamp = (seconds) => {
  if (seconds === undefined) return '00:00';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

// 计算当前活跃的字幕片段
const activeChunkIndex = computed(() => {
  if (!props.subtitlesData?.chunks?.length || props.currentTime <= 0) {
    return -1;
  }
  
  return props.subtitlesData.chunks.findIndex(chunk => {
    const [start, end] = chunk.timestamp;
    return props.currentTime >= start && props.currentTime <= end;
  });
});

// 监听活跃片段变化，自动滚动到对应位置
watch(activeChunkIndex, async (newIndex, oldIndex) => {
  console.log('活跃片段变化:', newIndex, '当前时间:', props.currentTime);
  if (newIndex >= 0 && newIndex !== oldIndex && subtitleScrollbar.value) {
    await nextTick();
    
    // 使用 chunkRefs 直接获取对应的元素
    const activeElement = chunkRefs.value[newIndex];
    if (activeElement) {
      // 获取滚动容器
      const scrollContainer = subtitleScrollbar.value.$refs.wrap;
      if (scrollContainer) {
        const elementTop = activeElement.offsetTop;
        const elementHeight = activeElement.offsetHeight;
        const containerHeight = scrollContainer.clientHeight;
        
        // 计算目标滚动位置（将元素居中显示）
        const targetScrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2);
        
        // 平滑滚动到目标位置
        scrollContainer.scrollTo({
          top: Math.max(0, targetScrollTop),
          behavior: 'smooth'
        });
      }
    }
  }
});

// 判断字幕片段的播放状态
const getChunkStatus = (chunk, index) => {
  if (props.currentTime <= 0) {
    return 'normal';
  }
  
  const [start, end] = chunk.timestamp;
  
  if (index === activeChunkIndex.value) {
    return 'active';
  } else if (props.currentTime > end) {
    return 'passed';
  } else if (props.currentTime < start) {
    return 'upcoming';
  } else {
    return 'normal';
  }
};

// 计算字幕统计信息
const subtitleStats = computed(() => {
  const chunks = props.subtitlesData?.chunks || [];
  const totalDuration = chunks.reduce((sum, chunk) => {
    const [start, end] = chunk.timestamp;
    return sum + (end - start);
  }, 0);
  
  return {
    totalChunks: chunks.length,
    totalDuration: Math.round(totalDuration),
    averageLength: chunks.length > 0 ? Math.round(props.subtitlesData.text.length / chunks.length) : 0
  };
});
</script>

<template>
  <div class="subtitles-container">
    <!-- 整体布局容器 -->
    <div class="subtitles-layout">
      <!-- 完整文本区域 -->
      <div class="text-section">
        <div class="section-header">
          <h4 class="section-title">
            <i class="title-icon">📄</i>
            完整文本
            <span v-if="processedText" class="processed-indicator">已优化</span>
          </h4>
          <div class="text-actions">
            <!-- 标点处理按钮 -->
            <el-button
              v-if="needsPunctuation && !processedText"
              size="small"
              type="warning"
              @click="smartAddPunctuation"
              :icon="Edit"
              :loading="isProcessingText"
              text
              class="punctuation-btn"
            >
              添加标点
            </el-button>
            <el-button
              v-if="processedText"
              size="small"
              type="info"
              @click="resetText"
              text
              class="reset-btn"
            >
              重置
            </el-button>
            <el-button
              size="small"
              type="primary"
              @click="copySubtitleText"
              :icon="CopyDocument"
              text
              class="copy-btn"
            >
              复制文本
            </el-button>
          </div>
        </div>
        
        <!-- 标点提示 -->
        <div v-if="needsPunctuation && !processedText" class="punctuation-notice">
          <span class="notice-text">
            📝 检测到文本缺少标点符号，建议添加标点以提高可读性
          </span>
        </div>
        
        <div class="text-preview-container" :class="{ 'has-processed': processedText }">
          <el-scrollbar height="85px">
            <p class="text-preview-content">
              {{ displayText }}
            </p>
          </el-scrollbar>
        </div>
      </div>

      <!-- 字幕列表区域 -->
      <div class="subtitles-list-container">
        <div class="section-header">
          <h4 class="section-title">
            <i class="title-icon">⏰</i>
            字幕时间轴
          </h4>
          <div class="header-info">
            <div class="stats-compact">
              <span class="stat-compact">{{ subtitleStats.totalChunks }}段</span>
              <span class="stat-compact">{{ formatTimestamp(subtitleStats.totalDuration) }}</span>
              <span class="stat-compact">均{{ subtitleStats.averageLength }}字</span>
            </div>
            <span class="current-time" v-if="currentTime > 0">
              当前: {{ formatTimestamp(currentTime) }}
            </span>
          </div>
        </div>
        <el-scrollbar ref="subtitleScrollbar" height="calc(65vh - 200px)" class="subtitle-scrollbar">
          <div class="subtitle-list-content">
            <div
              v-for="(chunk, index) in subtitlesData.chunks"
              :key="index"
              :ref="(el) => setChunkRef(el, index)"
              class="subtitle-chunk"
              :class="getChunkStatus(chunk, index)"
              @click="seekToTime(chunk.timestamp[0])"
            >
              <div class="chunk-header">
                <div class="subtitle-timestamp">
                  <i class="time-icon">🕐</i>
                  {{ formatTimestamp(chunk.timestamp[0]) }} - 
                  {{ formatTimestamp(chunk.timestamp[1]) }}
                </div>
                <div class="chunk-actions">
                  <el-button
                    size="small"
                    type="primary"
                    :icon="VideoPlay"
                    circle
                    @click.stop="seekToTime(chunk.timestamp[0])"
                    class="play-btn"
                    title="跳转播放"
                  />
                </div>
              </div>
              <div class="subtitle-text">{{ chunk.text }}</div>
              <div class="chunk-footer">
                <span class="chunk-duration">
                  时长: {{ formatTimestamp(chunk.timestamp[1] - chunk.timestamp[0]) }}
                </span>
                <span class="chunk-index">#{{ index + 1 }}</span>
              </div>
            </div>
          </div>
        </el-scrollbar>
      </div>
    </div>
  </div>
</template>

<style scoped>
.subtitles-container {
  height: 100%;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 12px;
  padding: 20px;
}

.subtitles-layout {
  display: flex;
  height: calc(100% - 2rem);
  flex-direction: column;
  gap: 20px;
}

.text-section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
}

.section-header {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}

.section-title {
  font-weight: 600;
  color: #1e293b;
  font-size: 16px;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.title-icon {
  font-size: 18px;
}

.processed-indicator {
  font-size: 11px;
  color: #059669;
  background: #d1fae5;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}

.text-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.header-info {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.stats-compact {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 2px 8px;
  background: #f8fafc;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
}

.stat-compact {
  font-size: 11px;
  color: #64748b;
  font-weight: 500;
  padding: 2px 4px;
  background: white;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
  white-space: nowrap;
}

.current-time {
  font-size: 12px;
  color: #3b82f6;
  font-weight: 500;
  padding: 4px 8px;
  background: #dbeafe;
  border-radius: 6px;
  white-space: nowrap;
}

.punctuation-btn {
  transition: all 0.2s ease;
  color: #f59e0b;
}

.punctuation-btn:hover {
  transform: translateY(-1px);
  color: #d97706;
}

.reset-btn {
  transition: all 0.2s ease;
  color: #6b7280;
}

.reset-btn:hover {
  transform: translateY(-1px);
  color: #4b5563;
}

.copy-btn {
  transition: all 0.2s ease;
}

.copy-btn:hover {
  transform: translateY(-1px);
}

.punctuation-notice {
  margin-bottom: 12px;
  padding: 8px 12px;
  background: #fffbeb;
  border: 1px solid #fed7aa;
  border-radius: 6px;
  border-left: 4px solid #f59e0b;
}

.notice-text {
  font-size: 12px;
  color: #92400e;
  display: flex;
  align-items: center;
  gap: 6px;
}

.text-preview-container {
  border-radius: 8px;
  border: 2px solid #e2e8f0;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  padding: 16px;
  transition: all 0.2s ease;
}

.text-preview-container:hover {
  border-color: #3b82f6;
}

.text-preview-container.has-processed {
  border-color: #059669;
  background: linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%);
}

.text-preview-container.has-processed:hover {
  border-color: #047857;
}

.subtitles-list-container {
  flex: 1;
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.subtitle-scrollbar {
  border-radius: 8px;
}

.subtitle-list-content {
  padding: 8px;
}

.subtitle-chunk {
  margin-bottom: 16px;
  border-radius: 12px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  padding: 16px;
  border: 2px solid #e2e8f0;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.subtitle-chunk::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: #e2e8f0;
  transition: all 0.3s ease;
}

/* 默认状态 */
.subtitle-chunk.normal {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border-color: #e2e8f0;
}

.subtitle-chunk.normal:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  border-color: #3b82f6;
}

.subtitle-chunk.normal:hover::before {
  background: #3b82f6;
}

/* 活跃状态 */
.subtitle-chunk.active {
  background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
  border-color: #3b82f6;
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
  animation: activeGlow 2s ease-in-out infinite alternate;
}

@keyframes activeGlow {
  0% {
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
  }
  100% {
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
  }
}

.subtitle-chunk.active::before {
  background: #3b82f6;
  width: 6px;
  box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
}

/* 已播放状态 */
.subtitle-chunk.passed {
  background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
  opacity: 0.6;
  transform: none;
}

.subtitle-chunk.passed:hover {
  opacity: 0.8;
  transform: translateY(-1px);
}

.subtitle-chunk.passed::before {
  background: #64748b;
}

/* 即将播放状态 */
.subtitle-chunk.upcoming {
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  border-color: #f59e0b;
}

.subtitle-chunk.upcoming::before {
  background: #f59e0b;
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.subtitle-timestamp {
  font-size: 13px;
  color: #64748b;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}

.time-icon {
  font-size: 14px;
}

.chunk-actions {
  opacity: 0;
  transition: opacity 0.2s ease;
}

.subtitle-chunk:hover .chunk-actions {
  opacity: 1;
}

.play-btn {
  width: 28px;
  height: 28px;
  border: none;
  background: #3b82f6;
  color: white;
}

.play-btn:hover {
  background: #2563eb;
  transform: scale(1.1);
}

.subtitle-text {
  color: #1e293b;
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 12px;
  font-weight: 500;
}

.text-preview-content {
  line-height: 1.7;
  color: #475569;
  font-size: 14px;
  margin: 0;
}

.chunk-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  color: #94a3b8;
}

.chunk-duration {
  font-weight: 500;
}

.chunk-index {
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
}

/* 活跃状态下的特殊样式 */
.subtitle-chunk.active .subtitle-timestamp {
  color: #1d4ed8;
}

.subtitle-chunk.active .subtitle-text {
  color: #1e40af;
  font-weight: 600;
}

.subtitle-chunk.active .chunk-index {
  background: #3b82f6;
  color: white;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .subtitles-container {
    padding: 12px;
  }
  
  .section-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .text-actions {
    width: 100%;
    justify-content: flex-end;
  }
  
  .header-info {
    width: 100%;
    justify-content: space-between;
  }
  
  .stats-compact {
    gap: 4px;
  }
  
  .stat-compact {
    font-size: 10px;
    padding: 1px 3px;
  }
  
  .subtitle-chunk {
    padding: 12px;
  }
  
  .chunk-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .punctuation-notice {
    margin-bottom: 8px;
    padding: 6px 8px;
  }
  
  .notice-text {
    font-size: 11px;
  }
}

/* 滚动条美化 */
:deep(.el-scrollbar__bar) {
  opacity: 0.3;
}

:deep(.el-scrollbar__thumb) {
  background: #3b82f6;
  border-radius: 4px;
}

:deep(.el-scrollbar__bar.is-horizontal) {
  height: 6px;
}

:deep(.el-scrollbar__bar.is-vertical) {
  width: 6px;
}
</style>