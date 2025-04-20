<script lang="ts" setup>
import { ref } from 'vue';
import { Page } from '@vben/common-ui';
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
  ElMenu,
  ElMenuItem, // 添加菜单相关组件
  ElInfiniteScroll,
  ElScrollbar,
} from 'element-plus';

const videoSrc = ref('https://www.example.com/sample.mp4');

const subtitlesData = ref({
  chunks: [
    {
      text: 'Do you know what the greatest possession of a rich man is?',
      timestamp: [0.0, 4.0],
    },
    {
      text: 'The greatest wealth of the rich is not the luxury car, the mansion and the balance of the card, nor is it any random',
      timestamp: [4.0, 11.12],
    },
    {
      text: 'cultural relics and works of art. The greatest wealth of the rich is that of the poor.',
      timestamp: [11.12, 16.84],
    },
    {
      text: 'The way the world works is that 80% of hardworking people feed 20% of unearned people. The purpose',
      timestamp: [17.84, 25.14],
    },
    {
      text: 'of the 80% is to help the other 20% live better.',
      timestamp: [25.14, 29.0],
    },
    {
      text: 'And then the 20% are constantly designing rules to limit and control the 80%.',
      timestamp: [29.0, 36.0],
    },
    {
      text: 'Read books to learn, you need to improve your cognition.',
      timestamp: [36.0, 39.22],
    },
  ],
  text: 'Do you know what the greatest possession of a rich man is? The greatest wealth of the rich is not the luxury car, the mansion and the balance of the card, nor is it any random cultural relics and works of art. The greatest wealth of the rich is that of the poor. The way the world works is that 80% of hardworking people feed 20% of unearned people. The purpose of the 80% is to help the other 20% live better. And then the 20% are constantly designing rules to limit and control the 80%. Read books to learn, you need to improve your cognition.',
});

// 添加导航菜单激活状态
const activeTab = ref('summary');
// 添加缺失的时间戳格式化函数
const formatTimestamp = (seconds: number | undefined): string => {
  if (seconds === undefined) return '00:00';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};
// 菜单选择处理函数
const handleTabChange = (key: string) => {
  activeTab.value = key;
};
</script>

<template>
  <Page description="视频总结,字幕列表等" title="视频内容分析">
    <!-- 设置容器高度并使用flex布局 -->
    <div class="flex h-[calc(100vh-160px)] gap-4">
      <!-- 左侧卡片 - 占50%且高度100% -->
      <el-card class="h-full w-[35%] overflow-hidden shadow-md">
        <div class="flex h-full flex-col">
          <video controls :src="videoSrc" class="w-full flex-1"></video>
        </div>
      </el-card>

      <!-- 右侧卡片 - 占50%且高度100% -->
      <el-card class="h-full w-[65%] overflow-hidden shadow-md">
        <div class="flex h-full flex-col">
          <!-- 顶部导航菜单 -->
          <el-menu
            :default-active="activeTab"
            class="analysis-tabs border-0"
            mode="horizontal"
            @select="handleTabChange"
          >
            <el-menu-item index="summary">总结摘要</el-menu-item>
            <el-menu-item index="subtitles">字幕列表</el-menu-item>
            <el-menu-item index="process">分析过程</el-menu-item>
            <el-menu-item index="threat">威胁报告</el-menu-item>
          </el-menu>

          <!-- 内容区域，可滚动 -->
          <div class="flex-1 overflow-auto p-4">
            <!-- 总结摘要内容 -->
            <div v-if="activeTab === 'summary'">
              <h3 class="mb-4 text-lg font-medium">视频总结摘要</h3>
              <p class="text-gray-700">
                这段视频主要讲述了...这里是视频的总体摘要内容，可以包括关键点、主题和整体评估。
              </p>
              <!-- 可以添加更多内容 -->
            </div>
            <!-- 字幕列表内容 -->
            <div v-else-if="activeTab === 'subtitles'" class="h-full">

              <!-- 整体布局容器 -->
              <div class="flex h-[calc(100%-2rem)] flex-col">
                <!-- 完整文本区域 -->
                <h4 class="font-medium">完整文本:</h4>
                <div
                  class="mb-4 rounded-lg border border-gray-200 bg-gray-50 p-4"
                  style="height: 120px"
                >

                  <el-scrollbar height="75px">
                    <p class="leading-relaxed text-gray-700">
                      {{ subtitlesData.text }}
                    </p>
                  </el-scrollbar>
                </div>

                <!-- 字幕列表区域 -->
                <div class="flex flex-1 flex-col">
                  <div class="mb-2 flex items-center justify-between">
                    <h4 class="font-medium">字幕时间轴:</h4>
                    <span class="text-xs text-gray-500"
                      >共 {{ subtitlesData.chunks.length }} 个片段</span
                    >
                  </div>
                  <el-scrollbar
                    height="50vh"
                    class="subtitle-scrollbar rounded border border-gray-100"
                  >
                    <div class="p-1">
                      <div
                        v-for="(chunk, index) in subtitlesData.chunks"
                        :key="index"
                        class="m-3 rounded bg-gray-50 p-3 transition-colors hover:bg-gray-100"
                      >
                        <div class="mb-1 text-xs text-gray-500">
                          {{ formatTimestamp(chunk.timestamp[0]) }} -
                          {{ formatTimestamp(chunk.timestamp[1]) }}
                        </div>
                        <div class="text-gray-800">{{ chunk.text }}</div>
                      </div>
                    </div>
                  </el-scrollbar>
                </div>
              </div>
            </div>

            <!-- 分析过程内容 -->
            <div v-else-if="activeTab === 'process'">
              <h3 class="mb-4 text-lg font-medium">视频分析过程</h3>
              <div class="space-y-4">
                <div
                  v-for="i in 3"
                  :key="i"
                  class="border-l-4 border-blue-500 py-2 pl-4"
                >
                  <div class="font-medium">分析阶段 {{ i }}</div>
                  <div class="mt-1 text-gray-600">
                    执行了内容识别、语义分析等处理步骤...
                  </div>
                </div>
              </div>
            </div>

            <!-- 威胁报告内容 -->
            <div v-else-if="activeTab === 'threat'">
              <h3 class="mb-4 text-lg font-medium">内容威胁分析报告</h3>
              <div class="mb-4 rounded-lg bg-green-50 p-4">
                <div class="flex items-center">
                  <el-tag type="success" class="mr-2">安全</el-tag>
                  <span class="font-medium">未检测到明显威胁</span>
                </div>
                <p class="mt-2 text-gray-600">
                  此视频内容未发现违规或敏感内容，可以安全发布。
                </p>
              </div>
            </div>
          </div>
        </div>
      </el-card>
    </div>
  </Page>
</template>

<style scoped>
/* 自定义菜单样式 */

:deep(.el-menu-item) {
  height: 48px;
  line-height: 48px;
}

:deep(.el-menu--horizontal > .el-menu-item.is-active) {
  border-bottom: 2px solid #409eff;
  font-weight: 500;
}
</style>
