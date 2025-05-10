<script lang="ts" setup>
import type { UploadInstance, UploadProps } from 'element-plus';

import { ref } from 'vue';

import { Page } from '@vben/common-ui';

import { Plus } from '@element-plus/icons-vue';
import {
  ElButton,
  ElCard,
  ElDivider,
  ElInput,
  ElMessage,
  ElNotification,
  ElProgress,
  ElUpload,
} from 'element-plus';

const uploadRef = ref<UploadInstance>();
const videoUrl = ref<string>('');
const videoFile = ref<File | null>(null);
const isUploading = ref<boolean>(false);
const isProcessing = ref<boolean>(false);
const processProgress = ref<number>(0);
const summaryResult = ref<string>('');
const transcriptResult = ref<string>('');
const activeResult = ref<'' | 'summary' | 'transcript'>('');

// 上传前验证
const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  if (!file) {
    ElMessage.error('未选择文件!');
    return false;
  }
  const isVideo = file.type.startsWith('video/');
  if (!isVideo) {
    ElMessage.error('只能上传视频文件!');
    return false;
  }

  const isLt100M = file.size / 1024 / 1024 < 100;
  if (!isLt100M) {
    ElMessage.error('视频大小不能超过 100MB!');
    return false;
  }

  // 存储选择的文件但不自动上传
  videoFile.value = file;

  // 创建本地预览URL
  videoUrl.value = URL.createObjectURL(file);
  return false; // 阻止自动上传
};

// 上传视频
const handleUpload = async () => {
  if (!videoFile.value) {
    ElMessage.warning('请先选择视频文件');
    return;
  }

  isUploading.value = true;
  try {
    // 模拟上传过程
    await new Promise((resolve) => {
      let progress = 0;
      const timer = setInterval(() => {
        progress += 10;
        processProgress.value = progress;
        if (progress >= 100) {
          clearInterval(timer);
          resolve(true);
        }
      }, 300);
    });

    ElMessage.success('视频上传成功');
    ElNotification({
      title: '上传成功',
      message: `文件 ${videoFile.value.name} 已成功上传`,
      type: 'success',
    });
  } catch {
    ElMessage.error('上传失败，请重试');
  } finally {
    isUploading.value = false;
    processProgress.value = 0;
  }
};

// 视频总结
const generateSummary = async () => {
  if (!videoFile.value) {
    ElMessage.warning('请先上传视频');
    return;
  }

  isProcessing.value = true;
  activeResult.value = 'summary';
  summaryResult.value = '';

  try {
    // 模拟处理过程
    await new Promise((resolve) => {
      let progress = 0;
      const timer = setInterval(() => {
        progress += 5;
        processProgress.value = progress;
        if (progress >= 100) {
          clearInterval(timer);
          resolve(true);
        }
      }, 200);
    });

    // 模拟API返回结果
    summaryResult.value = `这是关于"${videoFile.value.name}"的视频总结：\n\n该视频主要讲述了人工智能在现代社会中的应用和发展趋势。视频从AI的基本概念出发，介绍了机器学习、深度学习和神经网络的核心原理。随后分析了AI在医疗、教育、交通和环保等领域的创新应用案例。最后，视频探讨了AI发展面临的伦理挑战和未来展望。`;

    ElMessage.success('视频总结生成成功');
  } catch {
    ElMessage.error('处理失败，请重试');
  } finally {
    isProcessing.value = false;
    processProgress.value = 0;
  }
};

// 提取视频文字
const extractTranscript = async () => {
  if (!videoFile.value) {
    ElMessage.warning('请先上传视频');
    return;
  }

  isProcessing.value = true;
  activeResult.value = 'transcript';
  transcriptResult.value = '';

  try {
    // 模拟处理过程
    await new Promise((resolve) => {
      let progress = 0;
      const timer = setInterval(() => {
        progress += 4;
        processProgress.value = progress;
        if (progress >= 100) {
          clearInterval(timer);
          resolve(true);
        }
      }, 150);
    });

    // 模拟API返回结果
    transcriptResult.value = `[00:00:05] 欢迎观看本视频，今天我们将探讨人工智能的发展与应用。
[00:01:20] 人工智能，简称AI，是计算机科学的一个分支，致力于创造能够模拟人类思维过程的智能机器。
[00:03:45] 机器学习是AI的核心技术之一，它通过算法使计算机能够从数据中学习。
[00:05:30] 深度学习则是机器学习的一个子领域，它基于人工神经网络的结构和功能。
[00:08:15] 在医疗领域，AI已经应用于疾病诊断、药物研发和个性化治疗方案。
[00:12:40] 教育方面，AI辅助学习系统能够为学生提供个性化的学习体验。
[00:15:55] 在交通领域，自动驾驶技术正逐步改变我们的出行方式。
[00:19:20] 环保方面，AI通过优化能源使用和监测污染来帮助应对气候变化。
[00:22:35] 然而，AI的快速发展也带来了隐私、安全和就业等方面的担忧。
[00:25:50] 展望未来，负责任的AI发展将需要技术创新与伦理考量并重。`;

    ElMessage.success('视频文字提取成功');
  } catch {
    ElMessage.error('处理失败，请重试');
  } finally {
    isProcessing.value = false;
    processProgress.value = 0;
  }
};

const handleRemove = () => {
  videoUrl.value = '';
  videoFile.value = null;
  summaryResult.value = '';
  transcriptResult.value = '';
  activeResult.value = '';
};
</script>

<template>
  <Page
    description="支持视频上传、生成视频总结和提取视频文字内容"
    title="视频处理"
  >
    <div>
      <ElCard>
        <template #header>
          <div>
            <span>视频上传</span>
            <ElButton
              @click="handleUpload"
              :disabled="!videoFile || isUploading"
              :loading="isUploading"
            >
              上传视频
            </ElButton>
          </div>
        </template>

        <ElUpload
          ref="uploadRef"
          drag
          :auto-upload="false"
          :show-file-list="false"
          :before-upload="beforeUpload"
        >
          <div v-if="!videoUrl">
            <el-icon>
              <Plus />
            </el-icon>
            <div>点击或拖拽视频文件到此区域上传</div>
          </div>
          <div v-else>
            <video :src="videoUrl" controls></video>
            <div>
              {{ videoFile ? videoFile.name : '' }}
              <ElButton @click.stop="handleRemove">删除</ElButton>
            </div>
          </div>
        </ElUpload>

        <ElProgress
          v-if="isUploading || isProcessing"
          :percentage="processProgress"
        />

        <ElDivider />

        <div>
          <ElButton
            @click="generateSummary"
            :disabled="!videoUrl || isProcessing"
          >
            生成视频总结
          </ElButton>
          <ElButton
            @click="extractTranscript"
            :disabled="!videoUrl || isProcessing"
          >
            提取视频文字
          </ElButton>
        </div>
      </ElCard>

      <ElCard v-if="activeResult">
        <template #header>
          <div>
            <span>{{
              activeResult === 'summary' ? '视频总结结果' : '视频文字稿'
            }}</span>
            <ElButton
              @click="
                navigator.clipboard.writeText(
                  activeResult === 'summary' ? summaryResult : transcriptResult,
                )
              "
            >
              复制
            </ElButton>
          </div>
        </template>

        <ElInput
          type="textarea"
          :model-value="
            activeResult === 'summary' ? summaryResult : transcriptResult
          "
          readonly
        />
      </ElCard>
    </div>
  </Page>
</template>
