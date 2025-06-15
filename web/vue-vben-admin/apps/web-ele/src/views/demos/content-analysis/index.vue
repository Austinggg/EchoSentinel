<script lang="ts" setup>
import type { UploadProps, UploadUserFile } from 'element-plus';

import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';

import { Page } from '@vben/common-ui';

import { 
  Promotion, 
  UploadFilled, 
  VideoCamera,
  Monitor,
  Warning,
  CircleCheck,
  Link as LinkIcon
} from '@element-plus/icons-vue';
import { 
  ElButton, 
  ElIcon, 
  ElInput, 
  ElMessage, 
  ElUpload,
  ElTabs,
  ElTabPane,
  ElCard,
  ElAlert,
  ElDivider,
  ElImage,
  ElTag
} from 'element-plus';

import { storeVideoByUrl } from '#/api/video';

const router = useRouter();

// 平台配置
const platforms = [
  {
    id: 'douyin',
    name: '抖音',
    icon: '/icons/tiktok.png',
    placeholder: '输入抖音视频链接进行数字人检测分析',
    example: 'https://www.douyin.com/video/7xxx',
    urlPattern: /douyin\.com\/video\/([^?]+)/,
    description: '支持抖音短视频数字人检测',
    color: '#fe2c55'
  },
  {
    id: 'tiktok', 
    name: 'TikTok',
    icon: '/icons/tiktok.png',
    placeholder: '输入TikTok视频链接进行数字人检测分析',
    example: 'https://www.tiktok.com/@username/video/7xxx',
    urlPattern: /tiktok\.com\/@[^\/]+\/video\/([^?]+)/,
    description: '支持TikTok短视频数字人检测',
    color: '#000000'
  },
  {
    id: 'bilibili',
    name: 'Bilibili',
    icon: '/icons/bilibili.png', 
    placeholder: '输入B站视频链接进行数字人检测分析',
    example: 'https://www.bilibili.com/video/BVxxx',
    urlPattern: /bilibili\.com\/video\/([^?]+)/,
    description: '支持B站视频数字人检测',
    color: '#00aeec'
  }
];

// 当前状态
const activePlatform = ref('douyin');
const videoUrlInput = ref('');
const isProcessing = ref(false);
const uploadAction = '/api/videos/upload';
const uploadRef = ref();
const fileList = ref<UploadUserFile[]>([]);

// 计算属性 - 当前平台信息
const activePlatformInfo = computed(() => {
  return platforms.find((p) => p.id === activePlatform.value) || platforms[0];
});

// 处理平台切换
const handlePlatformChange = (platform: string) => {
  videoUrlInput.value = '';
};

// 检测功能配置
const detectionFeatures = [
  {
    icon: 'VideoCamera',
    title: '面部真实性检测',
    description: '深度分析面部特征，识别AI生成痕迹'
  },
  {
    icon: 'Monitor', 
    title: '躯体异常检测',
    description: '检测肢体动作、比例等生物学异常'
  },
  {
    icon: 'CircleCheck',
    title: '整体特征评估', 
    description: '综合时空一致性和生成痕迹分析'
  },
  {
    icon: 'Warning',
    title: '智能风险评估',
    description: '多维度融合输出数字人概率评分'
  }
];

// 处理文件上传相关函数
const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
  console.log('File changed:', uploadFile, uploadFiles);
  fileList.value = uploadFiles.slice(-3);
};

const handleExceed: UploadProps['onExceed'] = (uploadFiles) => {
  ElMessage.warning(
    `最多只能上传3个文件，当前已选择${uploadFiles.length}个文件`,
  );
};

const handleSuccess: UploadProps['onSuccess'] = (response) => {
  if (response.code === 0) {
    ElMessage.success('视频上传成功，正在进行数字人检测分析...');
    
    // 跳转到分析结果页面
    if (response.data && response.data.video_id) {
      router.push(`/demos/content-analysis/detail/${response.data.video_id}`);
    }
  } else {
    ElMessage.error(response.message || '上传失败');
  }
};

const handleError: UploadProps['onError'] = (error, uploadFile) => {
  console.error('Upload error:', error, uploadFile);
  ElMessage.error('视频上传失败，请重试');
};

// 处理URL提交
const handleUrlSubmit = async () => {
  if (!videoUrlInput.value) {
    ElMessage.warning('请输入视频链接');
    return;
  }

  // 验证URL格式
  const urlMatch = videoUrlInput.value.match(activePlatformInfo.value.urlPattern);
  if (!urlMatch) {
    ElMessage.error(`请输入有效的${activePlatformInfo.value.name}视频链接`);
    return;
  }

  try {
    isProcessing.value = true;
    ElMessage.info('正在获取视频并启动数字人检测分析...');
    
    const result = await storeVideoByUrl(videoUrlInput.value);
    ElMessage.success('视频链接解析成功，数字人检测分析已启动！');

    // 清空输入
    videoUrlInput.value = '';

    // 跳转到分析结果页面
    if (result && result.video_id) {
      router.push(`/demos/content-analysis/detail/${result.video_id}`);
    }

    console.log('URL处理结果:', result);
  } catch (error) {
    console.error('URL处理错误:', error);
    ElMessage.error('视频链接处理失败，请检查链接是否有效');
  } finally {
    isProcessing.value = false;
  }
};

const handleSubmit = async () => {
  if (fileList.value.length === 0 && !videoUrlInput.value) {
    ElMessage.warning('请上传视频文件或输入视频链接进行数字人检测');
    return;
  }

  isProcessing.value = true;
  try {
    // 处理URL提交
    if (videoUrlInput.value) {
      await handleUrlSubmit();
    }

    // 处理文件上传
    if (fileList.value.length > 0 && uploadRef.value) {
      // 手动触发上传
      uploadRef.value.submit();
    } else if (!videoUrlInput.value) {
      ElMessage.success('数字人检测分析已启动！');
    }
  } catch (error) {
    console.error('处理失败:', error);
    ElMessage.error('操作失败，请重试');
  } finally {
    isProcessing.value = false;
  }
};
</script>

<template>
  <Page
    description="上传视频文件或输入视频链接，启动AI数字人智能检测分析系统"
    title="🤖 数字人检测分析"
  >
    <!-- 平台选择和URL输入区域 -->
    <ElCard class="input-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <ElIcon size="20"><LinkIcon /></ElIcon>
          <span>选择平台并输入视频链接</span>
        </div>
      </template>

      <!-- 平台选择标签 -->
      <ElTabs 
        v-model="activePlatform" 
        @tab-change="handlePlatformChange"
        type="border-card"
        class="platform-tabs"
      >
        <ElTabPane
          v-for="platform in platforms"
          :key="platform.id"
          :label="platform.name"
          :name="platform.id"
        >
          <template #label>
            <div class="tab-label">
              <ElImage :src="platform.icon" class="platform-icon" />
              <span>{{ platform.name }}</span>
            </div>
          </template>
        </ElTabPane>
      </ElTabs>

      <!-- URL输入框 -->
      <div class="url-input-section">
        <ElInput
          v-model="videoUrlInput"
          :placeholder="activePlatformInfo.placeholder"
          class="url-input"
          clearable
          :disabled="isProcessing"
          @keyup.enter="handleUrlSubmit"
          size="large"
        >
          <template #prefix>
            <ElIcon :color="activePlatformInfo.color"><LinkIcon /></ElIcon>
          </template>
          <template #append>
            <ElButton 
              type="primary" 
              @click="handleUrlSubmit" 
              :loading="isProcessing"
              :style="{ backgroundColor: activePlatformInfo.color, borderColor: activePlatformInfo.color }"
            >
              <ElIcon><VideoCamera /></ElIcon>
              {{ isProcessing ? '检测中...' : '开始检测' }}
            </ElButton>
          </template>
        </ElInput>
        
        <div class="platform-tips">
          <ElAlert
            :title="`${activePlatformInfo.name}视频链接示例: ${activePlatformInfo.example}`"
            type="info"
            :closable="false"
            show-icon
          />
        </div>
      </div>
    </ElCard>

    <ElDivider>
      <ElIcon><UploadFilled /></ElIcon>
      <span style="margin-left: 8px;">或上传本地视频文件</span>
    </ElDivider>

    <!-- 文件上传区域 -->
    <ElCard class="upload-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <ElIcon size="20"><UploadFilled /></ElIcon>
          <span>本地视频文件上传</span>
          <ElTag type="warning" size="small">支持多文件</ElTag>
        </div>
      </template>

      <ElUpload
        ref="uploadRef"
        v-model:file-list="fileList"
        class="digital-human-upload"
        drag
        :action="uploadAction"
        :auto-upload="false"
        multiple
        :limit="3"
        :on-exceed="handleExceed"
        :on-change="handleChange"
        :on-success="handleSuccess"
        :on-error="handleError"
        name="file"
      >
        <div class="upload-content">
          <ElIcon class="upload-icon" size="48" color="#409eff">
            <VideoCamera />
          </ElIcon>
          <div class="upload-text">
            <div class="upload-title">
              <strong>拖拽视频文件到此处或</strong>
              <em style="color: #409eff; margin-left: 4px;">点击上传</em>
            </div>
            <div class="upload-subtitle">
              🤖 上传后将自动进行数字人检测分析
            </div>
          </div>
        </div>
        
        <template #tip>
          <div class="upload-tips">
            <div class="tip-row">
              <ElIcon color="#67c23a"><CircleCheck /></ElIcon>
              <span>单个文件大小 ≤ 2GB</span>
            </div>
            <div class="tip-row">
              <ElIcon color="#67c23a"><CircleCheck /></ElIcon>
              <span>支持格式：MP4, MOV, AVI, MKV, WEBM, M4A, WAV</span>
            </div>
            <div class="tip-row">
              <ElIcon color="#e6a23c"><Warning /></ElIcon>
              <span>建议上传时长 ≤ 10分钟的视频以获得最佳检测效果</span>
            </div>
          </div>
        </template>
      </ElUpload>
    </ElCard>

    <!-- 提交按钮 -->
    <div class="submit-section">
      <ElButton 
        type="primary" 
        size="large"
        :loading="isProcessing" 
        @click="handleSubmit"
        class="submit-button"
      >
        <ElIcon size="18">
          <VideoCamera />
        </ElIcon>
        <span>{{ isProcessing ? '🤖 数字人检测进行中...' : '🚀 启动数字人检测' }}</span>
      </ElButton>
      
      <div class="submit-tips">
        <ElIcon color="#909399"><Monitor /></ElIcon>
        <span>检测过程通常需要 5-15 分钟，请耐心等待</span>
      </div>
    </div>
  </Page>
</template>

<style scoped>
/* 卡片样式 */
.input-card,
.upload-card {
  margin-bottom: 24px;
  border-radius: 12px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  font-size: 16px;
}

/* 平台标签样式 */
.platform-tabs {
  margin-bottom: 20px;
}

.platform-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}

.tab-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.platform-icon {
  width: 20px;
  height: 20px;
}

/* URL输入区域 */
.url-input-section {
  margin-top: 16px;
}

.url-input {
  margin-bottom: 12px;
}

.platform-tips {
  margin-top: 12px;
}

/* 上传区域样式 */
.digital-human-upload {
  width: 100%;
}

.digital-human-upload :deep(.el-upload-dragger) {
  border: 2px dashed #409eff;
  border-radius: 12px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
  transition: all 0.3s ease;
}

.digital-human-upload :deep(.el-upload-dragger:hover) {
  border-color: #66b1ff;
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 40px 20px;
}

.upload-icon {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

.upload-text {
  text-align: center;
}

.upload-title {
  font-size: 16px;
  margin-bottom: 8px;
  color: #303133;
}

.upload-subtitle {
  font-size: 14px;
  color: #606266;
}

.upload-tips {
  margin-top: 16px;
  padding: 16px;
  background: #f0f9ff;
  border-radius: 8px;
  border: 1px solid #e1f5fe;
}

.tip-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 13px;
  color: #606266;
}

.tip-row:last-child {
  margin-bottom: 0;
}

/* 提交区域 */
.submit-section {
  text-align: center;
  margin-top: 32px;
}

.submit-button {
  height: 50px;
  padding: 0 32px;
  font-size: 16px;
  border-radius: 25px;
  background: linear-gradient(135deg, #409eff 0%, #66b1ff 100%);
  border: none;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
  transition: all 0.3s ease;
}

.submit-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(64, 158, 255, 0.4);
}

.submit-tips {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 12px;
  font-size: 13px;
  color: #909399;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .upload-content {
    padding: 24px 16px;
  }
  
  .submit-button {
    width: 100%;
    max-width: 300px;
  }
}
</style>