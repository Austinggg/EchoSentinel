<script lang="ts" setup>
import { ref } from 'vue';
import { UploadFilled, Promotion } from '@element-plus/icons-vue';
import type { UploadProps, UploadUserFile } from 'element-plus';
import { Page } from '@vben/common-ui';

import { ElUpload, ElIcon, ElInput, ElButton, ElMessage } from 'element-plus';
import { storeVideoByUrl } from '#/api/video';

const videoUrlInput = ref('');
// 跟踪操作处理状态
const isProcessing = ref(false);
// 上传文件的action
const uploadAction = '/api/videos/upload';
// 添加一个ref引用上传组件
const uploadRef = ref();
//显示再前端的上传文件列表
const fileList = ref<UploadUserFile[]>([]);

// 修改handleChange函数，使其可以接收新上传的文件
const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
  console.log('File changed:', uploadFile, uploadFiles);
  // 将所有上传文件保存到fileList，最多保存3个
  fileList.value = uploadFiles.slice(-3);
};

// 处理文件超出限制
const handleExceed: UploadProps['onExceed'] = (uploadFiles) => {
  ElMessage.warning(`最多只能上传3个文件，当前已选择${uploadFiles.length}个文件`);
};

// 处理上传成功
const handleSuccess: UploadProps['onSuccess'] = (response) => {
  if (response.code === 0) {
    ElMessage.success('文件上传成功！');
  } else {
    ElMessage.error(response.data || '上传失败');
  }
};

// 处理上传错误
const handleError: UploadProps['onError'] = (error, uploadFile) => {
  console.error('Upload error:', error, uploadFile);
  ElMessage.error('文件上传失败，请重试');
};

// 处理URL提交
const handleUrlSubmit = async () => {
  if (!videoUrlInput.value) {
    ElMessage.warning('请输入视频链接');
    return;
  }

  try {
    isProcessing.value = true;
    const result = await storeVideoByUrl(videoUrlInput.value);
    ElMessage.success('链接提交成功！');

    // 清空输入
    videoUrlInput.value = '';

    // 可以在这里处理返回的文件信息
    console.log('URL处理结果:', result);
  } catch (error) {
    console.error('URL处理错误:', error);
    ElMessage.error('链接处理失败，请检查链接是否有效');
  } finally {
    isProcessing.value = false;
  }
};

const handleSubmit = async () => {
  if (fileList.value.length === 0 && !videoUrlInput.value) {
    ElMessage.warning('请上传视频文件或输入视频链接');
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
    } else {
      ElMessage.success('操作完成！');
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
    description="输入视频连接或者上传视频文件进行文字转录"
    title="视频转录工具"
  >
    <div>
      <el-input
        v-model="videoUrlInput"
        style="max-width: 100%; height: 40px"
        placeholder="输入抖音、B站、小红书等短视频平台的视频链接"
      >
        <template #prepend>Http://</template>
      </el-input>
    </div>

    <div
      :style="{
        boxShadow: '--el-box-shadow-dark',
        marginTop: '20px',
      }"
    >
      <el-upload
        ref="uploadRef"
        v-model:file-list="fileList"
        class="videoFileUpload"
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
      
        <el-icon class="el-icon--upload">
          <upload-filled />
        </el-icon>
        <div class="el-upload__text">
          <em>点击上传</em>或拖拽至此处可同时选择多个文件
          <br />
        </div>
        <template #tip>
          <div class="el-upload__tip" style="text-align: center">
            (单个文件大小 ≤2G) 支持格式：mp3, mp4, mov, m4a, wav, webm, avi, mkv
            等
          </div>
        </template>
      </el-upload>
    </div>
    <div style="text-align: center; margin-top: 20px;">
      <el-button
        type="primary"
        :loading="isProcessing"
        @click="handleSubmit"
      >
        <el-icon>
          <Promotion />
        </el-icon>
        提交
      </el-button>
    </div>
  </Page>
</template>