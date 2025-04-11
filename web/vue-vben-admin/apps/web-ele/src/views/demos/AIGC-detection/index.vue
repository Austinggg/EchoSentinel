<script lang="ts" setup>
import { ref } from 'vue'
import { 
  uploadVideo,
  processVideo,
  detectAIGC
} from '@/api/video.ts'

// 响应式状态
const videoUrl = ref('')
const processingType = ref<'summary' | 'transcript' | 'detect' | ''>('')
const results = reactive({
  summary: '',
  transcript: '',
  detection: ''
})

// 上传处理
const handleUpload = async (file: File) => {
  try {
    const { data } = await uploadVideo(file, progress => {
      // 更新进度条
    })
    videoUrl.value = data.url // 获取服务器返回的视频地址
  } catch (error) {
    // 错误处理
  }
}

// 数字人检测
const runDetection = async () => {
  processingType.value = 'detect'
  try {
    const { data } = await detectAIGC(videoUrl.value)
    results.detection = data.result
  } finally {
    processingType.value = ''
  }
}
</script>

<template>
  <Page description="支持视频上传、生成视频总结和提取视频文字内容" title="视频处理">
      <div>
          <ElCard>
              <template #header>
                  <div>
                      <span>视频上传</span>
                      <ElButton @click="handleUpload" :disabled="!videoFile || isUploading" :loading="isUploading">
                          上传视频
                      </ElButton>
                  </div>
              </template>

              <ElUpload ref="uploadRef" drag :auto-upload="false" :show-file-list="false"
                  :before-upload="beforeUpload">
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

              <ElProgress v-if="isUploading || isProcessing" :percentage="processProgress" />

              <ElDivider />

              <div>
                  <ElButton @click="generateSummary" :disabled="!videoUrl || isProcessing">
                      生成视频总结
                  </ElButton>
                  <ElButton @click="extractTranscript" :disabled="!videoUrl || isProcessing">
                      提取视频文字
                  </ElButton>
              </div>
          </ElCard>

          <ElCard v-if="activeResult">
              <template #header>
                  <div>
                      <span>{{ activeResult === 'summary' ? '视频总结结果' : '视频文字稿' }}</span>
                      <ElButton
                          @click="navigator.clipboard.writeText(activeResult === 'summary' ? summaryResult : transcriptResult)">
                          复制
                      </ElButton>
                  </div>
              </template>

              <ElInput type="textarea" :model-value="activeResult === 'summary' ? summaryResult : transcriptResult"
                  readonly />
          </ElCard>
      </div>
  </Page>
</template>