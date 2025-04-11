import axios from 'axios'
import type { AxiosProgressEvent } from 'axios'

const service = axios.create({
  baseURL: import.meta.env.VITE_APP_BASE_API,
  timeout: 30000
})

// 文件上传接口
export const uploadVideo = (file: File, onProgress: (progress: number) => void) => {
  const formData = new FormData()
  formData.append('video', file)
  
  return service.post('/api/v1/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent: AxiosProgressEvent) => {
      if (progressEvent.total) {
        onProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total))
      }
    }
  })
}

// 视频处理接口
export const processVideo = (type: 'summary' | 'transcript', videoId: string) => {
  return service.post(`/api/v1/process/${type}`, { videoId })
}

// 数字人检测专用接口
export const detectAIGC = (videoId: string) => {
  return service.post('/api/v1/detect/aigc', { videoId })
}