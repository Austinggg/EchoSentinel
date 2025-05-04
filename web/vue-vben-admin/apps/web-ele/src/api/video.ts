import { requestClient } from './request';
import type { AxiosProgressEvent } from 'axios';

// 定义视频文件信息接口
export interface VideoFileInfo {
  fileId: string;
  filename: string;
  size: number;
  mimeType: string;
  url: string;
  uploadTime?: string;
  hasAnalysis?: boolean;
}

// 定义多文件上传响应接口
export interface MultipleUploadResponse {
  files: VideoFileInfo[];
  count: number;
}

// 定义分析结果接口
export interface AnalysisResult {
  fileId: string;
  analysis: any; // 可以根据实际返回数据结构进一步定义
}

/**
 * 上传视频文件
 * @param file 要上传的文件
 * @param onProgress 可选的上传进度回调
 * @returns Promise<VideoFileInfo> 上传成功后的文件信息
 */
export function uploadVideoFile(file: File, onProgress?: (progressEvent: AxiosProgressEvent) => void) {
  const formData = new FormData();
  formData.append('file', file);
  
  return requestClient.post<VideoFileInfo>('/api/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: onProgress,
  });
}

/**
 * 上传多个视频文件
 * @param files 要上传的文件数组
 * @param onProgress 可选的上传进度回调
 * @returns Promise<MultipleUploadResponse> 上传成功后的文件信息
 */
export function uploadMultipleVideoFiles(files: File[], onProgress?: (progressEvent: AxiosProgressEvent) => void) {
  const formData = new FormData();
  
  // 最多上传3个文件
  const filesToUpload = files.slice(0, 3);
  
  filesToUpload.forEach(file => {
    formData.append('files[]', file);
  });
  
  return requestClient.post<MultipleUploadResponse>('/api/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: onProgress,
  });
}

/**
 * 获取视频文件URL
 * @param fileId 文件ID
 * @returns 视频文件的完整URL
 */
export function getVideoUrl(fileId: string): string {
  return `/api/videos/${fileId}`;
}

/**
 * 分析视频内容
 * @param fileId 文件ID
 * @returns Promise<AnalysisResult> 分析结果
 */
export function analyzeVideo(fileId: string) {
  return requestClient.post<AnalysisResult>(`/api/videos/${fileId}/analyze`);
}

/**
 * 通过URL存储视频
 * @param videoUrl 视频URL
 * @returns Promise<VideoFileInfo> 存储的视频文件信息
 */
export function storeVideoByUrl(videoUrl: string) {
  return requestClient.post<VideoFileInfo>('/api/videos/store-by-url', {
    url: videoUrl
  });
}

/**
 * 批量分析多个视频
 * @param fileIds 文件ID数组
 * @returns Promise<AnalysisResult[]> 所有分析结果
 */
export async function analyzeMultipleVideos(fileIds: string[]) {
  const promises = fileIds.map(fileId => analyzeVideo(fileId));
  return Promise.all(promises);
}

/**
 * 获取视频文件信息
 * @param fileId 文件ID
 * @returns Promise<VideoFileInfo> 视频文件信息
 */
export function getVideoInfo(fileId: string) {
  // 注意：这个接口需要后端提供，当前后端代码中没有明确的获取文件信息的API
  // 这里假设有这样的API，实际使用时可能需要修改
  return requestClient.get<VideoFileInfo>(`/api/videos/${fileId}/info`);
}