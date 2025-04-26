import { requestClient } from './request';//返回的是data

// 定义视频分析记录类型（仅包含前端需要显示的字段）
export interface VideoAnalysisRecord {
  id: string;
  title: string;
  cover: string;
  summary: string;
  threatLevel: 'low' | 'medium' | 'high' | 'processing';
  createTime: string;
  publishTime?: string;
  tags?: string[];
}

// 定义响应结构 - 简化为只有total和items
export interface VideoListResult {
  total: number;
  items: VideoAnalysisRecord[];
}

// 获取视频列表API - 不传递任何参数
export function getVideoList() {
  return requestClient.get<VideoListResult>('/videos/list');
}

// 获取视频详情API
export function getVideoDetail(id: string) {
  return requestClient.get<VideoAnalysisRecord>(`/videos/${id}`);
}

// 删除视频API
export function deleteVideo(id: string) {
  return requestClient.delete(`/videos/${id}`);
}

// 批量删除视频API
export function batchDeleteVideos(ids: string[]) {
  return requestClient.post('/videos/batch-delete', { ids });
}

// 上传视频API - 返回上传后的视频信息
export function uploadVideo(formData: FormData) {
  return requestClient.post<ApiResponse<VideoAnalysisRecord>>('/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
}

// API响应的顶层结构（仅用于upload等其他API）
export interface ApiResponse<T> {
  code: number;
  data: T;
  message: string;
}

// 更新视频分析状态API（例如手动触发分析）
export function analyzeVideo(id: string) {
  return requestClient.post<ApiResponse<{fileId: string; analysis: any}>>(`/videos/${id}/analyze`);
}