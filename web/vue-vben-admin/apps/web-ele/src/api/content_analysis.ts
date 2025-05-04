import { requestClient } from './request';

// 定义转录文本块的结构
export interface TranscriptChunk {
  text: string;       // 转录的文本内容
  timestamp: number[]; // 时间戳 [开始时间, 结束时间]
}

// 定义转录API的响应结构
export interface TranscriptResponse {
  audio_path: string | null;  // 音频文件路径(如果有)
  chunks: TranscriptChunk[];  // 分段转录文本数组
  filename: string;           // 原始文件名
  full_text: string;          // 完整转录文本(可选)
}

/**
 * 视频转录API
 * @param file - 要转录的视频文件
 * @returns Promise<TranscriptResponse> - 包含转录结果的Promise
 */
export const transcribeVideo = (file: File) => {
  const formData = new FormData();  // 创建表单数据对象
  formData.append('file', file);    // 添加文件到表单
  
  // 使用requestClient发送POST请求
  return requestClient.post<TranscriptResponse>(
    '/api/transcribe/file',  // API端点
    formData,                // 请求体数据
    {                        // 请求配置
        headers: {
            'Content-Type': 'multipart/form-data',  // 设置内容类型
        },
    }
  )
}
// 定义文件存储响应的接口
export interface FileStorageResponse {
  fileId: string;           // 存储后的文件ID
  url: string;              // 文件访问URL
  filename: string;         // 原始文件名
  size: number;             // 文件大小(字节)
  mimeType: string;         // 文件MIME类型
  createdAt: string;        // 创建时间
}

/**
 * 存储视频/音频文件API
 * @param file - 要存储的文件
 * @param metadata - 可选的文件元数据
 * @returns Promise<FileStorageResponse> - 包含存储结果的Promise
 */
export const storeMediaFile = (file: File, metadata?: Record<string, any>) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // 如果有元数据，添加到表单
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata));
  }
  
  return requestClient.post<FileStorageResponse>(
    '/api/media/store',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
};

/**
 * 通过URL上传视频
 * @param videoUrl - 视频的URL地址
 * @returns Promise<FileStorageResponse> - 包含存储结果的Promise
 */
export const storeMediaByUrl = (videoUrl: string) => {
  return requestClient.post<FileStorageResponse>(
    '/api/media/store-by-url',
    { url: videoUrl },
    {
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
};