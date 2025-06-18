<script lang="ts" setup>
import { ref, computed } from 'vue';
import { 
  Refresh, 
  Document, 
  Download, 
  ArrowDown, 
  Grid,     
  DataBoard 
} from '@element-plus/icons-vue';
import { 
  ElButton, 
  ElTag, 
  ElCard, 
  ElDivider, 
  ElIcon,
  ElDropdown,
  ElDropdownMenu,
  ElDropdownItem,
  ElMessage
} from 'element-plus';
import MarkdownIt from 'markdown-it';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';

// 创建markdown-it实例
const md = new MarkdownIt({
  html: true,
  breaks: true,
  linkify: true,
  typographer: true,
});

// 定义组件接收的props
const props = defineProps({
  summary: {
    type: String,
    default: ''
  },
  loading: {
    type: Boolean,
    default: false
  },
  videoTitle: {
    type: String,
    default: ''
  }
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['regenerate']);

// 重新生成摘要
const regenerateSummary = () => {
  emit('regenerate');
};

// 获取文件名前缀
const getFilePrefix = () => {
  const title = props.videoTitle || '视频摘要';
  const timestamp = new Date().toISOString().slice(0, 10);
  return `${title}_${timestamp}`;
};

// 导出为Markdown
const exportAsMarkdown = () => {
  try {
    const content = generateMarkdownContent();
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    downloadFile(blob, `${getFilePrefix()}.md`);
    ElMessage.success('Markdown摘要导出成功');
  } catch (error) {
    ElMessage.error('导出失败: ' + error.message);
  }
};

// 导出为PDF
const exportAsPDF = async () => {
  try {
    const pdf = new jsPDF();
    
    // 设置基本信息（使用英文）
    const summaryTitle = 'Video Summary';
    const videoTitle = props.videoTitle || 'Unknown Video';
    const timestamp = `Generated: ${new Date().toLocaleDateString()}`;
    
    // 设置字体和标题
    pdf.setFontSize(20);
    pdf.text(summaryTitle, 20, 30);
    
    pdf.setFontSize(16);
    pdf.text(`Video: ${videoTitle}`, 20, 50);
    
    // 添加基本信息
    pdf.setFontSize(12);
    pdf.text(timestamp, 20, 70);
    
    // 添加分割线
    pdf.line(20, 80, 190, 80);
    
    // 处理摘要内容
    pdf.setFontSize(14);
    pdf.text('Summary Content:', 20, 100);
    
    pdf.setFontSize(10);
    let yPosition = 120;
    
    // 简化处理：只添加英文摘要
    const englishSummary = `
This is a video content summary.

Note: Detailed summary content contains Chinese characters.
Please refer to the Markdown or JSON export for complete content.
    `.trim();
    
    const lines = englishSummary.split('\n');
    lines.forEach((line) => {
      if (yPosition > 250) {
        pdf.addPage();
        yPosition = 30;
      }
      
      if (line.trim()) {
        pdf.text(line.trim(), 20, yPosition);
        yPosition += 15;
      }
    });
    
    pdf.save(`${getFilePrefix()}.pdf`);
    ElMessage.success('PDF摘要导出成功（英文版本）');
  } catch (error) {
    ElMessage.error('PDF导出失败: ' + error.message);
  }
};

// 导出为Excel
const exportAsExcel = () => {
  try {
    const workbook = XLSX.utils.book_new();

    // 创建摘要概览工作表
    const overviewData = [
      ['视频摘要'],
      [''],
      ['视频标题', props.videoTitle || ''],
      ['生成时间', new Date().toLocaleString()],
      [''],
      ['摘要内容'],
      [props.summary.replace(/<[^>]*>/g, '').replace(/[#*_`]/g, '')],
    ];

    const overviewSheet = XLSX.utils.aoa_to_sheet(overviewData);
    XLSX.utils.book_append_sheet(workbook, overviewSheet, '视频摘要');

    XLSX.writeFile(workbook, `${getFilePrefix()}.xlsx`);
    ElMessage.success('Excel摘要导出成功');
  } catch (error) {
    ElMessage.error('Excel导出失败: ' + error.message);
  }
};

// 导出为JSON
const exportAsJSON = () => {
  try {
    const jsonData = {
      title: props.videoTitle || '视频摘要',
      timestamp: new Date().toISOString(),
      summary: props.summary.replace(/<[^>]*>/g, ''), // 移除HTML标签
      metadata: {
        exportTime: new Date().toISOString(),
        version: '1.0',
      },
    };

    const blob = new Blob([JSON.stringify(jsonData, null, 2)], {
      type: 'application/json;charset=utf-8',
    });
    downloadFile(blob, `${getFilePrefix()}.json`);
    ElMessage.success('JSON数据导出成功');
  } catch (error) {
    ElMessage.error('JSON导出失败: ' + error.message);
  }
};

// 生成Markdown内容
const generateMarkdownContent = () => {
  return `# ${props.videoTitle || '视频摘要'}

## 摘要内容

${props.summary.replace(/<[^>]*>/g, '')}

---
*摘要由EchoSentinel系统自动生成于 ${new Date().toLocaleString()}*
`;
};

// 通用下载函数
const downloadFile = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// 处理导出选择
const handleExport = (format) => {
  if (!props.summary) {
    ElMessage.warning('暂无摘要内容可导出');
    return;
  }
  
  switch (format) {
    case 'markdown':
      exportAsMarkdown();
      break;
    case 'pdf':
      exportAsPDF();
      break;
    case 'excel':
      exportAsExcel();
      break;
    case 'json':
      exportAsJSON();
      break;
    default:
      ElMessage.warning('不支持的导出格式');
  }
};
</script>

<template>
  <div class="summary-container">
    <!-- 摘要内容 -->
    <div class="summary-content">
      <el-card v-if="summary" class="content-card" shadow="hover">
        <div class="markdown-body" v-html="summary"></div>
      </el-card>
      
      <el-card v-else class="empty-content" shadow="never">
        <div class="no-content">
          <el-icon size="48" color="#c0c4cc"><Document /></el-icon>
          <h3>暂无摘要内容</h3>
          <p>点击下方按钮生成智能摘要</p>
        </div>
      </el-card>
    </div>

    <!-- 操作区域 -->
    <div class="action-section">
      <el-card class="action-card" shadow="hover">
        <div class="action-content">
          <div class="action-info">
            <h4>
              <el-icon><Refresh /></el-icon>
              重新生成摘要
            </h4>
            <div class="action-tips">
              <el-tag type="info" size="small" effect="plain">AI智能分析</el-tag>
              <el-tag type="success" size="small" effect="plain">多维度提取</el-tag>
              <el-tag type="warning" size="small" effect="plain">关键信息聚合</el-tag>
            </div>
          </div>
          <div class="action-buttons">
            <el-button
              type="primary"
              :loading="loading"
              @click="regenerateSummary"
              size="default"
              :icon="Refresh"
              :disabled="loading"
            >
              {{ loading ? '生成中...' : '重新生成' }}
            </el-button>
            
            <!-- 导出下拉菜单 -->
            <el-dropdown 
              @command="handleExport" 
              class="export-dropdown"
              :disabled="!summary"
            >
              <el-button 
                type="success" 
                size="default"
                :disabled="!summary"
              >
                <el-icon><Download /></el-icon>
                导出摘要
                <el-icon class="el-icon--right"><ArrowDown /></el-icon>
              </el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="markdown">
                    <el-icon><Document /></el-icon>
                    Markdown (.md)
                  </el-dropdown-item>
                  <el-dropdown-item command="pdf">
                    <el-icon><Document /></el-icon>
                    PDF文档 (.pdf)
                  </el-dropdown-item>
                  <el-dropdown-item command="excel">
                    <el-icon><Grid /></el-icon>
                    Excel表格 (.xlsx)
                  </el-dropdown-item>
                  <el-dropdown-item command="json">
                    <el-icon><DataBoard /></el-icon>
                    JSON数据 (.json)
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<style scoped>
.summary-container {
  padding: 0;
  max-width: 100%;
  /* 移除或减少顶部间距 */
  margin-top: -20px;;
  padding-top: 0;
}

/* 内容区域 */
.summary-content {
  margin-bottom: 24px;
  /* 使用负边距来减少上方空白 */
  margin-top: -20px; /* 根据实际情况调整这个值 */
}

.content-card {
  border-radius: 12px;
  border: none;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.content-card :deep(.el-card__body) {
  padding: 30px;
}

.empty-content {
  border: 2px dashed #e4e7ed;
  border-radius: 12px;
  background: #fafafa;
}

.no-content {
  text-align: center;
  padding: 40px 20px;
  color: #909399;
}

.no-content h3 {
  margin: 16px 0 8px 0;
  color: #606266;
  font-size: 18px;
  font-weight: 500;
}

.no-content p {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

/* 操作区域 */
.action-section {
  margin-top: 24px;
}

.action-card {
  border-radius: 12px;
  border: 1px solid #e8f4fd;
  background: linear-gradient(135deg, #f6f9fc 0%, #e8f4fd 100%);
}

.action-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
}

.action-info {
  flex: 1;
}

.action-info h4 {
  margin: 0 0 8px 0;
  color: #303133;
  font-size: 16px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-info p {
  margin: 0 0 12px 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
}

.action-tips {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.action-buttons {
  flex-shrink: 0;
  display: flex;
  gap: 12px;
  align-items: center;
}

.export-dropdown {
  margin-left: 0;
}

/* 下拉菜单项样式 */
:deep(.el-dropdown-menu__item) {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Markdown内容样式 */
:deep(.markdown-body) {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  line-height: 1.8;
  color: #2c3e50;
  word-break: break-word;
}

:deep(.markdown-body h1) {
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 20px 0;
  padding-bottom: 12px;
  border-bottom: 3px solid #409eff;
  position: relative;
}

:deep(.markdown-body h1::before) {
  content: '';
  position: absolute;
  bottom: -3px;
  left: 0;
  width: 60px;
  height: 3px;
  background: linear-gradient(90deg, #409eff, #67c23a);
  border-radius: 2px;
}

:deep(.markdown-body h2) {
  margin: 32px 0 16px 0;
  font-size: 20px;
  font-weight: 600;
  color: #409eff;
  background: linear-gradient(135deg, #ecf5ff 0%, #e8f4fd 100%);
  padding: 12px 16px;
  border-radius: 8px;
  border-left: 4px solid #409eff;
  position: relative;
}

:deep(.markdown-body h3) {
  margin: 24px 0 12px 0;
  font-size: 17px;
  font-weight: 600;
  color: #67c23a;
  position: relative;
  padding-left: 20px;
}

:deep(.markdown-body h3::before) {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 16px;
  background: #67c23a;
  border-radius: 2px;
}

:deep(.markdown-body p) {
  margin-bottom: 16px;
  line-height: 1.8;
  text-align: justify;
}

:deep(.markdown-body ul, .markdown-body ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}

:deep(.markdown-body li) {
  margin-bottom: 8px;
  line-height: 1.6;
}

:deep(.markdown-body li::marker) {
  color: #409eff;
  font-weight: bold;
}

/* 强调内容样式 */
:deep(.markdown-body strong) {
  color: #e6a23c;
  font-weight: bold;
  background: linear-gradient(135deg, rgba(255, 229, 100, 0.3), rgba(255, 239, 153, 0.3));
  padding: 2px 6px;
  border-radius: 4px;
  border-bottom: 2px solid rgba(230, 162, 60, 0.3);
}

:deep(.markdown-body em) {
  color: #909399;
  font-style: italic;
  background: rgba(144, 147, 153, 0.1);
  padding: 1px 4px;
  border-radius: 3px;
}

/* 引用样式 */
:deep(.markdown-body blockquote) {
  margin: 20px 0;
  padding: 16px 20px;
  background: linear-gradient(135deg, #f0f9eb 0%, #e8f5e8 100%);
  border-left: 4px solid #67c23a;
  border-radius: 0 8px 8px 0;
  font-style: italic;
  color: #67c23a;
  position: relative;
}

:deep(.markdown-body blockquote::before) {
  content: '"';
  position: absolute;
  left: 8px;
  top: 8px;
  font-size: 24px;
  color: #67c23a;
  opacity: 0.5;
}

/* 代码样式 */
:deep(.markdown-body code) {
  background: #f4f4f5;
  color: #e6a23c;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Monaco', 'Consolas', monospace;
  font-size: 0.9em;
  border: 1px solid #e9ecef;
}

/* 摘要开头段落特殊样式 */
:deep(.markdown-body > p:first-child) {
  font-size: 16px;
  background: linear-gradient(135deg, #f0f9eb 0%, #e8f5e8 100%);
  padding: 20px;
  border-radius: 10px;
  border-left: 5px solid #67c23a;
  font-weight: 500;
  margin-bottom: 25px;
  box-shadow: 0 2px 12px rgba(103, 194, 58, 0.1);
  position: relative;
}

:deep(.markdown-body > p:first-child::before) {
  content: '💡';
  position: absolute;
  left: -12px;
  top: 20px;
  background: white;
  padding: 4px;
  border-radius: 50%;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .action-content {
    flex-direction: column;
    text-align: center;
    gap: 16px;
  }
  
  .action-info {
    text-align: center;
  }
  
  .action-tips {
    justify-content: center;
  }
  
  .action-buttons {
    flex-direction: column;
    width: 100%;
    gap: 8px;
  }
  
  .action-buttons .el-button {
    width: 100%;
  }
  
  :deep(.markdown-body) {
    font-size: 14px;
  }
  
  .content-card :deep(.el-card__body) {
    padding: 20px;
  }
}

@media (max-width: 480px) {
  .summary-container {
    padding: 0;
  }
}
</style>