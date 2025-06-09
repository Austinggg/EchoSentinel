<script lang="ts" setup>
import { computed, ref } from 'vue';
import { 
  Refresh, 
  Download, 
  ArrowDown, 
  Document, 
  Grid,     
  DataBoard 
} from '@element-plus/icons-vue';
import {
  ElButton,
  ElCard,
  ElResult,
  ElSkeleton,
  ElDropdown,
  ElDropdownMenu,
  ElDropdownItem,
  ElMessage,
  ElTag,
  ElIcon
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
  reportData: {
    type: Object,
    default: () => null,
  },
  loading: {
    type: Boolean,
    default: false,
  },
  error: {
    type: String,
    default: null,
  },
  videoTitle: {
    type: String,
    default: '',
  },
});

// 定义需要向父组件发送的事件
const emit = defineEmits(['regenerate', 'export']);

// 重新生成报告
const regenerateReport = () => {
  emit('regenerate');
};

// 格式化日期
const formatDate = (date) => {
  if (!date) return '未知时间';
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

// 获取风险等级信息
const riskLevelInfo = computed(() => {
  if (!props.reportData || !props.reportData.risk_level)
    return { class: 'info', color: '#909399', text: '未评估' };

  const level = props.reportData.risk_level.toLowerCase();
  switch (level) {
    case 'low':
      return { class: 'success', color: '#67C23A', text: '低风险' };
    case 'medium':
      return { class: 'warning', color: '#E6A23C', text: '中等风险' };
    case 'high':
      return { class: 'danger', color: '#F56C6C', text: '高风险' };
    default:
      return { class: 'info', color: '#909399', text: '未评估' };
  }
});

// 获取文件名前缀
const getFilePrefix = () => {
  const title = props.videoTitle || '威胁分析报告';
  const timestamp = new Date().toISOString().slice(0, 10);
  return `${title}_${timestamp}`;
};

// 导出为Markdown
const exportAsMarkdown = () => {
  try {
    const content = generateMarkdownContent();
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    downloadFile(blob, `${getFilePrefix()}.md`);
    ElMessage.success('Markdown报告导出成功');
  } catch (error) {
    ElMessage.error('导出失败: ' + error.message);
  }
};

// 导出为PDF（修复中文乱码）
const exportAsPDF = async () => {
  try {
    const pdf = new jsPDF();
    
    // 设置基本信息（使用英文）
    const reportTitle = 'Threat Analysis Report';
    const videoTitle = props.videoTitle || 'Unknown Video';
    const riskLevel = `Risk Level: ${riskLevelInfo.value.text}`;
    const riskProb = `Risk Probability: ${(props.reportData.risk_probability * 100).toFixed(1)}%`;
    const timestamp = `Generated: ${formatDate(props.reportData?.timestamp)}`;
    
    // 设置字体和标题
    pdf.setFontSize(20);
    pdf.text(reportTitle, 20, 30);
    
    pdf.setFontSize(16);
    pdf.text(`Video: ${videoTitle}`, 20, 50);
    
    // 添加基本信息
    pdf.setFontSize(12);
    pdf.text(riskLevel, 20, 70);
    pdf.text(riskProb, 20, 85);
    pdf.text(timestamp, 20, 100);
    
    // 添加分割线
    pdf.line(20, 110, 190, 110);
    
    // 处理报告内容 - 将中文转换为拼音或英文描述
    pdf.setFontSize(14);
    pdf.text('Report Content:', 20, 130);
    
    pdf.setFontSize(10);
    let yPosition = 150;
    
    // 简化处理：只添加英文摘要
    const englishSummary = `
This is a threat analysis report for the video content.
Risk Assessment: ${riskLevelInfo.value.text}
Probability Score: ${(props.reportData.risk_probability * 100).toFixed(1)}%

Note: Detailed analysis content contains Chinese characters.
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
    ElMessage.success('PDF报告导出成功（英文版本）');
  } catch (error) {
    ElMessage.error('PDF导出失败: ' + error.message);
  }
};

// 导出为Excel
const exportAsExcel = () => {
  try {
    const workbook = XLSX.utils.book_new();

    // 创建报告概览工作表
    const overviewData = [
      ['威胁分析报告'],
      [''],
      ['视频标题', props.videoTitle || ''],
      ['风险等级', riskLevelInfo.value.text],
      ['风险概率', `${(props.reportData.risk_probability * 100).toFixed(1)}%`],
      ['生成时间', formatDate(props.reportData?.timestamp)],
      [''],
      ['报告内容'],
      [props.reportData.report.replace(/[#*_`]/g, '')],
    ];

    const overviewSheet = XLSX.utils.aoa_to_sheet(overviewData);
    XLSX.utils.book_append_sheet(workbook, overviewSheet, '威胁分析报告');

    // 如果有详细数据，可以创建额外的工作表
    if (props.reportData.details) {
      const detailsSheet = XLSX.utils.json_to_sheet(props.reportData.details);
      XLSX.utils.book_append_sheet(workbook, detailsSheet, '详细数据');
    }

    XLSX.writeFile(workbook, `${getFilePrefix()}.xlsx`);
    ElMessage.success('Excel报告导出成功');
  } catch (error) {
    ElMessage.error('Excel导出失败: ' + error.message);
  }
};

// 导出为JSON
const exportAsJSON = () => {
  try {
    const jsonData = {
      title: props.videoTitle || '威胁分析报告',
      timestamp: formatDate(props.reportData?.timestamp),
      riskLevel: riskLevelInfo.value.text,
      riskProbability: props.reportData.risk_probability,
      report: props.reportData.report,
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
  return `# ${props.videoTitle || '威胁分析报告'}

## 报告概览

- **风险等级**: ${riskLevelInfo.value.text}
- **风险概率**: ${(props.reportData.risk_probability * 100).toFixed(1)}%
- **生成时间**: ${formatDate(props.reportData?.timestamp)}

## 分析报告

${props.reportData.report}

---
*报告由EchoSentinel系统自动生成*
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
  <div class="threat-report">
    <div class="threat-report-header">
      <h3 class="section-heading">内容威胁分析报告</h3>
      <div class="report-timestamp">
        生成时间: {{ formatDate(reportData?.timestamp || new Date()) }}
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <!-- 错误状态 -->
    <el-result
      v-else-if="error"
      icon="error"
      :title="error"
      sub-title="无法获取分析报告数据"
    >
      <template #extra>
        <el-button type="primary" @click="regenerateReport">重试</el-button>
      </template>
    </el-result>

    <!-- 报告数据显示 -->
    <div v-else-if="reportData" class="analysis-report">
      <!-- 风险等级信息 -->
      <el-card class="risk-info-card" :class="`border-${riskLevelInfo.class}`">
        <div class="risk-info-header">
          <div class="risk-level-container">
            <el-tag
              :type="riskLevelInfo.class"
              size="large"
              effect="dark"
              class="risk-level-tag"
            >
              {{ riskLevelInfo.text }}
            </el-tag>
            <div class="risk-probability">
              风险概率:
              <span :style="{ color: riskLevelInfo.color }">
                {{ (reportData.risk_probability * 100).toFixed(1) }}%
              </span>
            </div>
          </div>
          <div class="action-buttons">
            <!-- 重新生成按钮 -->
            <el-button
              type="primary"
              @click="regenerateReport"
              :icon="Refresh"
              size="small"
            >
              重新生成
            </el-button>

            <!-- 导出下拉菜单 -->
            <el-dropdown @command="handleExport" class="export-dropdown">
              <el-button type="success" size="small">
                <el-icon><Download /></el-icon>
                导出报告
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

      <!-- 分析报告内容 -->
      <el-card class="report-content">
        <div class="report-container">
          <div
            class="markdown-body"
            v-html="md.render(reportData.report)"
          ></div>
        </div>
      </el-card>
    </div>

    <!-- 没有报告时显示 -->
    <div v-else>
      <el-result icon="info" title="暂无分析报告">
        <template #sub-title>
          <p>系统尚未对此视频生成分析报告，点击下方按钮生成。</p>
        </template>
        <template #extra>
          <el-button type="primary" @click="regenerateReport">
            生成分析报告
          </el-button>
        </template>
      </el-result>
    </div>
  </div>
</template>

<style scoped>
.threat-report {
  height: 100%;
  overflow: auto;
}

.threat-report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-heading {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 500;
}

.report-timestamp {
  font-size: 14px;
  color: #909399;
  font-style: italic;
}

.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 3rem;
  padding-bottom: 3rem;
}

.risk-info-card {
  margin-bottom: 1rem;
  border-top-width: 4px;
}

.risk-info-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.risk-level-container {
  display: flex;
  align-items: center;
}

.risk-level-tag {
  margin-right: 0.75rem;
}

.risk-probability {
  font-size: 1.125rem;
  font-weight: 500;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.export-dropdown {
  margin-left: 0.5rem;
}

.report-content {
  margin-bottom: 1rem;
}

.report-container {
  padding: 10px 5px;
}

.border-success {
  border-top-color: #67c23a;
}

.border-warning {
  border-top-color: #e6a23c;
}

.border-danger {
  border-top-color: #f56c6c;
}

.border-info {
  border-top-color: #909399;
}

/* 下拉菜单项样式 */
:deep(.el-dropdown-menu__item) {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Markdown样式保持不变 */
:deep(.markdown-body) {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  line-height: 1.8;
  color: #333;
  word-break: break-word;
}

:deep(.markdown-body h2) {
  margin-top: 28px;
  font-size: 20px;
  font-weight: 600;
  border-bottom: 2px solid #409eff;
  padding-bottom: 8px;
  color: #303133;
}

:deep(.markdown-body h3) {
  margin-top: 24px;
  font-size: 17px;
  font-weight: 600;
  color: #409eff;
  background-color: #ecf5ff;
  padding: 8px 12px;
  border-radius: 4px;
}

:deep(.markdown-body p:has(> ▲)) {
  background-color: #fef0f0;
  padding: 12px 16px;
  border-radius: 6px;
  border-left: 4px solid #f56c6c;
  margin-bottom: 20px;
}

:deep(.markdown-body p ▲) {
  color: #f56c6c;
  font-weight: bold;
  margin-right: 4px;
}

:deep(.markdown-body ol) {
  padding-left: 22px;
  margin-bottom: 20px;
}

:deep(.markdown-body ol li) {
  margin-bottom: 10px;
  padding-left: 6px;
}

:deep(.markdown-body strong) {
  color: #e6a23c;
  font-weight: bold;
  background-color: rgba(255, 229, 100, 0.3);
  padding: 0 4px;
  border-radius: 3px;
}

:deep(.markdown-body p strong:first-of-type) {
  display: inline-block;
  margin-right: 5px;
}

:deep(.markdown-body code) {
  color: #476582;
  background-color: rgba(27, 31, 35, 0.05);
  padding: 2px 5px;
  border-radius: 3px;
}

:deep(.markdown-body table) {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}

:deep(.markdown-body table th) {
  background: #f2f6fc;
  padding: 12px;
  border: 1px solid #ebeef5;
}

:deep(.markdown-body table td) {
  padding: 12px;
  border: 1px solid #ebeef5;
}

:deep(.markdown-body > p:first-child) {
  font-size: 16px;
  background-color: #fef0f0;
  padding: 15px;
  border-radius: 6px;
  border-left: 5px solid #f56c6c;
  font-weight: 500;
  margin-bottom: 25px;
}
</style>