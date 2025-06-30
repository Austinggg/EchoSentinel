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
    <!-- 简化头部 -->
    <div class="report-header">
      <div class="header-main">
        <h2 class="report-title">威胁分析报告</h2>
        <div class="report-meta">
          <span class="report-id">{{ reportData?.id || 'TR-' + Date.now().toString().slice(-6) }}</span>
          <span class="report-time">{{ formatDate(reportData?.timestamp || new Date()) }}</span>
        </div>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="6" animated />
    </div>

    <!-- 错误状态 -->
    <el-result
      v-else-if="error"
      icon="error"
      :title="error"
      sub-title="无法获取分析报告数据"
    >
      <template #extra>
        <el-button type="primary" @click="regenerateReport">重新分析</el-button>
      </template>
    </el-result>

    <!-- 报告内容 -->
    <div v-else-if="reportData" class="report-main">
      <!-- 风险概览 -->
      <div class="risk-overview">
        <div class="risk-header">
          <h3>风险评估</h3>
          <div class="actions">
            <el-button @click="regenerateReport" :icon="Refresh" size="small">重新分析</el-button>
            <el-dropdown @command="handleExport">
              <el-button :icon="Download" size="small">
                导出 <el-icon class="el-icon--right"><ArrowDown /></el-icon>
              </el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="pdf">PDF报告</el-dropdown-item>
                  <el-dropdown-item command="markdown">Markdown</el-dropdown-item>
                  <el-dropdown-item command="excel">Excel表格</el-dropdown-item>
                  <el-dropdown-item command="json">JSON数据</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
        
        <div class="risk-metrics">
          <div class="metric-item primary">
            <div class="metric-label">威胁等级</div>
            <div class="metric-value">
              <span class="risk-level" :class="riskLevelInfo.class">{{ riskLevelInfo.text }}</span>
            </div>
          </div>
          <div class="metric-item">
            <div class="metric-label">风险评分</div>
            <div class="metric-value">{{ (reportData.risk_probability * 100).toFixed(1) }}%</div>
          </div>
          <div class="metric-item">
            <div class="metric-label">分析状态</div>
            <div class="metric-value">
              <span class="status-complete">已完成</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 分析报告 -->
      <div class="analysis-section">
        <h3>分析报告</h3>
        <div class="report-content">
          <div
            class="markdown-body"
            v-html="md.render(reportData.report)"
          ></div>
        </div>
      </div>

      <!-- 报告信息 -->
      <div class="report-info">
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">分析引擎</span>
            <span class="info-value">EchoSentinel AI</span>
          </div>
          <div class="info-item">
            <span class="info-label">报告版本</span>
            <span class="info-value">v2.1</span>
          </div>
          <div class="info-item">
            <span class="info-label">置信度</span>
            <span class="info-value">{{ (reportData.risk_probability * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 无报告状态 -->
    <div v-else class="empty-state">
      <div class="empty-content">
        <div class="empty-icon">🛡️</div>
        <h3>暂无威胁分析报告</h3>
        <p>点击下方按钮开始分析视频内容的潜在威胁</p>
        <el-button type="primary" @click="regenerateReport">开始分析</el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.threat-report {
  height: 100%;
  background: #f8f9fa;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}

/* 头部样式 */
.report-header {
  background: #fff;
  border-bottom: 1px solid #ebeef5;
  padding: 20px 24px;
}

.header-main {
  max-width: 1000px;
  margin: 0 auto;
}

.report-title {
  font-size: 20px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 8px 0;
}

.report-meta {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: #8492a6;
}

.report-id {
  font-family: Monaco, Consolas, monospace;
  background: #f1f3f4;
  padding: 2px 6px;
  border-radius: 3px;
}

/* 主要内容 */
.report-main {
  max-width: 1000px;
  margin: 0 auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 风险概览 */
.risk-overview {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  overflow: hidden;
}

.risk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #ebeef5;
  background: #fafbfc;
}

.risk-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

.actions {
  display: flex;
  gap: 8px;
}

.risk-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1px;
  background: #ebeef5;
}

.metric-item {
  background: #fff;
  padding: 20px;
  text-align: center;
}

.metric-item.primary {
  background: #f8f9fa;
}

.metric-label {
  font-size: 13px;
  color: #8492a6;
  margin-bottom: 8px;
  font-weight: 500;
}

.metric-value {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

/* 风险等级样式 */
.risk-level {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 600;
}

.risk-level.success {
  background: #e8f5e8;
  color: #52c41a;
}

.risk-level.warning {
  background: #fff7e6;
  color: #fa8c16;
}

.risk-level.danger {
  background: #fff2f0;
  color: #ff4d4f;
}

.risk-level.info {
  background: #f0f0f0;
  color: #666;
}

.status-complete {
  color: #52c41a;
  font-weight: 500;
}

/* 分析部分 */
.analysis-section {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  overflow: hidden;
}

.analysis-section h3 {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
  padding: 16px 20px;
  border-bottom: 1px solid #ebeef5;
  background: #fafbfc;
}

.report-content {
  padding: 24px;
}

/* 报告信息 */
.report-info {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  padding: 16px 20px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.info-label {
  font-size: 13px;
  color: #8492a6;
  font-weight: 500;
}

.info-value {
  font-size: 13px;
  color: #2c3e50;
  font-family: Monaco, Consolas, monospace;
}

/* 空状态 */
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.empty-content {
  text-align: center;
  max-width: 300px;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-content h3 {
  font-size: 18px;
  color: #2c3e50;
  margin: 0 0 8px 0;
}

.empty-content p {
  color: #8492a6;
  margin-bottom: 20px;
  line-height: 1.5;
}

/* 加载状态 */
.loading-state {
  max-width: 1000px;
  margin: 24px auto;
  padding: 0 24px;
}

/* Markdown样式 - 简洁专业版 */
:deep(.markdown-body) {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #2c3e50;
}

:deep(.markdown-body h1),
:deep(.markdown-body h2) {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 20px 0 12px 0;
  padding-bottom: 6px;
  border-bottom: 1px solid #ebeef5;
}

:deep(.markdown-body h3) {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
  margin: 16px 0 8px 0;
}

:deep(.markdown-body p) {
  margin-bottom: 12px;
  line-height: 1.6;
}

:deep(.markdown-body strong) {
  color: #2c3e50;
  font-weight: 600;
  background: #fff7e6;
  padding: 1px 4px;
  border-radius: 2px;
}

:deep(.markdown-body ul),
:deep(.markdown-body ol) {
  padding-left: 20px;
  margin-bottom: 12px;
}

:deep(.markdown-body li) {
  margin-bottom: 4px;
}

:deep(.markdown-body blockquote) {
  margin: 12px 0;
  padding: 12px 16px;
  background: #f8f9fa;
  border-left: 3px solid #dcdfe6;
  color: #606266;
}

:deep(.markdown-body code) {
  background: #f1f3f4;
  color: #2c3e50;
  padding: 2px 4px;
  border-radius: 2px;
  font-family: Monaco, Consolas, monospace;
  font-size: 0.9em;
}

:deep(.markdown-body table) {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 13px;
}

:deep(.markdown-body table th) {
  background: #f8f9fa;
  color: #2c3e50;
  padding: 8px 12px;
  border: 1px solid #ebeef5;
  font-weight: 600;
}

:deep(.markdown-body table td) {
  padding: 8px 12px;
  border: 1px solid #ebeef5;
}

/* 威胁重点标注 */
:deep(.markdown-body p:has(strong:contains('▲'))) {
  background: linear-gradient(135deg, #fff2f0 0%, #ffebe8 100%);
  border-left: 3px solid #ff4d4f;
  padding: 12px 16px;
  margin: 16px 0;
  border-radius: 4px;
}

:deep(.markdown-body strong:contains('▲')) {
  color: #ff4d4f;
  background: transparent;
  font-weight: 700;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .report-main {
    padding: 16px;
    gap: 16px;
  }
  
  .risk-header {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }
  
  .actions {
    justify-content: center;
  }
  
  .risk-metrics {
    grid-template-columns: 1fr;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
  }
  
  .info-item {
    flex-direction: column;
    gap: 4px;
    align-items: flex-start;
  }
}
</style>