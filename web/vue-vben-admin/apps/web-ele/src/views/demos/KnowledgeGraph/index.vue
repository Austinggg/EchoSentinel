<script lang="ts" setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { Loading } from '@element-plus/icons-vue';

// 嵌入开源项目 OpenSPG KAG (Knowledge & AI Graph)
// 项目链接: https://github.com/OpenSPG/openspg
// 根据Apache 2.0许可使用
const targetUrl = ref('http://localhost:8887');
const iframeLoaded = ref(false);
const loadError = ref(false);

// 处理iframe加载状态
const handleIframeLoad = () => {
  iframeLoaded.value = true;
};

const handleIframeError = () => {
  loadError.value = true;
};
</script>

<template>
    <div class="knowledge-graph-container">
      <div class="iframe-wrapper">
        <!-- 加载指示器 -->
        <div v-if="!iframeLoaded && !loadError" class="loading-indicator">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在加载知识图谱应用...</span>
        </div>
        
        <!-- 错误提示 -->
        <div v-if="loadError" class="error-message">
          <el-alert
            title="无法加载知识图谱应用"
            type="error"
            description="请确保KAG应用(http://localhost:8887)已启动并且可以访问。"
            show-icon
            :closable="false"
          />
        </div>
        
        <!-- iframe嵌入应用 -->
        <iframe
          :src="targetUrl"
          @load="handleIframeLoad"
          @error="handleIframeError"
          frameborder="0"
          class="knowledge-graph-iframe"
          allow="fullscreen"
          title="Knowledge Graph Visualization"
        ></iframe>
      </div>
      
      <!-- 底部小型归属信息 -->
      <div class="footer-attribution">
        基于 <a href="https://github.com/OpenSPG/openspg" target="_blank">OpenSPG/KAG</a> (Apache 2.0)
      </div>
    </div>
  </template>

<style scoped>
.footer-attribution {
  padding: 6px 12px;
  text-align: right;
  font-size: 12px;
  color: #909399;
  background-color: transparent;
}

/* 修改iframe-wrapper样式使其占据更多空间 */
.iframe-wrapper {
  position: relative;
  flex: 1;
  min-height: 550px;
  overflow: hidden;
}
.knowledge-graph-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.project-attribution {
  padding: 16px;
  background-color: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
}

.attribution-text {
  color: #606266;
  font-size: 14px;
  margin: 8px 0;
}

.attribution-text a {
  color: #409eff;
  text-decoration: none;
}

.attribution-text a:hover {
  text-decoration: underline;
}

.iframe-wrapper {
  position: relative;
  flex: 1;
  min-height: 500px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.knowledge-graph-iframe {
  width: 100%;
  height: 100%;
  border: none;
}

.loading-indicator {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: #909399;
}

.error-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  max-width: 500px;
}
</style>