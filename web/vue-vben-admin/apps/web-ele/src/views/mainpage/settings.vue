<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue';
import axios from 'axios';
import {
  ElMessage,
  ElMessageBox,
  ElCard,
  ElTabs,
  ElTabPane,
  ElForm,
  ElFormItem,
  ElInput,
  ElInputNumber,
  ElSelect,
  ElOption,
  ElOptionGroup,
  ElButton,
  ElSwitch,
  ElTooltip,
  ElDivider,
  ElAlert,
  ElIcon,
  ElLink,
  ElScrollbar,
} from 'element-plus';

// 导入所需图标
import {
  Check,
  RefreshRight,
  VideoPlay,
  Picture,
  Connection,
  WarningFilled,
  CircleCheck,
  Microphone, // 新增图标
  DataAnalysis, // 新增图标
  Setting, // 新增图标
  Monitor, // 新增图标
} from '@element-plus/icons-vue';

// 当前活跃的标签页 - 修改默认标签为llm
const activeTab = ref('llm');
const saving = ref(false);

// 大语言模型配置 - 重命名为llmConfigs
const llmConfigs = {
  information_extraction: {
    label: '信息提取模型',
    description: '用于从视频内容中提取关键信息',
  },
  assessment: {
    label: '基础评估模型',
    description: '用于评估内容风险和分类',
  },
  report: {
    label: '报告生成模型',
    description: '用于生成最终分析报告',
  },
};

// 新增 - 远程服务模型配置
const remoteServiceConfigs = reactive({
  aigc_detection: {
    label: 'AIGC检测服务',
    icon: 'Monitor',
    enabled: true,
    description: '提供AI生成内容检测服务',
  },
  whisper: {
    label: '语音转写服务',
    icon: 'Microphone',
    enabled: true,
    description: '提供视频音频内容转写为文本的服务',
  },
  decision: {
    label: '决策推理服务',
    icon: 'DataAnalysis',
    enabled: true,
    description: '提供内容分析和决策推理的服务',
  },
  training: {
    label: '决策模型训练服务',
    icon: 'Setting',
    enabled: false,
    description: '提供模型训练和微调的服务',
  },
});

// 社交平台配置
const platformConfigs = reactive([
  {
    key: 'douyin',
    label: '抖音',
    icon: 'VideoPlay',
    enabled: true,
    description: '用于访问抖音平台内容，获取视频和账号信息',
    helpLink: 'https://example.com/how-to-get-douyin-cookie',
  },
  {
    key: 'tiktok',
    label: 'TikTok',
    icon: 'VideoPlay',
    enabled: false,
    description: '用于访问TikTok平台内容，获取海外视频和账号信息',
    helpLink: 'https://example.com/how-to-get-tiktok-cookie',
  },
  {
    key: 'bilibili',
    label: 'Bilibili',
    icon: 'Picture',
    enabled: false,
    description: '用于访问B站内容，获取视频和UP主信息',
    helpLink: 'https://example.com/how-to-get-bilibili-cookie',
  },
  {
    key: 'kag_api',
    label: 'KAG API',
    icon: 'Connection',
    enabled: true,
    description: '用于访问KAG API系统',
    helpLink: 'https://example.com/how-to-get-kag-api-cookie',
  },
]);

// 表单数据
const formData = reactive({
  // 是否使用本地模型
  local_information_extraction: false,
  local_assessment: false,
  local_report: false,

  // 模型配置
  information_extraction: {
    description: '信息提取模型',
    local_ollama: {
      base_url: 'http://localhost:11434/api/chat',
      model: 'deepseek-r1:14b',
      description: 'ollama格式的本地模型',
    },
    remote_openai: {
      api_key: 'sk-lqdncjtdqqaxrvgurqffyzcvsslqvrpczbmbiamxchewamgt',
      base_url: 'https://api.siliconflow.cn/v1',
      model: 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    },
  },
  assessment: {
    description: '评估模型',
    local_ollama: {
      base_url: 'http://localhost:11434/api/chat',
      model: 'deepseek-r1:14b',
      description: 'ollama格式的本地模型',
    },
    remote_openai: {
      api_key: '',
      base_url: 'https://api.siliconflow.cn/v1',
      model: 'Qwen/Qwen3-30B-A3B',
    },
  },
  report: {
    description: '报告生成模型',
    local_ollama: {
      base_url: 'http://localhost:11434/api/chat',
      model: 'deepseek-r1:14b',
      description: 'ollama格式的本地模型',
    },
    remote_openai: {
      api_key: '',
      base_url: 'https://api.siliconflow.cn/v1',
      model: 'Qwen/Qwen3-30B-A3B',
    },
  },
  remoteServices: {
    whisper: {
      address: 'http://localhost:3000',
    },
    decision: {
      address: 'http://localhost:3000',
    },
    training: {
      address: 'http://localhost:3000',
    },
    aigc_detection: {
      address: 'http://localhost:3000',
    },
  },

  // Cookie配置
  cookies: {
    douyin: '',
    tiktok: '',
    bilibili: '',
    kag_api: '',
  },

  // 爬虫配置
  crawler: {
    base_url: 'http://localhost:8080',
    max_concurrent_tasks: 3,
  },

  // 重试配置
  retry: {
    max_retries: 5,
    retry_delay: 1,
  },
});

// 加载设置
const loadSettings = async () => {
  try {
    const response = await axios.get('/api/system/settings');
    if (response.data && response.data.code === 200) {
      const settings = response.data.data;

      // 更新本地/远程模型开关
      formData.local_information_extraction =
        settings.local_information_extraction || false;
      formData.local_assessment = settings.local_assessment || false;
      formData.local_report = settings.local_report || false;

      // 更新模型配置
      if (settings.information_extraction_model) {
        formData.information_extraction = {
          ...formData.information_extraction,
          ...settings.information_extraction_model,
        };
      }

      if (settings.assessment_model) {
        formData.assessment = {
          ...formData.assessment,
          ...settings.assessment_model,
        };
      }

      if (settings.report_model) {
        formData.report = { ...formData.report, ...settings.report_model };
      }

      // 更新KAG API Cookie
      if (settings.KAG_API && settings.KAG_API.cookie) {
        formData.cookies.kag_api = settings.KAG_API.cookie;
      }

      // 更新重试设置
      if (settings.retry) {
        formData.retry = { ...formData.retry, ...settings.retry };
      }
      // 更新远程服务模型配置
      if (settings.remote_services) {
        Object.keys(settings.remote_services).forEach((key) => {
          if (formData.remoteServices[key]) {
            formData.remoteServices[key].address =
              settings.remote_services[key].address || '';

            // 如果有enabled状态，也更新它
            if (
              remoteServiceConfigs[key] &&
              settings.remote_services[key].enabled !== undefined
            ) {
              remoteServiceConfigs[key].enabled =
                settings.remote_services[key].enabled;
            }
          }
        });
      }

      // 更新其他Cookie (需要从另一个API获取)
      const cookieResponse = await axios.get('/api/system/platform-cookies');
      if (cookieResponse.data && cookieResponse.data.code === 200) {
        const cookies = cookieResponse.data.data;
        if (cookies.douyin) formData.cookies.douyin = cookies.douyin;
        if (cookies.tiktok) formData.cookies.tiktok = cookies.tiktok;
        if (cookies.bilibili) formData.cookies.bilibili = cookies.bilibili;
      }

      ElMessage.success('设置已加载');
    }
  } catch (error) {
    console.error('加载设置失败:', error);
    ElMessage.error('加载设置失败，请检查网络连接');
  }
};

// 保存设置
// 保存设置 - 修改以保存远程服务模型配置
const saveSettings = async () => {
  try {
    saving.value = true;

    // 转换为API需要的格式
    const settings = {
      // 大语言模型配置
      local_information_extraction: formData.local_information_extraction,
      local_assessment: formData.local_assessment,
      local_report: formData.local_report,
      information_extraction_model: formData.information_extraction,
      assessment_model: formData.assessment,
      report_model: formData.report,

      // 远程服务模型配置
      remote_services: Object.keys(formData.remoteServices).reduce(
        (acc, key) => {
          acc[key] = {
            address: formData.remoteServices[key].address,
            enabled: remoteServiceConfigs[key].enabled,
          };
          return acc;
        },
        {},
      ),

      // 其他设置
      KAG_API: {
        cookie: formData.cookies.kag_api,
      },
      retry: formData.retry,
    };

    // 保存主要设置
    await axios.post('/api/system/settings', settings);
    // 保存社交平台Cookie
    await axios.post('/api/system/crawler-settings', formData.crawler);
    ElMessage.success('设置已保存');
  } catch (error) {
    console.error('保存设置失败:', error);
    ElMessage.error('保存设置失败，请检查网络连接');
  } finally {
    saving.value = false;
  }
};
const testRemoteService = async (serviceKey) => {
  try {
    ElMessage.info(`正在测试${remoteServiceConfigs[serviceKey].label}连接...`);
    const address = formData.remoteServices[serviceKey].address;

    // 确保地址以正确格式结尾
    const healthEndpoint = address.endsWith('/')
      ? `${address}health`
      : `${address}/health`;

    const response = await axios.get(healthEndpoint, { timeout: 5000 });

    if (response.status === 200) {
      // 检查服务状态
      if (response.data[serviceKey]?.status === 'healthy') {
        ElMessage.success(
          `${remoteServiceConfigs[serviceKey].label}连接成功且状态正常!`,
        );
      } else {
        ElMessage.warning(
          `${remoteServiceConfigs[serviceKey].label}连接成功，但服务状态异常!`,
        );
      }
    } else {
      throw new Error(`连接失败，HTTP状态码: ${response.status}`);
    }
  } catch (error) {
    console.error(`测试远程服务失败:`, error);
    ElMessage.error(`测试失败: ${error.message || '连接错误'}`);
  }
};

// 重置设置
const resetSettings = () => {
  ElMessageBox.confirm(
    '确定要重置所有设置吗？这将恢复到上次保存的状态。',
    '确认重置',
    {
      confirmButtonText: '确认',
      cancelButtonText: '取消',
      type: 'warning',
    },
  )
    .then(() => {
      loadSettings();
      ElMessage.info('设置已重置');
    })
    .catch(() => {});
};

// 测试本地模型连接
const testLocalModel = async (modelKey) => {
  try {
    ElMessage.info('正在测试本地模型连接...');
    const response = await axios.post('/api/system/test-local-model', {
      model_type: modelKey,
      base_url: formData[modelKey].local_ollama.base_url,
      model_name: formData[modelKey].local_ollama.model,
    });

    if (response.data && response.data.code === 200) {
      ElMessage.success(`本地${llmConfigs[modelKey].label}连接成功!`); // 修改这里：modelConfigs -> llmConfigs
    } else {
      throw new Error(response.data.message || '连接失败');
    }
  } catch (error) {
    console.error('测试本地模型失败:', error);
    ElMessage.error(`测试失败: ${error.message || '连接错误'}`);
  }
};

// 测试远程模型API - 修复变量名
const testRemoteModel = async (modelKey) => {
  try {
    ElMessage.info('正在测试远程API连接...');
    const response = await axios.post('/api/system/test-remote-model', {
      model_type: modelKey,
      api_key: formData[modelKey].remote_openai.api_key,
      base_url: formData[modelKey].remote_openai.base_url,
      model_name: formData[modelKey].remote_openai.model,
    });

    if (response.data && response.data.code === 200) {
      ElMessage.success(`远程${llmConfigs[modelKey].label}API连接成功!`); // 修改这里：modelConfigs -> llmConfigs
    } else {
      throw new Error(response.data.message || '连接失败');
    }
  } catch (error) {
    console.error('测试远程API失败:', error);
    ElMessage.error(`测试失败: ${error.message || '连接错误'}`);
  }
};
// 测试平台Cookie
const testPlatformConnection = async (platform) => {
  try {
    ElMessage.info(`正在测试${getPlatformLabel(platform)}连接...`);
    const response = await axios.post('/api/system/test-platform-cookie', {
      platform,
      cookie: formData.cookies[platform],
    });

    if (response.data && response.data.code === 200) {
      ElMessage.success(`${getPlatformLabel(platform)}Cookie有效!`);
    } else {
      throw new Error(response.data.message || 'Cookie无效');
    }
  } catch (error) {
    console.error('测试Cookie失败:', error);
    ElMessage.error(`测试失败: ${error.message || 'Cookie无效或已过期'}`);
  }
};

// 测试爬虫连接
const testCrawlerConnection = async () => {
  try {
    ElMessage.info('正在测试爬虫服务连接...');
    const response = await axios.post('/api/system/test-crawler', {
      base_url: formData.crawler.base_url,
    });

    if (response.data && response.data.code === 200) {
      ElMessage.success('爬虫服务连接成功!');
    } else {
      throw new Error(response.data.message || '连接失败');
    }
  } catch (error) {
    console.error('测试爬虫连接失败:', error);
    ElMessage.error(`测试失败: ${error.message || '连接错误'}`);
  }
};

// 确认清除缓存
const confirmClearCache = () => {
  ElMessageBox.confirm(
    '确定要清除所有系统缓存吗？这可能会临时影响系统性能。',
    '清除缓存',
    {
      confirmButtonText: '确认',
      cancelButtonText: '取消',
      type: 'warning',
    },
  )
    .then(async () => {
      try {
        const response = await axios.post('/api/system/clear-cache');
        if (response.data && response.data.code === 200) {
          ElMessage.success('系统缓存已清除');
        } else {
          throw new Error(response.data.message || '操作失败');
        }
      } catch (error) {
        console.error('清除缓存失败:', error);
        ElMessage.error(`操作失败: ${error.message || '未知错误'}`);
      }
    })
    .catch(() => {});
};

// 确认重启服务
const confirmRestart = () => {
  ElMessageBox.confirm(
    '确定要重启系统服务吗？这将导致正在处理的任务中断。',
    '重启服务',
    {
      confirmButtonText: '确认',
      cancelButtonText: '取消',
      type: 'warning',
      distinguishCancelAndClose: true,
    },
  )
    .then(async () => {
      try {
        const response = await axios.post('/api/system/restart-service');
        if (response.data && response.data.code === 200) {
          ElMessage.success('服务重启指令已发送，系统将在几分钟内重启');
        } else {
          throw new Error(response.data.message || '操作失败');
        }
      } catch (error) {
        console.error('重启服务失败:', error);
        ElMessage.error(`操作失败: ${error.message || '未知错误'}`);
      }
    })
    .catch(() => {});
};

// 获取平台名称
const getPlatformLabel = (platformKey) => {
  const platform = platformConfigs.find((p) => p.key === platformKey);
  return platform ? platform.label : platformKey;
};

// 页面加载时获取设置
onMounted(() => {
  loadSettings();
});
</script>

<template>
  <div class="settings-container">
    <el-card class="settings-card">
      <!-- 头部保持不变 -->
      <template #header>
        <div class="card-header">
          <h2>系统设置</h2>
          <div class="action-buttons">
            <el-button type="primary" @click="saveSettings" :loading="saving">
              <el-icon><Check /></el-icon> 保存设置
            </el-button>
            <el-button @click="resetSettings" :disabled="saving">
              <el-icon><RefreshRight /></el-icon> 重置
            </el-button>
          </div>
        </div>
      </template>

      <el-tabs v-model="activeTab" tab-position="left" class="settings-tabs">
        <!-- 模型设置标签页 -->
        <el-tab-pane label="大语言模型" name="llm">
          <h3>大语言模型配置</h3>
          <p class="tab-description">配置系统使用的大语言模型及其相关参数</p>
          <el-alert
            type="info"
            show-icon
            :closable="false"
            title="API兼容性提示"
            style="margin-bottom: 20px"
          >
            <template #default>
              <p>
                本系统仅支持兼容OpenAI格式的API接口，请确保您选择的模型和API端点符合此格式要求。
              </p>
            </template>
          </el-alert>
          <div
            class="model-section"
            v-for="(modelType, key) in llmConfigs"
            :key="key"
          >
            <div class="section-header">
              <h4>{{ modelType.label }}</h4>
              <el-tooltip
                :content="`使用${formData[`local_${key}`] ? '本地' : '远程'}模型`"
              >
                <el-switch
                  v-model="formData[`local_${key}`]"
                  :active-text="'本地模型'"
                  :inactive-text="'远程API'"
                  class="model-switch"
                />
              </el-tooltip>
            </div>

            <!-- 本地模型配置 -->
            <div v-if="formData[`local_${key}`]" class="model-config-form">
              <el-form-item label="Ollama 基础URL">
                <el-input
                  v-model="formData[key].local_ollama.base_url"
                  placeholder="例如: http://localhost:11434/api/chat"
                />
              </el-form-item>
              <el-form-item label="模型名称">
                <el-select
                  v-model="formData[key].local_ollama.model"
                  placeholder="选择模型"
                >
                  <el-option label="deepseek-r1:14b" value="deepseek-r1:14b" />
                  <el-option label="llama3:8b" value="llama3:8b" />
                  <el-option label="mixtral:8x7b" value="mixtral:8x7b" />
                  <el-option label="qwen:14b" value="qwen:14b" />
                </el-select>
              </el-form-item>
              <el-button
                size="small"
                type="success"
                @click="testLocalModel(key)"
              >
                测试连接
              </el-button>
            </div>

            <!-- 远程模型配置 -->
            <div v-else class="model-config-form">
              <el-form-item label="API Key">
                <el-input
                  v-model="formData[key].remote_openai.api_key"
                  placeholder="输入API Key"
                  type="password"
                  show-password
                />
              </el-form-item>
              <el-form-item label="基础URL">
                <el-input
                  v-model="formData[key].remote_openai.base_url"
                  placeholder="例如: https://api.siliconflow.cn/v1"
                />
              </el-form-item>
              <el-form-item label="模型名称">
                <el-select
                  v-model="formData[key].remote_openai.model"
                  placeholder="选择模型"
                >
                  <el-option-group label="通义千问系列">
                    <el-option label="Qwen/Qwen3-8B" value="Qwen/Qwen3-8B" />
                    <el-option
                      label="Qwen/Qwen3-30B-A3B"
                      value="Qwen/Qwen3-30B-A3B"
                    />
                    <el-option label="Qwen/Qwen2-72B" value="Qwen/Qwen2-72B" />
                  </el-option-group>
                  <el-option-group label="其他模型">
                    <el-option label="gpt-4o" value="gpt-4o" />
                    <el-option label="gpt-3.5-turbo" value="gpt-3.5-turbo" />
                    <el-option label="Claude-3-Opus" value="Claude-3-Opus" />
                  </el-option-group>
                </el-select>
              </el-form-item>
              <el-button
                size="small"
                type="success"
                @click="testRemoteModel(key)"
              >
                测试API
              </el-button>
            </div>

            <el-divider v-if="key !== 'report'" />
          </div>
        </el-tab-pane>

        <!-- 新增 - 远程服务模型配置标签页 -->
        <el-tab-pane label="远程服务模型" name="remoteServices">
          <h3>远程服务模型配置</h3>
          <p class="tab-description">配置部署在远程的特定功能模型服务</p>

          <div class="model-service-container">
            <div
              class="model-section"
              v-for="(service, key) in remoteServiceConfigs"
              :key="key"
            >
              <div class="section-header">
                <h4>
                  <el-icon v-if="service.icon"
                    ><component :is="service.icon"
                  /></el-icon>
                  {{ service.label }}
                </h4>
                <el-switch
                  v-model="service.enabled"
                  class="service-switch"
                  :active-text="'启用'"
                />
              </div>

              <div v-if="service.enabled" class="model-config-form">
                <el-alert type="info" :closable="false" show-icon>
                  <template #default>
                    <p>{{ service.description }}</p>
                  </template>
                </el-alert>

                <el-form-item label="服务地址">
                  <el-input
                    v-model="formData.remoteServices[key].address"
                    placeholder="例如: http://192.168.1.100:3000"
                  >
                    <template #append>
                      <el-button @click="testRemoteService(key)"
                        >测试连接</el-button
                      >
                    </template>
                  </el-input>
                </el-form-item>
              </div>
              <el-divider
                v-if="key !== Object.keys(remoteServiceConfigs).pop()"
              />
            </div>
          </div>
        </el-tab-pane>

        <!-- 其他标签页保持不变 -->
        <!-- 社交平台配置标签页 -->
        <el-tab-pane label="社交平台配置" name="cookies">
          <h3>社交平台Cookie设置</h3>
          <p class="tab-description">配置各社交平台的访问凭证，用于内容获取</p>

          <div
            v-for="platform in platformConfigs"
            :key="platform.key"
            class="cookie-section"
          >
            <div class="section-header">
              <h4>
                <el-icon v-if="platform.icon"
                  ><component :is="platform.icon"
                /></el-icon>
                {{ platform.label }}
              </h4>
              <el-switch
                v-model="platform.enabled"
                class="platform-switch"
                :active-text="'启用'"
              />
            </div>

            <div v-if="platform.enabled" class="platform-config-form">
              <el-alert
                type="info"
                show-icon
                :closable="false"
                :title="`请配置${platform.label}的Cookie信息`"
              >
                <template #default>
                  <p>{{ platform.description }}</p>
                  <el-link
                    type="primary"
                    :href="platform.helpLink"
                    target="_blank"
                    >如何获取Cookie?</el-link
                  >
                </template>
              </el-alert>

              <el-form-item :label="`${platform.label} Cookie`">
                <el-input
                  v-model="formData.cookies[platform.key]"
                  type="textarea"
                  :rows="3"
                  placeholder="粘贴完整Cookie字符串"
                  show-password
                />
              </el-form-item>

              <el-button
                size="small"
                type="success"
                @click="testPlatformConnection(platform.key)"
              >
                测试连接
              </el-button>
            </div>
            <el-divider v-if="platform.key !== 'bilibili'" />
          </div>
        </el-tab-pane>

        <!-- 爬虫配置标签页 -->
        <el-tab-pane label="爬虫配置" name="crawler">
          <h3>爬虫服务配置</h3>
          <p class="tab-description">配置爬虫服务的连接参数</p>

          <el-form-item label="爬虫容器地址">
            <el-input
              v-model="formData.crawler.base_url"
              placeholder="例如: http://crawler-service:8080"
            >
              <template #append>
                <el-button @click="testCrawlerConnection">测试连接</el-button>
              </template>
            </el-input>
          </el-form-item>

          <el-form-item label="最大并发任务数">
            <el-input-number
              v-model="formData.crawler.max_concurrent_tasks"
              :min="1"
              :max="10"
            />
          </el-form-item>

          <el-divider />

          <h3>重试机制设置</h3>
          <div class="retry-settings">
            <el-form-item label="最大重试次数">
              <el-input-number
                v-model="formData.retry.max_retries"
                :min="1"
                :max="10"
              />
            </el-form-item>

            <el-form-item label="重试延迟(秒)">
              <el-input-number
                v-model="formData.retry.retry_delay"
                :min="1"
                :max="60"
                :step="1"
              />
            </el-form-item>
          </div>
        </el-tab-pane>

        <!-- 系统维护标签页 -->
        <el-tab-pane label="系统维护" name="system">
          <h3>系统维护</h3>
          <p class="tab-description">系统日志和维护操作</p>

          <div class="maintenance-actions">
            <el-card class="action-card">
              <template #header>
                <div class="action-header">
                  <h4>日志管理</h4>
                </div>
              </template>
              <p>查看和下载系统运行日志</p>
              <el-button type="primary">查看日志</el-button>
            </el-card>

            <el-card class="action-card danger-zone">
              <template #header>
                <div class="action-header">
                  <h4>危险操作区</h4>
                </div>
              </template>
              <p>以下操作可能会影响系统数据，请谨慎操作</p>
              <div class="danger-actions">
                <el-button type="danger" @click="confirmClearCache"
                  >清除缓存</el-button
                >
                <el-button type="danger" @click="confirmRestart"
                  >重启服务</el-button
                >
              </div>
            </el-card>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.settings-container {
  padding: 20px;
  height: 100%;
}

.settings-card {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h2 {
  margin: 0;
}

.action-buttons {
  display: flex;
  gap: 10px;
}

.settings-tabs {
  height: 100%;
}

:deep(.el-tabs__content) {
  padding: 0 20px;
  overflow-y: auto;
  height: calc(100vh - 200px);
}

.tab-description {
  color: #606266;
  margin-bottom: 20px;
}

/* 模型配置 */
.model-section {
  margin-bottom: 20px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.section-header h4 {
  margin: 0;
  font-weight: 500;
}

.model-switch,
.platform-switch {
  margin-left: 20px;
}

.model-config-form,
.platform-config-form {
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
}

/* Cookie配置 */
.cookie-section {
  margin-bottom: 20px;
}

/* 爬虫配置 */
.retry-settings {
  display: flex;
  gap: 20px;
}

/* 系统维护 */
.maintenance-actions {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.action-card {
  width: 100%;
  max-width: 400px;
  margin-bottom: 20px;
}

.action-header h4 {
  margin: 0;
}

.danger-zone {
  border: 1px solid #f56c6c;
}

.danger-actions {
  display: flex;
  gap: 10px;
}

.el-form-item {
  margin-bottom: 18px;
}

@media (max-width: 768px) {
  .retry-settings {
    flex-direction: column;
    gap: 0;
  }
}
</style>
