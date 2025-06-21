<script lang="ts" setup>
import {
  ref,
  onMounted,
  computed,
  watch,
  onBeforeUnmount,
  nextTick,
  h,
} from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import * as echarts from 'echarts/core';
import { PieChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
// 注册ECharts组件
echarts.use([
  PieChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);
import {
  ElAlert,
  ElAvatar,
  ElButton,
  ElCard,
  ElEmpty,
  ElIcon,
  ElImage,
  ElLoading,
  ElMessage,
  ElPagination,
  ElTable,
  ElTableColumn,
  ElTag,
  ElTooltip,
  ElInput,
  ElSelect,
  ElOption,
  ElMessageBox,
  ElProgress,
  ElDivider,
} from 'element-plus';

import {
  ArrowLeft,
  VideoPlay,
  Picture,
  Share,
  Star,
  Search,
  SortDown,
  Refresh,
  Timer,
  WarningFilled,
  InfoFilled,
  CircleCheckFilled,
  CircleCloseFilled,
  Delete,
  Loading,
  ChatLineRound,
  Location,
  UserFilled,
  Plus, // 添加这个 - 关注图标可能用到
  Collection, // 添加这个 - 收藏图标
} from '@element-plus/icons-vue';
const loadingStats = ref(false);
const statsData = ref({
  total_videos: 0,
  analyzed_videos: 0,
  pending_videos: 0,
  risk_distribution: [],
  analysis_status: [],
});
const analysisTemplate = ref('light'); // 默认完整分析
const analysisTemplates = ref([
  {
    value: 'full',
    label: '完整分析',
    description: '包含所有分析步骤',
  },
  {
    value: 'light',
    label: '轻量分析',
    description: '基础内容分析和数字人检测',
  },
  {
    value: 'content',
    label: '内容分析',
    description: '专注内容安全评估',
  },
]);
// 添加获取预计时间的函数
const getEstimatedTime = (template, videoCount = 1) => {
  const timePerVideo = {
    'full': 25,      // 完整分析：25分钟
    'light': 5,      // 轻量分析：5分钟
    'content': 3     // 内容分析：3分钟
  };
  
  const minutes = (timePerVideo[template] || 5) * videoCount;
  
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return remainingMinutes > 0 ? `${hours}小时${remainingMinutes}分钟` : `${hours}小时`;
  }
  
  return `${minutes}分钟`;
};
// 图表引用
const analysisStatusChart = ref(null);
const riskDistributionChart = ref(null);

// ECharts实例
let statusChartInstance = null;
let riskChartInstance = null;
const openDouyinProfile = () => {
  if (accountInfo.value?.sec_uid) {
    const url = `https://www.douyin.com/user/${accountInfo.value.sec_uid}`;
    window.open(url, '_blank');
  }
};
// 加载统计数据
const loadStats = async () => {
  if (!accountInfo.value?.id) return;

  try {
    loadingStats.value = true;

    const response = await axios.get(
      `/api/account/${accountInfo.value.id}/stats`,
    );

    if (response.data.code === 200) {
      statsData.value = response.data.data;

      // 初始化图表
      nextTick(() => {
        initCharts();
      });
    }
  } catch (error) {
    console.error('加载统计数据失败:', error);
    ElMessage.error('加载统计数据失败');
  } finally {
    loadingStats.value = false;
  }
};

// 初始化图表
const initCharts = () => {
  console.log('开始初始化图表', statsData.value);

  // 使用setTimeout确保DOM完全渲染
  setTimeout(() => {
    try {
      // 状态分析图
      if (analysisStatusChart.value) {
        console.log(
          '容器1尺寸:',
          analysisStatusChart.value.clientWidth,
          analysisStatusChart.value.clientHeight,
        );

        // 先销毁旧实例
        if (statusChartInstance) statusChartInstance.dispose();

        // 重新创建实例
        statusChartInstance = echarts.init(analysisStatusChart.value);

        // 应用选项
        const option = {
          tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)',
          },
          legend: {
            orient: 'vertical',
            left: 10,
            data: statsData.value.analysis_status.map((item) => item.name),
          },
          series: [
            {
              name: '分析状态',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2,
              },
              data: statsData.value.analysis_status,
              color: ['#67C23A', '#E6A23C', '#909399'],
            },
          ],
        };

        console.log('应用选项:', option);
        statusChartInstance.setOption(option);
        console.log('图表1初始化完成');
      }

      // 风险分析图
      if (riskDistributionChart.value) {
        console.log(
          '容器2尺寸:',
          riskDistributionChart.value.clientWidth,
          riskDistributionChart.value.clientHeight,
        );

        if (riskChartInstance) riskChartInstance.dispose();

        riskChartInstance = echarts.init(riskDistributionChart.value);

        const option = {
          tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)',
          },
          legend: {
            orient: 'vertical',
            left: 10,
            data: statsData.value.risk_distribution.map((item) => item.name),
          },
          series: [
            {
              name: '风险分布',
              type: 'pie',
              radius: ['40%', '70%'],
              avoidLabelOverlap: false,
              itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2,
              },
              data: statsData.value.risk_distribution,
              color: ['#67C23A', '#E6A23C', '#F56C6C', '#909399'],
            },
          ],
        };

        riskChartInstance.setOption(option);
        console.log('图表2初始化完成');
      }
    } catch (error) {
      console.error('图表初始化错误:', error);
    }
  }, 300); // 给DOM渲染足够的时间
};

// 监听窗口大小变化
const resizeCharts = () => {
  if (statusChartInstance) statusChartInstance.resize();
  if (riskChartInstance) riskChartInstance.resize();
};
const route = useRoute();
const router = useRouter();

// 获取路由参数
const platform = computed(() => route.query.platform as string);
const userId = computed(() => route.query.userId as string);

// 状态变量
const loading = ref(false);
const error = ref('');
const contentList = ref([]);
const totalItems = ref(0);
const accountInfo = ref(null);
const fetchingVideos = ref(false);

// 表格多选相关
const multipleSelection = ref([]);
const multipleTableRef = ref();

// 分页相关
const currentPage = ref(1);
const pageSize = ref(10);

// 搜索和过滤
const searchText = ref('');
const sortField = ref('create_time');
const sortOrder = ref('desc');

// 使用计算属性来确定平台名称
const platformName = computed(() => {
  switch (platform.value) {
    case 'douyin':
      return '抖音';
    case 'tiktok':
      return 'TikTok';
    case 'bilibili':
      return 'Bilibili';
    default:
      return platform.value;
  }
});
// 批量分析相关状态
const batchAnalyzing = ref(false);

// 检查是否有可分析的视频
const hasVideosToAnalyze = computed(() => {
  return multipleSelection.value.some(
    (video) => video.analysis_status !== 'completed' && !video.analyzing,
  );
});

const batchAnalyzeVideos = async () => {
  if (batchAnalyzing.value) {
    ElMessage.info('批量分析任务正在进行中，请稍候');
    return;
  }

  // 过滤出未分析和未在分析中的视频
  const videosToAnalyze = multipleSelection.value.filter(
    (video) => video.analysis_status !== 'completed' && !video.analyzing,
  );

  if (videosToAnalyze.length === 0) {
    ElMessage.info('没有可分析的视频，已选视频已全部分析或正在分析中');
    return;
  }

  try {
    // 获取当前选择的模板信息
    const selectedTemplate = analysisTemplates.value.find(
      (t) => t.value === analysisTemplate.value,
    );
    const templateName = selectedTemplate ? selectedTemplate.label : '未知模板';
    const templateDesc = selectedTemplate ? selectedTemplate.description : '';

    // 显示美化的确认对话框
    await ElMessageBox.confirm(
      '', // 主要内容放在message中
      '批量分析确认',
      {
        confirmButtonText: '立即开始分析',
        cancelButtonText: '取消',
        type: 'info',
        customClass: 'batch-analysis-confirm-dialog',
        showCancelButton: true,
        closeOnClickModal: false,
        // 修改 batchAnalyzeVideos 函数中的消息结构
        message: h('div', { class: 'analysis-confirm-content' }, [
          // 头部标题区
          h('div', { class: 'confirm-header' }, [
            h('div', { class: 'confirm-title' }, '即将启动批量分析'),
            h('div', { class: 'confirm-subtitle' }, '请确认以下分析配置'),
          ]),

          // 主要信息卡片 - 改为单列布局
          h('div', { class: 'confirm-info-card' }, [
            // 视频数量信息
            h('div', { class: 'info-row' }, [
              h('div', { class: 'info-label' }, [
                h('span', { style: 'margin-right: 6px;' }, '📹'),
                h('span', '分析对象'),
              ]),
              h(
                'div',
                { class: 'info-value highlight' },
                `${videosToAnalyze.length} 个视频`,
              ),
            ]),

            // 分析模板信息
            h('div', { class: 'info-row' }, [
              h('div', { class: 'info-label' }, [
                h('span', { style: 'margin-right: 6px;' }, '⚙️'),
                h('span', '分析模板'),
              ]),
              h('div', { class: 'info-value' }, [
                h('span', { class: 'template-name' }, templateName),
                h('div', { class: 'template-desc' }, templateDesc),
              ]),
            ]),
          ]),

          // 预计信息 - 使用新的时间计算
          h('div', { class: 'estimate-info' }, [
            h('div', { class: 'estimate-item' }, [
              h('span', { class: 'estimate-label' }, '⏱️ 预计用时：'),
              h(
                'span',
                { class: 'estimate-value' },
                getEstimatedTime(
                  analysisTemplate.value,
                  videosToAnalyze.length,
                ),
              ),
            ]),
            h('div', { class: 'estimate-item' }, [
              h('span', { class: 'estimate-label' }, '🔄 处理方式：'),
              h('span', { class: 'estimate-value' }, '逐个分析'),
            ]),
          ]),
        ]),
      },
    );

    batchAnalyzing.value = true;

    // 创建进度提示
    const loadingInstance = ElLoading.service({
      lock: true,
      text: `正在提交${templateName}分析任务 (0/${videosToAnalyze.length})`,
      background: 'rgba(0, 0, 0, 0.7)',
    });

    // 处理每个视频
    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < videosToAnalyze.length; i++) {
      const video = videosToAnalyze[i];

      // 更新加载提示
      loadingInstance.setText(
        `正在提交${templateName}分析任务 (${i + 1}/${videosToAnalyze.length})`,
      );

      try {
        // 设置分析中状态
        video.analyzing = true;
        video.analysis_progress = 0;

        // 提交分析请求
        const response = await axios.post(
          `/api/account/videos/${video.aweme_id}/analyze`,
          {
            template: analysisTemplate.value,
          },
          {
            headers: {
              'Content-Type': 'application/json',
            },
          },
        );

        if (response.data.code === 200) {
          successCount++;

          // 启动定时器检查分析状态
          if (analysisTimers.value[video.aweme_id]) {
            clearInterval(analysisTimers.value[video.aweme_id]);
          }

          analysisTimers.value[video.aweme_id] = setInterval(() => {
            checkAnalysisStatus(video);
          }, 3000); // 每3秒检查一次
        } else {
          throw new Error(response.data.message || '启动分析失败');
        }
      } catch (error) {
        console.error(`视频 ${video.aweme_id} 分析失败:`, error);
        failCount++;
        video.analyzing = false;
      }

      // 短暂延迟，避免API请求过于频繁
      await new Promise((resolve) => setTimeout(resolve, 300));
    }

    // 关闭加载提示
    loadingInstance.close();

    // 显示结果
    if (successCount > 0 && failCount === 0) {
      ElMessage.success(
        `成功提交 ${successCount} 个视频的${templateName}分析任务`,
      );
    } else if (successCount > 0 && failCount > 0) {
      ElMessage.warning(
        `成功提交 ${successCount} 个视频${templateName}分析任务，${failCount} 个视频提交失败`,
      );
    } else {
      ElMessage.error(`所有视频${templateName}分析任务提交失败`);
    }
  } catch (error) {
    if (error === 'cancel') {
      ElMessage.info('已取消批量分析');
    } else {
      console.error('批量分析失败:', error);
      ElMessage.error(`批量分析失败: ${error.message || '未知错误'}`);
    }
  } finally {
    batchAnalyzing.value = false;
  }
};
// 格式化数字(显示为1.2k, 3.5w等)
const formatNumber = (num) => {
  if (!num) return '0';
  if (num >= 10000) {
    return (num / 10000).toFixed(1) + 'w';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'k';
  }
  return num;
};

// 格式化日期
const formatDate = (timestamp) => {
  if (!timestamp) return '-';
  const date = new Date(
    typeof timestamp === 'number' ? timestamp * 1000 : timestamp,
  );
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
};
// 存储视频分析状态
const videoAnalysisStatus = ref({});
// 存储分析定时器
const analysisTimers = ref({});

// 获取分析按钮类型
const getAnalysisButtonType = (row) => {
  if (row.analyzing) return 'warning';
  if (row.analysis_status === 'completed') return 'success';
  if (row.analysis_status === 'failed') return 'danger';
  return 'primary';
};

// 获取分析按钮文字
const getAnalysisButtonText = (row) => {
  if (row.analyzing) return '分析中...';
  if (row.analysis_status === 'completed') return '已分析';
  if (row.analysis_status === 'failed') return '分析失败';
  return '分析视频';
};

// 获取风险等级标签类型
const getRiskLevelType = (level) => {
  if (!level) return 'info';
  switch (level.toLowerCase()) {
    case 'low':
      return 'success';
    case 'medium':
      return 'warning';
    case 'high':
      return 'danger';
    default:
      return 'info';
  }
};

// 获取风险等级文字
const getRiskLevelText = (level) => {
  if (!level) return '未知';
  switch (level.toLowerCase()) {
    case 'low':
      return '低风险';
    case 'medium':
      return '中风险';
    case 'high':
      return '高风险';
    default:
      return '未知';
  }
};

// 修改 analyzeVideo 函数
const analyzeVideo = async (row) => {
  if (row.analyzing) {
    ElMessage.info('视频正在分析中，请稍候');
    return;
  }

  if (row.analysis_status === 'completed') {
    ElMessage.info('视频已经分析过了');
    return;
  }

  try {
    // 获取当前选择的模板信息
    const selectedTemplate = analysisTemplates.value.find(
      (t) => t.value === analysisTemplate.value,
    );
    const templateName = selectedTemplate ? selectedTemplate.label : '未知模板';
    const templateDesc = selectedTemplate ? selectedTemplate.description : '';

    // 显示美化的确认对话框
    await ElMessageBox.confirm(
      '', // 主要内容放在message中
      '分析确认',
      {
        confirmButtonText: '开始分析',
        cancelButtonText: '取消',
        type: 'info',
        customClass: 'single-analysis-confirm-dialog',
        showCancelButton: true,
        closeOnClickModal: false,
        message: h('div', { class: 'analysis-confirm-content single' }, [
          // 头部区域
          h('div', { class: 'confirm-header' }, [
            h('div', { class: 'confirm-title' }, '视频分析确认'),
            h('div', { class: 'confirm-subtitle' }, '即将对此视频进行深度分析'),
          ]),

          // 分析配置卡片
          h('div', { class: 'confirm-info-card' }, [
            h('div', { class: 'info-row' }, [
              h('div', { class: 'info-label' }, [
                h('span', { style: 'margin-right: 8px;' }, '⚙️'),
                h('span', '分析模板'),
              ]),
              h('div', { class: 'info-value' }, [
                h('span', { class: 'template-name' }, templateName),
                h('div', { class: 'template-desc' }, templateDesc),
              ]),
            ]),
          ]),

          // 快速提示
          h('div', { class: 'quick-tips' }, [
            h(
              'div',
              { class: 'tip-item' },
              `⏱️ 预计用时：${getEstimatedTime(analysisTemplate.value)}`,
            ),
            h('div', { class: 'tip-item' }, '📊 完成后可查看详细报告'),
          ]),
        ]),
      },
    );

    // 设置分析中状态
    row.analyzing = true;
    row.analysis_progress = 0;

    // 调用分析API
    const response = await axios.post(
      `/api/account/videos/${row.aweme_id}/analyze`,
      {
        template: analysisTemplate.value,
      },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      },
    );

    if (response.data.code === 200) {
      ElMessage.success(`${templateName}分析任务已启动`);

      // 启动定时器检查分析状态
      if (analysisTimers.value[row.aweme_id]) {
        clearInterval(analysisTimers.value[row.aweme_id]);
      }

      analysisTimers.value[row.aweme_id] = setInterval(() => {
        checkAnalysisStatus(row);
      }, 3000); // 每3秒检查一次
    } else {
      throw new Error(response.data.message || '启动分析失败');
    }
  } catch (error) {
    if (error === 'cancel') {
      ElMessage.info('已取消分析');
    } else {
      console.error('分析视频失败:', error);
      ElMessage.error(`分析失败: ${error.message || '未知错误'}`);
    }
    row.analyzing = false;
  }
};
// 检查分析状态
const checkAnalysisStatus = async (row) => {
  try {
    const response = await axios.get(
      `/api/account/videos/${row.aweme_id}/analysis-status`,
    );

    if (response.data.code === 200) {
      const statusData = response.data.data;
      const previousStatus = row.analysis_status; // 保存之前的状态

      // 更新状态
      row.analysis_status = statusData.status;

      if (statusData.status === 'processing') {
        row.analysis_progress = statusData.progress || 0;
      } else if (statusData.status === 'completed') {
        // 分析完成
        row.analyzing = false;
        row.risk_level = statusData.risk_level;
        row.risk_probability = statusData.risk_probability;

        // 停止定时器
        if (analysisTimers.value[row.aweme_id]) {
          clearInterval(analysisTimers.value[row.aweme_id]);
          delete analysisTimers.value[row.aweme_id];

          // 只有当视频是从"处理中"变为"已完成"时，才显示消息
          // 或者是手动触发的分析（row.analyzing为true时）
          ElMessage.success('视频分析已完成');
        }
      } else if (statusData.status === 'failed') {
        // 分析失败
        row.analyzing = false;

        // 停止定时器
        if (analysisTimers.value[row.aweme_id]) {
          clearInterval(analysisTimers.value[row.aweme_id]);
          delete analysisTimers.value[row.aweme_id];
          ElMessage.error(`分析失败: ${statusData.error || '未知错误'}`);
        }
      }
    }
  } catch (error) {
    console.error('检查分析状态失败:', error);
  }
};

// 查看分析报告
const viewAnalysisReport = (row) => {
  if (row.video_file_id) {
    router.push(`/demos/analysis-records/analysis?id=${row.video_file_id}`);
  } else {
    ElMessage.warning('无法查看分析报告，缺少视频文件ID');
  }
};
// 在加载视频列表后检查分析状态
const checkAllAnalysisStatus = () => {
  contentList.value.forEach((row) => {
    // 如果有video_file_id，检查分析状态
    if (row.video_file_id) {
      checkAnalysisStatus(row);
    }
  });
};
// 加载用户信息
const loadUserInfo = async () => {
  try {
    loading.value = true;

    // 尝试从数据库获取用户信息
    const dbResponse = await axios.get(
      `/api/account/by-secuid/${userId.value}`,
    );

    if (dbResponse.data.code === 200 && dbResponse.data.data) {
      // 数据库中已有用户数据
      accountInfo.value = dbResponse.data.data;
      console.log('从数据库加载用户信息成功:', accountInfo.value);

      // 有了用户ID，加载其视频列表
      loadVideosFromDB();
      return;
    }

    // 如果数据库没有，尝试从抖音API获取
    if (platform.value === 'douyin') {
      const response = await axios.get(
        `/api/douyin/web/handler_user_profile?sec_user_id=${userId.value}`,
      );

      if (response.data.code === 200) {
        accountInfo.value = response.data.data.user;
        console.log('从抖音API加载用户信息成功:', accountInfo.value);
      } else {
        throw new Error(response.data.message || '获取用户信息失败');
      }
    }
  } catch (err) {
    console.error('加载用户信息失败:', err);
    error.value = '获取用户信息失败';
  } finally {
    loading.value = false;
  }
};

// 从数据库加载视频
const loadVideosFromDB = async () => {
  if (!accountInfo.value?.id) {
    console.log('无法加载视频：缺少用户ID');
    return;
  }

  try {
    loading.value = true;
    console.log('正在从数据库加载视频列表...');

    // 构建API请求参数
    const params = {
      page: currentPage.value,
      per_page: pageSize.value,
      sort_by: sortField.value,
      sort_order: sortOrder.value,
      search: searchText.value || undefined,
    };

    // 从后端API获取视频列表
    const response = await axios.get(
      `/api/account/${accountInfo.value.id}/videos`,
      { params },
    );

    if (response.data.code === 200) {
      contentList.value = response.data.data.videos || [];
      totalItems.value = response.data.data.total || 0;
      console.log('视频列表加载成功，共', totalItems.value, '条记录');
      // 加载完视频列表后检查分析状态
      checkAllAnalysisStatus();
    } else {
      throw new Error(response.data.message || '获取视频列表失败');
    }
  } catch (err) {
    console.error('加载视频列表失败:', err);
    error.value = err.message || '获取视频列表失败';
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
};

// 获取最新视频
const fetchLatestVideos = async () => {
  if (!accountInfo.value?.id) {
    ElMessage.warning('无法获取视频：缺少用户ID');
    return;
  }

  try {
    fetchingVideos.value = true;

    // 显示加载中提示
    const loading = ElLoading.service({
      lock: true,
      text: '正在获取最新视频...',
      background: 'rgba(0, 0, 0, 0.7)',
    });

    // 调用后端API获取最新视频
    const response = await axios.post(
      `/api/account/${accountInfo.value.id}/fetch_videos`,
      {
        max_videos: 30, // 最多获取30个视频
      },
    );

    if (response.data.code === 200) {
      const videosAdded = response.data.data.videos_added;
      ElMessage.success(`成功获取${videosAdded}个视频`);

      // 重新加载视频列表
      await loadVideosFromDB();
    } else {
      throw new Error(response.data.message || '获取视频失败');
    }
  } catch (err) {
    console.error('获取最新视频失败:', err);
    ElMessage.error(err.message || '获取视频失败，请稍后重试');
  } finally {
    fetchingVideos.value = false;
    if (ElLoading.service) {
      ElLoading.service().close();
    }
  }
};

// 处理搜索
const handleSearch = () => {
  currentPage.value = 1; // 重置到第一页
  loadVideosFromDB();
};

// 处理排序变化
const handleSortChange = (column) => {
  if (column.prop) {
    sortField.value = column.prop;
    sortOrder.value = column.order === 'ascending' ? 'asc' : 'desc';
    loadVideosFromDB();
  }
};

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val;
  loadVideosFromDB();
};

// 处理每页大小变化
const handleSizeChange = (val) => {
  pageSize.value = val;
  currentPage.value = 1; // 重置到第一页
  loadVideosFromDB();
};

// 使用路由名称导航
const handleRowClick = (row) => {
  localStorage.setItem('lastProfileId', accountInfo.value.id);

  router.push({
    name: 'VideoProcessingDetails',
    query: {
      awemeId: row.aweme_id,
      id: row.video_file_id,
    },
  });
};

// 处理多选变化
const handleSelectionChange = (val) => {
  multipleSelection.value = val;
  console.log('已选择视频:', multipleSelection.value.length);
};

// 清除选择
const clearSelection = () => {
  if (multipleTableRef.value) {
    multipleTableRef.value.clearSelection();
  }
};

// 根据分享URL生成短链接
const getShortShareUrl = (url) => {
  if (!url) return '-';
  try {
    const urlObj = new URL(url);
    return urlObj.hostname + '/...';
  } catch (e) {
    return url.substring(0, 20) + '...';
  }
};

// 监听搜索关键词变化
watch(searchText, (value) => {
  if (!value) {
    handleSearch(); // 当搜索框清空时，自动重新加载
  }
});
// 在视频列表加载后加载统计数据
watch(
  () => contentList.value,
  (newVal, oldVal) => {
    // 只在初始加载时触发一次，而不是每次列表变化都触发
    if (newVal && newVal.length > 0 && (!oldVal || oldVal.length === 0)) {
      loadStats();
    }
  },
);

// 初始加载
// 组件挂载时的处理
onMounted(() => {
  // 添加窗口大小变化监听
  window.addEventListener('resize', resizeCharts);

  // 初始加载
  if (!platform.value || !userId.value) {
    error.value = '缺少必要的参数';
    return;
  }
  loadUserInfo();

  // 调试
  console.log(
    '组件已挂载，图表容器:',
    analysisStatusChart.value,
    riskDistributionChart.value,
  );
});
// 组件卸载时清理所有定时器
onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts);
  if (statusChartInstance) statusChartInstance.dispose();
  if (riskChartInstance) riskChartInstance.dispose();

  Object.values(analysisTimers.value).forEach((timer) => {
    clearInterval(timer);
  });
  analysisTimers.value = {};
});
</script>

<template>
  <div
    v-if="$route.path === '/main/analysis-tasks/user-content'"
    class="user-content-container"
  >
    <!-- 错误提示 -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      show-icon
      :closable="false"
      class="error-alert"
    />

    <!-- 重新设计的用户信息卡片 -->
    <el-card v-if="accountInfo" class="user-card">
      <template #header>
        <div class="card-header">
          <span class="card-header-title">账号详情</span>
          <div class="card-header-actions">
            <!-- 新增：访问抖音主页按钮 -->
            <el-button
              type="primary"
              size="small"
              plain
              @click="openDouyinProfile"
              v-if="accountInfo && accountInfo.sec_uid"
            >
              <el-icon><Share /></el-icon>
              访问抖音主页
            </el-button>
            <el-button
              type="primary"
              :loading="fetchingVideos"
              @click="fetchLatestVideos"
              size="small"
              plain
            >
              <el-icon><Refresh /></el-icon>
              获取最新视频
            </el-button>
            <el-button
              type="primary"
              size="small"
              plain
              @click="loadStats"
              :loading="loadingStats"
            >
              <el-icon><Refresh /></el-icon>
              刷新分析数据
            </el-button>
          </div>
        </div>
      </template>

      <!-- 简化的用户信息布局 -->
      <div class="user-info-container">
        <!-- 左侧：用户头像 -->
        <div class="user-avatar-section">
          <div class="account-avatar-container">
            <el-avatar
              :size="100"
              :src="accountInfo.avatar_medium || accountInfo.avatar"
              class="account-avatar"
            />
            <div v-if="accountInfo.custom_verify" class="verified-badge">
              <el-tooltip :content="accountInfo.custom_verify">
                <el-icon><CircleCheckFilled /></el-icon>
              </el-tooltip>
            </div>
          </div>
        </div>

        <!-- 右侧：基本用户信息 -->
        <div class="user-details-section">
          <div class="account-header">
            <h2 class="account-name">{{ accountInfo.nickname }}</h2>
            <div class="account-id">
              ID: {{ accountInfo.sec_uid?.substring(0, 20) }}...
            </div>
          </div>

          <div class="account-stats">
            <div class="stat-item">
              <div class="stat-value">
                {{ formatNumber(accountInfo.following_count) }}
              </div>
              <div class="stat-label">关注</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">
                {{ formatNumber(accountInfo.follower_count) }}
              </div>
              <div class="stat-label">粉丝</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">
                {{ formatNumber(accountInfo.total_favorited || 0) }}
              </div>
              <div class="stat-label">获赞</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">
                {{ formatNumber(accountInfo.aweme_count) }}
              </div>
              <div class="stat-label">作品</div>
            </div>
          </div>

          <!-- 用户地理位置 -->
          <div v-if="accountInfo.ip_location" class="account-location">
            <el-icon><Location /></el-icon> {{ accountInfo.ip_location }}
          </div>

          <!-- 用户简介 -->
          <div v-if="accountInfo.signature" class="account-bio">
            {{ accountInfo.signature }}
          </div>
        </div>
      </div>
      <div class="analysis-section">
        <el-divider content-position="center">内容分析概览</el-divider>
        <div class="analysis-overview-section">
          <div class="stats-container">
            <div v-if="loadingStats" class="stats-loading">
              <el-icon class="rotating"><Loading /></el-icon>
              <span>加载统计数据...</span>
            </div>
            <div v-else-if="!statsData.total_videos" class="stats-empty">
              <el-empty description="暂无统计数据">
                <template #description>
                  <p>尚未采集到视频数据或尚未进行分析</p>
                </template>
              </el-empty>
            </div>
            <div v-else class="stats-content">
              <!-- 统计数字 -->
              <div class="stats-summary">
                <div class="stat-item">
                  <div class="stat-value">{{ statsData.total_videos }}</div>
                  <div class="stat-label">总视频数</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ statsData.analyzed_videos }}</div>
                  <div class="stat-label">已分析</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ statsData.pending_videos }}</div>
                  <div class="stat-label">待分析</div>
                </div>
              </div>

              <!-- 图表区域 -->
              <div class="charts-container">
                <!-- 分析状态图表 -->
                <div class="chart-wrapper">
                  <h4>分析状态分布</h4>
                  <div
                    ref="analysisStatusChart"
                    class="echarts-container"
                  ></div>
                </div>

                <!-- 风险级别图表 -->
                <div class="chart-wrapper">
                  <h4>风险级别分布</h4>
                  <div
                    ref="riskDistributionChart"
                    class="echarts-container"
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </el-card>
    <!-- 视频列表表格 -->
    <el-card class="content-table-card">
      <div class="table-header">
        <div class="table-title">
          <h3>发布视频列表</h3>
          <span class="video-count">共 {{ totalItems }} 条内容</span>
        </div>
        <!-- 新增：分析模板选择器 - 放在中间位置 -->
        <div class="template-selector-section">
          <span class="template-label">分析模板：</span>
          <el-select
            v-model="analysisTemplate"
            size="small"
            style="width: 140px"
          >
            <el-option
              v-for="template in analysisTemplates"
              :key="template.value"
              :label="template.label"
              :value="template.value"
            >
              <div
                style="
                  display: flex;
                  justify-content: space-between;
                  width: 100%;
                "
              >
                <span>{{ template.label }}</span>
                <span style="color: #8492a6; font-size: 12px; margin-left: 8px">
                  {{ template.description }}
                </span>
              </div>
            </el-option>
          </el-select>
        </div>
        <!-- 中间：表格操作区域 -->
        <div v-if="multipleSelection.length > 0" class="table-operations">
          <span class="selected-count"
            >已选择 {{ multipleSelection.length }} 项</span
          >
          <el-button size="small" @click="clearSelection">清除选择</el-button>
          <el-button
            size="small"
            type="primary"
            @click="batchAnalyzeVideos"
            :loading="batchAnalyzing"
            :disabled="!hasVideosToAnalyze"
          >
            批量分析
          </el-button>
        </div>

        <!-- 右侧：搜索框 -->
        <div class="search-box">
          <el-input
            v-model="searchText"
            placeholder="搜索标题或标签"
            class="search-input"
            clearable
            @keyup.enter="handleSearch"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <el-button type="primary" @click="handleSearch">搜索</el-button>
        </div>
      </div>

      <!-- 数据表格 -->
      <el-table
        ref="multipleTableRef"
        :data="contentList"
        border
        stripe
        style="width: 100%"
        v-loading="loading"
        @selection-change="handleSelectionChange"
        @sort-change="handleSortChange"
      >
        <el-table-column type="selection" width="55" />

        <!-- 封面列 -->
        <el-table-column label="封面" width="100" align="center">
          <template #default="{ row }">
            <el-image
              :src="row.cover_url"
              fit="cover"
              style="width: 70px; height: 90px; border-radius: 4px"
              :preview-src-list="[row.cover_url]"
            >
              <template #error>
                <div class="image-placeholder">
                  <el-icon><Picture /></el-icon>
                </div>
              </template>
            </el-image>
          </template>
        </el-table-column>

        <!-- 标题/描述列 -->
        <el-table-column label="标题" prop="desc" min-width="240">
          <template #default="{ row }">
            <div class="video-title-cell">
              <el-tooltip :content="row.desc" placement="top" effect="light">
                <div class="multiline-text video-title">
                  {{ row.desc || '无标题' }}
                </div>
              </el-tooltip>

              <!-- 标签 -->
              <div class="video-tags" v-if="row.tags && row.tags.length > 0">
                <el-tag
                  v-for="(tag, index) in row.tags.slice(0, 3)"
                  :key="index"
                  size="small"
                  effect="plain"
                  class="video-tag"
                >
                  {{ tag }}
                </el-tag>
                <el-tag size="small" effect="plain" v-if="row.tags.length > 3">
                  +{{ row.tags.length - 3 }}
                </el-tag>
              </div>
            </div>
          </template>
        </el-table-column>

        <!-- 发布时间列 -->
        <el-table-column
          label="发布时间"
          prop="create_time"
          width="120"
          sortable="custom"
        >
          <template #default="{ row }">
            <div class="time-cell">
              <el-icon><Timer /></el-icon>
              <span>{{ formatDate(row.create_time) }}</span>
            </div>
          </template>
        </el-table-column>
        <!-- 添加风险等级列 -->
        <el-table-column label="风险等级" width="120" align="center">
          <template #default="{ row }">
            <div v-if="row.analysis_status === 'completed'">
              <el-tag :type="getRiskLevelType(row.risk_level)" effect="dark">
                {{ getRiskLevelText(row.risk_level) }}
              </el-tag>
            </div>
            <div v-else-if="row.analyzing">
              <el-progress
                type="circle"
                :width="30"
                :stroke-width="4"
                :percentage="row.analysis_progress || 0"
              />
            </div>
            <div v-else>
              <el-tag type="info" effect="plain">未分析</el-tag>
            </div>
          </template>
        </el-table-column>
        <!-- 点赞数列 -->
        <el-table-column
          label="点赞数"
          prop="digg_count"
          width="100"
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><Star /></el-icon>
              <span>{{
                formatNumber(row.statistics?.digg_count || row.digg_count)
              }}</span>
            </div>
          </template>
        </el-table-column>

        <!-- 评论数列 -->
        <el-table-column
          label="评论数"
          prop="comment_count"
          width="100"
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><ChatLineRound /></el-icon>
              <!-- 替换 Chat -->
              <span>{{
                formatNumber(row.statistics?.comment_count || row.comment_count)
              }}</span>
            </div>
          </template>
        </el-table-column>

        <!-- 分享数列 -->
        <el-table-column
          label="分享数"
          prop="share_count"
          width="100"
          sortable="custom"
          align="center"
        >
          <template #default="{ row }">
            <div class="stat-cell">
              <el-icon><Share /></el-icon>
              <span>{{
                formatNumber(row.statistics?.share_count || row.share_count)
              }}</span>
            </div>
          </template>
        </el-table-column>

        <!-- 分享链接列 -->
        <el-table-column label="分享链接" width="150">
          <template #default="{ row }">
            <el-tooltip :content="row.share_url" placement="top" effect="light">
              <a :href="row.share_url" target="_blank" class="share-link">
                {{ getShortShareUrl(row.share_url) }}
              </a>
            </el-tooltip>
          </template>
        </el-table-column>
        <!-- 在操作列添加分析按钮 -->
        <!-- 操作列按钮 -->
        <el-table-column label="操作" width="240" fixed="right" align="center">
          <template #default="{ row }">
            <el-button
              size="small"
              type="primary"
              link
              @click.stop="handleRowClick(row)"
            >
              分析详情
            </el-button>

            <!-- 只有未分析完成的视频才显示分析按钮 -->
            <el-button
              v-if="row.analysis_status !== 'completed'"
              size="small"
              :type="getAnalysisButtonType(row)"
              link
              :loading="row.analyzing"
              @click.stop="analyzeVideo(row)"
            >
              {{ getAnalysisButtonText(row) }}
            </el-button>

            <!-- 已分析视频显示分析报告按钮 -->
            <el-button
              v-if="row.analysis_status === 'completed'"
              size="small"
              type="success"
              link
              @click.stop="viewAnalysisReport(row)"
            >
              分析报告
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <!-- 分页组件 -->
      <div class="pagination-container" v-if="totalItems > 0">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="totalItems"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>

      <!-- 空状态 -->
      <el-empty
        v-if="contentList.length === 0 && !loading"
        description="暂无视频数据"
      >
        <template #default>
          <div class="empty-action">
            <p>此账号尚未收集视频数据</p>
            <el-button type="primary" @click="fetchLatestVideos">
              <el-icon><Refresh /></el-icon>
              获取视频数据
            </el-button>
          </div>
        </template>
      </el-empty>
    </el-card>
  </div>
  <router-view v-else />
</template>

<style scoped>
/* 页面容器样式 */
.user-info-container {
  display: flex;
  gap: 30px;
  margin-bottom: 30px;
}

.user-avatar-section {
  flex-shrink: 0;
}

.user-details-section {
  flex: 1;
}

.account-name {
  font-size: 22px;
  margin: 0 0 4px 0;
}

.account-id {
  color: #909399;
  font-size: 14px;
  margin-bottom: 15px;
}

.account-stats {
  display: flex;
  gap: 30px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.account-location {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #606266;
  margin: 10px 0;
  font-size: 14px;
}

/* 在移动设备上调整为上下布局 */
@media (max-width: 768px) {
  .user-info-container {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .account-stats {
    justify-content: center;
  }

  .account-location,
  .account-bio {
    justify-content: center;
  }
}
.user-content-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* 错误提示样式 */
.error-alert {
  margin-bottom: 20px;
}

/* 用户卡片样式 */
.user-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header-title {
  font-size: 18px;
  font-weight: 500;
}

.card-header-actions {
  display: flex;
  gap: 10px;
}

/* 用户头像和信息样式 */
.account-avatar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  margin-bottom: 16px;
}

.account-avatar {
  border: 3px solid #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  margin-bottom: 12px;
}

.verified-badge {
  position: absolute;
  bottom: 65px;
  right: calc(50% - 50px);
  background: #409eff;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  border: 2px solid white;
}

.account-name {
  font-size: 22px;
  margin: 0 0 4px 0;
  text-align: center;
}

.account-id {
  color: #909399;
  font-size: 14px;
  text-align: center;
  margin-bottom: 12px;
}

.account-info-col {
  display: flex;
  flex-direction: column;
}

.account-stats {
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.account-bio {
  margin: 16px 0;
  line-height: 1.5;
  color: #606266;
  word-break: break-word;
}

.account-location {
  color: #606266;
  margin-bottom: 8px;
  font-size: 14px;
}

/* 分析概览区域样式 */
.analysis-section {
  margin-top: 20px;
}

/* 表格区域样式 */
.content-table-card {
  margin-bottom: 20px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.table-title {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.table-title h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.video-count {
  color: #909399;
  font-size: 14px;
}

.table-operations {
  display: flex;
  align-items: center;
  gap: 10px;
}

.selected-count {
  background-color: #f0f9eb;
  color: #67c23a;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 13px;
}

.search-box {
  display: flex;
  gap: 10px;
}

.search-input {
  width: 220px;
}

/* 视频标题与标签样式 */
.video-title-cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.multiline-text {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.5;
}

.video-title {
  font-weight: 500;
  color: #303133;
}

.video-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.video-tag {
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 数据单元格样式 */
.stat-cell,
.time-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
}

.time-cell {
  justify-content: flex-start;
}

.image-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 70px;
  height: 90px;
  background-color: #f5f7fa;
  color: #909399;
}

/* 分页和链接样式 */
.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.share-link {
  color: #409eff;
  text-decoration: none;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 空状态样式 */
.empty-action {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #909399;
}

/* 表格行点击效果 */
:deep(.el-table__row) {
  cursor: pointer;
}

:deep(.el-table__row:hover) {
  background-color: rgba(64, 158, 255, 0.08) !important;
}

/* 图表相关样式 */
.stats-container {
  min-height: 300px;
}

.charts-container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: space-around;
  margin-top: 20px;
}

.chart-wrapper {
  flex: 1 1 400px;
  min-width: 300px;
  border: 1px solid #eee;
  padding: 10px;
}

.echarts-container {
  height: 300px;
  width: 100%;
  background-color: #fafafa;
}

/* 统计数据加载状态 */
.stats-loading,
.stats-empty {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 300px;
  color: #909399;
}

.rotating {
  animation: rotate 2s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 统计数字样式 */
.stats-summary {
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
  padding: 20px;
  background: #f8f8f8;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.stat-item {
  padding: 15px 25px;
  text-align: center;
  transition: all 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-5px);
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #606266;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .account-avatar-col,
  .account-info-col {
    text-align: center;
  }

  .account-stats {
    justify-content: center;
  }

  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .card-header-actions {
    width: 100%;
    justify-content: space-around;
  }

  .table-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .search-box {
    width: 100%;
  }

  .search-input {
    flex: 1;
  }
}
</style>
<style>
/* 修改按钮样式 - 去掉紫色渐变 */
.batch-analysis-confirm-dialog .el-button--primary {
  background: #409eff !important;
  border: 1px solid #409eff !important;
  border-radius: 8px !important;
  padding: 10px 20px !important;
  font-weight: 500 !important;
}

.batch-analysis-confirm-dialog .el-button--primary:hover {
  background: #66b1ff !important;
  border-color: #66b1ff !important;
}

.single-analysis-confirm-dialog .el-button--primary {
  background: #409eff !important;
  border: 1px solid #409eff !important;
  border-radius: 8px !important;
  padding: 10px 20px !important;
  font-weight: 500 !important;
}

.single-analysis-confirm-dialog .el-button--primary:hover {
  background: #66b1ff !important;
  border-color: #66b1ff !important;
}

/* 修复布局问题 - 信息卡片使用单列布局 */
.confirm-info-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
  border: 1px solid #e3e6ea;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0;
  min-height: 24px;
}

.info-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  color: #495057;
  font-size: 14px;
  flex: 0 0 auto;
  min-width: 80px;
}

.info-value {
  text-align: right;
  flex: 1;
  margin-left: 12px;
}

.info-value.highlight {
  color: #409eff;
  font-weight: 600;
  font-size: 16px;
}

.template-name {
  color: #409eff;
  font-weight: 600;
  font-size: 14px;
  display: block;
  text-align: right;
}

.template-desc {
  color: #6c757d;
  font-size: 12px;
  margin-top: 2px;
  line-height: 1.3;
  text-align: right;
  max-width: none;
}

/* 响应式优化 */
@media (max-width: 600px) {
  .info-row {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    margin-bottom: 8px;
  }

  .info-label {
    font-size: 13px;
    min-width: auto;
  }

  .info-value {
    text-align: left;
    margin-left: 0;
    width: 100%;
  }

  .template-name,
  .template-desc {
    text-align: left;
  }
}
/* 优化后的确认对话框样式 */
.batch-analysis-confirm-dialog .el-message-box {
  min-width: 480px !important;
  max-width: 520px !important;
  border-radius: 12px !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15) !important;
}

.batch-analysis-confirm-dialog .el-message-box__header {
  padding: 20px 20px 15px !important;
  border-bottom: 1px solid #f0f0f0 !important;
}

.batch-analysis-confirm-dialog .el-message-box__title {
  font-size: 18px !important;
  font-weight: 600 !important;
  color: #2c3e50 !important;
}

.batch-analysis-confirm-dialog .el-message-box__content {
  padding: 0 !important;
}

.batch-analysis-confirm-dialog .el-message-box__btns {
  padding: 15px 20px 20px !important;
  border-top: 1px solid #f0f0f0 !important;
}

.batch-analysis-confirm-dialog .el-button--primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 20px !important;
  font-weight: 500 !important;
}

.batch-analysis-confirm-dialog .el-button--default {
  border-radius: 8px !important;
  padding: 10px 20px !important;
}

.single-analysis-confirm-dialog .el-message-box {
  min-width: 420px !important;
  max-width: 460px !important;
  border-radius: 12px !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15) !important;
}

.single-analysis-confirm-dialog .el-message-box__header {
  padding: 20px 20px 15px !important;
  border-bottom: 1px solid #f0f0f0 !important;
}

.single-analysis-confirm-dialog .el-message-box__title {
  font-size: 18px !important;
  font-weight: 600 !important;
  color: #2c3e50 !important;
}

.single-analysis-confirm-dialog .el-message-box__content {
  padding: 0 !important;
}

.single-analysis-confirm-dialog .el-message-box__btns {
  padding: 15px 20px 20px !important;
  border-top: 1px solid #f0f0f0 !important;
}

.single-analysis-confirm-dialog .el-button--primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 20px !important;
  font-weight: 500 !important;
}

/* 优化确认内容布局 */
.analysis-confirm-content {
  padding: 16px 20px !important;
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.confirm-header {
  text-align: center;
  margin-bottom: 20px;
}

.confirm-title {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 6px;
}

.confirm-subtitle {
  font-size: 13px;
  color: #7f8c8d;
}

/* 紧凑的信息卡片样式 */
.confirm-info-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
  border: 1px solid #e3e6ea;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  min-height: 24px;
}

.info-row:last-child {
  margin-bottom: 0;
}

.info-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  color: #495057;
  font-size: 14px;
  flex: 0 0 auto;
}

.info-value {
  text-align: right;
  flex: 1;
  margin-left: 12px;
}

.info-value.highlight {
  color: #667eea;
  font-weight: 600;
  font-size: 16px;
}

.template-name {
  color: #667eea;
  font-weight: 600;
  font-size: 14px;
  display: block;
  text-align: right;
}

.template-desc {
  color: #6c757d;
  font-size: 12px;
  margin-top: 2px;
  line-height: 1.3;
  text-align: right;
}

/* 更紧凑的预计信息 */
.estimate-info {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 16px;
}

.estimate-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  font-size: 13px;
}

.estimate-item:last-child {
  margin-bottom: 0;
}

.estimate-label {
  color: #856404;
  font-weight: 500;
}

.estimate-value {
  color: #856404;
  font-weight: 600;
}

/* 更紧凑的提示区域 */
.tips-section {
  background: #e7f3ff;
  border: 1px solid #b3d8ff;
  border-radius: 8px;
  padding: 12px;
}

.tips-title {
  font-weight: 600;
  color: #0066cc;
  margin-bottom: 8px;
  font-size: 13px;
}

.tips-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tip-item {
  color: #0066cc;
  font-size: 12px;
  line-height: 1.3;
}

/* 更紧凑的快速提示 */
.quick-tips {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  padding: 12px;
}

.quick-tips .tip-item {
  color: #495057;
  font-size: 12px;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  background: white;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

/* 单个分析对话框的优化 */
.analysis-confirm-content.single {
  padding: 14px 20px !important;
}

.analysis-confirm-content.single .confirm-header {
  margin-bottom: 16px;
}

.analysis-confirm-content.single .confirm-info-card {
  margin-bottom: 12px;
}

/* 模板选择器也优化一下 */
.template-selector-section {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 6px;
  border: 1px solid #dee2e6;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.template-label {
  font-size: 13px;
  color: #495057;
  font-weight: 500;
  white-space: nowrap;
}

/* 响应式优化 */
@media (max-width: 600px) {
  .batch-analysis-confirm-dialog .el-message-box,
  .single-analysis-confirm-dialog .el-message-box {
    min-width: 90vw !important;
    max-width: 95vw !important;
    margin: 0 2.5vw !important;
  }

  .info-row {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    margin-bottom: 10px;
  }

  .info-label {
    font-size: 13px;
  }

  .info-value {
    text-align: left;
    margin-left: 0;
    width: 100%;
  }

  .template-name,
  .template-desc {
    text-align: left;
  }

  .estimate-item {
    flex-direction: column;
    gap: 2px;
    margin-bottom: 8px;
  }

  .quick-tips {
    grid-template-columns: 1fr;
    gap: 8px;
  }

  .quick-tips .tip-item {
    font-size: 11px;
    padding: 6px;
  }
}

/* 额外的紧凑样式 */
@media (min-width: 601px) {
  .analysis-confirm-content {
    max-width: 100%;
    overflow: hidden;
  }

  .template-desc {
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}
</style>
