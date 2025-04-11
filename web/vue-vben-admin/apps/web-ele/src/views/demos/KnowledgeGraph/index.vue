<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue';

import { Page } from '@vben/common-ui';

import {
    ElButton,
    ElCard,
    ElUpload,
    ElMessage,
    ElNotification,
    ElProgress,
    ElSpace,
    ElInput,
    ElDivider,
    ElSelect,
    ElOption,
} from 'element-plus';
import type { UploadProps, UploadFile, UploadInstance } from 'element-plus';
import { Plus } from '@element-plus/icons-vue';

// 文件上传相关
const uploadRef = ref<UploadInstance>();
const textFile = ref<File | null>(null);
const isUploading = ref<boolean>(false);
const isProcessing = ref<boolean>(false);
const processProgress = ref<number>(0);

// 知识图谱相关
const graphContainer = ref<HTMLElement | null>(null);
const graphData = ref({
    nodes: [],
    links: []
});
const sampleTexts = ref([
    {
        label: '人工智能领域',
        value: 'ai'
    },
    {
        label: '医疗健康领域',
        value: 'medical'
    },
    {
        label: '金融科技领域',
        value: 'fintech'
    }
]);
const selectedSample = ref('ai');
const customText = ref('');
const showCustomInput = ref(false);

// 上传前验证
const beforeUpload: UploadProps['beforeUpload'] = (file) => {
    if (!file) {
        ElMessage.error('未选择文件!');
        return false;
    }
    const isText = file.type === 'text/plain' || file.name.endsWith('.txt');
    if (!isText) {
        ElMessage.error('只能上���TXT文本文件!');
        return false;
    }

    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
        ElMessage.error('文件大小不能超过 10MB!');
        return false;
    }

    // 存储选择的文件但不自动上传
    textFile.value = file;
    return false; // 阻止自动上传
};

// 上传文本文件
const handleUpload = async () => {
    if (!textFile.value) {
        ElMessage.warning('请先选择文本文件');
        return;
    }

    isUploading.value = true;
    try {
        // 模拟上传过程
        await new Promise(resolve => {
            let progress = 0;
            const timer = setInterval(() => {
                progress += 10;
                processProgress.value = progress;
                if (progress >= 100) {
                    clearInterval(timer);
                    resolve(true);
                }
            }, 300);
        });

        ElMessage.success('文件上传成功');
        ElNotification({
            title: '上传成功',
            message: `文件 ${textFile.value.name} 已成功上传`,
            type: 'success',
        });
        
        // 上传成功后生成知识图谱
        generateKnowledgeGraph();
    } catch (error) {
        ElMessage.error('上传失败，请重试');
    } finally {
        isUploading.value = false;
        processProgress.value = 0;
    }
};

// 使用样例文本
const useSampleText = () => {
    showCustomInput.value = false;
    generateKnowledgeGraph();
};

// 使用自定义文本
const useCustomText = () => {
    showCustomInput.value = true;
};

// 提交自定义文本
const submitCustomText = () => {
    if (!customText.value.trim()) {
        ElMessage.warning('请输入文本内容');
        return;
    }
    generateKnowledgeGraph();
};

// 生成知识图谱
const generateKnowledgeGraph = async () => {
    isProcessing.value = true;
    processProgress.value = 0;

    try {
        // 模拟处理过程
        await new Promise(resolve => {
            let progress = 0;
            const timer = setInterval(() => {
                progress += 5;
                processProgress.value = progress;
                if (progress >= 100) {
                    clearInterval(timer);
                    resolve(true);
                }
            }, 200);
        });

        // 根据选择的样例或上传的文件生成不同的图谱数据
        if (selectedSample.value === 'ai') {
            graphData.value = getAIGraphData();
        } else if (selectedSample.value === 'medical') {
            graphData.value = getMedicalGraphData();
        } else if (selectedSample.value === 'fintech') {
            graphData.value = getFintechGraphData();
        } else if (textFile.value || customText.value) {
            // 如果是上传的文件或自定义文本，生成通用图谱
            graphData.value = getAIGraphData(); // 这里仅用于演示
        }

        // 渲染图谱
        renderGraph();
        
        ElMessage.success('知识图谱生成成功');
    } catch (error) {
        ElMessage.error('处理失败，请重试');
    } finally {
        isProcessing.value = false;
        processProgress.value = 0;
    }
};
// 渲染图谱
const renderGraph = () => {
    if (!graphContainer.value) return;

    // 清空容器
    graphContainer.value.innerHTML = '';

    // 创建SVG元素
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    // 渲染节点和连线
    graphData.value.nodes.forEach(node => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', node.x);
        circle.setAttribute('cy', node.y);
        circle.setAttribute('r', 20);
        circle.setAttribute('fill', '#3498db');
        svg.appendChild(circle);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', node.x);
        text.setAttribute('y', node.y + 5);
        text.setAttribute('fill', '#fff');
        text.setAttribute('font-size', '12');
        text.setAttribute('text-anchor', 'middle');
        text.textContent = node.label;
        svg.appendChild(text);
    });

    graphData.value.links.forEach(link => {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', link.source.x);
        line.setAttribute('y1', link.source.y);
        line.setAttribute('x2', link.target.x);
        line.setAttribute('y2', link.target.y);
        line.setAttribute('stroke', '#2ecc71');
        line.setAttribute('stroke-width', 2);
        svg.appendChild(line);
    });

    graphContainer.value.appendChild(svg);
};