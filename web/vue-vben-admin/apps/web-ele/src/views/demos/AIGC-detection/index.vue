<script setup lang="ts">
import type { Column, UploadProps, UploadUserFile } from 'element-plus';

import type { CellRenderProps, EchoDetectionData } from './types';

import { computed, h, onMounted, ref } from 'vue';

import { Page } from '@vben/common-ui';

import {
  ElButton,
  ElCard,
  ElImage,
  ElMessage,
  ElTableV2,
  ElTag,
  ElUpload,
} from 'element-plus';

import { requestClient } from '#/api/request';

import { Alignment } from './types';

const fileList = ref<UploadUserFile[]>([]);

const handleRemove: UploadProps['onRemove'] = (_file, _uploadFiles) => {
  // console.log(file, uploadFiles);
};

const handlePreview: UploadProps['onPreview'] = (_uploadFile) => {
  // console.log(uploadFile);
};

const handleExceed: UploadProps['onExceed'] = (files, uploadFiles) => {
  ElMessage.warning(
    `The limit is 3, you selected ${files.length} files this time, add up to ${
      files.length + uploadFiles.length
    } totally`,
  );
};

// const beforeRemove: UploadProps['beforeRemove'] = (uploadFile, uploadFiles) => {
//   return ElMessageBox.confirm(
//     `Cancel the transfer of ${uploadFile.name} ?`,
//   ).then(
//     () => true,
//     () => false,
//   );
// };

// 调整表格宽度
const divRef = ref<HTMLDivElement>();
const tableWidth = ref(1000);
const cellWidth = computed(() => tableWidth.value / 5);
onMounted(() => {
  updateTableWidth();
  window.addEventListener('resize', updateTableWidth);
});
const updateTableWidth = () => {
  if (divRef.value) {
    tableWidth.value = divRef.value.clientWidth;
  }
};
// 表格数据
const videoCoverVNode = (props: CellRenderProps) => {
  return h('div', { style: { margin: '10px', height: '180px' } }, [
    h(ElImage, {
      src: `/api/videos/${props.rowData.id}/thumbnail`,
      fit: 'scale-down',
      previewSrcList: [`/api/videos/${props.rowData.id}/thumbnail`],
      style: { height: '100%' },
    }),
  ]);
};
const tagVNode = (props: CellRenderProps) => {
  return h(
    ElTag,
    {
      size: 'large',
      style: {
        fontSize: '14px',
      },
    },
    () => props.cellData,
  );
};
const columns = computed<Column[]>(() => [
  // {
  //   key: 'id',
  //   dataKey: 'id',
  //   title: 'ID',
  //   width: cellWidth.value,
  // },
  {
    key: 'filename',
    dataKey: 'filename',
    title: '视频标题',
    width: cellWidth.value,
    align: Alignment.CENTER,
  },
  {
    key: 'videoCover',
    dataKey: 'videoCover',
    title: '视频封面',
    width: cellWidth.value,
    align: Alignment.CENTER,
    cellRenderer: (props) => videoCoverVNode(props),
  },
  {
    key: 'face',
    dataKey: 'face',
    title: '面部',
    width: cellWidth.value,
    align: Alignment.CENTER,
    cellRenderer: (props) => tagVNode(props),
  },
  {
    key: 'body',
    dataKey: 'body',
    title: '躯干',
    width: cellWidth.value,
    align: Alignment.CENTER,
    cellRenderer: (props) => tagVNode(props),
  },
  {
    key: 'whole',
    dataKey: 'whole',
    title: '整体',
    width: cellWidth.value,
    align: Alignment.CENTER,
    cellRenderer: (props) => tagVNode(props),
  },
]);

const tableData = ref<EchoDetectionData[]>([]);
const fetchData = async () => {
  const response = await requestClient.get<EchoDetectionData[]>(
    '/aigc-detection/tableData',
  );
  tableData.value = response;
};
onMounted(() => {
  fetchData();
});
</script>
<template>
  <Page title="数字人识别">
    <div class="">
      <ElCard>
        <ElUpload
          v-model:file-list="fileList"
          class="upload-demo"
          action="/api/videos/upload?aigc=true"
          multiple
          :on-preview="handlePreview"
          :on-remove="handleRemove"
          :limit="3"
          :on-exceed="handleExceed"
        >
          <ElButton type="primary">点击上传</ElButton>
          <template #tip>
            <div class="el-upload__tip">请上传MP4文件。</div>
          </template>
        </ElUpload>
        <div ref="divRef" class="w-full">
          <ElTableV2
            :columns="columns"
            :data="tableData"
            :width="tableWidth"
            :height="600"
            :row-height="200"
            fixed
          />
        </div>
      </ElCard>
    </div>
  </Page>
</template>
