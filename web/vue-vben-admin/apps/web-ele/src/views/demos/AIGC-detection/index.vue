<script setup lang="ts">
import type { Column, UploadProps, UploadUserFile } from 'element-plus';

import type { EchoDetectionData } from './types';

import { computed, onMounted, ref } from 'vue';

import { Page } from '@vben/common-ui';

import {
  ElButton,
  ElCard,
  ElMessage,
  ElTableV2,
  ElUpload,
  TableV2Alignment,
} from 'element-plus';

import { requestClient } from '#/api/request';

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
const cellWidth = computed(() => tableWidth.value / 4);
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
const columnDefs = [
  { key: 'id', title: 'ID' },
  { key: 'face', title: '面部' },
  { key: 'body', title: '躯干' },
  { key: 'whole', title: '整体' },
];

const columns = computed<Column[]>(() =>
  columnDefs.map(({ key, title }) => ({
    align: TableV2Alignment.CENTER,
    key,
    dataKey: key,
    title,
    width: cellWidth.value,
  })),
);

const tableData = ref<EchoDetectionData[]>([
  {
    id: 'demo',
    face: 'face Demo',
    body: 'body Demo',
    whole: 'whole Demo',
  },
]);
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
            fixed
          />
        </div>
      </ElCard>
    </div>
  </Page>
</template>
