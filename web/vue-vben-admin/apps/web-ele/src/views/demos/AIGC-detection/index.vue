<script setup lang="ts">
import type { UploadProps, UploadUserFile } from 'element-plus';

import type { EchoIdentify, EchoIdentifyTableRow } from './types';

import { computed, onMounted, ref } from 'vue';

import { Page } from '@vben/common-ui';

import {
  ElButton,
  ElCard,
  ElDivider,
  ElMessage,
  ElTableV2,
  ElUpload,
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
const dividerRef = ref<HTMLDivElement>();
const tableWidth = ref(1000);
const cellWidth = computed(() => tableWidth.value / 4);
onMounted(() => {
  updateTableWidth();
  window.addEventListener('resize', updateTableWidth);
});
const updateTableWidth = () => {
  if (dividerRef.value) {
    tableWidth.value = dividerRef.value.clientWidth;
  }
};
// 表格数据
const columns = computed(() => [
  { key: 'id', dataKey: 'id', title: 'id', width: cellWidth.value },
  { key: 'face', dataKey: 'face', title: 'face', width: cellWidth.value },
  { key: 'body', dataKey: 'body', title: 'body', width: cellWidth.value },
  { key: 'whole', dataKey: 'whole', title: 'whole', width: cellWidth.value },
]);
const data = ref<EchoIdentifyTableRow[]>([
  { id: '!', face: '1', body: '2', whole: '3' },
]);
const fetchData = async () => {
  const response = await requestClient.get<EchoIdentify[]>(
    '/aigc-detection/tableData',
  );
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
        <div ref="dividerRef">
          <ElDivider />
        </div>
        <ElTableV2
          :columns="columns"
          :data="data"
          :width="tableWidth"
          :height="600"
          fixed
        />
      </ElCard>
    </div>
  </Page>
</template>
