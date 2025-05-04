<script lang="ts" setup>
import { ApiComponent } from '@vben/common-ui';

import { ElTable, ElTableColumn, ElTag } from 'element-plus';

import { requestClient } from '#/api/request';

const emit = defineEmits([
  'updateProfile',
  'changeProfileCardLoading',
  'setActiveValue',
]);
async function getAllDemosApi() {
  return requestClient.get('/userAnalyse/demo');
}
function handleRowClick(row: Record<string, any>) {
  emit('changeProfileCardLoading');
  emit('updateProfile', row.sec_uid);
  emit('setActiveValue', 1);
}
</script>
<template>
  <ApiComponent
    :api="getAllDemosApi"
    :component="ElTable"
    :immediate="true"
    options-prop-name="data"
    @row-click="handleRowClick"
  >
    <ElTableColumn prop="nickname" label="NickName" />
    <ElTableColumn label="Tag">
      <template #default="scope">
        <ElTag
          :type="scope.row.tag === '正常' ? 'primary' : 'warning'"
          disable-transitions
        >
          {{ scope.row.tag }}
        </ElTag>
      </template>
    </ElTableColumn>
  </ApiComponent>
</template>
