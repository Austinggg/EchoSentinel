<script lang="ts" setup>
import { requestClient } from '#/api/request';
import { ApiComponent } from '@vben/common-ui';

import { ElTable, ElTableColumn, ElTag } from 'element-plus';

async function getAllDemosApi() {
  return requestClient.get('/userAnalyse/demo');
}
const emit = defineEmits(['updateProfile', 'changeProfileCardLoading','setActiveValue']);
function handleRowClick(row: Record<string, any>) {
  emit('changeProfileCardLoading');
  console.log('Clicked row:', row.sec_uid);
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
    <el-table-column prop="nickname" label="NickName" />
    <el-table-column label="Tag">
      <template #default="scope">
        <el-tag
          :type="scope.row.tag === '正常' ? 'primary' : 'warning'"
          disable-transitions
          >{{ scope.row.tag }}</el-tag
        >
      </template></el-table-column
    ></ApiComponent
  >
</template>
