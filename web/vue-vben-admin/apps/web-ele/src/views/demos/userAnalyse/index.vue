<script setup lang="ts">
import { reactive } from 'vue';
import { ref } from 'vue';
import { Page } from '@vben/common-ui';
import {
  ElCard,
  ElTable,
  ElTableColumn,
  ElTag,
  ElForm,
  ElFormItem,
  ElInput,
  ElButton,
  ElSteps,
  ElStep,
} from 'element-plus';
const tableData = [
  {
    username: 'Tom',
    tag: '危险',
  },
  {
    username: 'Tom',
    tag: '危险',
  },
  {
    username: 'Tom',
    tag: '危险',
  },
  {
    username: 'Tom',
    tag: '正常',
  },
];
const form = reactive({
  username: '',
  gender: '',
  region: '',
  date1: '',
  date2: '',
  delivery: false,
  type: [],
  resource: '',
  desc: '',
});
const onSubmit = () => {
  console.log('submit!');
};

const active = ref(0);

const next = () => {
  if (active.value++ > 2) active.value = 0;
};
</script>
<template>
  <Page title="异常用户分析模块">
    <el-card class="mb-5">
      <el-steps :active="active" finish-status="success">
        <el-step title="Step 1 选择用户" />
        <el-step title="Step 2 获取指标" />
        <el-step title="Step 3 分析结果" />
        <el-step title="Step 4 查找相似" />
      </el-steps>
      <el-button style="margin-top: 12px" @click="next"
        >Next step</el-button
      ></el-card
    >
    <div class="flex flex-wrap gap-5">
      <el-card class="">
        <template #header> 1 </template>
        <el-table :data="tableData" style="width: 100%">
          <el-table-column prop="username" label="Username" />
          <el-table-column prop="tag" label="Tag"></el-table-column>
          <el-table-column label="Tag">
            <template #default="scope">
              <el-tag type="primary" disable-transitions>{{
                scope.row.tag
              }}</el-tag>
            </template></el-table-column
          >
        </el-table> </el-card
      ><el-card class=""
        ><template #header> 2 </template>
        <el-form :model="form" label-width="auto" inline style="max-width: 600px">
          <el-form-item label="用户名">
            <el-input v-model="form.username" readonly />
          </el-form-item>
          <el-form-item label="性别">
            <el-input v-model="form.gender" />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" @click="onSubmit">Create</el-button>
            <el-button>Cancel</el-button>
          </el-form-item>
        </el-form></el-card
      ><el-card class="w-96"><template #header> 3 </template></el-card>
      <el-card class="w-96"><template #header> 4 </template></el-card>
    </div>
  </Page>
</template>
