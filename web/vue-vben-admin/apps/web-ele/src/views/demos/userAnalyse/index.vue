<script setup lang="ts">
import { reactive } from 'vue';
import { requestClient } from '#/api/request';
import { ref } from 'vue';
import { Page } from '@vben/common-ui';
import {
  ElCard,
  ElForm,
  ElFormItem,
  ElInput,
  ElButton,
  ElSteps,
  ElStep,
  ElImage,
  ElAffix,
} from 'element-plus';
const userProfile = reactive({
  sec_uid: 'MS4wLjABAAAALxPAZM7qgk5yrE_L-Qu4eZW_L2MJ-ApSH6yXNdoOShU',
  nickname: '',
  gender: '',
  city: '',
  province: '',
  country: '',
  aweme_count: '',
  follower_count: '',
  following_count: '',
  total_favorited: '',
  favoriting_count: '',
  user_age: '',
  ip_location: '',
  covers: [],
});
// 进度条
const active = ref(0);
const next = () => {
  if (active.value++ > 2) active.value = 0;
};
async function setActiveValue(value: number) {
  active.value = value;
}

async function onSubmit(secUid: string) {
  console.log('submit!');
  let data = await requestClient.post('/userAnalyse/getProfile', {
    sec_uid: secUid,
  });
  Object.assign(userProfile, {
    ...data,
    gender: data.gender === 1 ? '男' : data.gender === 2 ? '女' : '未知', // 三元运算符处理性别
  });
  changeProfileCardLoading();
}

import ApiDemoTable from './ApiDemoTable.vue';
const profileCardLoading = ref(false);
function changeProfileCardLoading() {
  profileCardLoading.value = !profileCardLoading.value;
}
const analyseCardLoading = ref(false);
async function changeAnalyseCardLoading() {
  analyseCardLoading.value = !analyseCardLoading.value;
}
const analyseText = ref('');
async function handleAnalyse(sec_uid: string) {
  setActiveValue(2);
  changeAnalyseCardLoading();
  let data = await requestClient.post('/userAnalyse/getRank', {
    sec_uid: sec_uid,
  });
  console.log(data);
  analyseText.value = '重构误差：' + data.loss;
  changeAnalyseCardLoading();
  setActiveValue(3);
}
</script>
<template>
  <Page title="异常用户分析模块">
    <el-affix :offset="100">
      <el-card class="mb-5">
        <el-steps :active="active" finish-status="success">
          <el-step title="Step 1 选择用户" />
          <el-step title="Step 2 获取指标" />
          <el-step title="Step 3 分析结果" />
        </el-steps>
        <el-button style="margin-top: 12px" @click="next"
          >Next step</el-button
        ></el-card
      >
    </el-affix>
    <div class="flex flex-wrap gap-5">
      <el-card class="flex-1">
        <template #header>用户示例</template>
        <ApiDemoTable
          @updateProfile="onSubmit"
          @changeProfileCardLoading="changeProfileCardLoading"
          @setActiveValue="setActiveValue"
        ></ApiDemoTable>
      </el-card>
      <el-card class=""
        ><template #header> 用户信息 </template>
        <el-form
          :model="userProfile"
          label-width="auto"
          inline
          style="max-width: 600px"
          v-loading="profileCardLoading"
        >
          <el-form-item label="用户名">
            <el-input v-model="userProfile.nickname" readonly />
            <!-- <el-text>{{ form.username }}</el-text> -->
          </el-form-item>
          <el-form-item label="性别">
            <el-input v-model="userProfile.gender" readonly />
          </el-form-item>
          <el-form-item label="城市">
            <el-input v-model="userProfile.city" readonly />
          </el-form-item>
          <el-form-item label="省">
            <el-input v-model="userProfile.province" readonly />
          </el-form-item>
          <el-form-item label="地区">
            <el-input v-model="userProfile.country" readonly />
          </el-form-item>
          <el-form-item label="作品数量">
            <el-input v-model="userProfile.aweme_count" readonly />
          </el-form-item>
          <el-form-item label="粉丝数量">
            <el-input v-model="userProfile.follower_count" readonly />
          </el-form-item>
          <el-form-item label="关注数量">
            <el-input v-model="userProfile.following_count" readonly />
          </el-form-item>
          <el-form-item label="喜欢的作品">
            <el-input v-model="userProfile.favoriting_count" readonly />
          </el-form-item>
          <el-form-item label="关注数量">
            <el-input v-model="userProfile.following_count" readonly />
          </el-form-item>
          <el-form-item label="年龄">
            <el-input v-model="userProfile.user_age" readonly />
          </el-form-item>
          <el-form-item label="IP属地">
            <el-input v-model="userProfile.ip_location" readonly />
          </el-form-item>
          <el-form-item label="封面">
            <div class="flex flex-nowrap gap-5">
              <div v-for="url in userProfile.covers" :key="url" class="block">
                <el-image
                  style="width: 100px; height: 100px"
                  :src="url"
                  fit="contain"
                >
                  <template #error> <div class="image-slot"></div> </template
                ></el-image>
              </div>
            </div>
          </el-form-item> </el-form></el-card
      ><el-card class="w-screen"
        ><template #header> 3 </template>
        <div v-loading="analyseCardLoading">
          <p v-show="analyseText == ''">模型处理中</p>
          <p>{{ analyseText }}</p>
        </div>
        <template #footer>
          <el-button type="primary" @click="handleAnalyse(userProfile.sec_uid)"
            >开始分析</el-button
          ></template
        >
      </el-card>
    </div>
  </Page>
</template>
