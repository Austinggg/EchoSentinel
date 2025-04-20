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
  ElProgress,
  ElAvatar,
  ElTable,
  ElTableColumn,
  ElDescriptions,
  ElDescriptionsItem,
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
//加载状态
const profileCardLoading = ref(false);
const analyseCardLoading = ref(false);
const similalCardLoading = ref(false);
function changeProfileCardLoading(value = !profileCardLoading.value) {
  profileCardLoading.value = value;
}
function changeAnalyseCardLoading(value = !analyseCardLoading.value) {
  analyseCardLoading.value = value;
}
function changeSimilarCardLoading(value = !similalCardLoading.value) {
  similalCardLoading.value = value;
}
const lossInfo = ref();
async function handleAnalyse(sec_uid: string) {
  setActiveValue(2);
  changeAnalyseCardLoading(true);
  let data = await requestClient.post('/userAnalyse/getRank', {
    sec_uid: sec_uid,
  });
  console.log(data);
  changeAnalyseCardLoading(false);
  percentage.value = data.anomaly_score;
  lossInfo.value = data.loss;
  setActiveValue(3);
}
const percentage = ref(0);
const colors = [
  { color: '#f56c6c', percentage: 20 },
  { color: '#e6a23c', percentage: 40 },
  { color: '#5cb87a', percentage: 60 },
  { color: '#1989fa', percentage: 80 },
  { color: '#6f7ad3', percentage: 100 },
];

const similarCluster = ref();
async function getSimilarCluster() {
  let data = await requestClient.post('/userAnalyse/similarCluster', {
    sec_uid: userProfile.sec_uid,
  });
  // console.log(data);
  similarCluster.value = data.similarCluster;
  return 'finish';
}
const similarUser = ref();
async function getSimilarUser() {
  let data = await requestClient.post('/userAnalyse/similarUser', {
    sec_uid: userProfile.sec_uid,
  });
  // console.log(data);
  similarUser.value = data.similarUser;
  return 'finish';
}
async function getSimilar() {
  changeSimilarCardLoading(true);
  let r1 = await getSimilarCluster();
  let r2 = await getSimilarUser();
  if (r1 == 'finish' && r2 == 'finish') changeSimilarCardLoading(false);
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
        ><template #header> 异常检测 </template>
        <div v-loading="analyseCardLoading" class="flex flex-nowrap gap-5">
          <div class="flex flex-1 items-center justify-center">
            <el-descriptions title="" :border="true" :column="1">
              <el-descriptions-item label="本次样本重构误差">{{
                lossInfo
              }}</el-descriptions-item>
              <el-descriptions-item label="异常分数说明"
                >重构损失越大，异常分数越大，越有可能是异常用户</el-descriptions-item
              >
            </el-descriptions>
          </div>
          <div class="flex flex-1 items-center justify-center">
            <el-progress
              type="dashboard"
              :percentage="percentage"
              :color="colors"
            >
              <template #default="{ percentage }">
                <span class="percentage-value block text-xl font-bold"
                  >{{ percentage }}%</span
                >
                <span class="percentage-label mt-2 block">异常分数</span>
              </template></el-progress
            >
          </div>
        </div>

        <template #footer>
          <el-button type="primary" @click="handleAnalyse(userProfile.sec_uid)"
            >开始异常分析</el-button
          ></template
        >
      </el-card>
      <el-card class="w-screen">
        <template #header>相似集群以及相似用户 </template>

        <div class="flex flex-nowrap gap-5" v-loading="similalCardLoading">
          <div class="flex-1">
            <el-table :data="similarCluster">
              <el-table-column prop="cluster_id" label="集群" width="120px" />

              <el-table-column prop="avatar_list" label="包括用户">
                <template #default="scope">
                  <div class="-space-x-2">
                    <el-avatar
                      v-for="item in scope.row.avatar_list"
                      :src="item"
                    ></el-avatar>
                  </div>
                </template>
              </el-table-column>
            </el-table>
          </div>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 w-px bg-gray-200"></div>
          </div>
          <div class="flex-1">
            <el-table :data="similarUser">
              <el-table-column prop="avatar_medium" label="用户头像">
                <template #default="scope">
                  <el-avatar :src="scope.row.avatar_medium"></el-avatar>
                </template>
              </el-table-column>
              <el-table-column prop="nickname" label="昵称" />
              <el-table-column prop="similarity" label="相似性" />
            </el-table>
          </div>
        </div>
        <template #footer>
          <el-button @click="getSimilar" type="primary">相似分析</el-button>
        </template>
      </el-card>
    </div>
  </Page>
</template>
