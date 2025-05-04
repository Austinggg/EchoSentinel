<script setup lang="ts">
import { reactive, ref } from 'vue';

import { Page } from '@vben/common-ui';

import {
  ElAffix,
  ElAvatar,
  ElButton,
  ElCard,
  ElDescriptions,
  ElDescriptionsItem,
  ElForm,
  ElFormItem,
  ElImage,
  ElInput,
  ElProgress,
  ElStep,
  ElSteps,
  ElTable,
  ElTableColumn,
} from 'element-plus';

import { requestClient } from '#/api/request';

import ApiDemoTable from './ApiDemoTable.vue';
import ClusterScatter from './ClusterScatter.vue';

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
// const next = () => {
//   if (active.value++ > 2) active.value = 0;
// };
async function setActiveValue(value: number) {
  active.value = value;
}

async function onSubmit(secUid: string) {
  const data = await requestClient.post('/userAnalyse/getProfile', {
    sec_uid: secUid,
  });

  const getGenderText = (gender: number): string => {
    switch (gender) {
      case 1: {
        return '男';
      }
      case 2: {
        return '女';
      }
      default: {
        return '未知';
      }
    }
  };

  Object.assign(userProfile, {
    ...data,
    gender: getGenderText(data.gender),
  });

  changeProfileCardLoading();
}
// 加载状态
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
  const data = await requestClient.post('/userAnalyse/getRank', {
    sec_uid,
  });

  changeAnalyseCardLoading(false);
  percentage.value = data.anomaly_score;
  lossInfo.value = data.loss;
  setActiveValue(3);
}
const percentage = ref(0);
const colors = [
  { color: '#67c23a', percentage: 0 }, // 正常 - 绿色
  { color: '#b3e19d', percentage: 20 }, // 较正常 - 浅绿
  { color: '#409eff', percentage: 40 }, // 一般 - 蓝色
  { color: '#e6a23c', percentage: 60 }, // 较异常 - 橙色
  { color: '#f56c6c', percentage: 80 }, // 异常 - 红色
  { color: '#c45656', percentage: 100 }, // 严重异常 - 深红
];
const similarCluster = ref();
async function getSimilarCluster() {
  const data = await requestClient.post('/userAnalyse/similarCluster', {
    sec_uid: userProfile.sec_uid,
  });
  // console.log(data);
  similarCluster.value = data.similarCluster;
  return 'finish';
}
const similarUser = ref();
async function getSimilarUser() {
  const data = await requestClient.post('/userAnalyse/similarUser', {
    sec_uid: userProfile.sec_uid,
  });
  // console.log(data);
  similarUser.value = data.similarUser;
  return 'finish';
}
async function getSimilar() {
  changeSimilarCardLoading(true);
  const r1 = await getSimilarCluster();
  const r2 = await getSimilarUser();
  if (r1 === 'finish' && r2 === 'finish') changeSimilarCardLoading(false);
}
const childRef = ref();
const callChildMethodMarkPoint = () => {
  childRef.value?.markPoint(userProfile.sec_uid);
};
const callChildMethodMReDraw = () => {
  childRef.value?.reDraw();
};
</script>
<template>
  <Page title="异常用户分析模块">
    <ElAffix :offset="100">
      <ElCard class="mb-5">
        <ElSteps :active="active" finish-status="success">
          <ElStep title="Step 1 选择用户" />
          <ElStep title="Step 2 获取指标" />
          <ElStep title="Step 3 分析结果" />
        </ElSteps>
        <!-- <el-button style="margin-top: 12px" @click="next">Next step</el-button> -->
      </ElCard>
    </ElAffix>
    <div class="flex flex-wrap gap-5">
      <ElCard class="flex-1">
        <template #header>用户示例</template>
        <ApiDemoTable
          @update-profile="onSubmit"
          @change-profile-card-loading="changeProfileCardLoading"
          @set-active-value="setActiveValue"
        />
      </ElCard>
      <ElCard class="">
        <template #header> 用户信息 </template>
        <ElForm
          :model="userProfile"
          label-width="auto"
          inline
          style="max-width: 600px"
          v-loading="profileCardLoading"
        >
          <ElFormItem label="用户名">
            <ElInput v-model="userProfile.nickname" readonly />
          </ElFormItem>
          <ElFormItem label="性别">
            <ElInput v-model="userProfile.gender" readonly />
          </ElFormItem>
          <ElFormItem label="城市">
            <ElInput v-model="userProfile.city" readonly />
          </ElFormItem>
          <ElFormItem label="省">
            <ElInput v-model="userProfile.province" readonly />
          </ElFormItem>
          <ElFormItem label="地区">
            <ElInput v-model="userProfile.country" readonly />
          </ElFormItem>
          <ElFormItem label="作品数量">
            <ElInput v-model="userProfile.aweme_count" readonly />
          </ElFormItem>
          <ElFormItem label="粉丝数量">
            <ElInput v-model="userProfile.follower_count" readonly />
          </ElFormItem>
          <ElFormItem label="关注数量">
            <ElInput v-model="userProfile.following_count" readonly />
          </ElFormItem>
          <ElFormItem label="喜欢的作品">
            <ElInput v-model="userProfile.favoriting_count" readonly />
          </ElFormItem>
          <ElFormItem label="关注数量">
            <ElInput v-model="userProfile.following_count" readonly />
          </ElFormItem>
          <ElFormItem label="年龄">
            <ElInput v-model="userProfile.user_age" readonly />
          </ElFormItem>
          <ElFormItem label="IP属地">
            <ElInput v-model="userProfile.ip_location" readonly />
          </ElFormItem>
          <ElFormItem label="封面">
            <div class="flex flex-nowrap gap-5">
              <div v-for="url in userProfile.covers" :key="url" class="block">
                <ElImage
                  style="width: 100px; height: 100px"
                  :src="url"
                  fit="contain"
                >
                  <template #error> <div class="image-slot"></div> </template>
                </ElImage>
              </div>
            </div>
          </ElFormItem>
        </ElForm>
      </ElCard>
      <ElCard class="w-screen">
        <template #header> 异常检测 </template>
        <div v-loading="analyseCardLoading" class="flex flex-nowrap gap-5">
          <div class="flex flex-1 items-center justify-center">
            <ElDescriptions title="" :border="true" :column="1">
              <ElDescriptionsItem label="本次样本重构误差">
                {{ lossInfo }}
              </ElDescriptionsItem>
              <ElDescriptionsItem label="异常分数说明">
                重构损失越大，异常分数越大，越有可能是异常用户
              </ElDescriptionsItem>
            </ElDescriptions>
          </div>
          <div class="flex flex-1 items-center justify-center">
            <ElProgress
              type="dashboard"
              :percentage="percentage"
              :color="colors"
            >
              <template #default="{ percentage: progressPercentage }">
                <span class="percentage-value block text-xl font-bold">
                  {{ progressPercentage }}%
                </span>
                <span class="percentage-label mt-2 block">异常分数</span>
              </template>
            </ElProgress>
          </div>
        </div>

        <template #footer>
          <ElButton type="primary" @click="handleAnalyse(userProfile.sec_uid)">
            开始分析
          </ElButton>
        </template>
      </ElCard>
      <ElCard class="w-screen" v-show="false">
        <template #header>相似集群以及相似用户 </template>

        <div class="flex flex-nowrap gap-5" v-loading="similalCardLoading">
          <div class="flex-1">
            <ElTable :data="similarCluster">
              <ElTableColumn prop="cluster_id" label="集群" width="120px" />

              <ElTableColumn prop="avatar_list" label="包括用户">
                <template #default="scope">
                  <div class="-space-x-2">
                    <ElAvatar
                      v-for="(item, index) in scope.row.avatar_list"
                      :key="index"
                      :src="item"
                    />
                  </div>
                </template>
              </ElTableColumn>
            </ElTable>
          </div>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 w-px bg-gray-200"></div>
          </div>
          <div class="flex-1">
            <ElTable :data="similarUser">
              <ElTableColumn prop="avatar_medium" label="用户头像">
                <template #default="scope">
                  <ElAvatar :src="scope.row.avatar_medium" />
                </template>
              </ElTableColumn>
              <ElTableColumn prop="nickname" label="昵称" />
              <ElTableColumn prop="similarity" label="相似性" />
            </ElTable>
          </div>
        </div>
        <template #footer>
          <ElButton @click="getSimilar" type="primary">相似分析</ElButton>
        </template>
      </ElCard>
      <ElCard class="w-screen">
        <template #header>用户集群展示</template>
        <ClusterScatter ref="childRef" :sec-uid="userProfile.sec_uid" />
        <template #footer>
          <ElButton type="primary" @click="callChildMethodMarkPoint()">
            显示分析用户位置
          </ElButton>
          <ElButton type="primary" @click="callChildMethodMReDraw()">
            恢复初始
          </ElButton>
        </template>
      </ElCard>
    </div>
  </Page>
</template>
