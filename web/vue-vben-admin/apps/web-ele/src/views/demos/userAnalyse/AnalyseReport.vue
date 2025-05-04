<script setup lang="ts">
import { ref, watch } from 'vue';

import {
  ElAvatar,
  ElDescriptions,
  ElDescriptionsItem,
  ElLink,
  ElTable,
  ElTableColumn,
} from 'element-plus';

import { requestClient } from '#/api/request';

const props = defineProps({
  nickname: {
    type: String,
    default: '',
  },
  secUid: {
    default: '',
    type: String,
  },
  anomalyScore: {
    default: 0,
    type: Number,
  },
});
const similarUser = ref([]);
async function getSimilarUser() {
  const res = await requestClient.post('/userAnalyse/similarUser', {
    sec_uid: props.secUid,
  });
  similarUser.value = res.similarUser;
}
watch(
  () => props.secUid,
  async () => {
    await getSimilarUser();
  },
);
const uidToUrl = (sec_uid: String) => {
  return sec_uid ? `https://www.douyin.com/user/${sec_uid}` : '';
};
const similarTableClick = (row: any, _column: any, _event: Event) => {
  window.open(uidToUrl(row.sec_uid));
};
</script>
<template>
  <ElDescriptions class="margin-top" :column="3" border label-width="80">
    <ElDescriptionsItem width="160">
      <template #label>
        <div class="flex items-center justify-center">账号昵称</div>
      </template>
      {{ props.nickname }}
    </ElDescriptionsItem>
    <ElDescriptionsItem width="160">
      <template #label>
        <div class="flex items-center justify-center">异常分数</div>
      </template>
      {{ props.anomalyScore }}%
    </ElDescriptionsItem>
    <ElDescriptionsItem>
      <template #label>
        <div class="flex items-center justify-center">账号链接</div>
      </template>
      <ElLink :href="uidToUrl(props.secUid)" target="_blank">
        {{ uidToUrl(props.secUid) }}
      </ElLink>
    </ElDescriptionsItem>
    <ElDescriptionsItem>
      <template #label>
        <div class="flex items-center justify-center">相似用户</div>
      </template>
      <ElTable :data="similarUser" @row-click="similarTableClick">
        <ElTableColumn prop="avatar_medium" label="用户头像">
          <template #default="scope">
            <ElAvatar :src="scope.row.avatar_medium" />
          </template>
        </ElTableColumn>
        <ElTableColumn prop="nickname" label="昵称" />
        <ElTableColumn prop="similarity" label="相似性" />
      </ElTable>
    </ElDescriptionsItem>
  </ElDescriptions>
</template>
