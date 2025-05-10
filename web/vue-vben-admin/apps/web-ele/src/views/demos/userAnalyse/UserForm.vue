<script setup lang="ts">
import type { UserProfile } from './types';

import { ref, watch } from 'vue';

import { ElForm, ElFormItem, ElInput } from 'element-plus';

const props = defineProps<{
  loading?: boolean;
  userProfile: UserProfile;
}>();
const emit = defineEmits<{
  (e: 'update:userProfile', value: UserProfile): void;
}>();
const localUserProfile = ref({ ...props.userProfile });

// 监听变化并通知父组件
watch(
  localUserProfile,
  (newVal) => {
    emit('update:userProfile', newVal);
  },
  { deep: true },
);
</script>
<template>
  <ElForm
    :model="userProfile"
    label-width="auto"
    inline
    style="max-width: 600px"
    v-loading="props.loading"
  >
    <ElFormItem label="用户名">
      <ElInput v-model="localUserProfile.nickname" readonly />
    </ElFormItem>
    <ElFormItem label="性别">
      <ElInput v-model="localUserProfile.gender" readonly />
    </ElFormItem>
    <ElFormItem label="城市">
      <ElInput v-model="localUserProfile.city" readonly />
    </ElFormItem>
    <ElFormItem label="省">
      <ElInput v-model="localUserProfile.province" readonly />
    </ElFormItem>
    <ElFormItem label="地区">
      <ElInput v-model="localUserProfile.country" readonly />
    </ElFormItem>
    <ElFormItem label="作品数量">
      <ElInput v-model="localUserProfile.aweme_count" readonly />
    </ElFormItem>
    <ElFormItem label="粉丝数量">
      <ElInput v-model="localUserProfile.follower_count" readonly />
    </ElFormItem>
    <ElFormItem label="关注数量">
      <ElInput v-model="localUserProfile.following_count" readonly />
    </ElFormItem>
    <ElFormItem label="喜欢的作品">
      <ElInput v-model="localUserProfile.favoriting_count" readonly />
    </ElFormItem>
    <ElFormItem label="关注数量">
      <ElInput v-model="localUserProfile.following_count" readonly />
    </ElFormItem>
    <ElFormItem label="年龄">
      <ElInput v-model="localUserProfile.user_age" readonly />
    </ElFormItem>
    <ElFormItem label="IP属地">
      <ElInput v-model="localUserProfile.ip_location" readonly />
    </ElFormItem>
    <ElFormItem label="封面">
      <div class="flex flex-nowrap gap-5">
        <div v-for="url in localUserProfile.covers" :key="url" class="block">
          <ElImage style="width: 100px; height: 100px" :src="url" fit="contain">
            <template #error> <div class="image-slot"></div> </template>
          </ElImage>
        </div>
      </div>
    </ElFormItem>
  </ElForm>
</template>
