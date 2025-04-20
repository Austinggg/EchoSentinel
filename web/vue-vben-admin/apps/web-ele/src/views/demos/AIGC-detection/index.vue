<script setup lang="ts">
import { Page } from '@vben/common-ui';
import { ElCard, ElUpload, ElIcon, ElDialog } from 'element-plus';
import { ref } from 'vue';
import { Delete, Download, Plus, ZoomIn } from '@element-plus/icons-vue';
import type { UploadFile } from 'element-plus';
const dialogImageUrl = ref('');
const dialogVisible = ref(false);
const disabled = ref(false);

const handleRemove = (file: UploadFile) => {
  console.log(file);
};

const handlePictureCardPreview = (file: UploadFile) => {
  dialogImageUrl.value = file.url!;
  dialogVisible.value = true;
};

const handleDownload = (file: UploadFile) => {
  console.log(file);
};
</script>
<template>
  <Page title="数字人识别">
    <el-card class="mb-5 w-full"
      ><template #header>上传部分</template>
      <el-upload action="#" list-type="picture-card" :auto-upload="false">
        <el-icon><Plus /></el-icon>

        <template #file="{ file }">
          <div>
            <img
              class="el-upload-list__item-thumbnail"
              :src="file.url"
              alt=""
            />
            <span class="el-upload-list__item-actions">
              <span
                class="el-upload-list__item-preview"
                @click="handlePictureCardPreview(file)"
              >
                <el-icon><zoom-in /></el-icon>
              </span>
              <span
                v-if="!disabled"
                class="el-upload-list__item-delete"
                @click="handleDownload(file)"
              >
                <el-icon><Download /></el-icon>
              </span>
              <span
                v-if="!disabled"
                class="el-upload-list__item-delete"
                @click="handleRemove(file)"
              >
                <el-icon><Delete /></el-icon>
              </span>
            </span>
          </div>
        </template>
      </el-upload>
      <el-dialog v-model="dialogVisible">
        <img w-full :src="dialogImageUrl" alt="Preview Image" />
      </el-dialog>
    </el-card>

    <div class="flex flex-wrap gap-5">
      <el-card class="flex-1"><template #header>整体</template></el-card>
      <el-card class="flex-1"><template #header>躯干</template></el-card>
      <el-card class="flex-1"><template #header>面部</template></el-card>
    </div>
  </Page>
</template>
