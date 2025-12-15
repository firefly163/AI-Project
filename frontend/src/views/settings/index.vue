<template>
  <div class="settings-page">
    <el-card class="section-card">
      <template #header>
        <div class="card-title">{{ t('settings.appSettings') }}</div>
      </template>
      <el-form label-width="120px">
        <el-form-item :label="t('settings.theme')">
          <el-select v-model="appStore.theme" placeholder="Select" style="width: 220px">
            <el-option :label="t('settings.themeLight')" value="light" />
            <el-option :label="t('settings.themeDark')" value="dark" />
            <el-option :label="t('settings.themeSystem')" value="system" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('settings.language')">
          <el-select v-model="appStore.language" placeholder="Select" style="width: 220px">
            <el-option label="中文" value="zh" />
            <el-option label="English" value="en" />
            <el-option label="日本語" value="ja" />
          </el-select>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="section-card">
      <template #header>
        <div class="card-title">{{ t('settings.globalSettings') }}</div>
      </template>
      <el-form label-width="120px">
        <el-form-item :label="t('settings.autoForward')">
          <el-switch v-model="appStore.autoForward" />
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { useAppStore } from '@/stores/appStore'
import { useI18n } from '@/utils/i18n'

const appStore = useAppStore()
const { t } = useI18n()
</script>

<style scoped>
.settings-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px;
}

.section-card {
  border-radius: 12px;
}

.card-title {
  font-weight: 700;
  font-size: 16px;
}
</style>
