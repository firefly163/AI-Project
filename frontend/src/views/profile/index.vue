<template>
  <div class="profile-page">
    <el-card class="avatar-card">
      <div class="avatar-wrapper">
        <el-upload
          class="avatar-uploader"
          :show-file-list="false"
          :before-upload="handleBeforeUpload"
          :on-change="handleAvatarChange"
        >
          <img v-if="avatar" :src="avatar" class="avatar" />
          <div v-else class="avatar placeholder">上传头像</div>
        </el-upload>
      </div>
    </el-card>

    <el-card class="info-card">
      <div class="info-row">
        <span class="label">Authored by</span>
        <span class="value">{{ author }}</span>
      </div>
      <div class="info-row">
        <span class="label">Version</span>
        <span class="value">{{ version }}</span>
      </div>
      <div class="info-row">
        <span class="label">License</span>
        <span class="value">MIT</span>
      </div>
      <div class="info-column">
        <span class="label">Special thanks to</span>
        <ul class="credits-list">
          <li>
            <span class="credit-name">annali07</span>
            <span class="credit-role">luna-sama</span>
          </li>
          <li>
            <span class="credit-name">RVC-Boss</span>
            <span class="credit-role">GPT-SoVITS</span>
          </li>
          <li>
            <span class="credit-name">huangyf2013320506</span>
            <span class="credit-role">magic_conch</span>
          </li>
        </ul>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import pkg from '../../../package.json'

const AVATAR_KEY = 'profile-avatar'
const avatar = ref('')
const version = '1.0'
const author = 'Soulw1nd, firefly163'

const loadAvatar = () => {
  const saved = localStorage.getItem(AVATAR_KEY)
  if (saved) avatar.value = saved
}

const handleBeforeUpload = (file) => {
  const isImg = file.type.startsWith('image/')
  if (!isImg) {
    return false
  }
  return true
}

const handleAvatarChange = (uploadFile) => {
  const file = uploadFile.raw
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => {
    avatar.value = reader.result
    localStorage.setItem(AVATAR_KEY, avatar.value)
  }
  reader.readAsDataURL(file)
}

onMounted(loadAvatar)
</script>

<style scoped>
.profile-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px;
}

.avatar-card {
  display: flex;
  justify-content: center;
  border-radius: 12px;
}

.avatar-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
}

.avatar-uploader {
  cursor: pointer;
}

.avatar {
  width: 140px;
  height: 140px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid #e5e7eb;
}

.avatar.placeholder {
  width: 140px;
  height: 140px;
  border-radius: 50%;
  background: #f3f4f6;
  color: #6b7280;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px dashed #d1d5db;
}

.info-card {
  border-radius: 12px;
  background: #f7f8fa;
}

.info-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 4px;
  border-bottom: 1px dashed #e5e7eb;
}

.info-column {
  display: flex;
  flex-direction: column;
  padding: 10px 4px;
}

.credits-list {
  margin: 8px 0 0 0;
  padding: 0;
  list-style: none;
}

.credits-list li {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 14px;
}

.credit-name {
  color: #374151;
  font-weight: 500;
}

.credit-role {
  color: #6b7280;
  font-size: 13px;
}

.label {
  color: #6b7280;
}

.value {
  font-weight: 600;
  color: #111827;
}
</style>

