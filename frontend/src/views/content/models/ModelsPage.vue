<template>
  <div class="models-page">
    <div class="header">
      <h2 class="page-title">模型管理</h2>
      <el-button type="primary" size="small" @click="saveModels">保存设置</el-button>
    </div>

    <div class="models-grid">
      <div v-for="model in models" :key="model.id" class="model-card">
        <div class="model-header">
          <div class="avatar-upload" @click="triggerAvatarUpload(model.id)">
            <img :src="model.avatar" class="avatar-img" />
            <div class="upload-overlay">更换头像</div>
            <input
              type="file"
              :ref="setFileInputRef(model.id)"
              style="display: none"
              accept="image/*"
              @change="(e) => handleAvatarChange(e, model)"
            />
          </div>
          <template v-if="model.editing">
            <el-input
              v-model="model.editName"
              size="small"
              class="name-input"
              @keyup.enter="confirmEdit(model)"
            />
            <el-button link type="primary" size="small" @click="confirmEdit(model)">确定</el-button>
            <el-button link size="small" @click="cancelEdit(model)">取消</el-button>
          </template>
          <template v-else>
            <div class="model-name">
              {{ model.name }}
              <el-icon class="edit-icon" @click="startEdit(model)"><Edit /></el-icon>
            </div>
          </template>
        </div>

        <div class="feature-row">
          <span>语音</span>
          <el-switch v-model="model.features.voice" />
        </div>
        <div class="feature-row">
          <span>翻译</span>
          <el-switch v-model="model.features.translate" />
        </div>
        <div class="feature-row">
          <span>默认翻译</span>
          <el-switch v-model="model.features.autoTranslate" />
        </div>
        <div class="feature-row">
          <span>语言</span>
          <el-select v-model="model.features.language" size="small" class="lang-select">
            <el-option label="中文" value="zh" />
            <el-option label="日文" value="ja" />
          </el-select>
        </div>
      </div>
    </div>

    <el-card class="info-card" v-loading="infoLoading">
      <template #header>
        <div class="card-header">服务器信息</div>
      </template>
      <div class="info-grid">
        <div v-for="(val, key) in serverInfo" :key="key" class="info-item">
          <span class="info-key">{{ key }}</span>
          <span class="info-value">{{ val }}</span>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Edit } from '@element-plus/icons-vue'
import { get } from '@/utils/request'
import { API } from '@/api/config'
import defaultAvatar1 from '@/assets/images/women_default1.png'
import defaultAvatar2 from '@/assets/images/women_default2.png'
import hailuoAvatar from '@/assets/images/hailuo2.png'

const defaultAvatars = [defaultAvatar1, defaultAvatar2, hailuoAvatar]
const STORAGE_KEY = 'models-settings'

const makeModel = (id, name, avatar) => {
  return {
    id,
    name,
    avatar: avatar || hailuoAvatar,
    editing: false,
    editName: name,
    features: reactive({
      voice: true,
      translate: true,
      autoTranslate: false,
      language: 'zh'
    })
  }
}

const models = reactive([
  makeModel('luna', '桜小路ルナ', defaultAvatar1),
  makeModel('asahi', '小倉朝日', defaultAvatar2)
])

const fileInputs = new Map()
const setFileInputRef = (id) => (el) => {
  if (el) fileInputs.set(id, el)
}

const triggerAvatarUpload = (id) => {
  const el = fileInputs.get(id)
  if (el) el.click()
}

const handleAvatarChange = (event, model) => {
  const file = event.target.files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => {
    model.avatar = reader.result
    saveModels()
  }
  reader.onerror = () => {
    ElMessage.error('读取头像失败，请重试')
  }
  reader.readAsDataURL(file)
}

const startEdit = (model) => {
  model.editing = true
  model.editName = model.name
}

const cancelEdit = (model) => {
  model.editing = false
  model.editName = model.name
}

const confirmEdit = (model) => {
  model.name = model.editName?.trim() || model.name
  model.editing = false
  saveModels()
}

const serverInfo = ref({
  '桜小路ルナ': 'Loading...',
  '小倉朝日': 'Loading...'
})
const infoLoading = ref(false)
const infoError = ref(false)

const fetchInfo = async () => {
  infoLoading.value = true
  infoError.value = false
  try {
    const res = await get(API.INFO)
    serverInfo.value = {
      '桜小路ルナ': '在线 (v1.0)',
      '小倉朝日': '在线 (v1.0)',
      '服务端口': '8000',
      '支持功能': res?.supports?.join?.(', ') || '文本对话, 语音合成'
    }
  } catch (e) {
    infoError.value = true
    serverInfo.value = {
      '桜小路ルナ': '在线 (v1.0)',
      '小倉朝日': '在线 (v1.0)',
      '服务端口': '8000',
      '支持功能': '文本对话, 语音合成'
    }
  } finally {
    infoLoading.value = false
  }
}

const hydrateModels = () => {
  const savedRaw = localStorage.getItem(STORAGE_KEY)
  if (!savedRaw) return
  try {
    const saved = JSON.parse(savedRaw)
    if (Array.isArray(saved) && saved.length) {
      models.splice(
        0,
        models.length,
        ...saved.map((m) =>
          makeModel(
            m.id,
            m.name,
            m.avatar
          )
        )
      )
      // 应用功能设置
      models.forEach((m, idx) => {
        const savedModel = saved[idx]
        if (savedModel?.features) {
          m.features.voice = !!savedModel.features.voice
          m.features.translate = !!savedModel.features.translate
          m.features.autoTranslate = !!savedModel.features.autoTranslate
          m.features.language = savedModel.features.language || 'ja'
        }
      })
    }
  } catch (e) {
    console.warn('解析本地模型设置失败，将使用默认值', e)
  }
}

const saveModels = () => {
  const payload = models.map((m) => ({
    id: m.id,
    name: m.name,
    avatar: m.avatar,
    features: m.features
  }))
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  ElMessage.success('设置已保存')
}

onMounted(() => {
  hydrateModels()
  fetchInfo()
})
</script>

<style scoped>
.models-page {
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-title {
  font-size: 18px;
  font-weight: bold;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.model-card {
  background: #ffffff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.avatar-upload {
  position: relative;
  width: 48px;
  height: 48px;
  cursor: pointer;
  border-radius: 50%;
  overflow: hidden;
}

.avatar-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.upload-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  color: white;
  font-size: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
}

.avatar-upload:hover .upload-overlay {
  opacity: 1;
}

.model-name {
  font-weight: bold;
  font-size: 16px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.edit-icon {
  cursor: pointer;
  color: #999;
  font-size: 14px;
}

.edit-icon:hover {
  color: #409eff;
}

.name-input {
  width: 140px;
}

.feature-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  color: #606266;
}

.lang-select {
  width: 80px;
}

.info-card {
  border-radius: 12px;
}

.card-header {
  font-weight: bold;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.info-key {
  font-size: 12px;
  color: #909399;
}

.info-value {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
}

.info-error {
  margin-top: 12px;
  font-size: 12px;
  color: #e6a23c;
}
</style>
