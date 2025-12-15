<template>
  <div class="models-page">
    <div class="models-grid">
      <div
        v-for="model in models"
        :key="model.id"
        class="model-card"
      >
        <div class="avatar-wrapper" @click="triggerAvatarUpload(model.id)">
          <img :src="model.avatar" alt="avatar" />
          <input
            class="file-input"
            type="file"
            accept="image/*"
            :ref="setFileInputRef(model.id)"
            @change="(e) => handleAvatarChange(e, model)"
          />
          <span class="avatar-tip">点击更换头像</span>
        </div>

        <div class="name-row">
          <template v-if="!model.editing">
            <span class="model-name" @click="startEdit(model)">{{ model.name }}</span>
          </template>
          <template v-else>
            <el-input
              v-model="model.editName"
              size="small"
              class="name-input"
              @keyup.enter="confirmEdit(model)"
            />
            <div class="name-actions">
              <el-button size="small" @click="cancelEdit(model)">取消</el-button>
              <el-button type="primary" size="small" @click="confirmEdit(model)">完成</el-button>
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
      <div v-if="infoError" class="info-error">获取服务器信息失败，已显示本地占位数据。</div>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { get } from '@/utils/request'
import { API } from '@/api/config'
import defaultAvatar1 from '@/assets/images/women_default1.png'
import defaultAvatar2 from '@/assets/images/women_default2.png'
import hailuoAvatar from '@/assets/images/hailuo2.png'

const defaultAvatars = [defaultAvatar1, defaultAvatar2, hailuoAvatar]
const STORAGE_KEY = 'models-settings'

const makeModel = (id, name, avatar) => {
  const modelAvatar = avatar || defaultAvatars[Math.floor(Math.random() * defaultAvatars.length)]
  return {
    id,
    name,
    avatar: modelAvatar,
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
  makeModel('luna', '桜小路ルナ'),
  makeModel('asahi', '小倉朝日')
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
  model: '-',
  adapter: '-',
  supports: '-',
  version: '-'
})
const infoLoading = ref(false)
const infoError = ref(false)

const fetchInfo = async () => {
  infoLoading.value = true
  infoError.value = false
  try {
    const res = await get(API.INFO)
    // 兼容不同数据形态
    serverInfo.value = {
      model: res?.model || res?.data?.model || '未知模型',
      adapter: res?.adapter || res?.data?.adapter || 'unknown',
      supports: res?.supports?.join?.(', ') || res?.data?.supports?.join?.(', ') || 'N/A',
      version: res?.version || res?.data?.version || 'N/A'
    }
  } catch (e) {
    infoError.value = true
    serverInfo.value = {
      model: '占位模型',
      adapter: 'mock-adapter',
      supports: 'text, audio',
      version: 'dev'
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
          m.features.language = savedModel.features.language || 'zh'
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
    features: {
      voice: m.features.voice,
      translate: m.features.translate,
      autoTranslate: m.features.autoTranslate,
      language: m.features.language
    }
  }))
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
}

watch(
  models,
  () => saveModels(),
  { deep: true }
)

onMounted(() => {
  hydrateModels()
  fetchInfo()
})
</script>

<style scoped>
.models-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

.model-card {
  background: #ffffff;
  border: 1px solid #e5eaf3;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
}

.avatar-wrapper {
  position: relative;
  width: 96px;
  height: 96px;
  border-radius: 50%;
  overflow: hidden;
  border: 2px solid #f1f5f9;
  cursor: pointer;
}

.avatar-wrapper img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.avatar-tip {
  position: absolute;
  bottom: 4px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 10px;
  opacity: 0;
  transition: opacity 0.2s ease;
  pointer-events: none;
}

.avatar-wrapper:hover .avatar-tip {
  opacity: 1;
}

.file-input {
  display: none;
}

.name-row {
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  justify-content: center;
}

.model-name {
  font-weight: 700;
  cursor: pointer;
}

.name-actions {
  display: flex;
  gap: 8px;
}

.name-input {
  width: 140px;
}

.feature-row {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px dashed #eef1f6;
}

.feature-row:last-child {
  border-bottom: none;
}

.lang-select {
  width: 120px;
}

.info-card {
  border-radius: 12px;
}

.card-header {
  font-weight: 700;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px 16px;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.info-key {
  color: #6b7280;
  min-width: 72px;
}

.info-value {
  color: #111827;
}

.info-error {
  margin-top: 8px;
  color: #e11d48;
  font-size: 12px;
}
</style>

