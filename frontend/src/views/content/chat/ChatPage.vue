<template>
  <div class="ai-practice-container">
    <!-- 左侧历史对话记录 -->
    <div class="history-panel">
      <div class="new-chat-container">
        <button class="new-chat-btn" @click="newConversation">
          {{ t('chat.newChat') }}
        </button>
      </div>
      <ul class="history-list">
        <li
          v-for="(item, index) in historyList"
          :key="item.id"
          @click="selectConversation(index)"
          :class="{ active: currentConversationIndex === index }"
          @mouseenter="hoverHistoryId = item.id"
          @mouseleave="hoverHistoryId = ''"
        >
          <template v-if="item.renaming">
            <input
              v-model="renameDraft"
              class="rename-input"
              @keyup.enter.stop="applyRename(item)"
              @click.stop
            />
            <div class="rename-actions" @click.stop>
              <el-button size="small" @click="applyRename(item)">{{ t('chat.confirm') }}</el-button>
            </div>
          </template>
          <template v-else>
            <span class="title-text">{{ item.title }}</span>
            <div class="history-actions" :class="{ visible: hoverHistoryId === item.id }" @click.stop>
              <el-dropdown trigger="click">
                <span class="ellipsis">...</span>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item @click="startRename(item)">{{ t('chat.rename') }}</el-dropdown-item>
                    <el-dropdown-item divided class="danger" @click="deleteConversation(item.id)">{{ t('chat.delete') }}</el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </div>
          </template>
        </li>
      </ul>
    </div>

    <!-- 右侧对话页面 -->
    <div class="chat-wrapper">
      <div class="chat-panel">
        <!-- 上半部分聊天界面 -->
        <div class="chat-messages" ref="chatMessagesRef">
          <div
            v-for="message in currentConversation.messages"
            :key="message.id"
            :class="['message', message.role]"
            @mouseenter="hoverMessageId = message.id"
            @mouseleave="hoverMessageId = ''"
          >
            <div class="avatar">
              <img
                v-if="message.role === 'assistant'"
                :src="getModelAvatar(message.modelId)"
                alt="AI Avatar"
              >
              <img
                v-else
                src="@/assets/images/user.png"
                alt="User"
              >
            </div>
            <div class="content-wrapper">
              <div
                v-if="message.role === 'assistant' && message.replyFrom?.content"
                class="reply-meta"
              >
                {{ t('chat.from') }} {{ message.replyFrom.sender || t('chat.user') }}：{{ (message.replyFrom.content || '').slice(0, 10) + '.....' }}
              </div>

              <div
                class="content"
                :class="{ loading: message.loading }"
              >
                <template v-if="message.editing">
                  <el-input
                    v-model="message.editDraft"
                    type="textarea"
                    :rows="3"
                    resize="none"
                    class="edit-input"
                  />
                  <div class="edit-actions">
                    <el-button size="small" @click="cancelEdit(message)">{{ t('chat.cancel') }}</el-button>
                    <el-button type="primary" size="small" @click="confirmEdit(message)">{{ t('chat.confirm') }}</el-button>
                  </div>
                </template>
                <template v-else>
                  <div class="content-text">
                    <span v-if="message.loading">{{ t('chat.modelThinking', { name: getModelName(message.modelId) }) }}</span>
                    <template v-else>
                      <span>{{ message.content }}</span>
                    </template>
                  </div>
                </template>
              </div>

              <!-- 操作栏（悬停显示，但预留高度避免抖动） -->
              <div class="action-bar" :class="{ visible: hoverMessageId === message.id }">
                <template v-if="message.role === 'user'">
                  <el-tooltip :content="t('chat.forward')">
                    <el-button text circle size="small" @click="openForwardDialog(message)">
                      <el-icon><Promotion /></el-icon>
                    </el-button>
                  </el-tooltip>
                  <el-tooltip :content="t('chat.copy')">
                    <el-button text circle size="small" @click="copyText(message.content)">
                      <el-icon><CopyDocument /></el-icon>
                    </el-button>
                  </el-tooltip>
                  <el-tooltip v-if="!message.locked" :content="t('chat.edit')">
                    <el-button text circle size="small" @click="startEdit(message)">
                      <el-icon><Edit /></el-icon>
                    </el-button>
                  </el-tooltip>
                  <el-tooltip v-else :content="t('chat.locked')">
                    <el-button text circle size="small" disabled>
                      <el-icon><Edit /></el-icon>
                    </el-button>
                  </el-tooltip>
                </template>
                <template v-else>
                  <el-popover
                    placement="top"
                    :width="300"
                    trigger="hover"
                    :content="message.translation || t('chat.translating')"
                    :disabled="!canTranslate(message.modelId)"
                    @show="translateMessage(message)"
                  >
                    <template #reference>
                      <div style="display: inline-block;">
                        <el-button
                          text
                          circle
                          size="small"
                          :disabled="!canTranslate(message.modelId)"
                        >
                          <el-icon><Refresh /></el-icon>
                        </el-button>
                      </div>
                    </template>
                  </el-popover>
                  
                  <el-tooltip :content="t('chat.play')" :disabled="!canPlay(message.modelId) || !message.audioUrl">
                    <el-button
                      text
                      circle
                      size="small"
                      :disabled="!canPlay(message.modelId) || !message.audioUrl"
                      @click="playAudio(message)"
                    >
                      <el-icon><VideoPlay /></el-icon>
                    </el-button>
                  </el-tooltip>
                  <el-tooltip :content="t('chat.forward')">
                    <el-button text circle size="small" @click="openForwardDialog(message)">
                      <el-icon><Promotion /></el-icon>
                    </el-button>
                  </el-tooltip>
                  <el-tooltip :content="t('chat.copy')">
                    <el-button text circle size="small" @click="copyText(message.content)">
                      <el-icon><CopyDocument /></el-icon>
                    </el-button>
                  </el-tooltip>
                </template>
              </div>
            </div>
          </div>
        </div>

        <!-- 输入框 -->
        <div class="input-area">
          <div class="input-wrapper">
            <input
                v-model="userInput"
                @keyup.enter="sendMessage"
                :placeholder="t('chat.inputPlaceholder')"
                type="text"
            >
            <div class="button-group">
                  <el-button
                      class="send-button"
                      circle
                      @click="sendMessage"
                      :disabled="!userInput.trim()"
                  >
                    <el-icon>
                      <Top/>
                    </el-icon>
                  </el-button>
            </div>
          </div>
        </div>

        <div class="disclaimer">
          {{ t('chat.disclaimer') }}
        </div>
      </div>
    </div>

    <!-- 模型选择面板 -->
    <el-dialog
      v-model="modelDialogVisible"
      :title="t('chat.modelSelectTitle')"
      width="420px"
    >
      <el-checkbox-group v-model="selectedModels">
        <el-checkbox
          v-for="item in modelOptions"
          :key="item.id"
          :label="item.id"
        >
          {{ item.name }}
        </el-checkbox>
      </el-checkbox-group>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="modelDialogVisible = false">{{ t('chat.cancel') }}</el-button>
          <el-button type="primary" @click="confirmModelSelection">{{ t('chat.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 转发面板 -->
    <el-dialog
      v-model="forwardDialogVisible"
      :title="t('chat.forwardTitle')"
      width="480px"
    >
      <div class="forward-list">
        <el-checkbox-group v-model="forwardSelectedModels">
          <el-checkbox
            v-for="item in modelOptions"
            :key="item.id"
            :label="item.id"
          >
            <span class="model-item">
              <img :src="item.avatar" class="model-avatar" />
              {{ item.name }}
            </span>
          </el-checkbox>
        </el-checkbox-group>
      </div>
      <div class="forward-footer-extra">
        <el-checkbox v-model="forwardSetDefault">{{ t('chat.setDefault') }}</el-checkbox>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="forwardDialogVisible = false">{{ t('chat.cancel') }}</el-button>
          <el-button type="primary" @click="sendForward">{{ t('chat.send') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Top, Promotion, CopyDocument, Edit, Refresh, VideoPlay } from '@element-plus/icons-vue'
import { callLunaLLM, callAsahiLLM, getLunaTTS, getAsahiTTS } from '@/api/models'
import { translateHistoryForModel } from '@/utils/chatHistory'
import { translateTextBaidu, languageToBaidu } from '@/utils/baiduTranslate'
import lunaAvatar from '@/assets/images/women_default1.png'
import asahiAvatar from '@/assets/images/women_default2.png'
import hailuoAvatar from '@/assets/images/hailuo2.png'
import { useI18n } from '@/utils/i18n'
import { useAppStore } from '@/stores/appStore'

const { t } = useI18n()
const appStore = useAppStore()

const genId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`

const MODEL_STORAGE_KEY = 'models-settings'
const CHAT_STORAGE_KEY = 'chat-conversations'
const MAX_HISTORY = 24 // role+content 共24条（12轮）

const defaultModelOptions = [
  { id: 'luna', name: '桜小路ルナ', avatar: lunaAvatar, features: { voice: true, translate: true, autoTranslate: false, language: 'ja' } },
  { id: 'asahi', name: '小倉朝日', avatar: asahiAvatar, features: { voice: true, translate: true, autoTranslate: false, language: 'ja' } },
]

const modelOptions = ref([...defaultModelOptions])

const loadModelOptions = () => {
  try {
    const raw = localStorage.getItem(MODEL_STORAGE_KEY)
    if (raw) {
      const saved = JSON.parse(raw)
      if (Array.isArray(saved) && saved.length) {
        modelOptions.value = saved.map((m) => ({
          id: m.id,
          name: m.name,
          avatar: m.avatar || hailuoAvatar,
          features: {
            voice: !!m.features?.voice,
            translate: !!m.features?.translate,
            autoTranslate: !!m.features?.autoTranslate,
            language: m.features?.language || 'ja'
          }
        }))
        return
      }
    }
  } catch (e) {
    console.warn('读取模型配置失败，将使用默认值', e)
  }
  modelOptions.value = [...defaultModelOptions]
}

const makeDefaultConversation = () => ({
  id: genId(),
  title: '蟹堡制作指南',
  selectedModels: ['luna'],
  messages: [],
  historyRecords: []
})

const historyList = ref([makeDefaultConversation()])

const hydrateConversations = () => {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY)
    if (raw) {
      const saved = JSON.parse(raw)
      if (Array.isArray(saved) && saved.length) {
        historyList.value = saved.map((c) => ({
          ...c,
          messages: c.messages || [],
          selectedModels: c.selectedModels || [],
          historyRecords: c.historyRecords || []
        }))
        if (currentConversationIndex.value >= historyList.value.length) {
          currentConversationIndex.value = 0
        }
        return
      }
    }
  } catch (e) {
    console.warn('读取会话失败，将使用默认会话', e)
  }
  historyList.value = [makeDefaultConversation()]
}

const persistConversations = () => {
  localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(historyList.value))
}

const currentConversationIndex = ref(0)
const userInput = ref('')
const chatMessagesRef = ref(null)
const hoverMessageId = ref('')
const hoverHistoryId = ref('')
const renameDraft = ref('')

const modelDialogVisible = ref(false)
const selectedModels = ref([])

const forwardDialogVisible = ref(false)
const forwardSelectedModels = ref([])
const forwardSetDefault = ref(false)
const forwardMessage = ref(null)

const currentConversation = computed(() => historyList.value[currentConversationIndex.value])

const ensureHistoryList = (conv) => {
  if (!conv.historyRecords) conv.historyRecords = []
}

const trimHistory = (conv) => {
  ensureHistoryList(conv)
  if (conv.historyRecords.length > MAX_HISTORY) {
    conv.historyRecords = conv.historyRecords.slice(conv.historyRecords.length - MAX_HISTORY)
  }
}

const pushHistoryRecord = (conv, role, content, message) => {
  ensureHistoryList(conv)
  conv.historyRecords.push({ role, content })
  trimHistory(conv)
  if (message) {
    message.historyIndex = conv.historyRecords.length - 1
  }
}

const getModelConfig = (modelId) => {
  const list = Array.isArray(modelOptions.value) ? modelOptions.value : defaultModelOptions
  return list.find((i) => i.id === modelId) || list[0]
}

const selectConversation = (index) => {
  if (index < 0 || index >= historyList.value.length) return
  currentConversationIndex.value = index
  nextTick(scrollToBottom)
}

const newConversation = () => {
  historyList.value.unshift({
    id: genId(),
    title: '新对话',
    selectedModels: [],
    messages: [],
    historyRecords: []
  })
  currentConversationIndex.value = 0
  selectedModels.value = []
  modelDialogVisible.value = true
  persistConversations()
}

const confirmModelSelection = () => {
  currentConversation.value.selectedModels = [...selectedModels.value]
  modelDialogVisible.value = false
  persistConversations()
}

const ensureModelsSelected = () => {
  if (!currentConversation.value.selectedModels?.length) {
    selectedModels.value = []
    modelDialogVisible.value = true
    return false
  }
  return true
}

const translateText = async (text, targetLang, sourceLang = 'auto') => {
  return translateTextBaidu(text, targetLang, sourceLang)
}

const buildReplyMeta = (sender, content) => ({
  sender: sender || t('chat.user'),
  content: content || ''
})

const sendMessage = async () => {
  if (!userInput.value.trim()) return
  if (!ensureModelsSelected()) return

  const content = userInput.value.trim()
  const userMsg = {
    id: genId(),
    role: 'user',
    content,
    originalContent: content,
    locked: true
  }
  currentConversation.value.messages.push(userMsg)
  pushHistoryRecord(currentConversation.value, 'user', content, userMsg)

  userInput.value = ''
  nextTick(scrollToBottom)
  persistConversations()

  const models = currentConversation.value.selectedModels
  models.forEach((mid) => handleModelResponse(mid, content, buildReplyMeta(t('chat.user'), content)))
}

const handleModelResponse = async (modelId, prompt, replyFrom) => {
  const loadingMessage = {
    id: genId(),
    role: 'assistant',
    modelId,
    content: '',
    loading: true,
    replyFrom,
    translation: '',
    showTranslation: false,
    audioUrl: ''
  }
  currentConversation.value.messages.push(loadingMessage)
  nextTick(scrollToBottom)
  persistConversations()

  try {
    const { text, audioUrl, translation } = await fetchModelReply(modelId, prompt)
    loadingMessage.content = text || t('chat.emptyResponse')
    loadingMessage.loading = false
    if (audioUrl) loadingMessage.audioUrl = audioUrl
    if (translation) {
      loadingMessage.translation = translation
      loadingMessage.showTranslation = true
    }
    pushHistoryRecord(currentConversation.value, getModelName(modelId), loadingMessage.content, loadingMessage)

    // Auto-Forward Logic
    if (appStore.autoForward && text) {
      const senderName = getModelName(modelId)
      // Check for Luna
      if (modelId !== 'luna' && (text.includes('桜小路ルナ') || text.includes('ルナ'))) {
        handleModelResponse('luna', text, buildReplyMeta(senderName, text))
      }
      // Check for Asahi
      if (modelId !== 'asahi' && (text.includes('小倉朝日') || text.includes('朝日'))) {
        handleModelResponse('asahi', text, buildReplyMeta(senderName, text))
      }
    }

  } catch (err) {
    loadingMessage.loading = false
    loadingMessage.content = t('chat.errorResponse')
    console.warn(err)
  } finally {
    nextTick(scrollToBottom)
    persistConversations()
  }
}

const fetchModelReply = async (modelId, prompt) => {
  const model = getModelConfig(modelId)
  const targetLang = model?.features?.language || 'ja'

  let translatedHistory = []
  let translatedPrompt = prompt
  try {
    translatedHistory = await translateHistoryForModel(
      currentConversation.value.historyRecords || [],
      languageToBaidu(targetLang),
      (txt) => translateText(txt, languageToBaidu(targetLang))
    )
    translatedPrompt = await translateText(prompt, languageToBaidu(targetLang))
  } catch (e) {
    console.warn('翻译历史或输入失败，使用原文继续', e)
    translatedHistory = currentConversation.value.historyRecords || []
    translatedPrompt = prompt
  }

  let text = ''
  if (modelId === 'luna') {
    text = await callLunaLLM(translatedPrompt, translatedHistory)
  } else {
    text = await callAsahiLLM(translatedPrompt, translatedHistory)
  }

  let audioUrl = ''
  if (model?.features?.voice && text) {
    try {
      const ttsRes = modelId === 'luna'
        ? await getLunaTTS(text, targetLang)
        : await getAsahiTTS(text, targetLang)
      if (ttsRes?.ok) {
        audioUrl = ttsRes.audioUrl
      } else if (ttsRes?.error) {
        ElMessage.error('语音生成失败，已关闭本条语音播放')
      }
    } catch (e) {
      console.warn('TTS 调用失败', e)
    }
  }

  let translation = ''
  if (model?.features?.autoTranslate && text) {
    try {
      // Auto-translate uses user's preferred language (app setting)
      // Defaults to 'zh' if setting is missing or invalid for translation target
      const userLang = appStore.language === 'en' ? 'en' : (appStore.language === 'ja' ? 'jp' : 'zh')
      translation = await translateText(text, userLang, languageToBaidu(targetLang))
    } catch (e) {
      console.warn('自动翻译失败', e)
    }
  }

  return { text, audioUrl, translation }
}

const openForwardDialog = (message) => {
  forwardMessage.value = message
  forwardSelectedModels.value = []
  forwardSetDefault.value = false
  forwardDialogVisible.value = true
}

const sendForward = () => {
  if (!forwardSelectedModels.value.length) {
    ElMessage.warning(t('chat.selectAtLeastOne'))
    return
  }
  if (forwardSetDefault.value) {
    currentConversation.value.selectedModels = [...forwardSelectedModels.value]
  }

  const content = forwardMessage.value.content
  const sender =
    forwardMessage.value.role === 'assistant'
      ? getModelName(forwardMessage.value.modelId)
      : t('chat.user')
  const replyMeta = buildReplyMeta(sender, content)

  forwardSelectedModels.value.forEach((mid) => handleModelResponse(mid, content, replyMeta))
  forwardDialogVisible.value = false
  nextTick(scrollToBottom)
  persistConversations()
}

const copyText = async (text) => {
  try {
    await navigator.clipboard.writeText(text || '')
    ElMessage.success(t('chat.copied'))
  } catch (e) {
    ElMessage.error(t('chat.copyFailed'))
  }
}

const startEdit = (message) => {
  if (message.role !== 'user') return
  if (message.locked) return
  message.editing = true
  message.editDraft = message.content
}

const cancelEdit = (message) => {
  message.editing = false
  message.editDraft = message.content
}

const confirmEdit = (message) => {
  if (message.role !== 'user') return
  message.content = message.editDraft || ''
  message.originalContent = message.content
  message.editing = false
  if (typeof message.historyIndex === 'number' && currentConversation.value.historyRecords?.[message.historyIndex]) {
    currentConversation.value.historyRecords[message.historyIndex].content = message.content
  }
  persistConversations()
}

const translateMessage = async (message) => {
  if (!canTranslate(message.modelId)) return

  if (message.translation) {
    // Already translated
    return
  }

  try {
    // Determine target language based on app settings (user's UI language)
    // Default to 'zh' if current app language is not supported or maps to 'zh'
    const userLang = appStore.language === 'en' ? 'en' : (appStore.language === 'ja' ? 'jp' : 'zh')
    
    message.translation = await translateText(
      message.content,
      userLang,
      languageToBaidu(getModelConfig(message.modelId)?.features?.language || 'ja')
    )
    message.showTranslation = true
  } catch (e) {
    ElMessage.error(t('chat.translateFailed'))
  }
}

const playAudio = (message) => {
  if (!canPlay(message.modelId) || !message.audioUrl) return
  const audio = new Audio(message.audioUrl)
  audio.play().catch(() => ElMessage.error(t('chat.playFailed')))
}

const getModelAvatar = (modelId) => {
  const m = getModelConfig(modelId)
  return m?.avatar || hailuoAvatar
}

const getModelName = (modelId) => {
  const m = getModelConfig(modelId)
  return m?.name || '神奇海螺'
}

const canTranslate = (modelId) => {
  const m = getModelConfig(modelId)
  return !!m?.features?.translate
}

const canPlay = (modelId) => {
  const m = getModelConfig(modelId)
  return !!m?.features?.voice
}

const scrollToBottom = () => {
  const chatMessages = chatMessagesRef.value
  if (chatMessages) {
    chatMessages.scrollTop = chatMessages.scrollHeight
  }
}

const startRename = (item) => {
  historyList.value.forEach((c) => (c.renaming = false))
  renameDraft.value = item.title
  item.renaming = true
}

const applyRename = (item) => {
  item.title = renameDraft.value.trim() || item.title
  item.renaming = false
  persistConversations()
}

const deleteConversation = (id) => {
  const idx = historyList.value.findIndex((c) => c.id === id)
  if (idx === -1) return
  historyList.value.splice(idx, 1)
  if (!historyList.value.length) {
    historyList.value.push(makeDefaultConversation())
  }
  currentConversationIndex.value = 0
  persistConversations()
}

watch(
  historyList,
  () => persistConversations(),
  { deep: true }
)

onMounted(() => {
  loadModelOptions()
  hydrateConversations()
})
</script>

<style scoped>
/* 样式保持不变 */
.ai-practice-container {
  display: flex;
  height: 100vh;
  font-family: Arial, sans-serif;
}

.history-panel {
  width: 280px;
  background: linear-gradient(135deg, rgba(230, 240, 255, 0.01), rgba(240, 230, 255, 0.01));
  background-color: #ffffff;
  padding: 20px;
  overflow-y: auto;
}

.new-chat-container {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.new-chat-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 13px;
  margin-top: 10px;
  margin-bottom: 5px;
  background: linear-gradient(to right, #0e111a, #1f2633);
  color: #c1c9d9;
  border: 1px solid #2f3745;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 14px;
  font-weight: bold;
}

.new-chat-btn:hover {
  background: linear-gradient(to right, #161c27, #252e3d);
  color: #e3e8f0;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}

.history-list {
  list-style-type: none;
  padding: 0;
}

.history-list li {
  padding: 10px;
  margin-bottom: 10px;
  background-color: #ffffff;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s;
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: space-between;
}

.history-list li:hover,
.history-list li.active {
  background-color: rgba(0, 105, 224, 0.15);
  color: #0052bc;
}

.title-text {
  flex: 1;
  min-width: 0;
}

.history-actions {
  opacity: 0;
  transition: opacity 0.2s ease;
}

.history-actions.visible {
  opacity: 1;
}

.ellipsis {
  display: inline-block;
  padding: 2px 6px;
  cursor: pointer;
  color: #6b7280;
}

.danger {
  color: #e11d48;
}

.rename-input {
  flex: 1;
  padding: 6px 8px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
}

.rename-actions {
  margin-left: 6px;
}

.chat-wrapper {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg,
  rgba(0, 105, 224, 0.08),
  rgba(0, 56, 148, 0.08)
  );
}

.chat-panel {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: transparent;
  box-shadow: none;
  padding-top: 12px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding-top: 20px;
  padding-left: 10%;
  padding-right: 10%;
  background-color: transparent;
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 105, 224, 0.3) transparent;
}

.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background-color: rgba(0, 105, 224, 0.3);
  border-radius: 3px;
}

.message {
  display: flex;
  margin-bottom: 20px;
}

.message .avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px;
  overflow: hidden;
}

.message .avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 50%;
  background-color: #ffffff;
}

.content-wrapper {
  position: relative;
  max-width: 80%;
  display: flex;
  flex-direction: column;
}

.reply-meta {
  background: rgba(0, 105, 224, 0.08);
  color: #1f2d3d;
  padding: 4px 8px;
  border-radius: 6px 6px 0 0;
  font-size: 12px;
  border: 1px solid rgba(0, 105, 224, 0.15);
  border-bottom: none;
}

.message .content {
  background-color: rgba(255, 255, 255, 1);
  padding: 12px 18px;
  border-radius: 0 10px 10px 10px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  font-size: 16px;
  line-height: 1.8;
}

.message.user .content {
  background-color: rgba(0, 105, 224, 0.12);
  color: black;
  border-radius: 10px 0 10px 10px;
}

.message.user {
  flex-direction: row-reverse;
}

.message.user .avatar {
  margin-right: 0;
  margin-left: 10px;
}

.content.loading {
  color: #5c6c7b;
}

.action-bar {
  display: flex;
  gap: 6px;
  margin-top: 6px;
  opacity: 0;
  transition: opacity 0.2s;
  min-height: 32px;
}

.action-bar.visible {
  opacity: 1;
}

.message:hover .action-bar {
  opacity: 1;
}

.edit-actions {
  margin-top: 8px;
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.edit-input :deep(.el-textarea__inner) {
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid #dcdfe6;
}

.edit-input :deep(.el-textarea__inner:focus) {
  border-color: #0069e0;
  box-shadow: 0 0 0 2px rgba(0, 105, 224, 0.12);
}

.input-area {
  padding: 20px 10% 0 10%;
  border-top: 0 solid #e0e0e0;
  background-color: transparent;
}

.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

input {
  width: 100%;
  padding: 12px 70px 12px 18px;
  border: 1px solid rgba(204, 204, 204, 0.5);
  border-radius: 25px;
  font-size: 16px;
  background-color: rgba(255, 255, 255, 0.7);
  transition: border-color 0.3s;
  height: 55px;
}

input:focus {
  outline: none;
  border-color: #0069e0;
}

input::placeholder {
  color: #969696;
}

.button-group {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: center;
}

.send-button {
  width: 40px;
  height: 40px;
  background: linear-gradient(to right, #0069e0, #0052bc);
  border: none;
  color: white;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.send-button:disabled {
  background: rgba(0, 105, 224, 0.1);
  color: rgba(0, 82, 188, 0.3);
  cursor: default;
}

.send-button :deep(.el-icon) {
  font-size: 24px;
}

.send-button:not(:disabled):hover {
  opacity: 0.9;
}

.disclaimer {
  font-size: 10px;
  color: #999;
  text-align: center;
  margin-top: 12px;
  margin-bottom: 12px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.forward-list {
  max-height: 260px;
  overflow-y: auto;
}

.model-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.model-avatar {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

.forward-footer-extra {
  margin-top: 12px;
}
</style>