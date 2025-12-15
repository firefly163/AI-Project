import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { addRecord, getHistory, updateLastUserContent, resetHistory } from '@/utils/chatHistory'

export const useChatStore = defineStore('chat', () => {
  const conversations = ref([])
  const activeId = ref('')
  const modelOptions = ref([])

  const activeConversation = computed(() =>
    conversations.value.find((c) => c.id === activeId.value) || null
  )

  const setConversations = (list) => {
    conversations.value = list || []
    if (list?.length && !activeId.value) {
      activeId.value = list[0].id
    }
  }

  const setActive = (id) => {
    activeId.value = id
  }

  const setModelOptions = (list) => {
    modelOptions.value = list || []
  }

  const pushHistory = (role, content) => addRecord(role, content)
  const getHistoryList = () => getHistory()
  const editLastUser = (content) => updateLastUserContent(content)
  const clearHistory = () => resetHistory()

  return {
    conversations,
    activeId,
    modelOptions,
    activeConversation,
    setConversations,
    setActive,
    setModelOptions,
    pushHistory,
    getHistoryList,
    editLastUser,
    clearHistory
  }
})

