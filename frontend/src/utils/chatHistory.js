// 聊天历史记录工具（前端内存+可自行持久化）
// 数据结构：[{ role: 'user' | 'modelId', content: '原始文本' }]

const MAX_PAIRS = 12 * 2 // 最近12轮，对应最多24条记录
let history = []

export function addRecord(role, content) {
  if (!role || content == null) return
  history.push({ role, content })
  if (history.length > MAX_PAIRS) {
    history = history.slice(history.length - MAX_PAIRS)
  }
}

export function getHistory() {
  return [...history]
}

export function updateLastUserContent(newContent) {
  for (let i = history.length - 1; i >= 0; i -= 1) {
    if (history[i].role === 'user') {
      history[i].content = newContent
      break
    }
  }
}

// 将原始历史转换为目标语言，translateFn: async (text, targetLang) => translatedText
export async function translateHistoryForModel(rawHistory, targetLang, translateFn) {
  if (!translateFn || !targetLang) return rawHistory || []
  const result = []
  for (const item of rawHistory || []) {
    const translated = await translateFn(item.content, targetLang)
    result.push({ role: item.role, content: translated || item.content })
  }
  return result
}

// 工具函数：重置（测试或切换会话时可用）
export function resetHistory() {
  history = []
}

