import axios from 'axios'

const LUNA_LLM_BASE = '/luna-llm'
const LUNA_TTS_BASE = '/luna-tts'
const ASAHI_LLM_BASE = '/asahi-llm'
const ASAHI_TTS_BASE = '/asahi-tts'

const createClient = (baseURL) =>
  axios.create({
    baseURL,
    timeout: 30000 // 部分响应较慢，放宽超时
  })

const lunaLLMClient = createClient(LUNA_LLM_BASE)
const lunaTTSClient = createClient(LUNA_TTS_BASE)
const asahiLLMClient = createClient(ASAHI_LLM_BASE)
const asahiTTSClient = createClient(ASAHI_TTS_BASE)

const sanitizeDownloadUrl = (url, proxyBase) => {
  if (!url) return ''
  let clean = url
  clean = clean.replace('0.0.0.0', '117.50.85.179')
  try {
    const parsed = new URL(clean.startsWith('http') ? clean : `http://${clean}`)
    return `${proxyBase}${parsed.pathname}${parsed.search || ''}`
  } catch (e) {
    return `${proxyBase}${clean.startsWith('/') ? clean : `/${clean}`}`
  }
}

async function downloadWavWithRetry(url, attempts = 3, timeoutMs = 10000) {
  if (!url) return { audioUrl: '', ok: false }
  let lastError
  for (let i = 0; i < attempts; i += 1) {
    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), timeoutMs)
      const res = await fetch(url, { signal: controller.signal })
      clearTimeout(timer)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const objectUrl = URL.createObjectURL(blob)
      return { audioUrl: objectUrl, ok: true }
    } catch (err) {
      lastError = err
      await new Promise((r) => setTimeout(r, 300 * (i + 1)))
    }
  }
  console.warn('下载音频失败', lastError)
  return { audioUrl: '', ok: false, error: lastError }
}

async function callModelLLM(client, userInput, historyArray = []) {
  const payload = {
    user: userInput || '',
    history: historyArray || []
  }
  const res = await client.post('/chat', payload)
  // 兼容不同返回结构
  const data = res?.data || res
  return data?.sentence || data?.data?.sentence || ''
}

async function callModelTTS(client, text, text_lang, proxyBase) {
  const res = await client.get('/speak', { params: { text, text_lang } })
  const data = res?.data || res
  const rawDownload =
    data?.download_url ||
    data?.data?.download_url ||
    data?.download_info?.download_url ||
    data?.data?.download_info?.download_url ||
    data?.url ||
    data?.data?.url
  const downloadUrl = sanitizeDownloadUrl(rawDownload, proxyBase)
  const { audioUrl, ok, error } = await downloadWavWithRetry(downloadUrl)
  return {
    downloadUrl,
    audioUrl,
    ok,
    error
  }
}

export async function callLunaLLM(userInput, historyArray) {
  return callModelLLM(lunaLLMClient, userInput, historyArray)
}

export async function getLunaTTS(text, text_lang = 'ja') {
  return callModelTTS(lunaTTSClient, text, text_lang, LUNA_TTS_BASE)
}

export async function callAsahiLLM(userInput, historyArray) {
  return callModelLLM(asahiLLMClient, userInput, historyArray)
}

export async function getAsahiTTS(text, text_lang = 'ja') {
  return callModelTTS(asahiTTSClient, text, text_lang, ASAHI_TTS_BASE)
}

