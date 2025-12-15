// 百度大模型文本翻译封装（前端使用）
// 文档: https://fanyi-api.baidu.com/ait/api/aiTextTranslate

const APP_ID = ''
const API_KEY = '' // Bearer Token
// 使用本地代理绕过 CORS
const ENDPOINT = '/baidu-api/ait/api/aiTextTranslate'

const langMap = {
  zh: 'zh',
  cn: 'zh',
  'zh-cn': 'zh',
  ja: 'jp',
  jp: 'jp',
  en: 'en',
  auto: 'auto'
}

const toBaiduLang = (lang) => langMap[lang?.toLowerCase()] || 'auto'

export async function translateTextBaidu(text, targetLang = 'zh', sourceLang = 'auto') {
  if (!text) return ''
  
  const from = toBaiduLang(sourceLang)
  const to = toBaiduLang(targetLang)

  const payload = {
    appid: APP_ID,
    from,
    to,
    q: text
  }

  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(payload)
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    
    // Check for Baidu API error codes
    if (data.error_code && data.error_code !== 52000) {
      console.warn('百度翻译API错误:', data.error_code, data.error_msg)
      throw new Error(`Baidu API Error: ${data.error_code} ${data.error_msg}`)
    }

    if (Array.isArray(data?.trans_result) && data.trans_result.length) {
      return data.trans_result.map((i) => i.dst).join('')
    }
    throw new Error('empty translation')
  } catch (err) {
    console.warn('翻译失败:', err)
    // Rethrow to let caller handle UI feedback
    throw err
  }
}

export const languageToBaidu = toBaiduLang
