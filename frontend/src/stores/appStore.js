import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useAppStore = defineStore('app', () => {
  const theme = ref(localStorage.getItem('app-theme') || 'system')
  const language = ref(localStorage.getItem('app-lang') || 'zh')
  const autoForward = ref(localStorage.getItem('app-auto-forward') === 'true')

  const applyTheme = (val) => {
    const isDark = val === 'dark' || (val === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  // Initial application
  applyTheme(theme.value)

  // Watch for system changes if mode is system
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
  const handleSystemChange = (e) => {
    if (theme.value === 'system') {
      applyTheme('system')
    }
  }
  mediaQuery.addEventListener('change', handleSystemChange)

  watch(theme, (val) => {
    localStorage.setItem('app-theme', val)
    applyTheme(val)
  })

  watch(language, (val) => {
    localStorage.setItem('app-lang', val)
    // Here we might reload or trigger i18n update
  })

  watch(autoForward, (val) => {
    localStorage.setItem('app-auto-forward', val)
  })

  return {
    theme,
    language,
    autoForward
  }
})
