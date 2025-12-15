import './assets/main.css'

import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import VueAudio from 'vue-audio-better'
import { createPinia } from 'pinia'

const app = createApp(App)
const pinia = createPinia()

// 全局注册 Element Plus 图标组件
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(VueAudio)
app.use(ElementPlus)
app.use(router)
app.use(pinia)
app.mount('#app')
