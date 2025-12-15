import { createRouter, createWebHistory } from 'vue-router'
import LoginPage from '../views/LoginPage.vue'
import MainPage from '@/views/content/MainPage.vue'
import ChatPage from '@/views/content/chat/ChatPage.vue'
import ModelsPage from '@/views/content/models/ModelsPage.vue'
import SettingsPage from '@/views/settings/index.vue'
import ProfilePage from '@/views/profile/index.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: LoginPage
    },
    {
      path: '/login',
      redirect: '/'
    },
    {
      path: '/main',
      redirect: '/chat'
    },
    {
      path: '/chat',
      component: MainPage,
      children: [
        {
          path: '',
          name: 'chat',
          component: ChatPage
        }
      ]
    },
    {
      path: '/models',
      component: MainPage,
      children: [
        {
          path: '',
          name: 'models',
          component: ModelsPage
        }
      ]
    },
    {
      path: '/settings',
      component: MainPage,
      children: [
        {
          path: '',
          name: 'settings',
          component: SettingsPage
        }
      ]
    },
    {
      path: '/profile',
      component: MainPage,
      children: [
        {
          path: '',
          name: 'profile',
          component: ProfilePage
        }
      ]
    }
  ]
})

export default router
