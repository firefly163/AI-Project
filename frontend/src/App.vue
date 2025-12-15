<script setup lang="ts">
import { RouterView } from 'vue-router'
import { useAppStore } from '@/stores/appStore'

// Initialize app store to apply theme settings
useAppStore()
</script>

<template>
  <RouterView />
</template>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
}

#app {
  padding: 0;
  height: 100%;
  width: 100%;
}
</style>
