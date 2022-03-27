<template>
  <div class="flex flex-col gap-4 h-screen">
  <Frame/>
  <router-view />
  </div>
</template>

<script>
import Frame from './components/Frame.vue'
export default {
  name: 'App',
  components: {
    Frame: Frame,
  },
  data () {
    return {
      drawer: null,
    }
  },
  watch: {
    drawer: {
      immediate: true,
      handler(drawer) {
        if (process.client) {
          if (drawer) document.body.style.setProperty("overflow", "hidden");
          else document.body.style.removeProperty("overflow");
        }
      }
    }
  }
}
</script>
