import Landing from '../components/Landing.vue'
import Home from '../components/Home.vue'
import Build from '../components/build/Build.vue'
import Authenticate from '../components/Authenticate.vue'
import Train from '../components/Train.vue'
import { createRouter, createWebHashHistory } from 'vue-router'
// import { RouteRecordRaw, RouteLocationNormalized, NavigationGuardNext } from 'vue-router';
import { auth } from '../main'

const guard = (to, from, next) => {
  try {
    auth.onAuthStateChanged((user) => {
      if (user) {
      next();
      }
      else {
      console.log('no user registered');
      next("/auth");
      }
    })
  } catch (error) {
    next("/auth")
  }
}


const routes = [
  { path: '/', component: Landing },
  { path: '/about', component: Home },
  { path: '/contact', component: Home },
  { path: '/auth', component: Authenticate },
  { path: '/home', 
    component: Home 
  },
  { path: '/build', 
    component: Build,
    beforeEnter: guard
  },
  { path: '/analyze', 
    component: Train,
    beforeEnter: guard
  },

]

const router = createRouter({
  // 4. Provide the history implementation to use. We are using the hash history for simplicity here.
  history: createWebHashHistory(),
  routes, // short for `routes: routes`
})

export default router;
