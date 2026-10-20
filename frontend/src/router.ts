import { createRouter, createWebHashHistory } from 'vue-router'
import Chat from './screens/Chat.vue'
import Gallery from './screens/Gallery.vue'
import Production from './screens/Production.vue'
import Quick from './screens/Quick.vue'
import RunView from './screens/RunView.vue'
import Runs from './screens/Runs.vue'
import Settings from './screens/Settings.vue'

/** Hash routes: the page is one static file behind an API, and a hash survives a reload without a
 *  server that knows the paths. */
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/quick' },
    { path: '/productions', component: Production },
    { path: '/productions/:id', component: Production, props: (route) => ({ id: Number(route.params.id) }) },
    { path: '/quick', component: Quick },
    { path: '/chat', component: Chat },
    { path: '/chat/:id', component: Chat, props: (route) => ({ id: Number(route.params.id) }) },
    { path: '/gallery', component: Gallery },
    { path: '/runs', component: Runs },
    { path: '/runs/:id', component: RunView, props: true },
    { path: '/settings', component: Settings },
    { path: '/:rest(.*)', redirect: '/' },
  ],
})

const TITLES: Record<string, string> = {
  '/productions': 'Productions',
  '/quick': 'Quick',
  '/gallery': 'Gallery',
  '/runs': 'Runs',
  '/chat': 'Chat',
  '/settings': 'Settings',
}

router.afterEach((to) => {
  const name = TITLES[to.path] ?? (to.path.startsWith('/runs/') ? String(to.params.id) : to.path.startsWith('/chat') ? 'Chat' : 'Production')
  document.title = `${name} — DiffLab`
})
