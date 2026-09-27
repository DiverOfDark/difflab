import { createRouter, createWebHashHistory } from 'vue-router'
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
    { path: '/', component: Production },
    { path: '/productions/:id', component: Production, props: (route) => ({ id: Number(route.params.id) }) },
    { path: '/quick', component: Quick },
    { path: '/runs', component: Runs },
    { path: '/runs/:id', component: RunView, props: true },
    { path: '/settings', component: Settings },
    { path: '/:rest(.*)', redirect: '/' },
  ],
})

const TITLES: Record<string, string> = {
  '/': 'Productions',
  '/quick': 'Quick',
  '/runs': 'Runs',
  '/settings': 'Settings',
}

router.afterEach((to) => {
  const name = TITLES[to.path] ?? (to.path.startsWith('/runs/') ? String(to.params.id) : 'Production')
  document.title = `${name} — DiffLab`
})
