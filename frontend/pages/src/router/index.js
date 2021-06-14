import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../views/Home.vue'

Vue.use(VueRouter)

const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/operate',
        name: 'Operate',
        component: () => import(/* webpackChunkName: "about" */ '../views/Operate.vue')
    },
    {
        path: '*',
        name: 'Error',
        component: () => import('../views/Error.vue')
    },
];

const router = new VueRouter({
    routes
})

export default router
