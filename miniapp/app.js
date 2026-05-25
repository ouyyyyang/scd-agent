// app.js
const store = require('./utils/store.js')

App({
  onLaunch() {
    // 首次启动播种演示数据到本地存储
    store.seedIfEmpty()
  },
  globalData: {},
})
