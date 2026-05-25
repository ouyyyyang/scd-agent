// pages/entry/entry.js — 录入体征数据（手动）
// WXML 绑定用英文 key，保存时映射回中文 metrics key（与 seed/store 一致，AI 提示词友好）。
const store = require('../../utils/store.js')

function today() {
  const d = new Date()
  const p = (n) => (n < 10 ? '0' + n : '' + n)
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

// 英文字段 → 中文 metrics key
const FIELD_MAP = {
  restingHr: '静息心率',
  systolic: '收缩压',
  diastolic: '舒张压',
  spo2: '血氧饱和度',
  hrv: '心率变异性HRV',
  pulse: '脉搏',
  sleep: '睡眠时长',
  steps: '日步数',
}

Page({
  data: {
    date: today(),
    f: { restingHr: '', systolic: '', diastolic: '', spo2: '', hrv: '', pulse: '', sleep: '', steps: '' },
  },

  onDate(e) { this.setData({ date: e.detail.value }) },
  onInput(e) { this.setData({ ['f.' + e.currentTarget.dataset.field]: e.detail.value }) },

  save() {
    const f = this.data.f
    const metrics = {}
    Object.keys(f).forEach((k) => { if (f[k] !== '') metrics[FIELD_MAP[k]] = Number(f[k]) })
    if (!Object.keys(metrics).length) {
      wx.showToast({ title: '请至少填写一项指标', icon: 'none' })
      return
    }
    store.addRecord({
      id: 'rec-' + this.data.date + '-' + Date.now(),
      来源: '手动',
      device: '手动录入',
      date: this.data.date,
      time: new Date().toTimeString().slice(0, 5),
      metrics,
      异常事件: [],
    })
    wx.showToast({ title: '已保存', icon: 'success' })
    setTimeout(() => wx.navigateBack(), 600)
  },
})
