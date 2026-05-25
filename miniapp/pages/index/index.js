// pages/index/index.js — 首页：今日概览 + 指标仪表盘 + 短期建议 + 风险评估
const store = require('../../utils/store.js')
const { buildCards } = require('../../utils/metrics.js')
const { assessRisk } = require('../../utils/llm.js')

Page({
  data: {
    record: null,
    profile: {},
    hero: null,            // 概览大数字
    primaryCards: [],
    auxCards: [],
    alerts: [],
    // AI
    loading: true,
    error: '',
    result: null,
    assessedId: '',
  },

  onShow() {
    const record = store.getLatestRecord()
    const profile = store.getProfile()
    if (!record) {
      this.setData({ loading: false, error: '暂无体征数据，去「录入」添加一条吧。' })
      return
    }
    const cards = buildCards(record)
    const m = record.metrics || {}
    this.setData({
      record,
      profile,
      hero: { hr: m.静息心率 != null ? m.静息心率 : '--', date: record.date },
      primaryCards: cards.filter((c) => c.primary),
      auxCards: cards.filter((c) => !c.primary),
      alerts: record.异常事件 || [],
    })
    // 数据没变就不重复请求 AI
    if (record.id !== this.data.assessedId) this.analyze(record, profile)
  },

  analyze(record, profile) {
    this.setData({ loading: true, error: '', result: null })
    assessRisk(record, profile)
      .then((result) => {
        const toClass = (lvl) => {
          lvl = lvl || ''
          return lvl.indexOf('高') >= 0 ? 'high' : lvl.indexOf('中') >= 0 ? 'mid' : 'low'
        }
        result.levelClass = toClass(result.riskLevel)
        ;(result.factors || []).forEach((f) => { f.levelClass = toClass(f.level) })
        const care = result.careLevel || ''
        result.careClass = care.indexOf('就医') >= 0 ? 'urgent' : care.indexOf('咨询') >= 0 ? 'consult' : 'observe'
        this.setData({ loading: false, result, assessedId: record.id })
      })
      .catch((err) => this.setData({ loading: false, error: err.message || String(err) }))
  },

  onRetry() {
    this.analyze(this.data.record, this.data.profile)
  },

  goEntry() {
    wx.navigateTo({ url: '../entry/entry' })
  },
})
