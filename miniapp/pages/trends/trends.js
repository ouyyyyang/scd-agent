// pages/trends/trends.js — 趋势图 + 历史列表 + 长期 AI 分析
const store = require('../../utils/store.js')
const { drawLineChart } = require('../../utils/chart.js')
const { analyzeTrend } = require('../../utils/llm.js')

const METRICS = [
  { name: '静息心率', key: '静息心率', unit: 'bpm', accent: '#ff2d55' },
  { name: 'HRV', key: '心率变异性HRV', unit: 'ms', accent: '#5e5ce6' },
  { name: '收缩压', key: '收缩压', unit: 'mmHg', accent: '#ff375f' },
  { name: '血氧', key: '血氧饱和度', unit: '%', accent: '#30d0c6' },
  { name: '睡眠', key: '睡眠时长', unit: 'h', accent: '#7d5fff' },
]

Page({
  data: {
    metrics: METRICS.map((m) => m.name),
    active: 0,
    stat: null,          // 当前指标的 max/min/avg
    history: [],
    // AI 趋势
    aiLoading: false,
    aiError: '',
    trend: null,
  },

  onShow() {
    const records = store.getRecords()
    this.records = records
    this.profile = store.getProfile()
    this.setData({
      history: records.slice().reverse().map((r) => ({
        date: r.date,
        hr: (r.metrics && r.metrics.静息心率) || '--',
        bp: `${(r.metrics && r.metrics.收缩压) || '--'}/${(r.metrics && r.metrics.舒张压) || '--'}`,
        alerts: (r.异常事件 && r.异常事件.length) || 0,
      })),
    })
    this.refreshChart()
  },

  onReady() { this.ready = true; this.refreshChart() },

  switchMetric(e) {
    this.setData({ active: Number(e.currentTarget.dataset.i) }, () => this.refreshChart())
  },

  refreshChart() {
    const m = METRICS[this.data.active]
    const series = (this.records || [])
      .map((r) => ({ label: r.date, value: r.metrics ? r.metrics[m.key] : null }))
      .filter((d) => d.value != null)
    if (series.length) {
      const vals = series.map((d) => d.value)
      this.setData({
        stat: {
          max: Math.max.apply(null, vals),
          min: Math.min.apply(null, vals),
          avg: Math.round(vals.reduce((a, b) => a + b, 0) / vals.length * 10) / 10,
          unit: m.unit,
        },
      })
    } else {
      this.setData({ stat: null })
    }
    this.drawTo(series, m.accent)
  },

  drawTo(series, color) {
    wx.createSelectorQuery().in(this).select('#chart')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res || !res[0]) return
        const canvas = res[0].node
        const dpr = (wx.getSystemInfoSync().pixelRatio) || 2
        canvas.width = res[0].width * dpr
        canvas.height = res[0].height * dpr
        const ctx = canvas.getContext('2d')
        ctx.scale(dpr, dpr)
        drawLineChart(ctx, { width: res[0].width, height: res[0].height, data: series, color })
      })
  },

  genTrend() {
    if (!this.records || this.records.length < 2) {
      wx.showToast({ title: '数据太少，无法分析趋势', icon: 'none' })
      return
    }
    this.setData({ aiLoading: true, aiError: '', trend: null })
    analyzeTrend(this.records, this.profile)
      .then((trend) => {
        const t = trend.trend || ''
        trend.trendClass = t.indexOf('警') >= 0 ? 'bad' : t.indexOf('好') >= 0 ? 'good' : 'mid'
        ;(trend.highlights || []).forEach((h) => {
          const d = h.direction || ''
          h.dirClass = d.indexOf('升') >= 0 ? 'up' : d.indexOf('降') >= 0 ? 'down' : 'flat'
        })
        this.setData({ aiLoading: false, trend })
      })
      .catch((err) => this.setData({ aiLoading: false, aiError: err.message || String(err) }))
  },
})
