// pages/qa/qa.js — AI 问答：基于用户档案与最近数据的多轮对话
const store = require('../../utils/store.js')
const { askDoctor } = require('../../utils/llm.js')

const SUGGESTIONS = ['我最近心率为什么偏高？', '我的血压正常吗？', '怎样改善我的睡眠？', '我需要去医院吗？']

Page({
  data: {
    messages: [{ role: 'assistant', text: '你好，我是你的 AI 心脏健康助手。可以问我关于你心率、血压、睡眠等数据的问题。' }],
    suggestions: SUGGESTIONS,
    input: '',
    loading: false,
    toView: '',
  },

  onShow() {
    this.profile = store.getProfile()
    this.record = store.getLatestRecord()
  },

  onInput(e) { this.setData({ input: e.detail.value }) },

  quick(e) { this.submit(e.currentTarget.dataset.q) },
  onSend() { this.submit(this.data.input) },

  submit(text) {
    text = (text || '').trim()
    if (!text || this.data.loading) return
    const messages = this.data.messages.concat([{ role: 'user', text }])
    this.setData({ messages, input: '', loading: true, suggestions: [] })
    this.scrollToEnd()

    const history = messages.map((m) => ({ role: m.role, content: m.text })).slice(-12)
    askDoctor(history, this.profile, this.record)
      .then((reply) => {
        this.setData({ messages: this.data.messages.concat([{ role: 'assistant', text: (reply || '').trim() }]), loading: false })
        this.scrollToEnd()
      })
      .catch((err) => {
        this.setData({ messages: this.data.messages.concat([{ role: 'assistant', text: '⚠ ' + (err.message || '请求失败') }]), loading: false })
        this.scrollToEnd()
      })
  },

  scrollToEnd() {
    this.setData({ toView: 'm' + (this.data.messages.length - 1) })
  },
})
