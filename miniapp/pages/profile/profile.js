// pages/profile/profile.js — 我的：健康档案 + 紧急求助
// 注意：WXML 绑定的数据 key 必须是英文（中文 key 在 WXML 里 .访问会编译报错）。
// 这里 form 用英文 key，存到 storage 时再映射回中文（便于 AI 提示词阅读）。
const store = require('../../utils/store.js')

const sexOptions = ['男', '女']

Page({
  data: {
    sexOptions,
    form: { sexIndex: 0, age: '', height: '', weight: '', hypertension: false, diabetes: false, smoking: false, familyHistory: false, meds: '' },
    bmi: '',
    defaultContact: null,
    saved: false,
  },

  onShow() {
    const p = store.getProfile()
    this.setData({
      form: {
        sexIndex: Math.max(0, sexOptions.indexOf(p.性别)),
        age: p.年龄 || '',
        height: p.身高 || '',
        weight: p.体重 || '',
        hypertension: !!p.高血压,
        diabetes: !!p.糖尿病,
        smoking: !!p.吸烟,
        familyHistory: !!p.家族史,
        meds: (p.用药 || []).join('、'),
      },
      defaultContact: this.pickDefault(),
    })
    this.calcBmi()
  },

  pickDefault() {
    const list = store.getContacts()
    const c = list.find((x) => x.是否默认) || list[0]
    return c ? { name: c.姓名, relation: c.关系, phone: c.电话 } : null
  },

  onInput(e) { this.setData({ ['form.' + e.currentTarget.dataset.field]: e.detail.value }, () => this.calcBmi()) },
  onSwitch(e) { this.setData({ ['form.' + e.currentTarget.dataset.field]: e.detail.value }) },
  onPicker(e) { this.setData({ ['form.' + e.currentTarget.dataset.field]: Number(e.detail.value) }) },

  calcBmi() {
    const { height, weight } = this.data.form
    const h = Number(height) / 100, w = Number(weight)
    this.setData({ bmi: h > 0 && w > 0 ? (w / (h * h)).toFixed(1) : '' })
  },

  save() {
    const f = this.data.form
    store.saveProfile({
      姓名: '我',
      性别: sexOptions[f.sexIndex],
      年龄: f.age,
      身高: f.height,
      体重: f.weight,
      高血压: f.hypertension,
      糖尿病: f.diabetes,
      吸烟: f.smoking,
      家族史: f.familyHistory,
      用药: f.meds ? f.meds.split(/[、,，]/).map((s) => s.trim()).filter(Boolean) : [],
    })
    wx.showToast({ title: '已保存', icon: 'success' })
  },

  call120() { wx.makePhoneCall({ phoneNumber: '120' }) },
  callDefault() { if (this.data.defaultContact) wx.makePhoneCall({ phoneNumber: this.data.defaultContact.phone }) },
  goContacts() { wx.navigateTo({ url: '../contacts/contacts' }) },
  goEntry() { wx.navigateTo({ url: '../entry/entry' }) },
})
