// pages/contacts/contacts.js — 紧急联系人：列表 / 新增 / 拨号 / 删除 + 一键 120
// storage 里联系人用中文 key，映射成英文 key 的视图模型给 WXML（中文 key 在 WXML .访问会报错）。
const store = require('../../utils/store.js')

Page({
  data: {
    contacts: [],
    showForm: false,
    form: { name: '', relation: '', phone: '' },
  },

  onShow() { this.refresh() },
  refresh() {
    this.setData({
      contacts: store.getContacts().map((c) => ({
        id: c.id,
        name: c.姓名,
        relation: c.关系,
        phone: c.电话,
        isDefault: !!c.是否默认,
        initial: (c.姓名 || '?').charAt(0),
      })),
    })
  },

  onInput(e) { this.setData({ ['form.' + e.currentTarget.dataset.field]: e.detail.value }) },

  openForm() { this.setData({ showForm: true, form: { name: '', relation: '', phone: '' } }) },
  closeForm() { this.setData({ showForm: false }) },
  noop() {},

  add() {
    const f = this.data.form
    if (!f.name || !f.phone) { wx.showToast({ title: '请填写姓名和电话', icon: 'none' }); return }
    store.upsertContact({
      id: 'c' + Date.now(),
      姓名: f.name, 关系: f.relation || '联系人', 电话: f.phone,
      是否默认: store.getContacts().length === 0,
    })
    this.setData({ showForm: false })
    this.refresh()
  },

  call(e) { wx.makePhoneCall({ phoneNumber: e.currentTarget.dataset.phone }) },
  call120() { wx.makePhoneCall({ phoneNumber: '120' }) },

  setDefault(e) {
    const id = e.currentTarget.dataset.id
    store.saveContacts(store.getContacts().map((c) => ({ ...c, 是否默认: c.id === id })))
    this.refresh()
  },

  remove(e) {
    const id = e.currentTarget.dataset.id
    wx.showModal({
      title: '删除联系人', content: '确定删除该紧急联系人？',
      success: (res) => { if (res.confirm) { store.removeContact(id); this.refresh() } },
    })
  },
})
