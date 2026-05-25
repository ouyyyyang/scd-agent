// utils/store.js — 本地存储层（wx.storage）。档案 / 体征记录 / 紧急联系人的读写。
const seed = require('../mockdata/seed.js')

const KEYS = {
  profile: 'profile',
  records: 'records',
  contacts: 'contacts',
  seeded: 'seeded_v1',
}

// 首次启动播种初始数据（仅一次）
function seedIfEmpty() {
  if (wx.getStorageSync(KEYS.seeded)) return
  wx.setStorageSync(KEYS.profile, seed.profile)
  wx.setStorageSync(KEYS.records, seed.records)
  wx.setStorageSync(KEYS.contacts, seed.contacts)
  wx.setStorageSync(KEYS.seeded, true)
}

/* ---------- 个人档案 ---------- */
function getProfile() {
  return wx.getStorageSync(KEYS.profile) || {}
}
function saveProfile(profile) {
  wx.setStorageSync(KEYS.profile, profile)
}

/* ---------- 体征记录（按日期升序存储） ---------- */
function getRecords() {
  return wx.getStorageSync(KEYS.records) || []
}
// 最新一条
function getLatestRecord() {
  const list = getRecords()
  return list.length ? list[list.length - 1] : null
}
// 新增一条（手动录入或设备同步），按 date+time 维持有序
function addRecord(record) {
  const list = getRecords()
  list.push(record)
  list.sort((a, b) => (a.date + (a.time || '')).localeCompare(b.date + (b.time || '')))
  wx.setStorageSync(KEYS.records, list)
  return record
}

/* ---------- 紧急联系人 ---------- */
function getContacts() {
  return wx.getStorageSync(KEYS.contacts) || []
}
function saveContacts(contacts) {
  wx.setStorageSync(KEYS.contacts, contacts)
}
function upsertContact(contact) {
  const list = getContacts()
  const i = list.findIndex((c) => c.id === contact.id)
  if (i >= 0) list[i] = contact
  else list.push(contact)
  saveContacts(list)
}
function removeContact(id) {
  saveContacts(getContacts().filter((c) => c.id !== id))
}

module.exports = {
  seedIfEmpty,
  getProfile, saveProfile,
  getRecords, getLatestRecord, addRecord,
  getContacts, saveContacts, upsertContact, removeContact,
}
