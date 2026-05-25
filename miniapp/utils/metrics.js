// utils/metrics.js — 把一条体征记录整理成「Apple Watch 风格」指标卡片
// 心脏相关指标为主（primary，靠前），睡眠/步数等为辅助（次要，靠后）。
// 每张卡：{ key, icon, name, value, unit, status: good|warn|bad, accent, primary }

function band(v, goodMin, goodMax, warnMin, warnMax) {
  if (v == null || v === '') return 'good'
  if (v >= goodMin && v <= goodMax) return 'good'
  if (v >= warnMin && v <= warnMax) return 'warn'
  return 'bad'
}

function buildCards(record) {
  const m = (record && record.metrics) || {}
  const all = [
    // —— 心脏指标（主）——
    { key: '静息心率', icon: '❤', name: '静息心率', value: m.静息心率, unit: 'bpm', accent: '#ff2d55', primary: true,
      status: band(m.静息心率, 50, 70, 45, 85) },
    { key: 'HRV', icon: '∿', name: '心率变异 HRV', value: m.心率变异性HRV, unit: 'ms', accent: '#5e5ce6', primary: true,
      status: band(m.心率变异性HRV, 50, 200, 30, 49) },
    { key: '收缩压', icon: '⤴', name: '收缩压', value: m.收缩压, unit: 'mmHg', accent: '#ff375f', primary: true,
      status: band(m.收缩压, 90, 130, 131, 139) },
    { key: '舒张压', icon: '⤵', name: '舒张压', value: m.舒张压, unit: 'mmHg', accent: '#ff6482', primary: true,
      status: band(m.舒张压, 60, 85, 86, 89) },
    { key: '血氧', icon: '◎', name: '血氧', value: m.血氧饱和度, unit: '%', accent: '#30d0c6', primary: true,
      status: band(m.血氧饱和度, 95, 100, 90, 94) },
    { key: '脉搏', icon: '✦', name: '脉搏', value: m.脉搏, unit: 'bpm', accent: '#ff9f0a', primary: true,
      status: band(m.脉搏, 50, 90, 91, 100) },
    // —— 辅助指标 ——
    { key: '睡眠', icon: '☾', name: '睡眠时长', value: m.睡眠时长, unit: 'h', accent: '#7d5fff', primary: false,
      status: band(m.睡眠时长, 7, 10, 6, 6.9) },
    { key: '步数', icon: '➜', name: '今日步数', value: m.日步数, unit: '步', accent: '#34c759', primary: false,
      status: band(m.日步数, 8000, 100000, 4000, 7999) },
  ]
  return all.filter((c) => c.value != null && c.value !== '')
}

module.exports = { buildCards, band }
