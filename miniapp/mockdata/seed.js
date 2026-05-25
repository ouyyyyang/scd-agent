// mockdata/seed.js — 首次启动时播种到本地存储的初始数据
// 模拟「用户本人」的个人档案 + 最近 14 天的穿戴设备记录 + 一个示例紧急联系人。
// 接入真实设备后，把设备/手动数据写进 store 即可，本文件仅作初始演示数据。

// 个人档案（作为 AI 评估上下文）
const profile = {
  姓名: '我',
  性别: '男',
  年龄: 46,
  身高: 172,
  体重: 74,
  高血压: true,
  糖尿病: false,
  吸烟: false,
  家族史: true,
  用药: ['氨氯地平 5mg/日'],
}

// 生成最近 14 天的记录（带轻微波动，便于看趋势）
function genRecords() {
  const days = 14
  const today = new Date('2026-05-22T08:00:00')
  const list = []
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today.getTime() - i * 86400000)
    const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    const wave = Math.sin(i / 2)
    const restingHr = Math.round(80 + wave * 6 + (i < 3 ? 4 : 0))
    const hrv = Math.round(40 - wave * 8 - (i < 3 ? 6 : 0))
    const systolic = Math.round(134 + wave * 8 + (i < 2 ? 6 : 0))
    const diastolic = Math.round(86 + wave * 4)
    const spo2 = Math.round(97 - (i < 2 ? 2 : 0))
    const sleep = +(6.4 + wave * 0.6).toFixed(1)
    const steps = Math.round(4500 + wave * 1500)
    const alerts = []
    if (i === 1) alerts.push('夜间心率偏高')
    if (i === 0) alerts.push('检测到 1 次疑似房颤')
    list.push({
      id: 'rec-' + date,
      来源: '设备',
      device: '智能手表',
      date,
      time: '08:00',
      metrics: {
        静息心率: restingHr,
        平均心率: restingHr + 10,
        最高心率: 150 + Math.round(wave * 10),
        心率变异性HRV: hrv,
        血氧饱和度: spo2,
        收缩压: systolic,
        舒张压: diastolic,
        脉搏: restingHr,
        体温: 36.5,
        睡眠时长: sleep,
        深睡比例: 14 + Math.round(wave * 3),
        日步数: steps,
        久坐时长: 9 - Math.round(wave),
      },
      异常事件: alerts,
    })
  }
  return list
}

function pad(n) {
  return n < 10 ? '0' + n : '' + n
}

// 示例紧急联系人（用户可在「我的 → 紧急联系人」里增删改）
const contacts = [
  { id: 'c1', 姓名: '家人', 关系: '配偶', 电话: '13800000000', 是否默认: true },
]

module.exports = {
  profile,
  records: genRecords(),
  contacts,
}
