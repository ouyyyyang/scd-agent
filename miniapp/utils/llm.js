// utils/llm.js — 调用大模型：风险评估 + 趋势分析 + AI 问答
const config = require('../config.js')

const RISK_PROMPT = `你是一名心血管内科医生助手。用户会提供个人健康档案和一份来自智能穿戴设备（手环/手表）的心脏监测数据，请综合评估其当前的心血管健康状况与风险。

要求：
1. 只输出一个 JSON 对象，不要输出任何解释文字、不要用 markdown 代码块包裹。
2. JSON 结构严格如下：
{
  "riskLevel": "低 | 中 | 高",
  "riskScorePercent": "5-8",
  "summary": "一句话总结整体情况",
  "careLevel": "可观察 | 建议咨询 | 尽快就医",
  "careNote": "一句话就医建议",
  "factors": [ { "name": "静息心率偏高", "level": "高", "impact": "持续偏高提示心脏负荷增大" } ],
  "shortTermTips": ["今天/本周可立即执行的 2-3 条具体建议"],
  "longTermTips": ["需长期坚持的健康管理目标 2-3 条"]
}
3. careLevel 根据风险与异常事件判断分级：指标基本正常→可观察；有明显异常或多项偏离→建议咨询；出现疑似房颤、严重心律失常、血氧过低等危险信号→尽快就医。
4. 结合年龄、性别、病史、用药与各项指标（心脏指标为主）。语言通俗，面向普通用户。数据缺失就忽略，不要编造。`

const TREND_PROMPT = `你是一名心血管内科医生助手。用户会提供个人档案和最近一段时间的每日心脏监测数据序列，请分析其变化趋势。

要求：
1. 只输出一个 JSON 对象，不要输出解释文字、不要用 markdown 代码块包裹。
2. JSON 结构严格如下：
{
  "trend": "向好 | 平稳 | 需警惕",
  "summary": "一两句话总结这段时间的整体变化趋势",
  "highlights": [ { "name": "静息心率", "direction": "上升 | 下降 | 平稳", "note": "简短说明" } ],
  "longTermTips": ["针对趋势的长期管理建议 2-3 条"]
}
3. 重点看心脏指标（静息心率、HRV、血压、血氧）的走向。语言通俗。`

const QA_PROMPT = `你是用户的 AI 心脏健康助手。请基于下面提供的「用户健康档案」和「最近监测数据」，用通俗、简洁的中文回答用户关于自身心脏健康的问题。
要求：回答简明（一般 3~6 句）；必要时提醒就医；不做确诊、不开具体处方；不要用 markdown 标记。`

/* ---------- 请求核心：返回模型输出的原始文本 ---------- */
function callRaw(messages) {
  return new Promise((resolve, reject) => {
    if (!config.apiKey || config.apiKey.indexOf('在这里填入') >= 0) {
      reject(new Error('还没配置 API Key，请打开 config.js 填写 apiKey'))
      return
    }
    wx.request({
      url: config.baseUrl,
      method: 'POST',
      timeout: 60000,
      header: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + config.apiKey },
      data: { model: config.model, temperature: 0.4, messages },
      success: (res) => {
        if (res.statusCode !== 200) {
          reject(new Error('接口返回 ' + res.statusCode + '：' + JSON.stringify(res.data)))
          return
        }
        try {
          resolve(res.data.choices[0].message.content)
        } catch (e) {
          reject(new Error('返回格式异常：' + e.message))
        }
      },
      fail: (err) => reject(new Error('请求失败：' + (err.errMsg || JSON.stringify(err)))),
    })
  })
}

// 期望 JSON 的调用
function callJSON(messages) {
  return callRaw(messages).then((content) => {
    let text = (content || '').trim()
    const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/)
    if (fence) text = fence[1].trim()
    const s = text.indexOf('{'); const e = text.lastIndexOf('}')
    if (s >= 0 && e > s) text = text.slice(s, e + 1)
    return JSON.parse(text)
  }).catch((err) => { throw new Error(err.message.indexOf('JSON') >= 0 ? '解析模型返回失败' : err.message) })
}

function profileLines(profile) {
  const p = profile || {}
  return Object.keys(p).map((k) => {
    let v = p[k]
    if (v === true) v = '是'; else if (v === false) v = '否'
    if (Array.isArray(v)) v = v.join('、') || '无'
    return `${k}：${v}`
  }).join('\n')
}

function metricLines(record) {
  const m = (record && record.metrics) || {}
  const lines = []
  Object.keys(m).forEach((k) => {
    let v = m[k]
    if (Array.isArray(v)) v = v.length ? v.join('、') : '无'
    if (v === '' || v == null) return
    lines.push(`${k}：${v}`)
  })
  if (record && record.异常事件 && record.异常事件.length) lines.push('异常事件：' + record.异常事件.join('、'))
  return lines.join('\n')
}

/* ---------- 单次风险评估 ---------- */
function assessRisk(record, profile) {
  const content = `【个人档案】\n${profileLines(profile)}\n\n【最近一次监测（${record.date}）】\n${metricLines(record)}`
  return callJSON([
    { role: 'system', content: RISK_PROMPT },
    { role: 'user', content: '请评估以下数据：\n' + content },
  ])
}

/* ---------- 长期趋势分析 ---------- */
function analyzeTrend(records, profile) {
  const pick = ['静息心率', '心率变异性HRV', '收缩压', '舒张压', '血氧饱和度', '睡眠时长']
  const header = '日期，' + pick.join('，')
  const rows = records.map((r) => {
    const m = r.metrics || {}
    return r.date + '，' + pick.map((k) => (m[k] != null ? m[k] : '')).join('，')
  })
  const content = `【个人档案】\n${profileLines(profile)}\n\n【最近 ${records.length} 天数据（CSV）】\n${header}\n${rows.join('\n')}`
  return callJSON([
    { role: 'system', content: TREND_PROMPT },
    { role: 'user', content: '请分析以下趋势：\n' + content },
  ])
}

/* ---------- AI 问答（多轮，注入数据上下文） ---------- */
function askDoctor(history, profile, record) {
  const ctx = `【用户健康档案】\n${profileLines(profile)}\n\n【最近监测（${(record && record.date) || '无'}）】\n${record ? metricLines(record) : '暂无数据'}`
  const messages = [{ role: 'system', content: QA_PROMPT + '\n\n' + ctx }].concat(history)
  return callRaw(messages)
}

module.exports = { assessRisk, analyzeTrend, askDoctor }
