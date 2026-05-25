// utils/chart.js — 用 canvas 2d 画一条简洁的趋势折线图（带渐变面积填充）
// 由页面取得 canvas node、ctx、宽高与 dpr 后调用。

function drawLineChart(ctx, opts) {
  const { width, height, data, color } = opts
  const pad = { top: 16, right: 14, bottom: 22, left: 14 }
  const W = width - pad.left - pad.right
  const H = height - pad.top - pad.bottom

  ctx.clearRect(0, 0, width, height)
  if (!data || data.length === 0) return

  const ys = data.map((d) => d.value)
  let min = Math.min.apply(null, ys)
  let max = Math.max.apply(null, ys)
  if (min === max) { min -= 1; max += 1 }
  const range = max - min
  min -= range * 0.15
  max += range * 0.15

  const n = data.length
  const x = (i) => pad.left + (n === 1 ? W / 2 : (W * i) / (n - 1))
  const y = (v) => pad.top + H - ((v - min) / (max - min)) * H

  // 面积渐变填充
  ctx.beginPath()
  ctx.moveTo(x(0), y(data[0].value))
  for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(data[i].value))
  ctx.lineTo(x(n - 1), pad.top + H)
  ctx.lineTo(x(0), pad.top + H)
  ctx.closePath()
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + H)
  grad.addColorStop(0, hexA(color, 0.28))
  grad.addColorStop(1, hexA(color, 0))
  ctx.fillStyle = grad
  ctx.fill()

  // 折线
  ctx.beginPath()
  ctx.moveTo(x(0), y(data[0].value))
  for (let i = 1; i < n; i++) ctx.lineTo(x(i), y(data[i].value))
  ctx.lineWidth = 3
  ctx.strokeStyle = color
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.stroke()

  // 末点高亮
  const lx = x(n - 1)
  const ly = y(data[n - 1].value)
  ctx.beginPath()
  ctx.arc(lx, ly, 6, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  ctx.beginPath()
  ctx.arc(lx, ly, 11, 0, Math.PI * 2)
  ctx.fillStyle = hexA(color, 0.18)
  ctx.fill()

  // 首末日期标签
  ctx.fillStyle = '#9aa0ab'
  ctx.font = '20px sans-serif'
  ctx.textBaseline = 'bottom'
  ctx.textAlign = 'left'
  ctx.fillText(shortDate(data[0].label), pad.left, height - 2)
  ctx.textAlign = 'right'
  ctx.fillText(shortDate(data[n - 1].label), width - pad.right, height - 2)
}

// '#rrggbb' + alpha → rgba()
function hexA(hex, a) {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${a})`
}

// '2026-05-22' → '5/22'
function shortDate(s) {
  const p = String(s || '').split('-')
  return p.length === 3 ? `${+p[1]}/${+p[2]}` : s
}

module.exports = { drawLineChart }
