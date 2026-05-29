// config.js — 大模型 API 配置（本地课堂作业用）
//
// ⚠️ 注意：把 apiKey 写在前端只适合本地测试 / 课堂作业。
//    正式上线时 key 会被任何人抓包看到，必须改成由你自己的后端中转调用。
//
// 下面默认填的是魔搭社区 ModelScope（OpenAI 兼容格式）。换成其它厂商只需改这三项：
//   · 魔搭社区：  https://api-inference.modelscope.cn/v1   model: deepseek-ai/DeepSeek-V4-Pro
//   · 智谱 GLM：  https://open.bigmodel.cn/api/paas/v4/chat/completions   model: glm-4-flash
//   · 通义千问：  https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions   model: qwen-plus
//   · Kimi：      https://api.moonshot.cn/v1/chat/completions   model: moonshot-v1-8k
//   · OpenAI：    https://api.openai.com/v1/chat/completions   model: gpt-4o-mini
module.exports = {
  baseUrl: 'https://api-inference.modelscope.cn/v1',
  apiKey: 'ms-d4013857-c6b2-4cbf-a0c2-83e60b8a4cae',
  model: 'deepseek-ai/DeepSeek-V4-Pro',
}
