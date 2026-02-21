# Grok AI 助手

Grok 全能插件：文生图、图生图、图生视频、智能对话（自动联网）、LLM Tool 调用，一站式 AI 多媒体体验。

## 功能

| 功能 | 命令 | 说明 |
|------|------|------|
| 文生图 | `/grok生图 [数量] [尺寸] 提示词` | 根据文字描述生成图片 |
| 图生图 | `/grok生图 提示词 + 图片` | 基于参考图片进行编辑/重绘 |
| 生视频 | `/grok视频 [尺寸] [时长] 提示词 [+图片可选]` | 支持文生视频与图生视频 |
| 智能对话 | `/grok 内容 [+图片/语音/文件可选]` | 与 Grok 对话，自动判断是否需要联网 |
| 帮助 | `/grok帮助` | 查看使用说明 |

## 配置说明

### API 配置

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `grok_api_url` | API 基础地址 | `https://api.x.ai` |
| `grok_api_key` | API 密钥 | 你的 xAI API Key |

**URL 配置说明**：只需填写基础 URL，插件会自动拼接正确的接口路径。

支持的 URL 格式（以下均可正常工作）：
- `https://api.x.ai`
- `https://api.x.ai/v1`
- `https://api.x.ai/v1/chat/completions`

### 模型配置

| 配置项 | 功能 | 默认值 | 接口 |
|--------|------|--------|------|
| `grok_image_model` | 文生图 | `grok-imagine-1.0` | `/v1/images/generations` |
| `grok_edit_model` | 图生图 | `grok-imagine-1.0-edit` | `/v1/images/edits` |
| `grok_video_model` | 生视频 | `grok-imagine-1.0-video` | `/v1/chat/completions` |
| `grok_search_model` | 对话/搜索 | `grok-4-fast` | `/v1/chat/completions` |

**模型说明**：所有模型均通过配置项读取，代码中的默认值仅作为备用。你可以根据 API 提供商支持的模型自行修改。

常见可用模型（参考 [grok2api](https://github.com/chenyme/grok2api)）：

| 模型 | 类型 | 说明 |
|------|------|------|
| `grok-imagine-1.0` | 图像生成 | 标准图像生成 |
| `grok-imagine-1.0-edit` | 图像编辑 | 基于参考图编辑 |
| `grok-imagine-1.0-video` | 视频生成 | 图片转视频 |
| `grok-3` / `grok-4` / `grok-4-fast` | 对话+搜索 | 支持对话和联网搜索 |

### 对话/搜索配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `grok_search_model` | 对话/搜索模型 | `grok-4-fast` |
| `grok_search_mode` | 搜索模式 | `auto` |
| `grok_search_enable_thinking` | 开启思考模式 | `true` |
| `grok_search_thinking_budget` | 思考 token 预算 | `32000` |
| `grok_search_timeout_seconds` | 请求超时时间 | `60` |
| `grok_search_show_sources` | 显示来源链接 | `false` |
| `grok_search_max_sources` | 最多显示来源数 | `5` |
| `grok_search_extra_body` | 额外请求体 (JSON) | `{}` |
| `grok_search_extra_headers` | 额外请求头 (JSON) | `{}` |
| `grok_search_enable_skill` | 启用 Skill 模式 | `false` |

**搜索模式说明**：
- `auto`（默认）：模型自动判断是否需要联网搜索
- `on`：始终使用联网搜索
- `off`：纯对话模式，不联网

### 其他配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `save_media` | 是否保存生成的媒体文件 | `false` |
| `user_whitelist` | 用户白名单（空=不限制） | `[]` |
| `user_blacklist` | 用户黑名单 | `[]` |
| `group_whitelist` | 群聊白名单（空=不限制） | `[]` |
| `group_blacklist` | 群聊黑名单 | `[]` |

## 使用示例

### 文生图

```
/grok生图 一只可爱的猫咪
/grok生图 4 1792x1024 日落海滩风景
/grok生图 1024x1024 赛博朋克城市夜景
/grok生图 720x1280 一只猫
```

参数说明：
- 数量：1-10（默认 1）
- 尺寸：`1024x1024` / `1024x1792` / `1280x720` / `1792x1024` / `720x1280`
- 不加尺寸参数时，默认使用：`1024x1792`
- 参数顺序任意，如 `4 1792x1024` 或 `1792x1024 4` 均可

### 图生图

发送图片或引用图片，附带命令：
```
/grok生图 把背景换成森林
/grok生图 转换为油画风格
/grok生图 4 添加下雪效果
```

说明：
- 自动读取原图分辨率，并映射到最近合法尺寸
- 支持数量参数
- 目标尺寸始终根据原图自动匹配，忽略手动尺寸
- 附带两张图片时，第 2 张会作为局部重绘蒙版

### 生视频（文生/图生）

可直接发文字，或发送图片/引用图片后附带命令：
```
/grok视频 让画面动起来
/grok视频 10 夜晚海边的慢镜头
/grok视频 1280x720 夜晚海边的慢镜头
/grok视频 让人物眨眼微笑
/grok视频 添加飘落的樱花
```

说明：
- 文生视频默认尺寸：`1280x720`
- 文生视频默认时长：`6` 秒（可选 `10` / `15`）
- 图生视频自动读取原图分辨率，并匹配最近合法尺寸
- 固定 `720p` 输出（脚本内固定）
- 自动启用增强策略（高细节、低噪点、时序稳定）

### 智能对话

```
/grok 你好，介绍一下你自己
/grok 帮我写一首关于春天的诗
/grok 今天的新闻有哪些（自动联网）
/grok 最新的 AI 技术进展（自动联网）
/grok 帮我总结这段语音内容（附语音）
/grok 请提炼这个文件的重点（附文件）
```

说明：
- 默认 `auto` 模式：模型自动判断是否需要联网
- 普通问题直接回答，时效性问题自动联网搜索
- 支持图片、语音、文件的多模态理解
- 可通过配置切换为始终联网或纯对话模式
- 支持作为 LLM Tool 被其他插件调用

## API 接口对照

本插件参考 [grok2api](https://github.com/chenyme/grok2api) 项目的接口规范：

| 功能 | 接口路径 | 请求格式 |
|------|----------|----------|
| 文生图 | `POST /v1/images/generations` | JSON |
| 图生图 | `POST /v1/images/edits` | multipart/form-data |
| 图生视频 | `POST /v1/chat/completions` | JSON (stream) |
| 对话/搜索 | `POST /v1/chat/completions` | JSON |

## 注意事项

1. **API 兼容性**：本插件兼容 xAI 官方 API 及 grok2api 等第三方代理服务
2. **模型名称**：不同 API 提供商支持的模型可能不同，请根据实际情况配置
3. **图片格式**：支持 PNG、JPG、WEBP、GIF、BMP 格式
4. **文生图合法尺寸**：`1024x1024`、`1024x1792`、`1280x720`、`1792x1024`、`720x1280`
5. **超时设置**：图片生成 120 秒，视频生成 300 秒，对话/搜索默认 60 秒
6. **文件保存**：开启 `save_media` 后，文件保存在插件数据目录的 `images/` 和 `videos/` 子目录
7. **LLM Tool**：对话/搜索功能可作为 LLM Tool 被其他插件或 Agent 调用
8. **Skill 模式**：开启后禁用 LLM Tool，改为通过 Skill 钩子响应
9. **搜索模式**：`auto` 模式下模型会自动判断是否需要联网，无需手动切换
10. **模型容错**：插件会自动探测可用模型，当前配置模型不可用时自动回退
