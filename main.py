import asyncio
import base64
import io
import json
import time
import re
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
Image = None
try:
    from PIL import Image
except ImportError:
    pass

import aiohttp
import aiofiles
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.core.platform.astr_message_event import AstrMessageEvent

try:
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.provider.func_tool_manager import FunctionToolManager
except ImportError:
    ProviderRequest = Any
    FunctionToolManager = None


class GrokPlugin(Star):
    """Grok 多媒体与联网搜索插件 - 支持生图、生视频、联网搜索"""

    ASPECT_RATIOS = {
        "1:1": "1:1", "方": "1:1", "方形": "1:1",
        "16:9": "16:9", "横": "16:9", "横屏": "16:9",
        "9:16": "9:16", "竖": "9:16", "竖屏": "9:16",
        "3:2": "3:2", "2:3": "2:3",
    }

    DEFAULT_ASPECT_RATIO = "9:16"
    DEFAULT_SEARCH_MODEL = "grok-4-fast"
    DEFAULT_SEARCH_TIMEOUT = 60.0
    DEFAULT_SEARCH_THINKING_BUDGET = 32000

    MAX_IMAGE_COUNT = 10
    MAX_STREAM_LINES = 10000
    MAX_RESPONSE_BYTES = 50 * 1024 * 1024
    MIN_BASE64_LENGTH = 100
    IMAGE_TIMEOUT = 120
    VIDEO_TIMEOUT = 300
    MAX_PROMPT_LENGTH = 4000

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_grok_suite")
        self.temp_dir = Path(self.plugin_data_dir) / "temp"
        self.image_dir = Path(self.plugin_data_dir) / "images"
        self.video_dir = Path(self.plugin_data_dir) / "videos"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.image_dir.mkdir(exist_ok=True, parents=True)
        self.video_dir.mkdir(exist_ok=True, parents=True)

    async def initialize(self):
        if Image is None:
            logger.warning("Pillow 未安装，部分功能受限")
        async with self._session_lock:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
        logger.info("Grok 多媒体与联网搜索插件初始化完成")

    async def terminate(self):
        async with self._session_lock:
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception as e:
                    logger.warning(f"关闭 session 时出错: {e}")
            self._session = None
        logger.info("Grok 多媒体与联网搜索插件已终止")

    # ==================== 工具方法 ====================

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """移除文本中的 Markdown 格式"""
        if not text:
            return ""
        # 移除代码块（包括语言标识符）
        text = re.sub(r'```(?:\w+)?\n?([\s\S]*?)```', r'\1', text)
        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 移除粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # 移除斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # 移除标题符号
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # 移除链接，保留文本
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 移除图片
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        # 移除水平线
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # 移除引用符号
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    ERROR_TRANSLATIONS = {
        "Session is closed": "会话已关闭，请重试",
        "Connection reset by peer": "连接被重置，请重试",
        "Connection refused": "连接被拒绝，请检查API地址",
        "Timeout": "请求超时，请重试",
        "TimeoutError": "请求超时，请重试",
        "Name or service not known": "无法解析API地址，请检查网络",
        "No route to host": "无法连接到服务器，请检查网络",
        "Network is unreachable": "网络不可达，请检查网络连接",
        "SSL": "SSL证书错误，请检查API地址",
        "Certificate": "证书验证失败",
        "Unauthorized": "API密钥无效或已过期",
        "Forbidden": "访问被拒绝，请检查权限",
        "Not Found": "API接口不存在，请检查配置",
        "Too Many Requests": "请求过于频繁，请稍后重试",
        "Rate limit": "已达到速率限制，请稍后重试",
        "Internal Server Error": "服务器内部错误，请稍后重试",
        "Bad Gateway": "网关错误，请稍后重试",
        "Service Unavailable": "服务暂时不可用，请稍后重试",
        "Gateway Timeout": "网关超时，请稍后重试",
        "Invalid API Key": "API密钥无效",
        "Insufficient quota": "API额度不足",
        "Model not found": "模型不存在，请检查配置",
        "Content policy": "内容违反使用政策",
        "Safety system": "触发安全系统限制",
    }

    def _translate_error(self, error: str) -> str:
        """将英文错误消息翻译为中文"""
        if not error:
            return "未知错误"

        error_lower = error.lower()

        # 检查是否匹配已知错误模式
        for en_pattern, zh_msg in self.ERROR_TRANSLATIONS.items():
            if en_pattern.lower() in error_lower:
                return zh_msg

        # 处理 HTTP 状态码
        if "状态码: 401" in error or "status: 401" in error_lower:
            return "API密钥无效或已过期"
        if "状态码: 403" in error or "status: 403" in error_lower:
            return "访问被拒绝"
        if "状态码: 404" in error or "status: 404" in error_lower:
            return "API接口不存在"
        if "状态码: 429" in error or "status: 429" in error_lower:
            return "请求过于频繁，请稍后重试"
        if "状态码: 5" in error or "status: 5" in error_lower:
            return "服务器错误，请稍后重试"

        # 处理 Errno 错误
        if "errno" in error_lower:
            if "104" in error:
                return "连接被重置，请重试"
            if "111" in error:
                return "连接被拒绝，请检查API地址"
            if "110" in error:
                return "连接超时，请重试"
            if "113" in error:
                return "无法连接到服务器"

        # 如果无法翻译，返回简化的错误
        # 移除技术细节，只保留关键信息
        if ":" in error:
            parts = error.split(":")
            # 尝试找到有意义的部分
            for part in reversed(parts):
                part = part.strip()
                if part and not part.startswith("[") and len(part) > 3:
                    # 如果还是英文，返回通用错误
                    if any(c.isalpha() and ord(c) > 127 for c in part):
                        return part  # 已经是中文
                    break

        return "请求失败，请稍后重试"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """确保 session 有效（线程安全）"""
        async with self._session_lock:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    def _parse_image_api_response(self, data: dict) -> List[Tuple[Optional[str], Optional[bytes]]]:
        """解析图片生成 API 响应，返回 [(url, bytes), ...]"""
        results = []
        # 标准 OpenAI 格式: {"data": [{"url": "..."} or {"b64_json": "..."}]}
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    if item.get("url"):
                        results.append((item["url"], None))
                    elif item.get("b64_json"):
                        try:
                            img_bytes = base64.b64decode(item["b64_json"])
                            results.append((None, img_bytes))
                        except Exception as e:
                            logger.warning(f"Base64 解码失败: {e}")

        # 其他格式: 尝试提取 URL 或 Base64
        if not results:
            url, b64, _ = self._parse_json_response(data)
            if url:
                results.append((url, None))
            elif b64:
                try:
                    img_bytes = base64.b64decode(b64)
                    results.append((None, img_bytes))
                except Exception as e:
                    logger.warning(f"Base64 解码失败: {e}")

        return results

    # ==================== API 调用 ====================

    @staticmethod
    def _detect_mime_type(data: bytes) -> str:
        """检测图片 MIME 类型"""
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        if data.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        if data.startswith((b'GIF87a', b'GIF89a')):
            return "image/gif"
        if data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WEBP':
            return "image/webp"
        if data.startswith(b'BM'):
            return "image/bmp"
        return "image/png"

    def _get_headers(self) -> dict:
        api_key = str(self.conf.get("grok_api_key", "")).strip()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        """获取 API 基础 URL，自动处理常见的 URL 格式问题

        用户只需填写基础 URL（如 https://api.x.ai），
        会自动移除多余的路径后缀，返回纯净的基础 URL
        """
        url = str(self.conf.get("grok_api_url", "https://api.x.ai")).rstrip("/")
        # 移除常见的端点后缀，只保留基础 URL
        suffixes = [
            "/v1/chat/completions", "/v1/images/generations", "/v1/images/edits",
            "/v1/video/generations", "/chat/completions", "/images/generations",
            "/images/edits", "/video/generations", "/v1"
        ]
        for suffix in suffixes:
            if url.endswith(suffix):
                url = url[:-len(suffix)]
        return url.rstrip("/")

    async def _generate_image(self, prompt: str, image_bytes: Optional[bytes] = None,
                               n: int = 1, aspect_ratio: str = "1:1") -> Tuple[List[Tuple[Optional[str], Optional[bytes]]], Optional[str]]:
        """调用 Grok 生图 API，返回 [(url_or_path, bytes), ...] 或错误

        文生图: POST /v1/images/generations (JSON)
        图生图: POST /v1/images/edits (multipart/form-data)
        """
        if image_bytes:
            return await self._edit_image(prompt, image_bytes, n, aspect_ratio)

        base_url = self._get_base_url()
        api_url = f"{base_url}/v1/images/generations"
        model = self.conf.get("grok_image_model", "grok-imagine-1.0")

        payload = {
            "model": model,
            "prompt": prompt,
            "n": max(1, min(n, self.MAX_IMAGE_COUNT)),
            "response_format": "url"
        }

        # 尺寸映射到比例
        size_map = {
            "16:9": "1280x720", 
            "9:16": "720x1280", 
            "1:1": "1024x1024",
            "2:3": "1024x1792",
            "3:2": "1792x1024"
        }                        
        if aspect_ratio and aspect_ratio in size_map:
            payload["size"] = size_map[aspect_ratio]

        try:
            session = await self._ensure_session()
            async with session.post(
                api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.IMAGE_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"[文生图] API 请求失败 (状态码: {resp.status}): {text[:200]}")
                    return [], self._translate_error(f"状态码: {resp.status}")

                raw_content = await resp.read()
                try:
                    data = json.loads(raw_content.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.error(f"JSON解析失败，响应前200字节: {raw_content[:200]}")
                    return [], "API响应格式异常"

                results = self._parse_image_api_response(data)
                if results:
                    return results, None
                return [], "未能从响应中提取图片"

        except asyncio.TimeoutError:
            return [], "请求超时，请重试"
        except Exception as e:
            logger.error(f"[文生图] 请求异常: {e}")
            return [], self._translate_error(str(e))

    async def _edit_image(self, prompt: str, image_bytes: bytes,
                          n: int = 1, aspect_ratio: str = "1:1") -> Tuple[List[Tuple[Optional[str], Optional[bytes]]], Optional[str]]:
        """调用 Grok 图片编辑 API (图生图)

        使用 /v1/images/edits 接口，multipart/form-data 格式
        """
        base_url = self._get_base_url()
        api_url = f"{base_url}/v1/images/edits"
        model = self.conf.get("grok_edit_model", "grok-imagine-1.0-edit")

        # 构建 multipart/form-data
        form = aiohttp.FormData()
        form.add_field('model', model)
        form.add_field('prompt', prompt)
        form.add_field('n', str(max(1, min(n, self.MAX_IMAGE_COUNT))))
        form.add_field('response_format', 'url')

        # 添加图片文件
        mime_type = self._detect_mime_type(image_bytes)
        ext = mime_type.split('/')[-1]
        if ext == 'jpeg':
            ext = 'jpg'
        form.add_field('image', image_bytes,
                       filename=f'image.{ext}',
                       content_type=mime_type)

        try:
            session = await self._ensure_session()
            headers = {"Authorization": f"Bearer {self.conf.get('grok_api_key', '')}"}
            async with session.post(
                api_url,
                headers=headers,
                data=form,
                timeout=aiohttp.ClientTimeout(total=self.IMAGE_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"[图生图] API 请求失败 (状态码: {resp.status}): {text[:200]}")
                    return [], self._translate_error(f"状态码: {resp.status}")

                raw_content = await resp.read()
                try:
                    data = json.loads(raw_content.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.error(f"JSON解析失败，响应前200字节: {raw_content[:200]}")
                    return [], "API响应格式异常"

                results = self._parse_image_api_response(data)
                if results:
                    return results, None
                return [], "未能从响应中提取图片"

        except asyncio.TimeoutError:
            return [], "请求超时，请重试"
        except Exception as e:
            logger.error(f"[图生图] 请求异常: {e}")
            return [], self._translate_error(str(e))

    async def _generate_video(self, prompt: str, image_bytes: bytes, aspect_ratio: str = "16:9") -> Tuple[Optional[str], Optional[str]]:
        """调用 Grok 生视频 API

        使用 /v1/chat/completions 接口，模型为 grok-imagine-1.0-video
        """
        base_url = self._get_base_url()
        api_url = f"{base_url}/v1/chat/completions"
        model = self.conf.get("grok_video_model", "grok-imagine-1.0-video")

        mime_type = self._detect_mime_type(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }]

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "video_config": {
                "aspect_ratio": aspect_ratio,
                "resolution_name": "720p"
            }
        }

        for attempt in range(3):
            try:
                session = await self._ensure_session()
                async with session.post(
                    api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.VIDEO_TIMEOUT)
                ) as resp:
                    if resp.status != 200:
                        if resp.status >= 500 and attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        text = await resp.text()
                        logger.error(f"[图生视频] API 请求失败 (状态码: {resp.status}): {text[:200]}")
                        return None, self._translate_error(f"状态码: {resp.status}")

                    media_bytes, media_url, error = await self._parse_media_response(resp, "video")
                    if error:
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        return None, error
                    if media_bytes:
                        # 返回 bytes 需要先保存为临时文件
                        filename = f"grok_video_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
                        file_path = self.temp_dir / filename
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(media_bytes)
                        return str(file_path), None
                    if media_url:
                        return media_url, None
                    return None, "API 响应中未包含有效视频内容"

            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt == 2:
                    return None, "请求超时，请重试"
                await asyncio.sleep(2)
            except Exception as e:
                if attempt == 2:
                    logger.error(f"[图生视频] 请求异常: {e}")
                    return None, self._translate_error(str(e))
                await asyncio.sleep(1)

        return None, "所有重试均失败"

    # ==================== 响应解析 ====================

    async def _parse_media_response(self, resp, media_type: str = "image") -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """
        统一解析媒体响应，支持流式/非流式、URL/Base64
        返回: (media_bytes, media_url, error_msg)
        """
        accumulated_text = []
        is_streaming = False
        raw_content = b""
        extracted_url = None
        extracted_base64 = None
        line_count = 0

        async for line in resp.content:
            line_count += 1
            if line_count > self.MAX_STREAM_LINES:
                return None, None, "响应行数超限"
            if len(raw_content) > self.MAX_RESPONSE_BYTES:
                return None, None, "响应数据过大"
            raw_content += line
            if not line or not line.strip():
                continue

            try:
                line_str = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            # SSE 流式解析
            payload_str = None
            if line_str.startswith('data: '):
                payload_str = line_str[6:]
            elif line_str.startswith('data:'):
                payload_str = line_str[5:]

            if payload_str is not None:
                is_streaming = True
                payload_str = payload_str.strip()
                if payload_str in ('[DONE]', 'done', ''):
                    continue

                try:
                    chunk = json.loads(payload_str)
                    # 提取文本内容
                    content = self._extract_text_content(chunk)
                    if content:
                        accumulated_text.append(content)
                    # 提取媒体数据
                    url, b64 = self._extract_media_from_chunk(chunk)
                    if url:
                        extracted_url = url
                    if b64:
                        extracted_base64 = b64
                except json.JSONDecodeError:
                    if payload_str.startswith(("http://", "https://")):
                        extracted_url = payload_str.split()[0]
                    elif self._is_base64(payload_str):
                        extracted_base64 = payload_str

        # 非流式响应处理
        if not is_streaming and raw_content:
            try:
                data = json.loads(raw_content.decode('utf-8'))
                # 处理各种 API 响应格式
                url, b64, text = self._parse_json_response(data)
                if url:
                    extracted_url = url
                if b64:
                    extracted_base64 = b64
                if text:
                    accumulated_text.append(text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # 从累积文本中提取媒体
        full_text = "".join(accumulated_text)
        if not extracted_url and not extracted_base64 and full_text:
            extracted_url = self._extract_url_from_text(full_text)
            if not extracted_url:
                extracted_base64 = self._extract_base64_from_text(full_text)

        # 返回结果
        if extracted_base64:
            try:
                media_bytes = base64.b64decode(extracted_base64)
                return media_bytes, None, None
            except Exception as e:
                return None, None, f"Base64 解码失败: {e}"

        if extracted_url:
            return None, extracted_url, None

        return None, None, "未能从响应中提取媒体内容"

    def _parse_json_response(self, data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """解析 JSON 响应，返回 (url, base64, text)"""
        url = None
        b64 = None
        text = None

        # OpenAI 图像生成格式: {"data": [{"url": "..."} or {"b64_json": "..."}]}
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    if item.get("url"):
                        url = item["url"]
                    if item.get("b64_json"):
                        b64 = item["b64_json"]
                    if item.get("revised_prompt"):
                        text = item["revised_prompt"]

        # Chat Completions 格式
        if "choices" in data:
            for choice in data.get("choices", []):
                msg = choice.get("message") or choice.get("delta") or {}
                content = msg.get("content")
                if content:
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "image_url":
                                    img_url = part.get("image_url", {}).get("url", "")
                                    if img_url.startswith("data:"):
                                        b64 = self._extract_base64_from_data_uri(img_url)
                                    else:
                                        url = img_url
                                elif part.get("type") == "text":
                                    text = part.get("text", "")

        # 直接字段
        for key in ("url", "image_url", "video_url", "media_url", "file_url"):
            if data.get(key):
                url = data[key]
                break

        for key in ("b64_json", "base64", "image_base64", "data"):
            val = data.get(key)
            if val and isinstance(val, str) and self._is_base64(val):
                b64 = val
                break

        for key in ("content", "text", "result", "output", "message"):
            val = data.get(key)
            if val and isinstance(val, str):
                text = val
                break

        return url, b64, text

    def _extract_media_from_chunk(self, chunk: dict) -> Tuple[Optional[str], Optional[str]]:
        """从流式块中提取媒体 URL 或 Base64"""
        url = None
        b64 = None

        # 递归搜索
        def search(obj):
            nonlocal url, b64
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("url", "image_url", "video_url") and isinstance(v, str):
                        if v.startswith("data:"):
                            b64 = self._extract_base64_from_data_uri(v)
                        elif v.startswith("http"):
                            url = v
                    elif k in ("b64_json", "base64") and isinstance(v, str):
                        b64 = v
                    else:
                        search(v)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(chunk)
        return url, b64

    @staticmethod
    def _extract_text_content(chunk: dict) -> Optional[str]:
        """从响应块中提取文本内容"""
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content", "")
            if isinstance(content, list):
                return "".join(
                    str(c.get("text", "")) if isinstance(c, dict) else str(c)
                    for c in content
                )
            return str(content) if content else ""
        for key in ("content", "text", "result", "output"):
            val = chunk.get(key)
            if val:
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    return "".join(
                        str(c.get("text", "")) if isinstance(c, dict) else str(c)
                        for c in val
                    )
        return None

    @staticmethod
    def _extract_url_from_text(text: str) -> Optional[str]:
        """从文本中提取媒体 URL"""
        if not text:
            return None
        text = text.strip()

        if text.startswith(("http://", "https://")):
            return text.split()[0].rstrip('.,;!?)\'\"')

        patterns = [
            r'<(?:video|source|img)[^>]*src=["\']([^"\']+)["\']',
            r'(https?://[^\s<>"\')\]\\]+\.(?:mp4|webm|mov|avi|mkv|png|jpg|jpeg|gif|webp|bmp)(?:[?#][^\s<>"\')\]\\]*)?)',
            r'!\[[^\]]*\]\(([^)]+)\)',
            r'"url"\s*:\s*"(https?://[^"]+)"',
            r'(https?://[^\s<>"\')\]\\]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                url = match.group(1)
                if url.startswith(("http://", "https://")):
                    return url
        return None

    @staticmethod
    def _extract_base64_from_text(text: str) -> Optional[str]:
        """从文本中提取 Base64 数据"""
        if not text:
            return None

        # data URI 格式
        match = re.search(r'data:[^;]+;base64,([A-Za-z0-9+/=]+)', text)
        if match:
            return match.group(1)

        # 纯 base64 字符串（至少100字符，避免误判）
        match = re.search(r'([A-Za-z0-9+/]{100,}={0,2})', text)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_base64_from_data_uri(data_uri: str) -> Optional[str]:
        """从 data URI 中提取 Base64"""
        if "base64," in data_uri:
            return data_uri.split("base64,", 1)[1]
        return None

    @staticmethod
    def _is_base64(s: str) -> bool:
        """检查字符串是否为有效的 Base64"""
        if not s or len(s) < 100:
            return False
        try:
            if re.match(r'^[A-Za-z0-9+/]+={0,2}$', s):
                base64.b64decode(s[:100])
                return True
        except Exception:
            pass
        return False

    # ==================== 联网搜索辅助方法 ====================

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        """将配置值转换为布尔值"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return default

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        """将配置值转换为整数"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        """将配置值转换为浮点数"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_json_config_dict(self, key: str) -> Dict[str, Any]:
        """解析 JSON 格式的配置项"""
        value = self.conf.get(key, {})
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                logger.warning(f"配置项 {key} JSON 解析失败: {exc}")
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _extract_urls_from_text(text: str) -> List[str]:
        """从文本中提取 URL 列表"""
        urls = re.findall(r"https?://[^\s)\]}>\"']+", text or "")
        seen = set()
        result: List[str] = []
        for url in urls:
            cleaned = url.rstrip(".,;:!?'\"")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    @staticmethod
    def _coerce_search_json(text: str) -> Optional[Dict[str, Any]]:
        """尝试将搜索响应文本解析为 JSON 对象"""
        if not text:
            return None
        cleaned = text.strip()
        # 移除 Markdown 代码块包装
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            return None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _parse_search_message(self, message_content: Any) -> Tuple[str, List[Dict[str, str]], str]:
        """解析搜索响应消息，返回 (content, sources, raw)"""
        if isinstance(message_content, str):
            message_text = message_content.strip()
        elif isinstance(message_content, list):
            parts: List[str] = []
            for item in message_content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            message_text = "".join(parts).strip()
        else:
            message_text = str(message_content or "").strip()

        parsed = self._coerce_search_json(message_text)
        sources: List[Dict[str, str]] = []
        raw = ""

        if parsed is None:
            content = message_text
            raw = message_text
        else:
            content = str(parsed.get("content") or "").strip()
            source_list = parsed.get("sources")
            if isinstance(source_list, list):
                for item in source_list:
                    if isinstance(item, dict) and item.get("url"):
                        sources.append({
                            "url": str(item.get("url")),
                            "title": str(item.get("title") or ""),
                            "snippet": str(item.get("snippet") or ""),
                        })
            if not content:
                content = message_text

        if not sources:
            for url in self._extract_urls_from_text(content):
                sources.append({"url": url, "title": "", "snippet": ""})

        return content, sources, raw

    def _search_show_sources(self) -> bool:
        """是否显示搜索来源"""
        return self._to_bool(self.conf.get("grok_search_show_sources", False), False)

    def _search_max_sources(self) -> int:
        """最大显示来源数量"""
        value = self._to_int(self.conf.get("grok_search_max_sources", 5), 5)
        return 5 if value < 0 else value

    def _search_skill_enabled(self) -> bool:
        """是否启用 Skill 模式"""
        return self._to_bool(self.conf.get("grok_search_enable_skill", False), False)

    async def _perform_web_search(self, query: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """执行联网搜索/对话，支持图片理解"""
        started = time.time()
        query = (query or "").strip()
        if not query and not image_bytes:
            return {
                "ok": False,
                "error": "请输入问题内容",
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }

        if query and len(query) > self.MAX_PROMPT_LENGTH:
            return {
                "ok": False,
                "error": f"输入内容过长，最大支持 {self.MAX_PROMPT_LENGTH} 字符",
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }

        api_key = str(self.conf.get("grok_api_key", "")).strip()
        if not api_key:
            return {
                "ok": False,
                "error": "未配置 API 密钥",
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }

        model = str(self.conf.get("grok_search_model", self.DEFAULT_SEARCH_MODEL)).strip() or self.DEFAULT_SEARCH_MODEL
        timeout = self._to_float(self.conf.get("grok_search_timeout_seconds", self.DEFAULT_SEARCH_TIMEOUT), self.DEFAULT_SEARCH_TIMEOUT)
        if timeout <= 0:
            timeout = self.DEFAULT_SEARCH_TIMEOUT

        enable_thinking = self._to_bool(self.conf.get("grok_search_enable_thinking", True), True)
        thinking_budget = self._to_int(self.conf.get("grok_search_thinking_budget", self.DEFAULT_SEARCH_THINKING_BUDGET), self.DEFAULT_SEARCH_THINKING_BUDGET)
        if thinking_budget < 0:
            thinking_budget = self.DEFAULT_SEARCH_THINKING_BUDGET

        # 搜索模式: auto/on/off
        search_mode = str(self.conf.get("grok_search_mode", "auto")).strip().lower()
        if search_mode not in {"auto", "on", "off"}:
            search_mode = "auto"

        # 根据搜索模式设置 system prompt 和 search_parameters
        if search_mode == "off":
            # 纯对话模式，不联网
            system_prompt = (
                "You are a helpful assistant. "
                "IMPORTANT: Do NOT use Markdown formatting - respond in plain text only."
            )
            search_parameters = None
        elif search_mode == "on":
            # 始终联网搜索
            system_prompt = (
                "You are a web research assistant. Use live web search/browsing when answering. "
                "Return ONLY a single JSON object with keys: "
                "content (string), sources (array of objects with url/title/snippet when possible). "
                "Keep content concise and evidence-backed. "
                "IMPORTANT: Do NOT use Markdown formatting in the content field - use plain text only."
            )
            search_parameters = {"mode": "on"}
        else:
            # auto 模式，由模型自动判断
            system_prompt = (
                "You are a helpful assistant with web search capabilities. "
                "If the user's question requires up-to-date information, current events, or facts you're unsure about, "
                "use web search to find accurate information. "
                "When you do search, return a JSON object with keys: "
                "content (string), sources (array of objects with url/title/snippet when possible). "
                "For general questions that don't need web search, respond normally. "
                "IMPORTANT: Do NOT use Markdown formatting - respond in plain text only."
            )
            search_parameters = {"mode": "auto"}

        # 构建用户消息（支持多模态）
        if image_bytes:
            mime_type = self._detect_mime_type(image_bytes)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            user_content = [
                {"type": "text", "text": query or "请描述这张图片"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        else:
            user_content = query

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "stream": False,
        }

        # 添加 search_parameters（如果需要联网）
        if search_parameters:
            payload["search_parameters"] = search_parameters

        if enable_thinking:
            payload["reasoning_effort"] = "high"
            if thinking_budget > 0:
                payload["reasoning_budget_tokens"] = thinking_budget

        extra_body = self._parse_json_config_dict("grok_search_extra_body")
        for key, value in extra_body.items():
            if key not in {"model", "messages", "stream"}:
                payload[key] = value

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        extra_headers = self._parse_json_config_dict("grok_search_extra_headers")
        for key, value in extra_headers.items():
            if str(key).lower() not in {"authorization", "content-type"}:
                headers[str(key)] = str(value)

        api_url = f"{self._get_base_url()}/v1/chat/completions"
        raw_text = ""

        try:
            session = await self._ensure_session()
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                raw_text = await resp.text()
                if resp.status != 200:
                    logger.warning(f"[对话/搜索] HTTP {resp.status}: {raw_text[:500]}")
                    return {
                        "ok": False,
                        "error": self._translate_error(f"状态码: {resp.status}"),
                        "content": "",
                        "sources": [],
                        "raw": raw_text[:2000] if raw_text else "",
                        "elapsed_ms": int((time.time() - started) * 1000),
                    }

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                return {
                    "ok": False,
                    "error": "响应解析失败，API 返回了非 JSON 格式的数据",
                    "content": "",
                    "sources": [],
                    "raw": raw_text[:2000] if raw_text else "",
                    "elapsed_ms": int((time.time() - started) * 1000),
                }

            if "error" in data and isinstance(data.get("error"), (dict, str)):
                error_info = data["error"]
                error_msg = (
                    error_info.get("message", str(error_info))
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
                return {
                    "ok": False,
                    "error": self._translate_error(error_msg),
                    "content": "",
                    "sources": [],
                    "raw": raw_text[:2000] if raw_text else "",
                    "elapsed_ms": int((time.time() - started) * 1000),
                }

            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                return {
                    "ok": False,
                    "error": "响应缺少 choices 字段",
                    "content": "",
                    "sources": [],
                    "raw": raw_text[:2000] if raw_text else "",
                    "elapsed_ms": int((time.time() - started) * 1000),
                }

            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            content, sources, raw = self._parse_search_message((message or {}).get("content"))

            if not content:
                return {
                    "ok": False,
                    "error": "API 返回了空响应",
                    "content": "",
                    "sources": [],
                    "raw": raw_text[:2000] if raw_text else "",
                    "elapsed_ms": int((time.time() - started) * 1000),
                }

            return {
                "ok": True,
                "content": content,
                "sources": sources,
                "raw": raw,
                "model": data.get("model") or model,
                "usage": data.get("usage") or {},
                "elapsed_ms": int((time.time() - started) * 1000),
            }
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": "请求超时，请稍后重试",
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }
        except aiohttp.ClientError as exc:
            return {
                "ok": False,
                "error": self._translate_error(str(exc)),
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }
        except Exception as exc:
            logger.error(f"[联网搜索] 请求异常: {exc}")
            return {
                "ok": False,
                "error": self._translate_error(str(exc)),
                "content": "",
                "sources": [],
                "raw": "",
                "elapsed_ms": int((time.time() - started) * 1000),
            }

    def _format_search_result(self, result: Dict[str, Any]) -> str:
        """格式化搜索结果为用户友好的消息（纯文本，无 Markdown）"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            raw = result.get("raw", "")
            # 尝试从 raw 中提取更具体的错误信息
            if raw:
                try:
                    error_data = json.loads(raw)
                    if isinstance(error_data.get("error"), dict):
                        error = error_data["error"].get("message", error)
                    elif isinstance(error_data.get("error"), str):
                        error = error_data["error"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return f"❌ 请求失败: {error}"

        content = self._strip_markdown(str(result.get("content", "")))
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        lines = [content]
        if self._search_show_sources() and sources:
            max_sources = self._search_max_sources()
            selected = sources[:max_sources] if max_sources > 0 else sources
            lines.append("\n来源:")
            for i, src in enumerate(selected, 1):
                url = src.get("url", "")
                title = src.get("title", "")
                if title:
                    lines.append(f"  {i}. {title}\n     {url}")
                else:
                    lines.append(f"  {i}. {url}")

        return "\n".join(lines)

    def _format_search_result_for_llm(self, result: Dict[str, Any]) -> str:
        """格式化搜索结果供 LLM 使用"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            raw = result.get("raw", "")
            return f"搜索失败: {error}\n{raw}" if raw else f"搜索失败: {error}"

        content = str(result.get("content", ""))
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        lines = [f"搜索结果:\n{content}"]

        if self._search_show_sources() and sources:
            max_sources = self._search_max_sources()
            selected = sources[:max_sources] if max_sources > 0 else sources
            lines.append("\n参考来源:")
            for i, src in enumerate(selected, 1):
                url = src.get("url", "")
                title = src.get("title", "")
                snippet = src.get("snippet", "")
                if title:
                    lines.append(f"  {i}. {title}")
                    lines.append(f"     {url}")
                else:
                    lines.append(f"  {i}. {url}")
                if snippet:
                    lines.append(f"     {snippet}")

        return "\n".join(lines)

    # ==================== 媒体处理 ====================

    async def _download_media(self, url: str) -> Optional[bytes]:
        try:
            session = await self._ensure_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"媒体下载失败: {e}")
            return None

    async def _load_bytes(self, src: str) -> Optional[bytes]:
        if Path(src).is_file():
            try:
                async with aiofiles.open(src, 'rb') as f:
                    return await f.read()
            except Exception as e:
                logger.debug(f"读取本地文件失败 ({src[:50]}): {e}")
                return None
        elif src.startswith("http"):
            return await self._download_media(src)
        elif src.startswith("base64://"):
            try:
                return base64.b64decode(src[9:])
            except Exception as e:
                logger.debug(f"Base64解码失败: {e}")
                return None
        return None

    async def _get_image_from_event(self, event: AstrMessageEvent) -> Optional[bytes]:
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply) and seg.chain:
                for s in seg.chain:
                    if isinstance(s, Comp.Image):
                        if s.url and (img := await self._load_bytes(s.url)):
                            return img
                        if s.file and (img := await self._load_bytes(s.file)):
                            return img
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)):
                    return img
        return None

    async def _save_and_send_media(self, event: AstrMessageEvent, url: str,
                                    media_bytes: bytes, media_type: str = "image"):
        save_media = self.conf.get("save_media", False)
        if media_type == "video":
            ext = "mp4"
        else:
            mime_type = self._detect_mime_type(media_bytes)
            ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp", "image/bmp": "bmp"}
            ext = ext_map.get(mime_type, "png")
        filename = f"grok_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"

        if save_media:
            save_dir = self.video_dir if media_type == "video" else self.image_dir
        else:
            save_dir = self.temp_dir

        file_path = (save_dir / filename).resolve()

        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(media_bytes)

            if media_type == "video":
                component = Comp.Video.fromFileSystem(path=str(file_path), name=filename)
            else:
                component = Comp.Image.fromFileSystem(path=str(file_path))

            yield event.chain_result([component])

        except Exception as e:
            logger.error(f"媒体处理失败: {e}")
            yield event.plain_result("❌ 发送失败，请到后台查看")
        finally:
            if not save_media:
                try:
                    await aiofiles.os.remove(file_path)
                except Exception:
                    pass

    async def _send_images_forward(self, event: AstrMessageEvent,
                                    images_data: List[Tuple[str, bytes]], failed_count: int = 0):
        """使用合并转发发送多张图片"""
        saved_files = []
        nodes = []
        save_media = self.conf.get("save_media", False)
        save_dir = self.image_dir if save_media else self.temp_dir

        try:
            self_id = event.get_self_id()
            try:
                self_id_int = int(self_id)
            except (ValueError, TypeError):
                self_id_int = 10000

            for i, (url, img_bytes) in enumerate(images_data):
                mime_type = self._detect_mime_type(img_bytes)
                ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp", "image/bmp": "bmp"}
                ext = ext_map.get(mime_type, "png")
                filename = f"grok_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.{ext}"
                file_path = (save_dir / filename).resolve()

                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(img_bytes)
                saved_files.append((file_path, save_media))

                nodes.append(
                    Comp.Node(
                        name="Grok",
                        uin=self_id_int,
                        content=[Comp.Image.fromFileSystem(path=str(file_path))]
                    )
                )

            # 如果有失败的图片，添加提示节点
            if failed_count > 0:
                nodes.append(
                    Comp.Node(
                        name="Grok",
                        uin=self_id_int,
                        content=[Comp.Plain(f"⚠️ {failed_count}张图片下载失败，请到后台查看")]
                    )
                )

            yield event.chain_result([Comp.Nodes(nodes)])

        except Exception as e:
            logger.error(f"合并转发发送失败: {e}")
            for i, (url, img_bytes) in enumerate(images_data):
                async for result in self._save_and_send_media(event, url, img_bytes, "image"):
                    yield result
        finally:
            for file_path, should_keep in saved_files:
                if not should_keep:
                    try:
                        await aiofiles.os.remove(file_path)
                    except Exception:
                        pass

    # ==================== 权限检查 ====================

    async def _check_permissions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        group_blacklist = self.conf.get("group_blacklist", [])
        if hasattr(event, 'get_group_id') and group_blacklist:
            try:
                group_id = event.get_group_id()
                if group_id and group_id in group_blacklist:
                    return False, None
            except Exception as e:
                logger.debug(f"获取群组ID失败: {e}")

        group_whitelist = self.conf.get("group_whitelist", [])
        if hasattr(event, 'get_group_id') and group_whitelist:
            try:
                group_id = event.get_group_id()
                if group_id and group_id not in group_whitelist:
                    return False, None
            except Exception as e:
                logger.debug(f"获取群组ID失败: {e}")

        user_blacklist = self.conf.get("user_blacklist", [])
        if event.get_sender_id() in user_blacklist:
            return False, None

        user_whitelist = self.conf.get("user_whitelist", [])
        if user_whitelist and event.get_sender_id() not in user_whitelist:
            return False, None

        return True, None

    # ==================== 参数解析 ====================

    def _get_aspect_ratio_from_image(self, image_bytes: bytes) -> Optional[str]:
        """从图片字节识别宽高比，返回最接近的支持比例"""
        if not Image:
            return None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    return None

                ratio = width / height

                # 支持的比例及其数值
                supported_ratios = {
                    "16:9": 16 / 9,   # 1.778
                    "3:2": 3 / 2,     # 1.5
                    "1:1": 1.0,       # 1.0
                    "2:3": 2 / 3,     # 0.667
                    "9:16": 9 / 16,   # 0.5625
                }

                # 找到最接近的比例
                closest = min(supported_ratios.items(), key=lambda x: abs(x[1] - ratio))
                return closest[0]
        except Exception as e:
            logger.warning(f"自动识别图片比例失败: {e}")
            return None

    def _parse_image_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """解析生图参数: [数量] [比例] 提示词（顺序任意）

        规则：
        - 只识别开头连续的独立参数词
        - 数量: 1-10 的独立数字
        - 比例: ASPECT_RATIOS 中的关键词
        - 遇到非参数词立即停止，后续全部作为提示词
        - 每种参数最多识别一次
        """
        params = {"n": 1, "aspect_ratio": self.DEFAULT_ASPECT_RATIO}
        parts = text.split()
        if not parts:
            return "", params

        prompt_start = 0
        found_n = False
        found_ratio = False

        # 最多检查前2个词（数量+比例，顺序任意）
        for i in range(min(2, len(parts))):
            p = parts[i]

            # 检查是否为数量(1-10的独立数字)
            if not found_n and p.isdigit() and 1 <= int(p) <= 10:
                params["n"] = int(p)
                prompt_start = i + 1
                found_n = True
            # 检查是否为比例
            elif not found_ratio and p in self.ASPECT_RATIOS:
                params["aspect_ratio"] = self.ASPECT_RATIOS[p]
                prompt_start = i + 1
                found_ratio = True
            else:
                # 遇到非参数词，停止解析
                break

        prompt = " ".join(parts[prompt_start:]).strip()
        return prompt, params

    # ==================== 命令 ====================

    @filter.command("grok生图", prefix_optional=True)
    async def on_image_request(self, event: AstrMessageEvent):
        """Grok 生图: /grok生图 [数量] [比例] <提示词> [+图片可选]"""
        api_key = self.conf.get("grok_api_key", "").strip()
        if not api_key:
            yield event.plain_result("❌ 未配置 API 密钥")
            return

        # 从 message_str 中移除命令前缀
        raw_input = event.message_str.strip()
        cmd = "grok生图"
        if raw_input.startswith(cmd):
            user_input = raw_input[len(cmd):].strip()
        else:
            user_input = raw_input

        if not user_input:
            yield event.plain_result("❌ 请输入提示词\n示例: /grok生图 一只可爱的猫咪")
            return

        can_proceed, _ = await self._check_permissions(event)
        if not can_proceed:
            yield event.plain_result("❌ 当前会话无权限使用此功能")
            return

        prompt_text, params = self._parse_image_params(user_input)
        if not prompt_text:
            yield event.plain_result("❌ 请输入提示词")
            return

        if len(prompt_text) > self.MAX_PROMPT_LENGTH:
            yield event.plain_result(f"❌ 提示词过长，最大支持 {self.MAX_PROMPT_LENGTH} 字符")
            return

        image_bytes = await self._get_image_from_event(event)
        mode = "图生图" if image_bytes else "文生图"

        n = params["n"]
        ratio = params["aspect_ratio"]
        yield event.plain_result(f"🎨 正在进行 [{mode}] · {n}张 · {ratio} ...")

        results, error = await self._generate_image(
            prompt_text, image_bytes, n=n, aspect_ratio=ratio
        )

        if error:
            yield event.plain_result(f"❌ [{mode}] 生成失败: {self._translate_error(error)}")
            return

        if not results:
            yield event.plain_result("❌ 未获取到图片")
            return

        # 处理所有图片（URL 需下载，bytes 直接使用）
        images_data = []
        failed_count = 0
        for i, (url_or_path, img_bytes) in enumerate(results):
            if img_bytes:
                images_data.append((url_or_path or f"image_{i}", img_bytes))
            elif url_or_path:
                downloaded = await self._download_media(url_or_path)
                if downloaded:
                    images_data.append((url_or_path, downloaded))
                else:
                    failed_count += 1

        if not images_data:
            yield event.plain_result("❌ 图片下载失败，请到后台查看")
            return

        # 单张图片直接发送，多张使用合并转发
        if len(images_data) == 1:
            async for result in self._save_and_send_media(event, images_data[0][0], images_data[0][1], "image"):
                yield result
            # 单张图片时，如果有失败的，单独提示
            if failed_count > 0:
                yield event.plain_result(f"⚠️ {failed_count}张图片下载失败，请到后台查看")
        else:
            async for result in self._send_images_forward(event, images_data, failed_count):
                yield result

    @filter.command("grok视频", prefix_optional=True)
    async def on_video_request(self, event: AstrMessageEvent):
        """Grok 图生视频: /grok视频 <提示词> + 图片"""
        api_key = self.conf.get("grok_api_key", "").strip()
        if not api_key:
            yield event.plain_result("❌ 未配置 API 密钥")
            return

        # 从 message_str 中移除命令前缀
        raw_input = event.message_str.strip()
        cmd = "grok视频"
        if raw_input.startswith(cmd):
            prompt_text = raw_input[len(cmd):].strip()
        else:
            prompt_text = raw_input

        if not prompt_text:
            yield event.plain_result("❌ 请输入提示词\n示例: /grok视频 让画面动起来")
            return

        if len(prompt_text) > self.MAX_PROMPT_LENGTH:
            yield event.plain_result(f"❌ 提示词过长，最大支持 {self.MAX_PROMPT_LENGTH} 字符")
            return

        can_proceed, _ = await self._check_permissions(event)
        if not can_proceed:
            yield event.plain_result("❌ 当前会话无权限使用此功能")
            return

        image_bytes = await self._get_image_from_event(event)
        if not image_bytes:
            yield event.plain_result("❌ 需要图片，请上传或引用图片")
            return

        # 自动识别图片方向
        aspect_ratio = self._get_aspect_ratio_from_image(image_bytes) or self.DEFAULT_ASPECT_RATIO

        yield event.plain_result(f"🎬 正在进行 [图生视频] · {aspect_ratio} ...")

        video_result, error = await self._generate_video(prompt_text, image_bytes, aspect_ratio)

        if error:
            yield event.plain_result(f"❌ [图生视频] 生成失败: {self._translate_error(error)}")
            return

        if not video_result:
            yield event.plain_result("❌ 未获取到视频")
            return

        save_media = self.conf.get("save_media", False)

        # 判断是本地文件路径还是 URL
        if Path(video_result).is_file():
            # 本地临时文件
            try:
                if save_media:
                    # 移动到视频保存目录
                    filename = Path(video_result).name
                    save_path = (self.video_dir / filename).resolve()
                    async with aiofiles.open(video_result, 'rb') as src:
                        content = await src.read()
                    async with aiofiles.open(save_path, 'wb') as dst:
                        await dst.write(content)
                    await aiofiles.os.remove(video_result)
                    component = Comp.Video.fromFileSystem(path=str(save_path), name=filename)
                else:
                    component = Comp.Video.fromFileSystem(path=video_result, name=Path(video_result).name)
                yield event.chain_result([component])
            except Exception as e:
                logger.error(f"视频发送失败: {e}")
                yield event.plain_result(f"❌ 视频发送失败: {self._translate_error(str(e))}")
            finally:
                if not save_media:
                    try:
                        await aiofiles.os.remove(video_result)
                    except Exception:
                        pass
        else:
            # 需要下载
            video_bytes = await self._download_media(video_result)
            if video_bytes:
                async for result in self._save_and_send_media(event, video_result, video_bytes, "video"):
                    yield result
            else:
                yield event.plain_result("❌ 视频下载失败，请到后台查看")

    @filter.command("grok帮助", prefix_optional=True)
    async def on_help(self, event: AstrMessageEvent):
        help_text = (
            "【Grok AI 助手】\n\n"
            "🎨 生图命令:\n"
            "/grok生图 [数量] [比例] 提示词\n"
            "• 数量: 1-10 (默认1)\n"
            "• 比例: 横/竖/方/16:9/9:16/1:1/3:2/2:3 (默认竖)\n"
            "• 可附带图片进行图生图\n\n"
            "示例:\n"
            "• /grok生图 一只猫\n"
            "• /grok生图 4 横 日落海滩\n"
            "• /grok生图 把背景换成森林 +图片\n\n"
            "━━━━━━━━━━━━━━\n"
            "🎬 视频命令:\n"
            "/grok视频 提示词 + 图片\n"
            "• 自动识别图片方向\n"
            "• 分辨率: 720p\n\n"
            "示例:\n"
            "• /grok视频 让画面动起来\n"
            "• /grok视频 让人物眨眼微笑\n\n"
            "━━━━━━━━━━━━━━\n"
            "💬 对话命令:\n"
            "/grok <内容> [+图片可选]\n"
            "• 智能对话，可自动联网获取最新信息\n"
            "• 可附带图片进行图片理解\n\n"
            "示例:\n"
            "• /grok 你好，介绍一下你自己\n"
            "• /grok 今天有什么新闻\n"
            "• /grok 这张图片里有什么 +图片"
        )
        yield event.plain_result(help_text)

    @filter.command("grok", prefix_optional=True)
    async def on_web_search(self, event: AstrMessageEvent):
        """Grok 对话/搜索: /grok <内容> [+图片可选]"""
        raw_input = event.message_str.strip()
        normalized_input = raw_input.lstrip("/")
        # 避免与其他 grok 命令冲突
        if normalized_input.startswith(("grok生图", "grok视频", "grok帮助")):
            return

        can_proceed, _ = await self._check_permissions(event)
        if not can_proceed:
            yield event.plain_result("❌ 当前会话无权限使用此功能")
            return

        cmd = "grok"
        query = normalized_input[len(cmd):].strip() if normalized_input.startswith(cmd) else normalized_input

        # 检测图片
        image_bytes = await self._get_image_from_event(event)

        if not query and not image_bytes:
            yield event.plain_result("使用方法: /grok <问题内容> [+图片可选]\n输入 /grok帮助 查看完整说明")
            return

        if query.lower() == "help" or query == "帮助":
            yield event.plain_result("使用方法: /grok <问题内容> [+图片可选]\n输入 /grok帮助 查看完整说明")
            return

        api_key = self.conf.get("grok_api_key", "").strip()
        if not api_key:
            yield event.plain_result("❌ 未配置 API 密钥")
            return

        result = await self._perform_web_search(query, image_bytes)
        yield event.plain_result(self._format_search_result(result))

    @filter.llm_tool(name="grok_web_search")
    async def grok_web_search_tool(self, event: AstrMessageEvent, query: str) -> str:
        """通过 Grok 进行实时联网搜索，获取最新信息和来源

        当需要搜索实时信息、最新新闻、API 版本、错误解决方案或验证过时/不确定信息时使用。

        Args:
            query(string): 搜索查询内容，应该是清晰具体的问题或关键词
        """
        query = (query or "").strip()
        if not query:
            return "搜索失败: 查询不能为空"

        can_proceed, _ = await self._check_permissions(event)
        if not can_proceed:
            return "搜索失败: 当前会话没有权限使用该工具"

        result = await self._perform_web_search(query)
        return self._format_search_result_for_llm(result)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在 LLM 请求时，如果启用了 Skill 则移除 grok_web_search 工具"""
        if not self._search_skill_enabled():
            return

        tool_set = getattr(req, "func_tool", None)
        if FunctionToolManager is not None and isinstance(tool_set, FunctionToolManager):
            req.func_tool = tool_set.get_full_tool_set()
            tool_set = req.func_tool

        if not tool_set:
            return
        if hasattr(tool_set, "remove_tool"):
            tool_set.remove_tool("grok_web_search")
            return
        if isinstance(tool_set, dict):
            tool_set.pop("grok_web_search", None)
