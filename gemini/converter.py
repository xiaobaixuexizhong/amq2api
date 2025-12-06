"""
Gemini 请求格式转换器
将 Claude API 格式转换为 Gemini API 格式
"""
import logging
import uuid
import random
from typing import Dict, Any, List, Optional, Union
from models import ClaudeRequest

logger = logging.getLogger(__name__)

# 默认 thinking budget
DEFAULT_THINKING_BUDGET = 1024


def get_thinking_config(thinking: Optional[Union[bool, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    根据 Claude 请求的 thinking 参数生成 Gemini thinkingConfig

    Args:
        thinking: Claude thinking 配置，可以是:
            - None: 默认启用 thinking
            - bool: True 启用，False 禁用
            - dict: {'type': 'enabled', 'budget_tokens': 1024} 等格式

    Returns:
        Gemini thinkingConfig 字典
    """
    # 默认启用 thinking
    if thinking is None:
        return {
            "includeThoughts": True,
            "thinkingBudget": DEFAULT_THINKING_BUDGET
        }

    # 布尔值
    if isinstance(thinking, bool):
        if thinking:
            return {
                "includeThoughts": True,
                "thinkingBudget": DEFAULT_THINKING_BUDGET
            }
        else:
            return {
                "includeThoughts": False
            }

    # 字典格式
    if isinstance(thinking, dict):
        # 检查是否启用
        thinking_type = thinking.get("type", "enabled")
        is_enabled = thinking_type == "enabled"

        if not is_enabled:
            return {
                "includeThoughts": False
            }

        # 获取 budget_tokens
        budget = thinking.get("budget_tokens", DEFAULT_THINKING_BUDGET)

        return {
            "includeThoughts": True,
            "thinkingBudget": budget
        }

    # 其他情况，默认启用
    return {
        "includeThoughts": True,
        "thinkingBudget": DEFAULT_THINKING_BUDGET
    }


def convert_claude_to_gemini(claude_req: ClaudeRequest, project: str) -> Dict[str, Any]:
    """
    将 Claude API 请求转换为 Gemini API 请求格式

    Args:
        claude_req: Claude 请求对象
        project: Gemini 项目 ID

    Returns:
        Gemini 请求字典
    """
    # 转换消息格式
    contents = []
    for msg in claude_req.messages:
        role = "user" if msg.role == "user" else "model"

        # 处理 content
        if isinstance(msg.content, str):
            parts = [{"text": msg.content}]
        elif isinstance(msg.content, list):
            parts = []
            pending_signature = None  # 保存待附加的 signature

            for i, item in enumerate(msg.content):
                if isinstance(item, dict):
                    if item.get("type") == "thinking":
                        # continue
                        # # 处理 thinking 内容块
                        part = {
                            "text": item.get("thinking", ""),
                            "thought": True
                        }
                        parts.append(part)
                        # 如果有 signature，保存到下一个 item（text 或 tool_use）
                        if "signature" in item:
                            pending_signature = item["signature"]
                    elif item.get("type") == "text":
                        part = {"text": item.get("text", "")}
                        # 如果有待附加的 signature，附加到这个 text part
                        if pending_signature:
                            part["thoughtSignature"] = pending_signature
                            pending_signature = None
                        parts.append(part)
                    elif item.get("type") == "image":
                        # 处理图片
                        source = item.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.get("media_type", "image/png"),
                                    "data": source.get("data", "")
                                }
                            })
                    elif item.get("type") == "tool_use":
                        # 处理工具调用
                        part = {
                            "functionCall": {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "args": item.get("input", {})
                            }
                        }
                        # 如果有待附加的 signature，附加到这个 tool_use part
                        if pending_signature:
                            part["thoughtSignature"] = pending_signature
                            pending_signature = None
                        parts.append(part)
                    elif item.get("type") == "tool_result":
                        # 处理工具结果
                        content = item.get("content", "")
                        if isinstance(content, list):
                            content = content[0].get("text", "") if content else ""
                        parts.append({
                            "functionResponse": {
                                "id": item.get("tool_use_id"),
                                "name": item.get("name", ""),
                                "response": {"output": content}
                            }
                        })
                else:
                    parts.append({"text": str(item)})

            # 如果循环结束后还有未附加的 signature，创建空 text part
            if pending_signature:
                parts.append({
                    "text": "",
                    "thoughtSignature": pending_signature
                })
        else:
            parts = [{"text": str(msg.content)}]

        contents.append({
            "role": role,
            "parts": parts
        })

    # 重新组织消息，确保 tool_use 后紧跟对应的 tool_result
    # contents = reorganize_tool_messages(contents)

    # 构建 Gemini 请求
    gemini_request = {
        "project": project,
        "requestId": f"agent-{uuid.uuid4()}",
        "request": {
            "contents": contents,
            "generationConfig": {
                "temperature": claude_req.temperature if claude_req.temperature is not None else 0.4,
                "topP": 1,
                "topK": 40,
                "candidateCount": 1,
                "maxOutputTokens": claude_req.max_tokens,
                "stopSequences": ["<|user|>", "<|bot|>", "<|context_request|>", "<|endoftext|>", "<|end_of_turn|>"],
                "thinkingConfig": get_thinking_config(claude_req.thinking)
            },
            "sessionId": "-3750763034362895578",
        },
        "model": map_claude_model_to_gemini(claude_req.model),
        "userAgent": "antigravity",
        "requestType": "agent"
    }

    # 添加 system instruction
    if claude_req.system:
        # 处理 system 字段（可能是字符串或列表）
        if isinstance(claude_req.system, str):
            # 简单字符串格式
            system_parts = [{"text": claude_req.system}]
        elif isinstance(claude_req.system, list):
            # 列表格式，提取所有 text 内容
            system_parts = []
            for item in claude_req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    system_parts.append({"text": item.get("text", "")})
        else:
            system_parts = [{"text": str(claude_req.system)}]

        gemini_request["request"]["systemInstruction"] = {
            "role": "user",
            "parts": system_parts
        }

    # 添加工具
    if claude_req.tools:
        gemini_request["request"]["tools"] = convert_tools(claude_req.tools)
        gemini_request["request"]["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "VALIDATED"
            }
        }

    return gemini_request


def map_claude_model_to_gemini(claude_model: str) -> str:
    """
    将 Claude 模型名称映射到 Gemini 模型名称
    如果请求的模型已经存在于支持列表中，则直接透传

    Args:
        claude_model: Claude 模型名称或 Gemini 模型名称

    Returns:
        Gemini 模型名称
    """
    # 支持的所有模型（直接透传）
    supported_models = {
        "gemini-2.5-flash", "gemini-2.5-flash-thinking", "gemini-2.5-pro",
        "gemini-3-pro-low", "gemini-3-pro-high", "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image", "gemini-2.5-flash-image",
        "claude-sonnet-4-5", "claude-sonnet-4-5-thinking", "claude-opus-4-5-thinking",
        "gpt-oss-120b-medium"
    }

    if claude_model in supported_models:
        return claude_model

    # Claude 标准模型名称映射
    model_mapping = {
        "claude-sonnet-4.5": "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-5",
        "claude-3-5-sonnet-20240620": "claude-sonnet-4-5",
        "claude-opus-4": "gemini-3-pro-high",
        "claude-haiku-4": "claude-haiku-4.5",
        "claude-3-haiku-20240307": "gemini-2.5-flash"
    }

    return model_mapping.get(claude_model, "claude-sonnet-4-5")


def reorganize_tool_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    重新组织消息，确保每个 tool_use 后紧跟对应的 tool_result
    保持 thinking 和 tool_use 的相对顺序，只移动 tool_result

    Args:
        contents: 原始消息列表

    Returns:
        重新组织后的消息列表
    """
    # 收集所有 tool_result
    tool_results = {}  # {tool_id: tool_result_part}

    for msg in contents:
        for part in msg.get("parts", []):
            if "functionResponse" in part:
                tool_id = part["functionResponse"].get("id")
                if tool_id:
                    tool_results[tool_id] = part

    # 如果没有工具调用，直接返回
    if not tool_results:
        return contents

    # 重新构建消息列表
    new_contents = []

    for msg in contents:
        parts = msg.get("parts", [])
        new_parts = []

        for part in parts:
            # 跳过 functionResponse，它们会被插入到对应的 functionCall 后面
            if "functionResponse" in part:
                continue

            new_parts.append(part)

            # 如果是 functionCall，立即插入对应的 functionResponse
            if "functionCall" in part:
                tool_id = part["functionCall"].get("id")
                if tool_id and tool_id in tool_results:
                    # 创建包含 tool_use 的 model 消息
                    new_contents.append({
                        "role": "model",
                        "parts": [part]
                    })
                    # 创建包含 tool_result 的 user 消息
                    new_contents.append({
                        "role": "user",
                        "parts": [tool_results[tool_id]]
                    })
                    new_parts.pop()  # 移除刚添加的 functionCall

        # 如果还有其他 parts（thinking, text 等），添加到消息中
        if new_parts:
            new_contents.append({
                "role": msg["role"],
                "parts": new_parts
            })

    return new_contents


def convert_tools(claude_tools: List[Any]) -> List[Dict[str, Any]]:
    """
    将 Claude 工具格式转换为 Gemini 工具格式

    Args:
        claude_tools: Claude 工具列表

    Returns:
        Gemini 工具列表
    """
    gemini_tools = []

    for tool in claude_tools:
        # 清理 JSON Schema，移除 Gemini 不支持的字段
        parameters = clean_json_schema(tool.input_schema)

        gemini_tool = {
            "functionDeclarations": [{
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }]
        }
        gemini_tools.append(gemini_tool)

    return gemini_tools


def clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理 JSON Schema，移除 Gemini 不支持的字段，并将验证要求追加到 description

    Args:
        schema: 原始 JSON Schema

    Returns:
        清理后的 JSON Schema
    """
    if not isinstance(schema, dict):
        return schema

    # 需要移除的验证字段
    validation_fields = {
        "minLength": "minLength",
        "maxLength": "maxLength",
        "minimum": "minimum",
        "maximum": "maximum",
        "minItems": "minItems",
        "maxItems": "maxItems",
    }

    # 需要完全移除的字段
    fields_to_remove = {"$schema", "additionalProperties"}

    # 收集验证信息
    validations = []
    for field, label in validation_fields.items():
        if field in schema:
            validations.append(f"{label}: {schema[field]}")

    # 递归清理 schema
    cleaned = {}
    for key, value in schema.items():
        if key in fields_to_remove or key in validation_fields:
            continue

        if key == "description" and validations:
            # 将验证要求追加到 description
            cleaned[key] = f"{value} ({', '.join(validations)})"
        elif isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_json_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    # 如果有验证信息但没有 description 字段，添加一个
    if validations and "description" not in cleaned:
        cleaned["description"] = f"Validation: {', '.join(validations)}"

    return cleaned