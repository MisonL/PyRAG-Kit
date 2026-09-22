import asyncio
import base64
import inspect
import struct
from types import SimpleNamespace

import pytest

import src.providers.__base__.model_provider as model_provider_module
import src.providers.anthropic as anthropic_provider_module
import src.providers.openai_compatible as openai_compatible_module
import src.providers.volcengine as volcengine_module
from src.providers.__base__.model_provider import (
    CompletionRequest,
    normalize_responses_input,
)
from src.providers.anthropic import AnthropicProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.google import GoogleProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.volcengine import VolcengineProvider

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "查询订单",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def test_anthropic_converts_openai_tool_schema():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    params = AnthropicProvider._build_message_params(provider, "hello", "system", TOOLS, 0.2)

    assert params["tools"] == [
        {
            "name": "lookup_order",
            "description": "查询订单",
            "input_schema": TOOLS[0]["function"]["parameters"],
        }
    ]


def test_anthropic_parse_rejects_user_profile_on_non_beta_sdk_path():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    with pytest.raises(ValueError, match="parse 不支持 user"):
        provider._build_message_params(
            prompt="hello",
            user="profile-1",
            for_parse=True,
        )


def test_anthropic_rejects_unlinked_tool_result():
    with pytest.raises(ValueError, match="tool_call_id"):
        AnthropicProvider._convert_messages([{"role": "tool", "content": "{}"}])


def test_anthropic_rejects_malformed_tool_call_arguments():
    with pytest.raises(ValueError, match="arguments.*有效 JSON"):
        AnthropicProvider._convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": "{"},
                        }
                    ],
                }
            ]
        )


def test_anthropic_rejects_image_content_without_url():
    with pytest.raises(ValueError, match="图片内容缺少有效 image_url"):
        AnthropicProvider._convert_content([{"type": "image_url", "image_url": {}}])


def test_anthropic_rejects_invalid_image_data_uri():
    with pytest.raises(ValueError, match="Base64"):
        AnthropicProvider._convert_content(
            [{"type": "image_url", "image_url": "data:image/png;base64,invalid"}]
        )


def test_anthropic_rejects_tool_call_without_id_or_name():
    with pytest.raises(ValueError, match="缺少 id"):
        AnthropicProvider._convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
                }
            ]
        )
    with pytest.raises(ValueError, match="function.name"):
        AnthropicProvider._convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "function": {"arguments": "{}"}}],
                }
            ]
        )


@pytest.mark.parametrize(
    "tool",
    [
        None,
        {"function": "not-an-object"},
        {"function": {"name": "lookup", "parameters": []}},
    ],
)
def test_anthropic_rejects_malformed_tool_definitions(tool):
    with pytest.raises(ValueError, match="工具定义"):
        AnthropicProvider._convert_tools([tool])


@pytest.mark.parametrize(
    "tool_choice",
    ["unsupported", {"type": "function"}, {"type": "tool"}],
)
def test_anthropic_rejects_malformed_tool_choices(tool_choice):
    with pytest.raises(ValueError, match="tool_choice"):
        AnthropicProvider._convert_tool_choice(tool_choice)


def test_anthropic_stream_errors_are_explicit():
    with pytest.raises(RuntimeError, match="上游失败"):
        AnthropicProvider._stream_event(
            SimpleNamespace(
                type="session.error",
                error=SimpleNamespace(message="上游失败"),
            )
        )


def test_google_converts_openai_tool_schema():
    converted = GoogleProvider._convert_tools(TOOLS)

    assert converted is not None
    declaration = converted[0].function_declarations[0]
    assert declaration.name == "lookup_order"
    assert declaration.description == "查询订单"
    assert declaration.parameters_json_schema == TOOLS[0]["function"]["parameters"]


def test_google_converts_standard_tool_result_to_function_response():
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order_id":"A-1"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"status":"paid"}',
            },
        ],
        None,
        None,
    )

    assert contents[0].role == "model"
    function_call = next(part.function_call for part in contents[0].parts if part.function_call)
    assert function_call.name == "lookup_order"
    function_response = contents[1].parts[0].function_response
    assert contents[1].role == "user"
    assert function_response.name == "lookup_order"
    assert function_response.id == "call-1"
    assert function_response.response == {"status": "paid"}


def test_google_merges_parallel_tool_results_into_one_user_turn():
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "lookup_order", "arguments": "{}"}},
                    {"id": "call-2", "function": {"name": "lookup_customer", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"status":"paid"}'},
            {"role": "tool", "tool_call_id": "call-2", "content": '{"name":"Ada"}'},
        ],
        None,
        None,
    )

    assert [content.role for content in contents] == ["model", "user"]
    assert len(contents[1].parts) == 2
    assert [part.function_response.name for part in contents[1].parts] == [
        "lookup_order",
        "lookup_customer",
    ]


def test_google_merges_consecutive_same_role_messages():
    contents = GoogleProvider._contents(
        [
            {"role": "user", "content": "第一条"},
            {"role": "user", "content": "第二条"},
            {"role": "assistant", "content": "答复"},
        ],
        None,
        None,
    )

    assert [content.role for content in contents] == ["user", "model"]
    assert [part.text for part in contents[0].parts] == ["第一条", "第二条"]


def test_google_rejects_external_media_urls_and_unknown_part_types():
    with pytest.raises(ValueError, match="仅支持 data: URI、gs:// URI 或 Google Files URI"):
        GoogleProvider._contents(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "https://example.com/a.png"}],
                }
            ],
            None,
            None,
        )

    allowed = GoogleProvider._contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "gs://bucket/a.png"},
                    {
                        "type": "file",
                        "url": "https://generativelanguage.googleapis.com/v1beta/files/file-1",
                    },
                ],
            }
        ],
        None,
        None,
    )
    assert allowed[0].parts[0].file_data.file_uri == "gs://bucket/a.png"
    assert allowed[0].parts[1].file_data.file_uri.endswith("/files/file-1")

    with pytest.raises(ValueError, match="不支持的消息内容块类型"):
        GoogleProvider._contents(
            [{"role": "user", "content": [{"type": "unknown_part", "value": "x"}]}],
            None,
            None,
        )


def test_google_media_file_ids_are_resolved_and_mime_types_forwarded():
    contents = GoogleProvider._contents(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": {"file_id": "file-image", "mime_type": "image/png"},
                    },
                    {
                        "type": "file",
                        "file": {"file_id": "file-document", "mime_type": "application/pdf"},
                    },
                ],
            }
        ],
        None,
        None,
    )

    image_data = contents[0].parts[0].file_data
    document_data = contents[0].parts[1].file_data
    assert image_data.file_uri.endswith("/v1beta/files/file-image")
    assert image_data.mime_type == "image/png"
    assert document_data.file_uri.endswith("/v1beta/files/file-document")
    assert document_data.mime_type == "application/pdf"


def test_google_vertex_rejects_gemini_files_ids():
    for value in (
        "file-image",
        "files/file-image",
        "https://generativelanguage.googleapis.com/v1beta/files/file-image",
    ):
        with pytest.raises(ValueError, match="Vertex/Enterprise"):
            GoogleProvider._contents(
                [{"role": "user", "content": [{"type": "input_image", "image_url": value}]}],
                None,
                None,
                vertex_mode=True,
            )


def test_google_vertex_accepts_cloud_storage_and_public_https_media():
    contents = GoogleProvider._contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "gs://bucket/image.png"},
                    {"type": "file", "url": "https://cdn.example.com/document.pdf?version=1"},
                ],
            }
        ],
        None,
        None,
        vertex_mode=True,
    )

    assert contents[0].parts[0].file_data.file_uri == "gs://bucket/image.png"
    assert (
        contents[0].parts[1].file_data.file_uri == "https://cdn.example.com/document.pdf?version=1"
    )


@pytest.mark.parametrize("file_id", ["", " ", "files/", "file/id"])
def test_google_media_file_ids_reject_invalid_values(file_id):
    with pytest.raises(ValueError, match="file_id|有效 URI"):
        GoogleProvider._contents(
            [{"role": "user", "content": [{"type": "input_image", "file_id": file_id}]}],
            None,
            None,
        )


def test_google_rejects_malformed_tool_arguments_and_results():
    with pytest.raises(ValueError, match="arguments.*有效 JSON"):
        GoogleProvider._contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": "{"},
                        }
                    ],
                }
            ],
            None,
            None,
        )

    with pytest.raises(ValueError, match="content.*有效 JSON"):
        GoogleProvider._contents(
            [{"role": "tool", "tool_call_id": "call-1", "name": "lookup", "content": "{"}],
            None,
            None,
        )


def test_google_rejects_invalid_data_uri_base64():
    with pytest.raises(ValueError, match="Base64"):
        GoogleProvider._contents(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "data:image/png;base64,not-base64"}
                    ],
                }
            ],
            None,
            None,
        )


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {}},
        {"type": "audio_url"},
        {"type": "video_url", "video_url": {}},
        {"type": "file", "file": {}},
    ],
)
def test_google_rejects_media_content_without_uri(part):
    with pytest.raises(ValueError, match="(图片|媒体)内容缺少有效 URI"):
        GoogleProvider._contents(
            [{"role": "user", "content": [part]}],
            None,
            None,
        )


def test_google_decodes_media_data_uri_exactly_once():
    contents = GoogleProvider._contents(
        [
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": "data:audio/wav;base64,YWJj"}],
            }
        ],
        None,
        None,
    )

    blob = contents[0].parts[0].inline_data
    assert blob.mime_type == "audio/wav"
    assert blob.data == b"abc"


def test_google_tool_call_message_with_null_content_does_not_emit_text():
    contents = GoogleProvider._contents(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ],
        None,
        None,
    )

    assert len(contents[0].parts) == 1
    assert contents[0].parts[0].function_call.name == "lookup"


def test_google_rejects_malformed_function_tool_choice():
    with pytest.raises(ValueError, match="function.name"):
        GoogleProvider._convert_tool_choice({"type": "function", "function": {}})


def test_anthropic_merges_parallel_tool_results_into_one_user_turn():
    converted = AnthropicProvider._convert_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "lookup_order", "arguments": "{}"}},
                    {"id": "call-2", "function": {"name": "lookup_customer", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"status":"paid"}'},
            {"role": "tool", "tool_call_id": "call-2", "content": '{"name":"Ada"}'},
        ]
    )

    assert [message["role"] for message in converted] == ["assistant", "user"]
    assert [block["tool_use_id"] for block in converted[1]["content"]] == ["call-1", "call-2"]


def test_volcengine_uses_ark_endpoint_and_ak_sk(monkeypatch):
    settings = SimpleNamespace(
        volc_access_key="access",
        volc_secret_key="secret",
        volc_base_url="https://ark.example/v3",
    )
    monkeypatch.setitem(VolcengineProvider.__init__.__globals__, "get_settings", lambda: settings)

    provider = VolcengineProvider("doubao-model")

    assert provider._client_options() == {
        "base_url": "https://ark.example/v3",
        "ak": "access",
        "sk": "secret",
    }
    assert provider._build_chat_request("hello", None, TOOLS, 0.4, False) == {
        "model": "doubao-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.4,
        "stream": False,
        "tools": TOOLS,
    }


def _responses_provider() -> OpenAICompatibleProvider:
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "responses-model"
    provider._provider = "openai"
    provider._protocol = "responses"
    return provider


def test_responses_request_converts_chat_tools_and_sets_strict_default():
    provider = _responses_provider()

    assert provider._build_responses_request("hello", "system", TOOLS, 0.4, True) == {
        "model": "responses-model",
        "input": "hello",
        "instructions": "system",
        "temperature": 0.4,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "lookup_order",
                "description": "查询订单",
                "parameters": TOOLS[0]["function"]["parameters"],
                "strict": False,
            }
        ],
    }


def test_responses_request_omits_implicit_temperature():
    provider = _responses_provider()

    request = provider._build_responses_request("hello", stream=False)

    assert "temperature" not in request


def test_responses_request_preserves_configured_temperature():
    provider = _responses_provider()
    provider._options = {"temperature": 0.2}

    request = provider._build_responses_request("hello", stream=False)

    assert request["temperature"] == 0.2


def _tool_call_history():
    return [
        {"role": "user", "content": "查订单"},
        {
            "role": "assistant",
            "content": "我来查询。",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup_order",
                        "arguments": '{"order_id":"A-1"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "lookup_customer",
                        "arguments": {"customer_id": "C-1"},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"status":"paid"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": {"name": "Ada"},
        },
    ]


def test_responses_input_converts_chat_tool_messages_and_parallel_calls():
    converted = normalize_responses_input(None, None, _tool_call_history())

    assert converted == [
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": "我来查询。"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup_order",
            "arguments": '{"order_id":"A-1"}',
        },
        {
            "type": "function_call",
            "call_id": "call-2",
            "name": "lookup_customer",
            "arguments": '{"customer_id":"C-1"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"status":"paid"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": '{"name":"Ada"}',
        },
    ]


def test_responses_input_keeps_prompt_string_and_validates_native_items():
    assert normalize_responses_input("hello", "system", None) == "hello"
    assert normalize_responses_input(
        None,
        None,
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "done",
            },
        ],
    ) == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "done",
        },
    ]


def test_responses_input_converts_chat_multimodal_content_blocks():
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请阅读图片和文件。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.png",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "file",
                        "file": {
                            "file_id": "file-1",
                            "filename": "说明.txt",
                        },
                    },
                ],
            }
        ],
    )

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "请阅读图片和文件。"},
                {
                    "type": "input_image",
                    "detail": "high",
                    "image_url": "https://example.com/image.png",
                },
                {
                    "type": "input_file",
                    "file_id": "file-1",
                    "filename": "说明.txt",
                },
            ],
        }
    ]

    from openai.types.responses.response_input_item_param import ResponseInputItemParam
    from pydantic import TypeAdapter

    TypeAdapter(list[ResponseInputItemParam]).validate_python(converted)


def test_responses_input_keeps_native_multimodal_items_in_sdk_shape():
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_image", "file_id": "file-2"},
                    {"type": "input_file", "file_url": "https://example.com/a.pdf"},
                ],
            }
        ],
    )

    assert converted[0]["content"] == [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "detail": "auto", "file_id": "file-2"},
        {"type": "input_file", "file_url": "https://example.com/a.pdf"},
    ]


def test_responses_input_converts_previous_output_text_for_follow_up_messages():
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "上一次答复"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {"type": "text", "text": "工具输出"},
                    {"type": "file", "url": "https://example.com/result.txt"},
                ],
            },
        ],
    )

    assert converted == [
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "上一次答复"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [
                {"type": "input_text", "text": "工具输出"},
                {
                    "type": "input_file",
                    "file_url": "https://example.com/result.txt",
                },
            ],
        },
    ]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "audio_url", "audio_url": "https://example.com/audio.wav"}],
        [{"type": "video_url", "video_url": "https://example.com/video.mp4"}],
        [{"type": "unknown_part", "value": "x"}],
        [{"type": "image_url", "image_url": {}}],
        [{"type": "input_file"}],
    ],
)
def test_responses_input_rejects_unsupported_or_incomplete_content_blocks(content):
    with pytest.raises(ValueError, match="(不受 Responses SDK 支持|缺少)"):
        normalize_responses_input(
            None,
            None,
            [{"role": "user", "content": content}],
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "missing id"}],
        [{"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}],
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{"}}],
            }
        ],
        [{"type": "function_call", "id": "item-1", "name": "lookup", "arguments": "{}"}],
    ],
)
def test_responses_input_rejects_unlinked_or_malformed_tool_items(messages):
    with pytest.raises(ValueError, match="(call_id|有效 JSON|函数名)"):
        normalize_responses_input(None, None, messages)


def test_openai_responses_request_converts_tool_history():
    provider = _responses_provider()

    request = provider._build_responses_request(
        messages=_tool_call_history(),
        stream=False,
    )

    assert request["input"][2]["type"] == "function_call"
    assert request["input"][2]["call_id"] == "call-1"
    assert request["input"][4] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"status":"paid"}',
    }


def test_openai_responses_token_count_converts_tool_history():
    provider = _responses_provider()
    params = provider._build_token_count_request(
        CompletionRequest(messages=_tool_call_history(), system_prompt=None)
    )

    assert params["input"][2]["type"] == "function_call"
    assert params["input"][4]["type"] == "function_call_output"


def test_openai_responses_token_count_converts_multimodal_content():
    provider = _responses_provider()
    params = provider._build_token_count_request(
        CompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "看图"},
                        {"type": "image_url", "image_url": "https://example.com/a.png"},
                    ],
                }
            ],
            system_prompt=None,
        )
    )

    assert params["input"][0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {
            "type": "input_image",
            "detail": "auto",
            "image_url": "https://example.com/a.png",
        },
    ]


def test_openai_responses_request_converts_multimodal_content():
    provider = _responses_provider()
    request = provider._build_responses_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {"type": "image_url", "image_url": "https://example.com/a.png"},
                ],
            }
        ],
        stream=False,
    )

    assert request["input"][0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {
            "type": "input_image",
            "detail": "auto",
            "image_url": "https://example.com/a.png",
        },
    ]


def test_ark_responses_request_converts_tool_history():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"server_verified_protocols": ["responses"]}

    request = provider._build_responses_request(
        CompletionRequest(messages=_tool_call_history(), system_prompt=None, stream=False)
    )

    assert request["input"][2]["type"] == "function_call"
    assert request["input"][2]["call_id"] == "call-1"
    assert request["input"][4]["type"] == "function_call_output"


def test_ark_responses_request_omits_implicit_temperature():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"server_verified_protocols": ["responses"]}

    request = provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))

    assert "temperature" not in request


def test_ark_responses_request_preserves_configured_temperature():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {
        "server_verified_protocols": ["responses"],
        "temperature": 0.2,
    }

    request = provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))

    assert request["temperature"] == 0.2


def test_ark_responses_request_converts_multimodal_content():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"server_verified_protocols": ["responses"]}

    request = provider._build_responses_request(
        CompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "看图"},
                        {"type": "image_url", "image_url": "https://example.com/a.png"},
                    ],
                }
            ],
            system_prompt=None,
            stream=False,
        )
    )

    assert request["input"][0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {
            "type": "input_image",
            "detail": "auto",
            "image_url": "https://example.com/a.png",
        },
    ]

    request = provider._build_responses_request(
        CompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "audio_url": "https://example.com/a.wav"},
                        {"type": "input_video", "file_id": "file-video", "fps": 1.5},
                    ],
                }
            ],
            system_prompt=None,
            stream=False,
        )
    )
    assert request["input"][0]["content"] == [
        {"type": "input_audio", "audio_url": "https://example.com/a.wav"},
        {"type": "input_video", "file_id": "file-video", "fps": 1.5},
    ]


def test_ark_responses_input_supports_audio_video_and_image_pixel_limit():
    converted = normalize_responses_input(
        None,
        None,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/a.wav"},
                    },
                    {
                        "type": "video_url",
                        "video_url": "https://example.com/v.mp4",
                        "fps": 2.0,
                    },
                    {
                        "type": "image_url",
                        "image_url": "https://example.com/i.png",
                        "detail": "high",
                        "image_pixel_limit": {
                            "min_pixels": 1024,
                            "max_pixels": 4096,
                        },
                    },
                ],
            }
        ],
        provider="ark",
    )

    assert converted[0]["content"] == [
        {"type": "input_audio", "audio_url": "https://example.com/a.wav"},
        {
            "type": "input_video",
            "video_url": "https://example.com/v.mp4",
            "fps": 2.0,
        },
        {
            "type": "input_image",
            "detail": "high",
            "image_url": "https://example.com/i.png",
            "image_pixel_limit": {"min_pixels": 1024, "max_pixels": 4096},
        },
    ]

    from pydantic import TypeAdapter
    from volcenginesdkarkruntime.types.responses.response_input_param import (
        ResponseInputParam,
    )

    TypeAdapter(ResponseInputParam).validate_python(converted)


def test_openai_responses_rejects_ark_only_multimodal_content():
    with pytest.raises(ValueError, match="不受 Responses SDK 支持"):
        normalize_responses_input(
            None,
            None,
            [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "audio_url": "https://example.com/a.wav"}],
                }
            ],
        )


def test_responses_request_converts_chat_tool_choice_to_flat_function_shape():
    provider = _responses_provider()

    request = provider._build_responses_request(
        "hello",
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "lookup_order"}},
        stream=False,
    )

    assert request["tool_choice"] == {"type": "function", "name": "lookup_order"}


def test_responses_request_preserves_native_tool_choice_forms():
    provider = _responses_provider()

    assert (
        provider._build_responses_request("hello", tool_choice="required", stream=False)[
            "tool_choice"
        ]
        == "required"
    )
    assert provider._build_responses_request(
        "hello", tool_choice={"type": "allowed_tools", "mode": "auto"}, stream=False
    )["tool_choice"] == {"type": "allowed_tools", "mode": "auto"}


def test_responses_rejects_malformed_tool_definitions():
    provider = _responses_provider()
    with pytest.raises(ValueError, match="工具定义"):
        provider._build_responses_request("hello", tools=[None], stream=False)
    with pytest.raises(ValueError, match="function"):
        provider._build_responses_request(
            "hello", tools=[{"type": "function", "function": "bad"}], stream=False
        )


def test_ark_model_sampling_options_merge_into_extra_body():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {
        "server_verified_protocols": ["responses"],
        "top_k": 20,
        "seed": 7,
        "extra_body": {"vendor_flag": True},
    }
    request = provider._build_responses_request(
        CompletionRequest(prompt="hello", stream=False, top_k=30)
    )
    assert request["extra_body"] == {"vendor_flag": True, "top_k": 30, "seed": 7}
    assert "server_verified_protocols" not in request


def test_openai_responses_rejects_conflicting_model_length_aliases():
    provider = _responses_provider()
    provider._options = {"max_tokens": 64, "max_completion_tokens": 128}

    with pytest.raises(ValueError, match="max_tokens.*max_completion_tokens"):
        provider._build_responses_request("hello", stream=False)


def test_ark_responses_rejects_conflicting_model_length_aliases():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {
        "max_tokens": 64,
        "max_completion_tokens": 128,
    }

    with pytest.raises(ValueError, match="max_tokens.*max_completion_tokens"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))


def test_ark_extra_body_sampling_conflicts_are_explicit():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"extra_body": {"seed": 1}}
    with pytest.raises(ValueError, match="seed.*重复"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False, seed=2))


def test_responses_sync_non_stream_extracts_output_text():
    provider = _responses_provider()
    calls = []

    class Responses:
        def create(self, **request):
            calls.append(request)
            return SimpleNamespace(status="completed", output_text="答复")

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    assert list(
        provider._invoke_responses({"model": "responses-model", "input": "hello", "stream": False})
    ) == ["答复"]
    assert calls == [{"model": "responses-model", "input": "hello", "stream": False}]


def test_responses_complete_uses_response_status_as_finish_reason():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            return SimpleNamespace(
                id="resp-1",
                status="completed",
                output_text="答复",
                usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5),
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    result = provider.complete(CompletionRequest(prompt="hello"))

    assert result.text == "答复"
    assert result.finish_reason == "completed"
    assert result.response_id == "resp-1"
    assert result.usage["total_tokens"] == 5


def test_openai_responses_complete_and_stream_paths_convert_tool_history():
    history = _tool_call_history()
    provider = _responses_provider()
    calls = []

    class Responses:
        def create(self, **request):
            calls.append(request)
            if request["stream"]:
                return iter(
                    [
                        SimpleNamespace(
                            type="response.completed",
                            response=SimpleNamespace(status="completed"),
                        )
                    ]
                )
            return SimpleNamespace(status="completed", output_text="答复")

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    assert provider.complete(CompletionRequest(messages=history, system_prompt=None)).text == "答复"
    list(provider.stream_events(CompletionRequest(messages=history, system_prompt=None)))

    assert calls[0]["input"][2]["type"] == "function_call"
    assert calls[0]["input"][4]["type"] == "function_call_output"
    assert calls[1]["input"] == calls[0]["input"]


def test_openai_responses_async_complete_and_stream_paths_convert_tool_history():
    history = _tool_call_history()
    provider = _responses_provider()
    calls = []

    class Responses:
        async def create(self, **request):
            calls.append(request)
            if request["stream"]:

                class Stream:
                    def __aiter__(self):
                        self._events = iter(
                            [
                                SimpleNamespace(
                                    type="response.completed",
                                    response=SimpleNamespace(status="completed"),
                                )
                            ]
                        )
                        return self

                    async def __anext__(self):
                        try:
                            return next(self._events)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc

                return Stream()
            return SimpleNamespace(status="completed", output_text="异步答复")

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        result = await provider.acomplete(CompletionRequest(messages=history, system_prompt=None))
        events = [
            event
            async for event in provider.astream_events(
                CompletionRequest(messages=history, system_prompt=None)
            )
        ]
        return result, events

    result, events = asyncio.run(collect())

    assert result.text == "异步答复"
    assert events[-1].type == "finish"
    assert calls[0]["input"][2]["type"] == "function_call"
    assert calls[1]["input"][4]["type"] == "function_call_output"


def test_openai_responses_async_stream_drains_after_completion():
    provider = _responses_provider()
    streams = []

    class Responses:
        async def create(self, **_request):
            class Stream:
                def __init__(self):
                    self._events = iter(
                        [
                            SimpleNamespace(
                                type="response.completed",
                                response=SimpleNamespace(status="completed"),
                            ),
                            SimpleNamespace(type="response.output_text.done"),
                        ]
                    )
                    self.exhausted = False
                    self.closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        self.exhausted = True
                        raise StopAsyncIteration from exc

                async def close(self):
                    self.closed = True

            stream = Stream()
            streams.append(stream)
            return stream

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    events = asyncio.run(collect())

    assert [event.type for event in events] == ["finish"]
    assert streams[0].exhausted is True
    assert streams[0].closed is True


def test_ark_responses_complete_and_stream_paths_convert_tool_history():
    history = _tool_call_history()
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = []

    class Responses:
        def create(self, **request):
            calls.append(request)
            if request["stream"]:
                return iter(
                    [
                        SimpleNamespace(
                            type="response.completed",
                            response=SimpleNamespace(status="completed"),
                        )
                    ]
                )
            return SimpleNamespace(status="completed", output_text="答复")

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    assert provider.complete(CompletionRequest(messages=history, system_prompt=None)).text == "答复"
    list(provider.stream_events(CompletionRequest(messages=history, system_prompt=None)))

    assert calls[0]["input"][2]["type"] == "function_call"
    assert calls[0]["input"][4]["type"] == "function_call_output"


def test_ark_responses_async_complete_and_stream_paths_convert_tool_history():
    history = _tool_call_history()
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    calls = []

    class Responses:
        async def create(self, **request):
            calls.append(request)
            if request["stream"]:

                class Stream:
                    def __aiter__(self):
                        self._events = iter(
                            [
                                SimpleNamespace(
                                    type="response.completed",
                                    response=SimpleNamespace(status="completed"),
                                )
                            ]
                        )
                        return self

                    async def __anext__(self):
                        try:
                            return next(self._events)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc

                return Stream()
            return SimpleNamespace(status="completed", output_text="异步答复")

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        result = await provider.acomplete(CompletionRequest(messages=history, system_prompt=None))
        events = [
            event
            async for event in provider.astream_events(
                CompletionRequest(messages=history, system_prompt=None)
            )
        ]
        return result, events

    result, events = asyncio.run(collect())

    assert result.text == "异步答复"
    assert events[-1].type == "finish"
    assert calls[0]["input"][2]["type"] == "function_call"
    assert calls[1]["input"][4]["type"] == "function_call_output"


def test_openai_chat_stream_events_include_tool_usage_and_finish():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider._model_name = "chat-model"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="答", tool_calls=None),
                                finish_reason=None,
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            id="call-1",
                                            index=0,
                                            function=SimpleNamespace(
                                                name="lookup", arguments='{"id":'
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            id=None,
                                            index=0,
                                            function=SimpleNamespace(name=None, arguments="1}"),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None), finish_reason="tool_calls"
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == [
        "text_delta",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_completed",
        "finish",
    ]
    assert events[1].tool_call["name"] == "lookup"
    assert events[3].tool_call["arguments"] == '{"id":1}'
    assert events[4].finish_reason == "tool_calls"
    assert events[4].usage["total_tokens"] == 5


def test_openai_chat_stream_events_preserve_text_and_finish_from_same_chunk():
    events = OpenAICompatibleProvider._chat_stream_events(
        SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="答", reasoning_content=None, refusal=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
        )
    )

    assert [event.type for event in events] == ["text_delta", "finish"]
    assert events[0].text == "答"
    assert events[1].finish_reason == "stop"


def test_openai_chat_stream_error_event_is_explicit():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "chat-model"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="error",
                        error=SimpleNamespace(message="chat upstream failed", status=400),
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    with pytest.raises(RuntimeError, match="chat upstream failed"):
        list(provider.stream_events(CompletionRequest(prompt="hello")))

    with pytest.raises(RuntimeError, match="chat upstream failed"):
        OpenAICompatibleProvider._chat_stream_events(
            SimpleNamespace(error=SimpleNamespace(message="chat upstream failed"))
        )


def test_ark_chat_stream_error_event_is_explicit():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._provider = "volcengine"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="error",
                        error=SimpleNamespace(message="ark upstream failed", status=400),
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    with pytest.raises(RuntimeError, match="ark upstream failed"):
        list(provider.stream_events(CompletionRequest(prompt="hello")))

    with pytest.raises(RuntimeError, match="ark upstream failed"):
        VolcengineProvider._chat_stream_events(
            SimpleNamespace(error=SimpleNamespace(message="ark upstream failed"))
        )


def test_openai_async_chat_stream_error_event_is_explicit():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "chat-model"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        async def create(self, **_request):
            class Stream:
                def __aiter__(self):
                    self._events = iter(
                        [
                            SimpleNamespace(
                                type="error",
                                error=SimpleNamespace(
                                    message="async chat upstream failed", status=400
                                ),
                            )
                        ]
                    )
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        raise StopAsyncIteration from exc

            return Stream()

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    with pytest.raises(RuntimeError, match="async chat upstream failed"):
        asyncio.run(collect())


def test_ark_async_chat_stream_error_event_is_explicit():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "chat_completions"
    provider._options = {}

    class Completions:
        async def create(self, **_request):
            class Stream:
                def __aiter__(self):
                    self._events = iter(
                        [
                            SimpleNamespace(
                                type="error",
                                error=SimpleNamespace(
                                    message="async ark upstream failed", status=400
                                ),
                            )
                        ]
                    )
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        raise StopAsyncIteration from exc

            return Stream()

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    with pytest.raises(RuntimeError, match="async ark upstream failed"):
        asyncio.run(collect())


def test_ark_chat_stream_events_preserve_text_and_finish_from_same_chunk():
    events = VolcengineProvider._chat_stream_events(
        SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="答", reasoning_content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
    )

    assert [event.type for event in events] == ["text_delta", "finish"]


def test_openai_chat_stream_events_propagate_refusal_delta():
    events = OpenAICompatibleProvider._chat_stream_events(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, refusal="不能处理"),
                    finish_reason=None,
                )
            ]
        )
    )

    assert len(events) == 1
    assert events[0].type == "refusal_delta"
    assert events[0].refusal == "不能处理"


def test_openai_chat_stream_events_preserve_parallel_tool_calls():
    chunk = SimpleNamespace(
        id="chat-1",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            index=0,
                            function=SimpleNamespace(name="lookup", arguments='{"id":'),
                        ),
                        SimpleNamespace(
                            id="call-2",
                            index=1,
                            function=SimpleNamespace(name="refund", arguments='{"order":'),
                        ),
                    ],
                ),
                finish_reason=None,
            )
        ],
    )

    events = OpenAICompatibleProvider._chat_stream_events(chunk)

    assert [event.type for event in events] == ["tool_call_delta", "tool_call_delta"]
    assert [event.tool_call["id"] for event in events] == ["call-1", "call-2"]
    assert [event.tool_call["index"] for event in events] == [0, 1]


def test_openai_responses_stream_exposes_tool_delta_and_completed_events():
    assert (
        OpenAICompatibleProvider._responses_stream_event(
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="function_call", call_id="call-1", name="lookup"),
            )
        )
        is None
    )

    delta = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="item-1",
            output_index=2,
            delta='{"id":',
        )
    )
    completed = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="item-1",
            output_index=2,
            item=SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="lookup",
                arguments='{"id":1}',
            ),
        )
    )

    assert delta.type == "tool_call_delta"
    assert delta.tool_call["index"] == 2
    assert completed.type == "tool_call_completed"
    assert completed.tool_call["id"] == "call-1"
    assert completed.tool_call["arguments"] == '{"id":1}'


def test_openai_responses_stream_deduplicates_tool_completion_events():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="lookup",
                arguments='{"id":1}',
            )
            return iter(
                [
                    SimpleNamespace(type="response.output_item.done", item=item, output_index=0),
                    SimpleNamespace(
                        type="response.function_call_arguments.done",
                        item=item,
                        output_index=0,
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == ["tool_call_completed", "finish"]


def test_openai_responses_stream_flushes_tool_delta_before_finish():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        item_id="item-1",
                        output_index=0,
                        name="lookup",
                        delta='{"id":1}',
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == ["tool_call_delta", "tool_call_completed", "finish"]


def test_ark_responses_stream_flushes_tool_delta_before_finish():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        item_id="item-1",
                        output_index=0,
                        name="lookup",
                        delta='{"id":1}',
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == ["tool_call_delta", "tool_call_completed", "finish"]


def test_ark_responses_stream_deduplicates_tool_completion_events():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="lookup",
                arguments='{"id":1}',
            )
            return iter(
                [
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        item_id="item-1",
                        output_index=0,
                        delta='{"id":',
                    ),
                    SimpleNamespace(
                        type="response.function_call_arguments.done",
                        item_id="item-1",
                        output_index=0,
                        name="lookup",
                        arguments='{"id":1}',
                    ),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=item,
                        output_index=0,
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == ["tool_call_delta", "tool_call_completed", "finish"]
    assert events[1].tool_call["id"] == "call-1"
    assert events[1].tool_call["arguments"] == '{"id":1}'


def test_openai_responses_arguments_done_is_complete_without_output_item():
    event = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="item-1",
            output_index=2,
            name="lookup",
            arguments='{"id":1}',
        )
    )

    assert event is not None
    assert event.type == "tool_call_completed"
    assert event.tool_call == {
        "id": "item-1",
        "index": 2,
        "type": "function_call",
        "name": "lookup",
        "arguments": '{"id":1}',
    }


def test_openai_responses_custom_tool_stream_keeps_added_metadata_and_raw_input():
    metadata = {}
    added = SimpleNamespace(
        type="response.output_item.added",
        output_index=1,
        item=SimpleNamespace(
            type="custom_tool_call",
            id="item-1",
            call_id="call-1",
            name="run_command",
            input="",
        ),
    )

    assert OpenAICompatibleProvider._responses_stream_event(added, metadata) is None
    delta = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.custom_tool_call_input.delta",
            item_id="item-1",
            output_index=1,
            delta="echo ",
        ),
        metadata,
    )
    completed = OpenAICompatibleProvider._responses_stream_event(
        SimpleNamespace(
            type="response.custom_tool_call_input.done",
            item_id="item-1",
            output_index=1,
            input="hello world",
        ),
        metadata,
    )

    # ``id`` 统一取调用标识符（call_id），与非流式路径及合并结果一致：
    # ``item_id`` 是 Responses 的输出项 ID，取它会让同一轮工具调用在 delta 与
    # completed 事件里得到不同的 id，调用方按 ``tool_call["id"]`` 回填
    # ``role=tool`` 时就会配不上。
    assert delta is not None and delta.tool_call == {
        "id": "call-1",
        "call_id": "call-1",
        "index": 1,
        "type": "custom_tool_call",
        "name": "run_command",
        "arguments": "echo ",
    }
    assert completed is not None and completed.tool_call == {
        "id": "call-1",
        "index": 1,
        "type": "custom_tool_call",
        "name": "run_command",
        "arguments": "hello world",
        "call_id": "call-1",
    }


def test_openai_responses_custom_tool_stream_events_emit_non_json_completion():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            item = SimpleNamespace(
                type="custom_tool_call",
                id="item-1",
                call_id="call-1",
                name="run_command",
                input="echo hello",
            )
            return iter(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="custom_tool_call",
                            id="item-1",
                            call_id="call-1",
                            name="run_command",
                            input="",
                        ),
                    ),
                    SimpleNamespace(
                        type="response.custom_tool_call_input.delta",
                        item_id="item-1",
                        output_index=0,
                        delta="echo ",
                    ),
                    SimpleNamespace(
                        type="response.custom_tool_call_input.done",
                        item_id="item-1",
                        output_index=0,
                        input="echo hello",
                    ),
                    SimpleNamespace(
                        type="response.output_item.done",
                        output_index=0,
                        item=item,
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == [
        "tool_call_delta",
        "tool_call_completed",
        "finish",
    ]
    assert events[0].tool_call["name"] == "run_command"
    assert events[1].tool_call["arguments"] == "echo hello"
    assert events[1].tool_call["type"] == "custom_tool_call"


def test_openai_responses_custom_tool_empty_input_is_valid():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="custom_tool_call",
                            id="item-1",
                            call_id="call-1",
                            name="ping",
                            input="",
                        ),
                    ),
                    SimpleNamespace(
                        type="response.custom_tool_call_input.done",
                        item_id="item-1",
                        output_index=0,
                        input="",
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(status="completed"),
                    ),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert events[0].type == "tool_call_completed"
    assert events[0].tool_call["arguments"] == ""


def test_openai_responses_custom_tool_async_stream_preserves_metadata():
    provider = _responses_provider()

    class Responses:
        async def create(self, **_request):
            events = [
                SimpleNamespace(
                    type="response.output_item.added",
                    output_index=0,
                    item=SimpleNamespace(
                        type="custom_tool_call",
                        id="item-1",
                        call_id="call-1",
                        name="run_command",
                        input="",
                    ),
                ),
                SimpleNamespace(
                    type="response.custom_tool_call_input.delta",
                    item_id="item-1",
                    output_index=0,
                    delta="echo hello",
                ),
                SimpleNamespace(
                    type="response.custom_tool_call_input.done",
                    item_id="item-1",
                    output_index=0,
                    input="echo hello",
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(status="completed"),
                ),
            ]

            class Stream:
                def __aiter__(self):
                    self._events = iter(events)
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        raise StopAsyncIteration from exc

            return Stream()

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    events = asyncio.run(collect())
    assert [event.type for event in events] == [
        "tool_call_delta",
        "tool_call_completed",
        "finish",
    ]
    assert events[1].tool_call["type"] == "custom_tool_call"
    assert events[1].tool_call["name"] == "run_command"


def test_google_response_text_excludes_thought_parts():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="思考", thought=True),
                        SimpleNamespace(text="答案", thought=False),
                    ]
                )
            )
        ]
    )

    assert GoogleProvider._response_text(response) == "答案"


def test_anthropic_tool_start_arguments_use_string_contract():
    event = AnthropicProvider._stream_event(
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", id="toolu-1", name="lookup", input={"id": 1}
            ),
        ),
        "msg-1",
    )

    assert event is not None
    assert event.tool_call["arguments"] == '{"id":1}'


def test_anthropic_user_profile_adds_required_beta_header():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    params = provider._build_message_params(prompt="hi", user="profile-1")

    assert params["user_profile_id"] == "profile-1"
    assert params["extra_headers"]["anthropic-beta"] == "user-profiles-2026-08-18"


def test_anthropic_stream_deduplicates_message_finish_events():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    class StreamManager:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        type="message_delta",
                        delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=1),
                    ),
                    SimpleNamespace(type="message_stop"),
                ]
            )

    class Messages:
        def stream(self, **_kwargs):
            return StreamManager()

    provider._get_client = lambda: SimpleNamespace(messages=Messages())
    events = list(provider.stream_events(CompletionRequest(prompt="hello")))

    assert [event.type for event in events] == ["finish"]


def test_anthropic_non_refusal_stop_details_do_not_create_refusal_result():
    response = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        stop_details=SimpleNamespace(
            type="max_tokens",
            explanation="normal completion detail",
        ),
    )

    result = AnthropicProvider._extract_result(response)

    assert result.refusal is None
    assert result.finish_reason == "end_turn"


def test_anthropic_non_refusal_stream_stop_details_emit_finish():
    event = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(
            stop_reason="end_turn",
            stop_details=SimpleNamespace(
                type="max_tokens",
                explanation="normal completion detail",
            ),
        ),
        usage=None,
    )

    converted = AnthropicProvider._stream_event(event, "msg-1")

    assert converted is not None
    assert converted.type == "finish"
    assert converted.finish_reason == "end_turn"
    assert converted.refusal is None


def test_anthropic_invoke_retries_stream_context_before_first_event(monkeypatch):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    attempts = []

    class StreamManager:
        def __init__(self, fail_on_enter):
            self.fail_on_enter = fail_on_enter
            self.text_stream = iter(["答复"])

        def __enter__(self):
            if self.fail_on_enter:
                raise RuntimeError("transient stream setup failure")
            return self

        def __exit__(self, *_args):
            return False

    class Messages:
        def stream(self, **_kwargs):
            attempts.append(True)
            return StreamManager(len(attempts) == 1)

    def fake_retry(call):
        last_error = None
        for _ in range(2):
            try:
                return call()
            except RuntimeError as error:
                last_error = error
        raise last_error

    monkeypatch.setattr(model_provider_module, "retry_sync_call", fake_retry)
    provider._get_client = lambda: SimpleNamespace(messages=Messages())

    assert list(provider.invoke("hello", stream=True)) == ["答复"]
    assert len(attempts) == 2


def test_anthropic_ainvoke_retries_async_stream_context_before_first_event(monkeypatch):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    attempts = []

    class AsyncStreamManager:
        def __init__(self, fail_on_enter):
            self.fail_on_enter = fail_on_enter
            self.text_stream = ["异步答复"]

        async def __aenter__(self):
            if self.fail_on_enter:
                raise RuntimeError("transient async stream setup failure")
            return self

        async def __aexit__(self, *_args):
            return False

    class Messages:
        def stream(self, **_kwargs):
            attempts.append(True)
            return AsyncStreamManager(len(attempts) == 1)

    async def fake_retry(call):
        last_error = None
        for _ in range(2):
            try:
                result = call()
                return await result if inspect.isawaitable(result) else result
            except RuntimeError as error:
                last_error = error
        raise last_error

    monkeypatch.setattr(model_provider_module, "retry_async_call", fake_retry)
    provider._get_aclient = lambda: SimpleNamespace(messages=Messages())

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", stream=True)]

    assert asyncio.run(consume()) == ["异步答复"]
    assert len(attempts) == 2


def test_anthropic_invoke_non_stream_refusal_is_explicit_error():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    response = SimpleNamespace(
        content=[SimpleNamespace(type="refusal", reason="不能回答")],
        stop_reason="refusal",
    )
    provider._get_client = lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_kwargs: response)
    )

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        list(provider.invoke("hello", stream=False))


def test_anthropic_invoke_stream_refusal_is_explicit_error():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    class StreamManager:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        type="content_block_start",
                        content_block=SimpleNamespace(type="refusal", reason="不能回答"),
                    ),
                    SimpleNamespace(type="message_stop"),
                ]
            )

        text_stream = iter(())

    provider._get_client = lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: StreamManager())
    )

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        list(provider.invoke("hello", stream=True))


def test_anthropic_text_stream_detects_refusal_before_consuming_text_stream():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    class RealLikeStream:
        current_message_snapshot = property(
            lambda _self: SimpleNamespace(
                content=[SimpleNamespace(type="refusal", reason="不能回答")],
                stop_reason="refusal",
            )
        )

        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        type="content_block_start",
                        content_block=SimpleNamespace(type="refusal", reason="不能回答"),
                    )
                ]
            )

        text_stream = iter(["这段文本不应先返回"])

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        next(AnthropicProvider._iter_invoke_stream_text(RealLikeStream()))


def test_anthropic_text_stream_checks_snapshot_before_yielding_text():
    class SnapshotStream:
        current_message_snapshot = property(
            lambda _self: SimpleNamespace(
                content=[SimpleNamespace(type="refusal", reason="拒答")],
                stop_reason="refusal",
            )
        )
        text_stream = iter(["不应输出"])

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        next(AnthropicProvider._iter_invoke_stream_text(SnapshotStream()))


def test_anthropic_stream_error_preserves_nested_error_message():
    event = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "error": {"message": "nested request detail"},
        },
    }

    with pytest.raises(RuntimeError, match="nested request detail"):
        AnthropicProvider._stream_event(event)


def test_openai_responses_non_stream_reasoning_summary_list_is_preserved():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[
                    SimpleNamespace(type="summary_text", text="先分析"),
                    SimpleNamespace(text="后结论"),
                ],
            )
        ]
    )

    assert OpenAICompatibleProvider._extract_reasoning(response) == "先分析后结论"


def test_ark_responses_non_stream_reasoning_summary_list_is_preserved():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(type="summary_text", text="Ark 思考")],
            )
        ],
        choices=[],
        output_text="",
        usage=None,
        status="completed",
        id="resp-1",
    )

    assert VolcengineProvider._extract_result(response).reasoning == "Ark 思考"


def test_anthropic_ainvoke_non_stream_refusal_is_explicit_error():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    response = SimpleNamespace(
        content=[SimpleNamespace(type="refusal", reason="异步不能回答")],
        stop_reason="refusal",
    )

    class Messages:
        async def create(self, **_kwargs):
            return response

    provider._get_aclient = lambda: SimpleNamespace(messages=Messages())

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", stream=False)]

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        asyncio.run(consume())


def test_anthropic_ainvoke_stream_refusal_is_explicit_error():
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}

    class AsyncStreamManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def __aiter__(self):
            async def events():
                yield SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="refusal", reason="异步不能回答"),
                )
                yield SimpleNamespace(type="message_stop")

            return events()

        text_stream = ()

    provider._get_aclient = lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: AsyncStreamManager())
    )

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", stream=True)]

    with pytest.raises(RuntimeError, match="拒绝|拒答"):
        asyncio.run(consume())


def test_responses_invoke_rejects_tools_for_text_only_interface():
    provider = _responses_provider()
    provider._get_client = lambda: pytest.fail("Responses invoke 不应初始化客户端")

    with pytest.raises(ValueError, match="tools|工具"):
        list(provider.invoke("hello", tools=TOOLS))


def test_responses_ainvoke_rejects_tools_for_text_only_interface():
    provider = _responses_provider()
    provider._get_aclient = lambda: pytest.fail("Responses ainvoke 不应初始化客户端")

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", tools=TOOLS)]

    with pytest.raises(ValueError, match="tools|工具"):
        asyncio.run(consume())


def test_chat_invoke_accepts_mapping_shaped_sdk_response():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = {"choices": [{"message": {"content": "字典答复"}}]}
    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: response))
    )

    assert list(provider.invoke("hello", stream=False)) == ["字典答复"]


def test_chat_ainvoke_accepts_mapping_shaped_sdk_response():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = {"choices": [{"message": {"content": "异步字典答复"}}]}

    class Completions:
        async def create(self, **_kwargs):
            return response

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", stream=False)]

    assert asyncio.run(consume()) == ["异步字典答复"]


def test_openai_chat_business_error_is_explicit_for_complete_and_invoke():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = SimpleNamespace(status="435", msg="Model not support", choices=None)
    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: response))
    )

    with pytest.raises(RuntimeError, match="435.*Model not support"):
        provider.complete(CompletionRequest(prompt="hello"))
    with pytest.raises(RuntimeError, match="435.*Model not support"):
        list(provider.invoke("hello", stream=False))


def test_openai_chat_business_error_is_explicit_for_async_complete_and_invoke():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = SimpleNamespace(status="435", msg="Model not support", choices=None)

    class Completions:
        async def create(self, **_kwargs):
            return response

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    with pytest.raises(RuntimeError, match="435.*Model not support"):
        asyncio.run(provider.acomplete(CompletionRequest(prompt="hello")))

    async def consume():
        return [item async for item in provider.ainvoke("hello", stream=False)]

    with pytest.raises(RuntimeError, match="435.*Model not support"):
        asyncio.run(consume())


def test_openai_chat_business_error_is_explicit_for_sync_and_async_streams():
    provider = object.__new__(OpenAICompatibleProvider)
    provider._model_name = "demo"
    provider._provider = "openai"
    provider._protocol = "chat_completions"
    provider._options = {}
    response = SimpleNamespace(status="435", msg="Model not support", choices=None)
    provider._get_client = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: iter([response])))
    )

    with pytest.raises(RuntimeError, match="435.*Model not support"):
        list(provider.stream_events(CompletionRequest(prompt="hello")))

    class AsyncCompletions:
        async def create(self, **_kwargs):
            class Stream:
                def __aiter__(self):
                    self._events = iter([response])
                    return self

                async def __anext__(self):
                    try:
                        return next(self._events)
                    except StopIteration as exc:
                        raise StopAsyncIteration from exc

            return Stream()

    provider._get_aclient = lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=AsyncCompletions())
    )

    async def collect():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hello"))]

    with pytest.raises(RuntimeError, match="435.*Model not support"):
        asyncio.run(collect())


def test_ark_responses_invoke_rejects_tools_for_text_only_interface():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_client = lambda: pytest.fail("Ark Responses invoke 不应初始化客户端")

    with pytest.raises(ValueError, match="tools|工具"):
        list(provider.invoke("hello", tools=TOOLS))


def test_ark_responses_ainvoke_rejects_tools_for_text_only_interface():
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-model"
    provider._protocol = "responses"
    provider._options = {"server_verified_protocols": ["responses"]}
    provider._get_aclient = lambda: pytest.fail("Ark Responses ainvoke 不应初始化客户端")

    async def consume():
        return [chunk async for chunk in provider.ainvoke("hello", tools=TOOLS)]

    with pytest.raises(ValueError, match="tools|工具"):
        asyncio.run(consume())


@pytest.mark.parametrize(
    "method_name",
    [
        "retrieve_response",
        "create_response",
        "delete_response",
        "list_response_input_items",
        "list_input_items",
    ],
)
def test_ark_response_resources_require_responses_protocol(method_name):
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "chat_completions"
    provider._get_client = lambda: pytest.fail("Ark Responses 资源不应初始化客户端")
    method = getattr(provider, method_name)

    args = {
        "retrieve_response": ("resp-1",),
        "create_response": (),
        "delete_response": ("resp-1",),
        "list_response_input_items": ("resp-1",),
        "list_input_items": ("resp-1",),
    }[method_name]
    kwargs = {"input": "hello"} if method_name == "create_response" else {}

    with pytest.raises(ValueError, match="仅适用于 responses 协议"):
        method(*args, **kwargs)


@pytest.mark.parametrize(
    "method_name",
    [
        "async_retrieve_response",
        "async_create_response",
        "async_delete_response",
        "async_list_response_input_items",
        "async_list_input_items",
    ],
)
def test_ark_async_response_resources_require_responses_protocol(method_name):
    provider = object.__new__(VolcengineProvider)
    provider._protocol = "chat_completions"
    provider._get_aclient = lambda: pytest.fail("Ark Async Responses 资源不应初始化客户端")
    method = getattr(provider, method_name)

    args = {
        "async_retrieve_response": ("resp-1",),
        "async_create_response": (),
        "async_delete_response": ("resp-1",),
        "async_list_response_input_items": ("resp-1",),
        "async_list_input_items": ("resp-1",),
    }[method_name]
    kwargs = {"input": "hello"} if method_name == "async_create_response" else {}

    async def invoke():
        await method(*args, **kwargs)

    with pytest.raises(ValueError, match="仅适用于 responses 协议"):
        asyncio.run(invoke())


def test_openai_complete_uses_retry_call_wrapper(monkeypatch):
    provider = _responses_provider()
    provider._provider = "openai"
    provider._base_url = None
    calls = []

    def fake_retry(call):
        calls.append(call)
        return call()

    monkeypatch.setattr(openai_compatible_module, "retry_sync_call", fake_retry)
    provider._get_client = lambda: SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_request: SimpleNamespace(status="completed", output_text="答复")
        )
    )

    assert provider.complete(CompletionRequest(prompt="hello")).text == "答复"
    assert len(calls) == 1


def test_openai_acomplete_uses_async_retry_call_wrapper(monkeypatch):
    provider = _responses_provider()
    provider._provider = "openai"
    provider._base_url = None
    calls = []

    async def fake_retry(call):
        calls.append(call)
        result = call()
        return await result if inspect.isawaitable(result) else result

    monkeypatch.setattr(openai_compatible_module, "retry_async_call", fake_retry)

    class Responses:
        async def create(self, **_request):
            return SimpleNamespace(status="completed", output_text="异步答复")

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    assert asyncio.run(provider.acomplete(CompletionRequest(prompt="hello"))).text == "异步答复"
    assert len(calls) == 1


def test_anthropic_complete_uses_retry_call_wrapper(monkeypatch):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    calls = []

    def fake_retry(call):
        calls.append(call)
        return call()

    monkeypatch.setattr(anthropic_provider_module, "retry_sync_call", fake_retry)
    provider._get_client = lambda: SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_request: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="答复")]
            )
        )
    )

    assert provider.complete(CompletionRequest(prompt="hello")).text == "答复"
    assert len(calls) == 1


def test_anthropic_acomplete_uses_async_retry_call_wrapper(monkeypatch):
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-test"
    provider._options = {}
    calls = []

    async def fake_retry(call):
        calls.append(call)
        result = call()
        return await result if inspect.isawaitable(result) else result

    monkeypatch.setattr(anthropic_provider_module, "retry_async_call", fake_retry)

    class Messages:
        async def create(self, **_request):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="异步答复")])

    provider._get_aclient = lambda: SimpleNamespace(messages=Messages())

    assert asyncio.run(provider.acomplete(CompletionRequest(prompt="hello"))).text == "异步答复"
    assert len(calls) == 1


def test_ark_acomplete_uses_async_retry_call_wrapper(monkeypatch):
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "doubao-test"
    provider._protocol = "chat_completions"
    provider._options = {}
    calls = []

    async def fake_retry(call):
        calls.append(call)
        result = call()
        return await result if inspect.isawaitable(result) else result

    monkeypatch.setattr(volcengine_module, "retry_async_call", fake_retry)

    class Completions:
        async def create(self, **_request):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="异步答复", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    provider._get_aclient = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    assert asyncio.run(provider.acomplete(CompletionRequest(prompt="hello"))).text == "异步答复"
    assert len(calls) == 1


def test_openai_and_ark_embedding_base64_is_decoded_to_float_vectors():
    encoded = base64.b64encode(struct.pack("<2f", 1.25, -2.5)).decode("ascii")

    openai_provider = object.__new__(OpenAICompatibleProvider)
    openai_provider._model_name = "embedding"
    openai_provider._provider = "openai"
    openai_provider._options = {}
    openai_provider._client = None
    openai_provider._get_client = lambda: SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=encoded)])
        )
    )
    assert openai_provider.embed_documents(["text"], encoding_format="base64") == [[1.25, -2.5]]

    ark_provider = object.__new__(VolcengineProvider)
    ark_provider._model_name = "embedding"
    ark_provider._options = {}
    ark_provider._get_client = lambda: SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=encoded)])
        )
    )
    assert ark_provider.embed_documents(["text"], encoding_format="base64") == [[1.25, -2.5]]


def test_responses_async_non_stream_extracts_output_text():
    provider = _responses_provider()
    calls = []

    class Responses:
        async def create(self, **request):
            calls.append(request)
            return SimpleNamespace(status="completed", output_text="异步答复")

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        return [
            chunk
            async for chunk in provider._ainvoke_responses(
                {"model": "responses-model", "input": "hello", "stream": False}
            )
        ]

    assert asyncio.run(collect()) == ["异步答复"]
    assert calls == [{"model": "responses-model", "input": "hello", "stream": False}]


def test_responses_sync_stream_yields_text_and_refusal_deltas():
    provider = _responses_provider()

    class Responses:
        def create(self, **request):
            assert request["stream"] is True
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="好"),
                    SimpleNamespace(type="response.refusal.delta", delta="的"),
                    SimpleNamespace(type="response.completed"),
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match="拒答"):
        list(
            provider._invoke_responses(
                {"model": "responses-model", "input": "hello", "stream": True}
            )
        )


def test_responses_sync_non_stream_refusal_is_explicit_error():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="refusal", refusal="不能回答")],
                    )
                ],
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match="拒答"):
        list(
            provider._invoke_responses(
                {"model": "responses-model", "input": "hello", "stream": False}
            )
        )


def test_responses_async_stream_yields_text_deltas():
    provider = _responses_provider()

    class AsyncEvents:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "events"):
                self.events = iter(
                    [
                        SimpleNamespace(type="response.output_text.delta", delta="异"),
                        SimpleNamespace(type="response.output_text.delta", delta="步"),
                        SimpleNamespace(type="response.completed"),
                    ]
                )
            try:
                return next(self.events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class Responses:
        async def create(self, **request):
            assert request["stream"] is True
            return AsyncEvents()

    provider._get_aclient = lambda: SimpleNamespace(responses=Responses())

    async def collect():
        return [
            chunk
            async for chunk in provider._ainvoke_responses(
                {"model": "responses-model", "input": "hello", "stream": True}
            )
        ]

    assert asyncio.run(collect()) == ["异", "步"]


@pytest.mark.parametrize(
    ("status", "details", "expected"),
    [
        ("failed", SimpleNamespace(), "Responses API 响应状态为 failed"),
        ("incomplete", SimpleNamespace(reason="max_output_tokens"), "max_output_tokens"),
    ],
)
def test_responses_non_stream_terminal_status_is_explicit_error(status, details, expected):
    provider = _responses_provider()
    response = SimpleNamespace(status=status, incomplete_details=details, error=None)

    class Responses:
        def create(self, **request):
            return response

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match=expected):
        list(
            provider._invoke_responses(
                {"model": "responses-model", "input": "hello", "stream": False}
            )
        )


def test_responses_stream_failed_event_is_explicit_error():
    provider = _responses_provider()

    class Responses:
        def create(self, **request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=SimpleNamespace(message="upstream failed"),
                        ),
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match="upstream failed"):
        list(
            provider._invoke_responses(
                {"model": "responses-model", "input": "hello", "stream": True}
            )
        )


def test_responses_stream_incomplete_event_is_explicit_error():
    provider = _responses_provider()

    class Responses:
        def create(self, **request):
            return iter(
                [
                    SimpleNamespace(
                        type="response.incomplete",
                        response=SimpleNamespace(
                            status="incomplete",
                            incomplete_details=SimpleNamespace(reason="content_filter"),
                        ),
                    )
                ]
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())

    with pytest.raises(RuntimeError, match="content_filter"):
        list(
            provider._invoke_responses(
                {"model": "responses-model", "input": "hello", "stream": True}
            )
        )


def test_responses_stream_error_event_is_explicit_error():
    with pytest.raises(RuntimeError, match="rate limit"):
        OpenAICompatibleProvider._stream_delta(SimpleNamespace(type="error", message="rate limit"))


def test_responses_cancelled_events_are_explicit_errors():
    with pytest.raises(RuntimeError, match="cancelled|取消"):
        OpenAICompatibleProvider._responses_stream_event(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="cancelled"),
            )
        )

    with pytest.raises(RuntimeError, match="cancelled|取消"):
        OpenAICompatibleProvider._stream_delta(SimpleNamespace(type="response.cancelled"))


def test_openai_responses_non_stream_extracts_refusal_output():
    provider = _responses_provider()

    class Responses:
        def create(self, **_request):
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="refusal", refusal="不能回答")],
                    )
                ],
            )

    provider._get_client = lambda: SimpleNamespace(responses=Responses())
    result = provider.complete(CompletionRequest(prompt="hello"))

    assert result.refusal == "不能回答"


def test_deepseek_chat_uses_max_tokens_for_explicit_and_configured_limits():
    provider = object.__new__(DeepSeekProvider)
    provider._model_name = "deepseek-chat"
    provider._protocol = "chat_completions"

    provider._options = {}
    request = provider._build_chat_request(prompt="hi", stream=False, max_tokens=128)
    assert request["max_tokens"] == 128
    assert "max_completion_tokens" not in request

    provider._options = {"max_tokens": 256}
    request = provider._build_chat_request(prompt="hi", stream=False)
    assert request["max_tokens"] == 256
    assert "max_completion_tokens" not in request
