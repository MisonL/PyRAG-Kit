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
    validate_complete_tool_call,
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


# `_convert_messages` 的三条 assistant `tool_calls[]` 形态判据在整套测试里从未执行
# （`sys.settrace` 行覆盖证实；把它们的 `raise` 换成 `pass` 后全量 3260 条仍全绿）。
# 既有用例 `test_anthropic_rejects_malformed_tool_call_arguments`／
# `test_anthropic_rejects_tool_call_without_id_or_name` 命中的都是**元素内部**字段
# （`arguments` 非法 JSON、缺 `id`、缺 `function.name`），三条形态判据都绕开了。
# 删除后行为确实改变（实测）：707 元素非对象 → 裸
# `AttributeError: 'int' object has no attribute 'get'`；710 `function` 非对象 →
# 裸 `AttributeError: 'str' object has no attribute 'get'`；730 `arguments` 非对象 →
# 裸 `TypeError: 'int' object is not iterable`（`dict(arguments)`）。
@pytest.mark.parametrize("call", [5, "bad", ["x"]])
def test_anthropic_rejects_non_object_tool_call_entry(call):
    """`tool_calls[]` 元素非对象时必须给带序号的 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match=r"tool_calls\[0\] 必须是对象"):
        AnthropicProvider._convert_messages([{"role": "assistant", "tool_calls": [call]}])


@pytest.mark.parametrize("function", ["bad", 5, ["x"]])
def test_anthropic_rejects_non_object_tool_call_function(function):
    """`tool_calls[].function` 非对象时给带字段名的 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match=r"tool_calls\[0\].function 必须是对象"):
        AnthropicProvider._convert_messages(
            [{"role": "assistant", "tool_calls": [{"function": function}]}]
        )


@pytest.mark.parametrize("arguments", [5, ["a"], 5.5])
def test_anthropic_rejects_non_object_tool_call_arguments(arguments):
    """`tool_calls[].function.arguments` 非 JSON 对象时给带字段名的 ``ValueError``。"""
    with pytest.raises(ValueError, match=r"tool_calls\[0\] 的 arguments 必须是 JSON 对象"):
        AnthropicProvider._convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-1", "function": {"name": "lookup", "arguments": arguments}}
                    ],
                }
            ]
        )


def test_anthropic_accepts_well_formed_tool_call_arguments():
    """收紧不能误伤：JSON 字符串与对象两种合法形态的 ``arguments`` 照常转换。"""
    for arguments in ('{"q": "x"}', {"q": "x"}):
        converted = AnthropicProvider._convert_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-1", "function": {"name": "lookup", "arguments": arguments}}
                    ],
                }
            ]
        )
        assert converted[0]["content"][0]["input"] == {"q": "x"}


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


def test_anthropic_unknown_tool_choice_string_names_the_value():
    """未知字符串必须点名取值，不能退化成通用兜底消息。

    删除守卫后 ``"bogus"`` 落到最后一条通用 ``ValueError``——它不再包含出错的
    取值，排查时无法分辨是「值不认识」还是「形态写错了」。变异扫描证实该守卫
    在整套测试里没有反向覆盖（删掉后全量仍绿）。
    """
    with pytest.raises(ValueError, match=r"不支持 tool_choice: bogus"):
        AnthropicProvider._convert_tool_choice("bogus")

    # 合法取值不受影响
    assert AnthropicProvider._convert_tool_choice("auto") == {"type": "auto"}


# `_convert_tools` 与 `_convert_tool_choice` 各有四条以上判据，上面两条用例只
# 覆盖了其中一部分：`test_anthropic_rejects_malformed_tool_definitions` 覆盖
# 「元素非对象」「function 非对象」「parameters 非对象」，但工具**缺 name** 与
# **description 非字符串**两条从未执行；`test_anthropic_rejects_malformed_tool_choices`
# 覆盖「未知字符串」「function 缺 name」「tool 缺 name」，但 **function 取值非对象**
# 那条从未执行。`sys.settrace` 行覆盖证实这三条在整套测试里从未执行；把它们的
# `raise` 换成 `pass` 后全量 3188 条仍然全绿（同一批里的「tools 非序列」「tool
# 缺 name」「tool_choice 兜底」三条已由既有用例钉住，变异后确实变红）。
#
# 删除后行为确实改变（实测）：
#   * 缺 name → **静默接受**，`{"function": {}}` 产出 `{'name': None, ...}`，把
#     `name: None` 的工具定义发给上游；
#   * description 非字符串 → **静默接受**，整数 `description` 原样进请求体；
#   * function 取值非对象 → 带指引的 `ValueError` 变成裸
#     `AttributeError: 'str' object has no attribute 'get'`。
@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {}},
        {"parameters": {}},
        {"description": "只有描述没有名字"},
        {"function": {"description": "也没有名字"}},
    ],
)
def test_anthropic_rejects_tool_definitions_without_a_name(tool):
    """工具定义缺 name 必须 ``ValueError``，不能静默产出 ``name=None``。"""
    with pytest.raises(ValueError, match="缺少 function.name"):
        AnthropicProvider._convert_tools([tool])


@pytest.mark.parametrize("description", [5, True, ["d"], {"d": 1}])
def test_anthropic_rejects_non_string_tool_description(description):
    """``description`` 必须是非字符串，不能原样进请求体。"""
    with pytest.raises(ValueError, match="description 必须是字符串"):
        AnthropicProvider._convert_tools([{"name": "lookup", "description": description}])


@pytest.mark.parametrize("function", ["bad", 5, ["x"]])
def test_anthropic_rejects_tool_choice_with_non_object_function(function):
    """``tool_choice.function`` 非对象时给带指引的 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="tool_choice.function 必须是对象"):
        AnthropicProvider._convert_tool_choice({"type": "function", "function": function})


def test_anthropic_accepts_well_formed_tools_and_tool_choice():
    """收紧不能误伤：合法工具定义与 tool_choice 照常转换。"""
    converted = AnthropicProvider._convert_tools(
        [
            {
                "type": "function",
                "function": {"name": "lookup", "description": "查表", "parameters": {}},
            }
        ]
    )

    assert converted == [{"name": "lookup", "description": "查表", "input_schema": {}}]
    assert AnthropicProvider._convert_tool_choice(
        {"type": "function", "function": {"name": "lookup"}}
    ) == {
        "type": "tool",
        "name": "lookup",
    }


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


# `GoogleProvider._contents` 的两条判据在整套测试里从未执行（`sys.settrace` 行覆盖
# 证实；把它们的 `raise` 换成 `pass` 后全量 3245 条仍然全绿）。删除后行为确实改变
# （实测）：
#   * 1641 工具结果既无 `tool_call_id` 也无 `name` → **静默接受**，产出一个没有
#     `name` 的 `FunctionResponse`，Google 侧无法把它关联回任何一次函数调用；
#   * 1881 assistant `tool_calls[]` 的 `arguments` 非对象 → 带字段名的
#     `ValueError` 变成裸 `TypeError: 'int' object is not iterable`（`dict(...)`）。
# 同函数的 1862／1865／1869 经甄别是**等价变异**，不补测试：上游
# `normalize_messages` 已用同措辞判据拦下（消息变成
# `messages[0].tool_calls[0] ...`），删除后收场不变。
@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": '{"status":"paid"}'},
        {"role": "tool", "tool_call_id": 5, "content": "x"},
    ],
)
def test_google_rejects_tool_result_without_correlatable_call_id(message):
    """工具结果既无 `tool_call_id` 也无 `name` 时必须报错，不能产出无名的响应块。"""
    with pytest.raises(ValueError, match="缺少可关联的 tool_call_id/name"):
        GoogleProvider._contents([message], None, None)


@pytest.mark.parametrize("arguments", [5, ["a"], 5.5])
def test_google_rejects_non_object_function_call_arguments(arguments):
    """assistant `tool_calls[].function.arguments` 非 JSON 对象时给带指引的 ``ValueError``。"""
    with pytest.raises(ValueError, match="arguments 必须是 JSON 对象"):
        GoogleProvider._contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup_order", "arguments": arguments},
                        }
                    ],
                }
            ],
            None,
            None,
        )


def test_google_accepts_json_string_and_object_function_call_arguments():
    """收紧不能误伤：JSON 字符串与对象两种合法形态的 ``arguments`` 照常转换。"""
    for arguments in ("{}", {"outer": 1}):
        contents = GoogleProvider._contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup_order", "arguments": arguments},
                        }
                    ],
                }
            ],
            None,
            None,
        )
        calls = [part.function_call for part in contents[0].parts if part.function_call is not None]
        assert len(calls) == 1
        assert calls[0].name == "lookup_order"
        assert isinstance(calls[0].args, dict)


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
    """缺 name 必须是带指引的 ``ValueError``，不能退化成 pydantic ``ValidationError``。

    ``match`` 必须钉住「缺少」前缀：删除守卫后落进 pydantic 校验，其消息里的
    ``allowed_function_names`` 恰好满足宽松正则 ``function.name``（``.`` 匹配
    ``_``），旧断言在变异后仍然变绿——变异扫描证实过这一盲区。
    """
    with pytest.raises(ValueError, match="缺少 function.name"):
        GoogleProvider._convert_tool_choice({"type": "function", "function": {}})


# `GoogleProvider._convert_tools` 的容器判据与 `_convert_tool_choice` 的
# 「function 非对象」判据在整套测试里从未执行（`sys.settrace` 行覆盖证实；
# 把它们的 `raise` 换成 `pass` 后全量 3238 条仍然全绿）。既有用例看似覆盖了同一
# 入口，其实都绕过了这两条：`test_google_convert_tools_rejects_malformed_definitions`
# 传的是「列表**内含**畸形元素」（`[{"type": ["google_search"]}]`），命中的是
# 元素级判据；`test_google_rejects_malformed_function_tool_choice` 传的
# `function={}` 是合法 Mapping，命中的是缺 name 那条。删除后行为确实改变（实测）：
#   * 1245 `tools=5` → 带序号的 `ValueError` 变成裸
#     `TypeError: 'int' object is not iterable`（`enumerate(tools)`）；
#   * 1333 `function` 非对象 → 带字段名的 `ValueError` 变成裸
#     `AttributeError: 'str' object has no attribute 'get'`。
@pytest.mark.parametrize("tools", [5, {"a": 1}, "oops", object()])
def test_google_convert_tools_rejects_non_sequence_container(tools):
    """``tools`` 非序列时必须给带指引的 ``ValueError``，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="Google 工具定义必须是列表"):
        GoogleProvider._convert_tools(tools)


@pytest.mark.parametrize("function", ["lookup", 5, ["x"]])
def test_google_rejects_tool_choice_with_non_object_function(function):
    """``tool_choice.function`` 非对象时给带字段名的 ``ValueError``，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="function tool_choice 的 function 必须是对象"):
        GoogleProvider._convert_tool_choice({"type": "function", "function": function})


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


# `VolcengineProvider._build_responses_request` 有一组「同一字段两处配置」与
# 「取值形态」判据，九条在整套测试里从未执行（`sys.settrace` 行覆盖证实；把它们
# 的 `raise` 换成 `pass` 后全量 3226 条仍然全绿）。删除后行为确实改变（实测）：
#   * 1218 模型 options 同时给 `conversation` 与 `session` → **静默接受**，只有
#     `session` 生效，`conversation` 被丢弃；
#   * 1222／1227／1256／1261 同一标量字段在模型 options 与 `extra_body`（或请求级
#     参数）两处出现 → **静默覆盖**，实测 `{"top_k": 1}` 被 `{"top_k": 2}` 顶掉；
#   * 1303／1311／1315 `session`／`conversation` 非对象 → 带字段名的 `ValueError`
#     变成裸 `TypeError: 'int' object is not iterable`（`dict(...)`）；
#   * 1336 请求级 `extra_body` 与模型 options 的同名键重复 → **静默覆盖**。
def _ark_responses_provider(**options):
    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._options = {"server_verified_protocols": ["responses"], **options}
    return provider


def test_ark_responses_rejects_conversation_and_session_together():
    """模型 options 同时给 `conversation` 与 `session` 时必须报错，不能静默丢弃其一。"""
    provider = _ark_responses_provider(conversation={"a": 1}, session={"b": 2})

    with pytest.raises(ValueError, match="不能同时设置 conversation 和 session"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))


@pytest.mark.parametrize("field", ["top_k", "seed"])
def test_ark_responses_rejects_scalar_option_duplicating_extra_body(field):
    """模型 options 的标量字段与其 `extra_body` 同名键重复时必须报错。"""
    provider = _ark_responses_provider(**{field: 1, "extra_body": {field: 2}})

    with pytest.raises(ValueError, match=rf"模型 options 的 {field} 与 extra_body 重复"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))


@pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
def test_ark_responses_rejects_sampling_field_duplicating_extra_body(field):
    """采样字段与模型 options 的 `extra_body` 重复时必须报错。"""
    provider = _ark_responses_provider(**{field: 0.5, "extra_body": {field: 0.5}})

    with pytest.raises(ValueError, match=rf"{field} 与模型 options.extra_body 重复"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))


def test_ark_responses_rejects_request_top_k_duplicating_extra_body():
    """请求级 `top_k` 与模型 options 的 `extra_body` 重复时必须报错。"""
    provider = _ark_responses_provider(extra_body={"top_k": 1})

    with pytest.raises(ValueError, match="top_k 与模型 options.extra_body 重复"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False, top_k=2))


@pytest.mark.parametrize("field", ["session", "conversation"])
def test_ark_responses_rejects_non_object_request_state(field):
    """请求级 `session`/`conversation` 非对象时给带字段名的 ``ValueError``，不是裸 ``TypeError``。"""
    provider = _ark_responses_provider()

    with pytest.raises(ValueError, match=rf"Ark Responses {field} 必须是对象"):
        provider._build_responses_request(
            CompletionRequest(prompt="hello", stream=False, **{field: 5})
        )


@pytest.mark.parametrize("field", ["session", "conversation"])
def test_ark_responses_rejects_non_object_configured_state(field):
    """模型 options 的 `session`/`conversation` 非对象时同样给带字段名的 ``ValueError``。"""
    provider = _ark_responses_provider(**{field: 5})

    with pytest.raises(ValueError, match=rf"模型 options.{field} 必须是对象"):
        provider._build_responses_request(CompletionRequest(prompt="hello", stream=False))


def test_ark_responses_request_extra_body_cannot_override_model_options():
    """请求级 `extra_body` 与模型 options 的同名键重复时必须报错，不能静默覆盖。"""
    provider = _ark_responses_provider(extra_body={"foo": 1})

    with pytest.raises(ValueError, match="extra_body 与模型 options 重复: foo"):
        provider._build_responses_request(
            CompletionRequest(prompt="hello", stream=False, extra_body={"foo": 2})
        )


def test_ark_responses_accepts_distinct_state_and_extra_body():
    """收紧不能误伤：合法的 `session` 与不重复的 `extra_body` 照常进请求。"""
    provider = _ark_responses_provider(extra_body={"foo": 1})

    request = provider._build_responses_request(
        CompletionRequest(prompt="hello", stream=False, session={"id": "s1"}, extra_body={"bar": 2})
    )

    assert request["session"] == {"id": "s1"}
    assert request["extra_body"] == {"foo": 1, "bar": 2}


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


# `_convert_responses_tools` 在 OpenAI 兼容与 Ark 两份实现里逐字同构，各有五条
# 元素级判据。既有用例只覆盖了「容器非序列」（`test_responses_converters_reject_non_sequence_tools`）
# 与「元素非对象」，另外五条从未执行：`sys.settrace` 行覆盖证实，把它们的 `raise`
# 换成 `pass` 后全量 3200 条仍然全绿（只有容器那条变异后变红）。删除后行为确实
# 改变（实测，两个 provider 同形）：
#   * 原生工具缺有效 `type` → **静默接受**，`{"native": 1}` 原样透传；
#   * 原生 `type=function` 缺 `name` → **静默接受**，产出没有 `name` 的工具；
#   * 原生 function 的 `parameters` 非对象 → 带序号的 `ValueError` 变成裸
#     `TypeError: 'int' object is not iterable`（`dict(parameters)`）；
#   * 统一 schema 缺 `function.name` → **静默接受**，产出 `name: None`；
#   * 统一 schema 的 `function.parameters` 非对象 → 同样变成裸 `TypeError`。
# 两个 provider 用同一组形态参数化，避免只钉住其中一份。
@pytest.mark.parametrize(
    "converter",
    [
        OpenAICompatibleProvider._convert_responses_tools,
        VolcengineProvider._convert_responses_tools,
    ],
    ids=["openai", "ark"],
)
@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        ([{"native": 1}], "缺少有效 type"),
        ([{"type": "function"}], "缺少 name"),
        ([{"type": "function", "name": "n", "parameters": 5}], "parameters 必须是对象"),
        ([{"function": {}}], "缺少 function.name"),
        ([{"function": {"name": "n", "parameters": 5}}], "function.parameters 必须是对象"),
    ],
    ids=[
        "missing-type",
        "native-missing-name",
        "native-bad-parameters",
        "missing-function-name",
        "bad-function-parameters",
    ],
)
def test_responses_tool_converters_reject_malformed_entries(converter, tools, expected):
    """元素级形态非法时必须给带指引的 ``ValueError``，不能静默接受或抛裸 ``TypeError``。"""
    with pytest.raises(ValueError, match=expected):
        converter(tools)


@pytest.mark.parametrize(
    "converter",
    [
        OpenAICompatibleProvider._convert_responses_tools,
        VolcengineProvider._convert_responses_tools,
    ],
    ids=["openai", "ark"],
)
def test_responses_tool_converters_accept_well_formed_entries(converter):
    """收紧不能误伤：原生工具与统一 schema 两种合法形态照常转换。"""
    native = converter([{"type": "function", "name": "lookup", "parameters": {"type": "object"}}])
    assert native[0]["name"] == "lookup"

    unified = converter([{"function": {"name": "lookup", "parameters": {"type": "object"}}}])
    assert unified[0]["name"] == "lookup"


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


def test_openai_chat_stream_survives_non_string_tool_call_type():
    """网关把片段里的 ``type`` 写成数组时，chat 路径不能抛裸 ``TypeError``。

    ``openai`` SDK 对 SSE chunk 的未知顶层字段用 ``extra="allow"`` 原样保留，
    ``{"type": ["function"]}`` 因此能构造出来。上一轮把判据从 ``==`` 改成集合
    成员测试后，这条路径从「静默通过」变成 ``TypeError: unhashable type``。
    """

    def chunk(arguments, tool_calls=None, finish_reason=None):
        if tool_calls is None:
            tool_calls = [
                SimpleNamespace(
                    id="call-1",
                    index=0,
                    type=["function"],
                    function=SimpleNamespace(name="lookup", arguments=arguments),
                )
            ]
        return SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model="chat-model",
        )

    accumulator: dict[object, dict[str, object]] = {}
    completed: set[object] = set()
    delivered = []
    for event in [
        *OpenAICompatibleProvider._chat_stream_events(chunk('{"id":1}')),
        *OpenAICompatibleProvider._chat_stream_events(chunk(None, [], "tool_calls")),
    ]:
        for emitted in OpenAICompatibleProvider._chat_completion_events(
            event, accumulator, completed
        ):
            if emitted.type == "tool_call_completed":
                delivered.append(emitted.tool_call)

    assert [call["arguments"] for call in delivered] == ['{"id":1}']


def test_validate_complete_tool_call_accepts_non_string_type():
    """``type`` 不是字符串时不能抛裸 ``TypeError``。

    ``type`` 来自网关 JSON，``openai`` SDK 对 SSE chunk 的未知字段用
    ``extra="allow"`` 原样保留，因此 ``{"type": ["function"]}`` 是能构造出来的。
    判据若直接做集合成员测试，``list``/``dict`` 会抛
    ``TypeError: unhashable type``——既不是调用方按 ``RuntimeError`` 做的分类，
    消息里也没有工具调用名。``merge_tool_call_fragment`` 与 ``enum_field`` 对同一
    形态都已有守卫，此处口径一致。
    """
    for tool_type in (["function"], {"name": "function"}, 1, None):
        result = validate_complete_tool_call(
            {"name": "lookup", "type": tool_type, "arguments": '{"id":1}'},
            "OpenAI Chat Completions",
        )

        assert result["name"] == "lookup"


def test_validate_complete_tool_call_rejects_unverifiable_arguments():
    """``json.loads`` 的另两类失败也必须收敛成带指引的 ``RuntimeError``。

    除 ``JSONDecodeError``（``ValueError`` 的子类）外，``json.loads`` 还会因整数
    位数超过 ``sys.get_int_max_str_digits()`` 抛 ``ValueError``、因嵌套超过解释器
    栈抛 ``RecursionError``。前者**不是** ``RuntimeError``，会穿透调用方按
    ``RuntimeError`` 做的错误分类，消息更是与「哪个工具调用」无关；两者都让一段
    结构完整、只是无法校验的 arguments 变成一条与工具调用无关的解析异常。
    """
    for arguments in ("9" * 4301, "[" * 2000 + "]" * 2000):
        with pytest.raises(RuntimeError, match="lookup"):
            validate_complete_tool_call(
                {"name": "lookup", "type": "function", "arguments": arguments},
                "OpenAI Chat Completions",
            )


def test_validate_complete_tool_call_requires_json_for_unknown_type():
    """除 ``custom_tool_call`` 外，任何 ``type`` 都必须带完整 JSON。

    同一函数里两条判据方向相反：第一条是**黑名单**（``type !=
    "custom_tool_call"`` 就必须有非空 arguments），第二条却是**白名单**
    （``type in {"function", "function_call"}`` 才校验 JSON）。于是任何不在白名单
    里的取值——网关改写过的 ``type``、大写形式、空串——同时满足「必须有参数」与
    「不必校验参数」，半截 JSON 被当作完整调用交付。

    白名单方向本身还带来第二个问题：集合成员测试对 ``list``/``dict`` 抛
    ``TypeError: unhashable type``（``tool_calls[].type`` 在 SDK 流式解析里走
    ``construct_type``，不做 ``Literal`` 校验，所以非字符串是可达的）。
    """
    for tool_type in ("weird", "FUNCTION", "", ["function"], {"name": "function"}, None):
        with pytest.raises(RuntimeError, match="lookup"):
            validate_complete_tool_call(
                {"name": "lookup", "type": tool_type, "arguments": '{"id":'},
                "OpenAI Chat Completions",
            )


def test_openai_chat_stream_rejects_unknown_type_with_incomplete_arguments():
    """端到端：未知 ``type`` 的半截 JSON 不能交付给调用方。

    用真实 SDK 的 ``construct_type`` 构造 chunk——这正是 SDK 流式路径的解析
    方式（``_process_response_data`` 在非严格校验下走 ``construct_type``，跳过
    已声明字段的 ``Literal`` 校验），因此这里的 ``type`` 取值是可真实到达的。
    """
    from openai._models import construct_type
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    # 真实流式形态：``finish_reason`` 与最后一个 delta 在同一个 chunk 上。
    delta = construct_type(
        type_=ChatCompletionChunk,
        value={
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "chat-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "tool_call",
                                "function": {"name": "lookup", "arguments": '{"id":'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )

    accumulator: dict[object, dict[str, object]] = {}
    completed: set[object] = set()
    with pytest.raises(RuntimeError, match="不是完整 JSON"):
        for event in OpenAICompatibleProvider._chat_stream_events(delta):
            OpenAICompatibleProvider._chat_completion_events(event, accumulator, completed)


def test_validate_complete_tool_call_keeps_custom_tool_arbitrary_text():
    """``custom_tool_call`` 的任意文本语义不受上述收敛影响。"""
    result = validate_complete_tool_call(
        {"name": "run_command", "type": "custom_tool_call", "arguments": "echo hello"},
        "OpenAI Responses",
    )

    assert result["arguments"] == "echo hello"


def test_openai_chat_stream_rejects_incomplete_tool_call_arguments():
    """Chat Completions 的工具调用 type 是 ``function``，半截 JSON 必须被拦下。

    Chat Completions 的流式片段只带 ``type: "function"``（Responses 用的是
    ``function_call``），因此 ``validate_complete_tool_call`` 的 JSON 判据若不
    把 ``function`` 算作函数调用，半截参数会被当作完整调用交付给调用方。
    """

    def chunk(arguments, finish_reason=None, tool_calls=None):
        if tool_calls is None:
            tool_calls = [
                SimpleNamespace(
                    id="call-1",
                    index=0,
                    function=SimpleNamespace(name="lookup", arguments=arguments),
                )
            ]
        return SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model="chat-model",
        )

    accumulator: dict[object, dict[str, object]] = {}
    completed: set[object] = set()
    with pytest.raises(RuntimeError, match="不是完整 JSON"):
        for event in [
            *OpenAICompatibleProvider._chat_stream_events(chunk('{"id":')),
            *OpenAICompatibleProvider._chat_stream_events(chunk(None, "tool_calls", [])),
        ]:
            OpenAICompatibleProvider._chat_completion_events(event, accumulator, completed)


def test_openai_chat_stream_accepts_complete_tool_call_arguments():
    """对照组：完整 JSON 必须仍然照常交付。"""

    def chunk(arguments, finish_reason=None, tool_calls=None):
        if tool_calls is None:
            tool_calls = [
                SimpleNamespace(
                    id="call-1",
                    index=0,
                    function=SimpleNamespace(name="lookup", arguments=arguments),
                )
            ]
        return SimpleNamespace(
            id="chat-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
            model="chat-model",
        )

    accumulator: dict[object, dict[str, object]] = {}
    completed: set[object] = set()
    delivered = []
    for event in [
        *OpenAICompatibleProvider._chat_stream_events(chunk('{"id":1}')),
        *OpenAICompatibleProvider._chat_stream_events(chunk(None, "tool_calls", [])),
    ]:
        for emitted in OpenAICompatibleProvider._chat_completion_events(
            event, accumulator, completed
        ):
            if emitted.type == "tool_call_completed":
                delivered.append(emitted.tool_call)

    assert [call["arguments"] for call in delivered] == ['{"id":1}']


def test_google_stream_rejects_incomplete_tool_call_arguments():
    """Google 的流式片段同样是 ``type: "function"``。"""
    accumulator: dict[object, dict[str, object]] = {
        ("index", 0): {
            "id": "call-1",
            "index": 0,
            "type": "function",
            "name": "lookup",
            "arguments": '{"order_id":',
        }
    }

    with pytest.raises(RuntimeError, match="不是完整 JSON"):
        GoogleProvider._complete_stream_tool_calls(accumulator, set(), "resp-1", None)


def test_anthropic_stream_rejects_incomplete_tool_call_arguments():
    """Anthropic 的流式片段同样是 ``type: "function"``。"""

    raw_events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="toolu-1", name="lookup", input={}),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"id":'),
        ),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use")),
        SimpleNamespace(type="message_stop"),
    ]

    class ClientManager:
        def __enter__(self):
            return iter(raw_events)

        def __exit__(self, *_exc):
            return False

    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-model"
    provider._options = {}
    provider._get_client = lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_params: ClientManager())
    )

    with pytest.raises(RuntimeError, match="不是完整 JSON"):
        list(provider.stream_events(CompletionRequest(prompt="hello")))


def test_ark_chat_stream_rejects_incomplete_tool_call_arguments():
    """Ark Chat Completions 的流式片段同样是 ``type: "function"``。"""

    class Completions:
        def create(self, **_request):
            return iter(
                [
                    SimpleNamespace(
                        id="chat-1",
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
                        ],
                        usage=None,
                        model="ark-model",
                    ),
                    SimpleNamespace(
                        id="chat-1",
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None, tool_calls=None),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=None,
                        model="ark-model",
                    ),
                ]
            )

    provider = object.__new__(VolcengineProvider)
    provider._model_name = "ark-model"
    provider._protocol = "chat"
    provider._options = {"server_verified_protocols": ["chat"]}
    provider._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    with pytest.raises(RuntimeError, match="不是完整 JSON"):
        list(provider.stream_events(CompletionRequest(prompt="hello")))


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


@pytest.mark.parametrize(
    "content",
    [
        [SimpleNamespace(type="text", text=None)],
        [SimpleNamespace(type="thinking", thinking=None)],
        [
            SimpleNamespace(type="thinking", thinking=None),
            SimpleNamespace(type="text", text=None),
        ],
        [SimpleNamespace(type="redacted_thinking", thinking=None)],
    ],
)
def test_anthropic_extract_result_tolerates_null_block_fields(content):
    """块字段存在但值为 None 时不得崩。

    ``field()`` 只在键不存在时返回默认值；键存在而值为 None 时仍返回 None，
    直接 ``"".join(...)`` 会抛 ``TypeError: expected str instance``。
    SDK 的 thinking / redacted_thinking 块确实会给出 None。
    """
    result = AnthropicProvider._extract_result(SimpleNamespace(content=content, usage=None))

    assert isinstance(result.text, str)
    assert isinstance(result.reasoning, str)


def test_anthropic_extract_result_preserves_real_text():
    """守卫不能把正常内容一起吞掉。"""
    result = AnthropicProvider._extract_result(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="推理"),
                SimpleNamespace(type="text", text="正文"),
            ],
            usage=None,
        )
    )

    assert result.text == "正文"
    assert result.reasoning == "推理"


@pytest.mark.parametrize(
    ("delta", "expected_type", "expected_value"),
    [
        (SimpleNamespace(type="text_delta", text=None), "text_delta", "text"),
        (SimpleNamespace(type="thinking_delta", thinking=None), "reasoning_delta", "reasoning"),
        (
            SimpleNamespace(type="input_json_delta", partial_json=None),
            "tool_call_delta",
            "tool_call",
        ),
    ],
)
def test_anthropic_stream_delta_tolerates_null_fields(delta, expected_type, expected_value):
    """流式 delta 的同一处缺陷：None 会让事件构造出非字符串或直接崩。"""
    event = AnthropicProvider._stream_event(
        SimpleNamespace(type="content_block_delta", index=0, delta=delta), "resp-1"
    )

    assert event.type == expected_type
    value = getattr(event, expected_value)
    if expected_value == "tool_call":
        assert value["arguments"] == ""
    else:
        assert value == ""


@pytest.mark.parametrize("tool_choice", [{"type": "tool"}, {"type": "TOOL"}, {"mode": "bogus"}])
def test_google_tool_choice_rejects_unknown_mode_instead_of_silent_enum(tool_choice):
    """未知模式必须显式报错。

    ``FunctionCallingConfigMode`` 是大小写不敏感枚举，未命中时会合成一个同名
    字符串枚举成员、只发 UserWarning，随后被静默发往服务端。OpenAI 风格的
    ``{"type": "tool"}`` 正会落到这里。
    """
    with pytest.raises(ValueError, match="tool_choice"):
        GoogleProvider._convert_tool_choice(tool_choice)


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [("auto", "AUTO"), ("required", "ANY"), ("any", "ANY"), ("ANY", "ANY"), ("none", "NONE")],
)
def test_google_tool_choice_maps_known_modes(tool_choice, expected):
    """已知模式仍须正常映射，收紧不能把合法输入一起拒掉。"""
    config = GoogleProvider._convert_tool_choice(tool_choice)

    assert config.function_calling_config.mode.name == expected


def test_google_tool_choice_normalizes_tool_config_errors():
    """未知键应归一为本项目的 ValueError，而不是裸的 pydantic ValidationError。"""
    with pytest.raises(ValueError, match="tool_choice 无效"):
        GoogleProvider._convert_tool_choice(
            {"function_calling_config": {"mode": "AUTO"}, "bogus": 1}
        )


@pytest.mark.parametrize(
    "response_format",
    [{"type": "json_schema"}, {"type": "json_schema", "json_schema": {"name": "r"}}],
)
def test_responses_json_schema_requires_schema(response_format):
    """缺 schema 不能回退成 schema 本身。

    那样会伪造出 ``{"type": "json_schema"}`` 的 schema 发给服务端，客户端以为
    拿到了结构化输出约束，实际没有。
    """
    with pytest.raises(ValueError, match="缺少 schema"):
        OpenAICompatibleProvider._convert_responses_format(response_format)


def test_responses_json_schema_preserves_caller_strict():
    """strict 必须透传，不能被硬编码的 True 覆盖。"""
    converted = OpenAICompatibleProvider._convert_responses_format(
        {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": False},
        }
    )

    assert converted["format"]["strict"] is False
    assert converted["format"]["name"] == "out"
    assert converted["format"]["schema"] == {"type": "object"}


def test_responses_json_schema_accepts_flat_shape():
    """扁平形态（schema/strict 与 type 同级）仍须支持。"""
    converted = OpenAICompatibleProvider._convert_responses_format(
        {"type": "json_schema", "name": "flat", "schema": {"type": "object"}}
    )

    assert converted["format"] == {
        "type": "json_schema",
        "name": "flat",
        "schema": {"type": "object"},
        "strict": True,
    }


# ── 回归：``max_tokens`` 的有限性必须先于 ``int()`` 判 ──
#
# ``_default_max_tokens`` 原先是「判类型 -> ``int(configured)`` -> 判正数与整数性」。
# ``int(float("inf"))`` 抛的是 ``OverflowError: cannot convert float infinity to
# integer``，``int(float("nan"))`` 抛 ``ValueError: cannot convert float NaN to
# integer``——两者都**不是**该方法承诺的「必须是正整数」消息，而 ``OverflowError``
# 连 ``except ValueError`` 都接不住。实测 ``options.max_tokens = inf`` 正是这种
# 收场，而同为非法取值的 ``1.5``／``0``／``-1``／``True`` 都能得到带指引的消息。


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan"), 1e400],
)
def test_anthropic_max_tokens_rejects_non_finite_values(value):
    """``inf``／``nan`` 必须报成「必须是正整数」，而不是 ``OverflowError``。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-5"
    provider._options = {"max_tokens": value}

    with pytest.raises(ValueError, match="max_tokens"):
        provider._default_max_tokens()


@pytest.mark.parametrize("value", [1.5, 0, -1, True, "100"])
def test_anthropic_max_tokens_rejects_other_invalid_values(value):
    """收紧不能误伤既有判据：其余非法取值照旧拒绝且消息一致。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-5"
    provider._options = {"max_tokens": value}

    with pytest.raises(ValueError, match="max_tokens"):
        provider._default_max_tokens()


def test_anthropic_max_tokens_accepts_valid_values():
    """收紧不能误伤：合法取值照常返回。"""
    provider = object.__new__(AnthropicProvider)
    provider._model_name = "claude-sonnet-4-5"

    provider._options = {"max_tokens": 100}
    assert provider._default_max_tokens() == 100

    provider._options = {"max_tokens": 8192.0}
    assert provider._default_max_tokens() == 8192

    provider._options = {}
    assert provider._default_max_tokens() > 0


# ── 回归：流式拒答文本的取值来自服务端，不得让它抛裸的整数转换错误 ──
#
# ``_stream_event`` 的两处拒答分支写的是
# ``refusal if isinstance(refusal, str) else str(refusal or "")``——非字符串走
# 裸 ``str()``。而 ``refusal`` 来自服务端返回的 JSON（``content_block`` 或
# ``delta`` 里的字段），形态不受本地控制：JSON 里一个超大整数字面量
# （``10**4301``）会让 ``str()`` 撞上 CPython 的整数位数上限，抛
# ``ValueError: Exceeds the limit``。
#
# 这个异常发生在**流式解析**过程中，即 ``_iterate_stream_events`` 的生成器里：
# 用户已经看到部分输出，随后拿到一条与「服务端拒答」毫无关系的整数转换错误，
# 而同一函数对正常拒答文本给出的是 ``StreamEvent(type="refusal_delta")``。
# 同一份输入里 ``42`` 被正常转成 ``"42"``、``None`` 变成 ``""``，只有位数超限
# 的那一个炸——收场方式取决于取值的**大小**而不是形态。
@pytest.mark.parametrize(
    "hostile_refusal",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_anthropic_stream_refusal_delta_survives_huge_integer(hostile_refusal):
    block = SimpleNamespace(type="refusal", refusal=hostile_refusal)
    event = SimpleNamespace(type="content_block_start", content_block=block)

    converted = AnthropicProvider._stream_event(event, "response-id")

    assert converted is not None
    assert converted.type == "refusal_delta"
    # 必须仍然是字符串，且不能是空的——空的会让调用方以为「没有拒答」。
    assert isinstance(converted.refusal, str)
    assert converted.refusal


def test_anthropic_stream_refusal_delta_survives_huge_integer_in_delta():
    """``content_block_delta`` 路径的同一判据（两处都要守住）。"""
    delta = SimpleNamespace(type="refusal_delta", refusal=10**4301)
    event = SimpleNamespace(type="content_block_delta", delta=delta)

    converted = AnthropicProvider._stream_event(event, "response-id")

    assert converted is not None
    assert converted.type == "refusal_delta"
    assert isinstance(converted.refusal, str)
    assert converted.refusal


def test_anthropic_stream_refusal_delta_keeps_plain_text_intact():
    """反向守卫：正常拒答文本必须原样保留，不能被量级描述改写。"""
    block = SimpleNamespace(type="refusal", refusal="我无法回答这个问题。")
    event = SimpleNamespace(type="content_block_start", content_block=block)

    converted = AnthropicProvider._stream_event(event, "response-id")

    assert converted.refusal == "我无法回答这个问题。"


# ── 回归：拒答详情（stop_details）的取值同样来自服务端，两处都要守卫 ──
#
# 上一轮修的是 ``_stream_event`` 里 ``refusal``／``delta.refusal`` 两条分支；
# 同一文件还有**两处**用同一写法处理 ``stop_details``：``explanation`` 与
# ``category`` 都来自服务端 JSON（``stop_details`` 是 Anthropic 的拒答详情对象），
# 而它们写的是 ``str(explanation or category or "<默认文案>")``——非字符串走裸
# ``str()``，JSON 里一个超大整数字面量就抛 ``ValueError: Exceeds the limit``。
#
# 这两处的后果与上一轮同源但方向相反：``stop_reason == "refusal"`` 是**安全拦截**
# 的信号，此处抛异常会让一次本该被显式上报的拒答变成一条与安全策略无关的整数
# 转换错误——用户看到的是「程序出错了」，而不是「请求被安全策略拒绝」。
#
# 非流式与流式（``message_delta``）两条路径各有一份，都要守住。
def _anthropic_refusal_response(explanation, category=None):
    stop_details = SimpleNamespace(type="refusal", explanation=explanation, category=category)
    return SimpleNamespace(
        stop_reason="refusal", stop_details=stop_details, id="response-id", content=[], usage=None
    )


@pytest.mark.parametrize(
    "hostile_value",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_anthropic_extract_result_survives_huge_integer_refusal_detail(hostile_value):
    result = AnthropicProvider._extract_result(_anthropic_refusal_response(hostile_value))

    assert isinstance(result.refusal, str)
    # 空串会让调用方以为「没有拒答」，静默吞掉一次安全拦截。
    assert result.refusal


@pytest.mark.parametrize(
    "hostile_value",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_anthropic_stream_refusal_stop_details_survives_huge_integer(hostile_value):
    stop_details = SimpleNamespace(type="refusal", explanation=hostile_value, category=None)
    delta = SimpleNamespace(stop_reason="refusal", stop_details=stop_details)
    event = SimpleNamespace(type="message_delta", delta=delta)

    converted = AnthropicProvider._stream_event(event, "response-id")

    assert converted is not None
    assert converted.type == "refusal_delta"
    assert isinstance(converted.refusal, str)
    assert converted.refusal


def test_anthropic_refusal_detail_keeps_plain_text_and_default():
    """反向守卫：正常文本原样保留，两者皆缺时仍给默认文案。"""
    assert (
        AnthropicProvider._extract_result(_anthropic_refusal_response("策略说明")).refusal
        == "策略说明"
    )
    # 只有 category 时用它
    assert (
        AnthropicProvider._extract_result(_anthropic_refusal_response(None, "cyber")).refusal
        == "cyber"
    )
    # 两者皆缺时给可读的默认文案，而不是空串
    assert (
        AnthropicProvider._extract_result(_anthropic_refusal_response(None, None)).refusal
        == "Anthropic 请求被安全策略拒绝"
    )


# ── 回归：finish_reason 归一化不得抛裸的整数转换错误 ──
#
# ``finish_reason`` 来自服务端响应（``stop_reason``／``finishReason`` 字段），
# 形态不受本地控制。两处归一化都用裸 ``str()``：
# ``GoogleProvider._normalize_finish_reason`` 的 ``str(raw)`` 与
# ``AnthropicProvider._stream_event`` 的 ``str(reason)``。JSON 里一个超大整数
# 字面量（``10**4301``）就让它们撞上 CPython 的整数位数上限，抛
# ``ValueError: Exceeds the limit``——而这两个位置都在**流式/非流式响应解析**
# 里，用户已经拿到部分输出后才看到一条与「停止原因」毫无关系的整数转换错误。
#
# 归一化的职责是「把枚举或原始值统一成协议字符串」，不是「原样渲染任意取值」：
# 位数超限时给可读描述（``safe_repr``）即可，仍能区分不同取值。
@pytest.mark.parametrize(
    "hostile_reason",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_google_normalize_finish_reason_survives_huge_integer(hostile_reason):
    from src.providers.google import GoogleProvider

    normalized = GoogleProvider._normalize_finish_reason(hostile_reason)

    assert isinstance(normalized, str)
    assert normalized


@pytest.mark.parametrize(
    "hostile_reason",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_google_extract_result_survives_huge_finish_reason(hostile_reason):
    part = SimpleNamespace(text="正文", function_call=None, thought=None, refusal=None)
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[part]),
                finish_reason=hostile_reason,
                refusal=None,
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        response_id="response-id",
    )

    result = GoogleProvider._extract_result(response)

    assert isinstance(result.finish_reason, str)


@pytest.mark.parametrize(
    "hostile_reason",
    [
        pytest.param(10**4301, id="huge-int-4301"),
        pytest.param(10**4000, id="huge-int-4000"),
    ],
)
def test_anthropic_stream_finish_reason_survives_huge_integer(hostile_reason):
    delta = SimpleNamespace(stop_reason=hostile_reason, stop_details=None)
    event = SimpleNamespace(type="message_delta", delta=delta, usage=None)

    converted = AnthropicProvider._stream_event(event, "response-id")

    assert converted is not None
    assert converted.type == "finish"
    assert isinstance(converted.finish_reason, str)
    assert converted.finish_reason


def test_finish_reason_normalization_keeps_enum_and_plain_values_intact():
    """反向守卫：枚举对象与普通字符串仍按原语义归一化。"""
    from src.providers.google import GoogleProvider

    # 普通字符串照旧大写化、去掉点号前缀
    assert GoogleProvider._normalize_finish_reason("stop") == "STOP"
    assert GoogleProvider._normalize_finish_reason("FinishReason.MAX_TOKENS") == "MAX_TOKENS"
    assert GoogleProvider._normalize_finish_reason(None) is None
    assert GoogleProvider._normalize_finish_reason("") is None

    # 枚举形态（有 name 属性）优先取 name
    enum_like = SimpleNamespace(name="stop", value=1)
    assert GoogleProvider._normalize_finish_reason(enum_like) == "STOP"

    # 只有 value 时取 value
    value_only = SimpleNamespace(value="max_tokens")
    assert GoogleProvider._normalize_finish_reason(value_only) == "MAX_TOKENS"


# ── Ark Responses 的 format／tools／tool_choice 形态判据 ──
#
# ``VolcengineProvider._convert_responses_format``／``_convert_responses_tools``／
# ``_convert_responses_tool_choice`` 与 OpenAI 兼容侧同构，但四条判据在整套测试里
# 从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后全量 3434 条仍全绿）。
# 既有用例只覆盖了容器层（``tools`` 非列表）与「``function`` 非对象」，下面这几条
# 全绕开了。删除后行为确实改变（实测）：
# * 863 ``json_schema`` 非 dict → 裸 ``AttributeError: 'int' object has no
#   attribute 'get'``（``schema.get("name")``）；
# * 878 ``type`` 既非 ``json_schema`` 也非 ``json_object``、且没有 ``format`` 键
#   → **静默返回空字典**，请求体里 ``format`` 直接消失，用户要的结构化输出约束
#   被无声丢弃；
# * 893 工具元素非 Mapping → 整数抛裸 ``TypeError: argument of type 'int' is not
#   iterable``、字符串抛 ``dictionary update sequence ...``；
# * 911 ``function`` 非对象 → 裸 ``AttributeError: 'str' object has no attribute
#   'get'``；
# * 942 未知字符串 → 消息从「不支持 tool_choice: x」退化成 959 那条通用消息；
# * 955 ``function`` tool_choice 缺 name → **静默接受**，产出 ``name: None``；
# * 959 非字符串非工具对象 → **静默返回 None**，请求不带 ``tool_choice``，等于
#   「让模型自己决定用不用工具」。


@pytest.mark.parametrize(
    "response_format",
    [{"type": "json_schema", "json_schema": 5}, {"type": "json_schema", "json_schema": "s"}],
    ids=["int", "str"],
)
def test_ark_responses_rejects_non_object_json_schema(response_format):
    with pytest.raises(ValueError, match="json_schema 必须是对象"):
        VolcengineProvider._convert_responses_format(response_format)


@pytest.mark.parametrize(
    "response_format",
    [{"type": "xml"}, {}, {"type": ""}, {"type": 5}],
    ids=["xml", "empty", "blank-type", "int-type"],
)
def test_ark_responses_rejects_unsupported_response_format(response_format):
    """不支持的 ``type`` 必须报错。

    删除守卫后实测静默返回 ``{}``——请求体里的 ``format`` 消失，用户配置的
    结构化输出约束被无声丢弃。
    """
    with pytest.raises(ValueError, match="仅支持 json_schema 或 json_object"):
        VolcengineProvider._convert_responses_format(response_format)


@pytest.mark.parametrize("tool", [5, "x", ["y"]], ids=["int", "str", "list"])
def test_ark_responses_tools_reject_non_mapping_entries(tool):
    with pytest.raises(ValueError, match=r"工具定义\[0\] 必须是对象"):
        VolcengineProvider._convert_responses_tools([tool])


def test_ark_responses_tools_reject_non_mapping_function():
    with pytest.raises(ValueError, match=r"工具定义\[0\]\.function 必须是对象"):
        VolcengineProvider._convert_responses_tools([{"function": "bad"}])


def test_ark_responses_tool_choice_rejects_unknown_string():
    """未知字符串必须给出指明取值的消息，而不是落到通用兜底。"""
    with pytest.raises(ValueError, match="不支持 tool_choice: x"):
        VolcengineProvider._convert_responses_tool_choice("x")


@pytest.mark.parametrize(
    "tool_choice",
    [{"type": "function", "function": {"x": 1}}, {"type": "function", "function": {}}],
    ids=["missing-name", "empty-function"],
)
def test_ark_responses_tool_choice_requires_function_name(tool_choice):
    """``function`` tool_choice 必须带 name。

    删除守卫后实测静默接受，产出 ``{"type": "function", "name": None}``。
    """
    with pytest.raises(ValueError, match="缺少 name"):
        VolcengineProvider._convert_responses_tool_choice(tool_choice)


@pytest.mark.parametrize(
    "tool_choice",
    [5, {}, {"type": "custom", "name": "n"}, []],
    ids=["int", "empty-dict", "custom-type", "list"],
)
def test_ark_responses_tool_choice_rejects_unsupported_shapes(tool_choice):
    """兜底判据必须真的兜住。

    删除后实测静默返回 ``None``——请求里干脆不带 ``tool_choice``，等于「让模型
    自己决定用不用工具」。
    """
    with pytest.raises(ValueError, match="必须是 auto、none、required 或工具对象"):
        VolcengineProvider._convert_responses_tool_choice(tool_choice)


def test_ark_responses_format_still_accepts_supported_shapes():
    """反向守卫：``json_schema``／``json_object``／已带 ``format`` 的形态照常转换。"""
    assert VolcengineProvider._convert_responses_format({"type": "json_object"}) == {
        "format": {"type": "json_object"}
    }
    assert VolcengineProvider._convert_responses_format({"type": "json_schema"}) == {
        "format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {}, "strict": True},
        }
    }
    assert VolcengineProvider._convert_responses_format({"format": {"type": "text"}}) == {
        "format": {"type": "text"}
    }


def test_ark_responses_tool_choice_still_accepts_supported_values():
    """反向守卫：别名与原生工具对象照常转换。"""
    assert VolcengineProvider._convert_responses_tool_choice("REQUIRED") == "required"
    assert VolcengineProvider._convert_responses_tool_choice(
        {"type": "function", "function": {"name": "f"}}
    ) == {"type": "function", "name": "f"}
    assert VolcengineProvider._convert_responses_tool_choice({"type": "web_search"}) == {
        "type": "web_search"
    }


# ── Anthropic tool_choice／图片 data URI 的四条形态判据 ──
#
# 这四条在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后
# 全量 3479 条仍全绿）。既有用例覆盖的是「未知字符串」「function 缺 name」与
# 「data URI 不是有效 Base64」，形态判据全绕开了。删除后行为确实改变（实测）：
# * 576 ``tool_choice`` 既非字符串也非 Mapping（整数）→ 裸 ``TypeError: 'int'
#   object is not iterable``（``dict(tool_choice)``）；列表则退化成兜底消息；
# * 600 非字符串非工具对象（空 dict、未知 ``type``）→ **静默返回 None**，
#   请求不带 ``tool_choice``，等于「让模型自己决定用不用工具」；
# * 627 data URI 无逗号 → 消息从「缺少逗号分隔符」退化成裸
#   ``ValueError: not enough values to unpack (expected 2, got 1)``；
# * 636 data URI 缺 MIME 或 base64 标记 → **静默接受**，产出一个
#   ``media_type`` 为空串（或 ``header`` 被当成 data）的图片块发给上游。


@pytest.mark.parametrize("tool_choice", [5, 1.5, object()], ids=["int", "float", "object"])
def test_anthropic_tool_choice_rejects_non_string_non_mapping(tool_choice):
    with pytest.raises(ValueError, match="必须是字符串或对象"):
        AnthropicProvider._convert_tool_choice(tool_choice)


@pytest.mark.parametrize(
    "tool_choice",
    [{}, {"type": "custom", "name": "n"}, {"type": ""}, {"type": 5}],
    ids=["empty", "custom", "blank-type", "int-type"],
)
def test_anthropic_tool_choice_rejects_unsupported_shapes(tool_choice):
    """兜底判据必须真的兜住，不能静默返回 None。"""
    with pytest.raises(ValueError, match="必须是 auto、any、none、tool 或 function 对象"):
        AnthropicProvider._convert_tool_choice(tool_choice)


def test_anthropic_image_data_uri_requires_comma():
    """``data:`` URI 缺逗号时必须给「缺少逗号分隔符」，而不是裸解包错误。"""
    with pytest.raises(ValueError, match="缺少逗号分隔符"):
        AnthropicProvider._convert_content(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64"}}]
        )


@pytest.mark.parametrize(
    "url",
    ["data:image/png,AAAA", "data:;base64,AAAA", "data:,AAAA", "data:image/png;utf8,AAAA"],
    ids=["no-base64-marker", "no-mime", "neither", "wrong-encoding"],
)
def test_anthropic_image_data_uri_requires_mime_and_base64(url):
    """``data:`` URI 必须同时带 MIME 类型与 ``base64`` 标记。

    删除守卫后实测静默接受，产出 ``media_type`` 为空串的图片块发给上游。
    """
    with pytest.raises(ValueError, match="必须包含 MIME 类型和 base64 标记"):
        AnthropicProvider._convert_content([{"type": "image_url", "image_url": {"url": url}}])


def test_anthropic_image_data_uri_still_accepts_well_formed_value():
    """反向守卫：合法 data URI 照常转成 base64 图片块。"""
    converted = AnthropicProvider._convert_content(
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUFB"}}]
    )

    assert converted == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "QUFB"},
        }
    ]


def test_anthropic_tool_choice_still_accepts_supported_values():
    """反向守卫：别名、原生工具对象与带 name 的裸对象照常转换。"""
    assert AnthropicProvider._convert_tool_choice("required") == {"type": "any"}
    assert AnthropicProvider._convert_tool_choice({"type": "tool", "name": "t"}) == {
        "type": "tool",
        "name": "t",
    }
    assert AnthropicProvider._convert_tool_choice({"name": "n"}) == {"type": "tool", "name": "n"}


# ── Google 配置边界与响应格式的十八条判据 ──
#
# 这些判据在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后
# 全量 3505 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 681 ``debug_config`` 含未知字段 → **静默接受**，``nope`` 被 pydantic 静默
#   忽略、产出一个什么都没设的 ``DebugConfig``；
# * 685 当前 SDK 没有 ``DebugConfig`` 时 → 漏出裸 ``NameError``，而不是
#   ``RuntimeError``「当前 google-genai SDK 不支持 debug_config」；
# * 689 ``DebugConfig`` 构造失败（``replay_id=5`` 这类非法值）→ **静默返回
#   None**，配置被无声丢弃；
# * 947 ``http_options`` 与请求级 HTTP 扩展同时出现 → **静默接受**，请求级值
#   覆盖 SDK 容器；
# * 951 ``config`` 与同级配置关键字同时出现 → **静默返回 config**，同级的
#   ``temperature`` 等被无声丢弃；
# * 979 Embedding 请求含不支持的参数 → **静默接受**，参数原样进 SDK；
# * 1127／1134／1137 ``response_format`` 及其 ``json_schema``／``schema``
#   非对象 → 裸 ``AttributeError: 'int' object has no attribute 'get'`` 或
#   ``TypeError: 'int' object is not iterable``；
# * 1144 ``type`` 既非 ``json_object`` 也非 ``json_schema`` → **静默返回
#   None**，结构化输出约束消失；
# * 1180 ``extra_body`` 与模型 options 重复 → **静默覆盖**，模型级值被请求级顶掉；
# * 1191 ``http_options`` 与请求级扩展冲突 → **静默接受**，两者同时生效；
# * 1391／1406／1412 同一量在两处配置（``max_tokens`` 与
#   ``max_output_tokens``、``thinking`` 与 ``thinking_config``、``response_format``
#   与请求参数）→ 前两者**静默丢弃**一个（1406 实测被 pydantic 拒绝成
#   ``ValidationError``），1412 **静默接受**请求级值；
# * 1477 data URI 无逗号 → 退化成裸 ``ValueError: not enough values to unpack``；
# * 1481 data URI 缺 MIME → **静默接受**，产出 ``mime_type=""`` 的块；
# * 1541 ``urlsplit`` 解析失败 → ``parsed`` 从未赋值，漏出裸 ``UnboundLocalError``。
#
# 同批的 1164／1171 经甄别是**等价变异**，不补测试：两条与共享 helper
# ``validate_secret_free_payload`` 抛**同一条消息的同一个 ``ValueError````，
# 删除后结论不变。


class _GoogleConfigStub:
    """``GenerateContentConfig`` 的最小替身，能记录构造参数。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _google_config_provider(**options):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-1.5-flash"
    provider._provider = "google"
    provider._options = dict(options)
    return provider


def test_google_debug_config_rejects_unknown_field():
    """``debug_config`` 含未知字段必须报错。

    删除后实测静默产出「什么都没设」的 ``DebugConfig``——配置写错了却看不出来。
    """
    with pytest.raises(ValueError, match="包含不支持的字段: nope"):
        GoogleProvider._coerce_debug_config({"nope": 1})


def test_google_debug_config_reports_missing_sdk_support(monkeypatch):
    """当前 SDK 没有 ``DebugConfig`` 时必须报「SDK 不支持」，不能漏出 ``NameError``。"""
    import sys

    monkeypatch.setitem(sys.modules, "google.genai.client", None)

    with pytest.raises(RuntimeError, match="SDK 不支持 debug_config"):
        GoogleProvider._coerce_debug_config({"client_mode": "x"})


@pytest.mark.parametrize(
    "debug_config",
    [{"replay_id": 5}, {"replays_directory": 5}, {"client_mode": 5}],
    ids=["replay-id", "replays-directory", "client-mode"],
)
def test_google_debug_config_reports_invalid_values(debug_config):
    """``DebugConfig`` 构造失败必须收敛成「配置无效」，不能静默返回 ``None``。"""
    with pytest.raises(ValueError, match="debug_config 配置无效"):
        GoogleProvider._coerce_debug_config(debug_config)


def test_google_debug_config_still_accepts_valid_fields():
    """反向守卫：合法字段照常转换。"""
    config = GoogleProvider._coerce_debug_config({"client_mode": "auto", "replay_id": "r1"})

    assert config.client_mode == "auto"
    assert config.replay_id == "r1"


def test_google_resource_config_rejects_http_options_with_request_level_fields():
    """``http_options`` 不能与请求级 HTTP 扩展同时传入。

    删除后实测静默接受，请求级值直接覆盖 SDK 容器——用户配置的两套 HTTP 设置
    谁生效取决于调用顺序。
    """
    with pytest.raises(ValueError, match="http_options 不能与请求级 HTTP 配置同时传入"):
        GoogleProvider._resource_config(
            _GoogleConfigStub,
            None,
            {"http_options": {"a": 1}, "extra_body": {"b": 2}},
        )


def test_google_resource_config_rejects_config_with_sibling_keywords():
    """``config`` 不能与同级配置关键字同时传入，否则同级值被无声丢弃。"""
    with pytest.raises(ValueError, match="config 不能与同级配置关键字同时传入"):
        GoogleProvider._resource_config(_GoogleConfigStub, {"x": 1}, {"temperature": 0.5})


def test_google_embedding_rejects_unsupported_request_kwargs():
    """Embedding 请求含不支持参数必须报错，不能静默进 SDK。"""
    with pytest.raises(ValueError, match="Embedding 不支持请求参数: bogus"):
        GoogleProvider._validate_embedding_kwargs({"bogus": 1})


def test_google_embedding_still_accepts_supported_request_kwargs():
    """反向守卫：受支持的参数照常通过（``None`` 值视为未设置）。"""
    assert (
        GoogleProvider._validate_embedding_kwargs(
            {"task_type": "RETRIEVAL_DOCUMENT", "auto_truncate": None}
        )
        is None
    )


@pytest.mark.parametrize("response_format", [5, "json_object", ["a"]], ids=["int", "str", "list"])
def test_google_response_format_rejects_non_mapping(response_format):
    with pytest.raises(ValueError, match="response_format 必须是对象"):
        GoogleProvider._response_format_values(response_format)


@pytest.mark.parametrize("json_schema", [5, "s", ["a"]], ids=["int", "str", "list"])
def test_google_response_format_rejects_non_mapping_json_schema(json_schema):
    with pytest.raises(ValueError, match=r"json_schema 必须是对象"):
        GoogleProvider._response_format_values({"type": "json_schema", "json_schema": json_schema})


@pytest.mark.parametrize("schema", [5, "s", ["a"]], ids=["int", "str", "list"])
def test_google_response_format_rejects_non_mapping_schema(schema):
    with pytest.raises(ValueError, match="必须是 JSON Schema 对象"):
        GoogleProvider._response_format_values(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )


@pytest.mark.parametrize(
    "response_format",
    [{"type": "xml"}, {}, {"type": ""}, {"type": 5}, {"schema": {}}],
    ids=["xml", "empty", "blank-type", "int-type", "schema-only"],
)
def test_google_response_format_rejects_unsupported_types(response_format):
    """不支持的 ``type`` 必须报错。

    删除后实测静默返回 ``None``——用户配置的结构化输出约束从请求里消失。
    """
    with pytest.raises(ValueError, match="仅支持 json_object 或 json_schema"):
        GoogleProvider._response_format_values(response_format)


def test_google_response_format_still_accepts_supported_types():
    """反向守卫：``json_object`` 与 ``json_schema`` 照常转换。"""
    assert GoogleProvider._response_format_values({"type": "json_object"}) == {
        "response_mime_type": "application/json"
    }
    assert GoogleProvider._response_format_values(
        {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    ) == {
        "response_mime_type": "application/json",
        "response_json_schema": {"type": "object"},
    }


_GOOGLE_HTTP_BASE = {
    "extra_body": None,
    "extra_headers": None,
    "extra_query": None,
    "timeout": None,
}


def test_google_extra_body_cannot_be_silently_overridden():
    """请求级 ``extra_body`` 与模型 options 同名键必须报错，不能静默覆盖。

    删除后实测模型级 ``{"k": 1}`` 被请求级 ``{"k": 2}`` 顶掉，谁生效取决于
    调用顺序且无任何提示。
    """
    provider = _google_config_provider(extra_body={"k": 1})

    with pytest.raises(ValueError, match="extra_body 与模型 options 重复: k"):
        provider._generation_http_options(**{**_GOOGLE_HTTP_BASE, "extra_body": {"k": 2}})


def test_google_http_options_cannot_coexist_with_request_level_extensions():
    """``http_options`` 与请求级 HTTP 扩展不能同时生效。

    删除后实测两者同时返回、都被套用——用户配置的两套 HTTP 设置发生叠加。
    """
    provider = _google_config_provider(http_options={"h": 1})

    with pytest.raises(ValueError, match="不能与请求级"):
        provider._generation_http_options(**{**_GOOGLE_HTTP_BASE, "extra_body": {"b": 2}})


def test_google_http_options_and_extra_body_still_merge_when_disjoint():
    """反向守卫：不冲突的 ``extra_body`` 与请求级扩展照常合并。"""
    provider = _google_config_provider(extra_body={"k": 1})

    merged, *_ = provider._generation_http_options(**{**_GOOGLE_HTTP_BASE, "extra_body": {"j": 2}})

    assert merged == {"k": 1, "j": 2}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"max_tokens": 10, "max_output_tokens": 20},
            "max_tokens 与 options.max_output_tokens 重复",
        ),
        (
            {"thinking": {"a": 1}, "thinking_config": {"b": 2}},
            "thinking 与 options.thinking_config 重复",
        ),
    ],
    ids=["max-tokens", "thinking"],
)
def test_google_generation_config_rejects_duplicate_scalar_options(options, message):
    """同一量在两处配置必须报错，不能静默丢弃其一。"""
    provider = _google_config_provider(**options)

    with pytest.raises(ValueError, match=message):
        provider._build_generation_config(None, None, None)


def test_google_generation_config_rejects_response_format_duplicate():
    """``response_format`` 同时出现在 options 与请求参数必须报错。"""
    provider = _google_config_provider(response_format={"type": "json_object"})

    with pytest.raises(ValueError, match="response_format 与模型 options 重复"):
        provider._build_generation_config(None, None, None, response_format={"type": "json_object"})


def test_google_decode_data_uri_requires_comma():
    with pytest.raises(ValueError, match="缺少逗号分隔符"):
        GoogleProvider._decode_data_uri("data:image/png;base64", "媒体")


def test_google_decode_data_uri_requires_mime_type():
    """``data:`` URI 缺 MIME 类型必须报错，不能产出 ``mime_type=""`` 的块。"""
    with pytest.raises(ValueError, match="缺少 MIME 类型"):
        GoogleProvider._decode_data_uri("data:;base64,QUFB", "媒体")


def test_google_decode_data_uri_still_accepts_well_formed_value():
    """反向守卫：合法 data URI 照常解码。"""
    assert GoogleProvider._decode_data_uri("data:image/png;base64,QUFB", "媒体") == (
        "image/png",
        b"AAA",
    )


def test_google_media_uri_reports_invalid_uri_readably():
    """``urlsplit`` 解析失败必须收敛成带上下文的 ``ValueError``。

    删除守卫后 ``parsed`` 从未赋值，漏出裸 ``UnboundLocalError``。
    """
    with pytest.raises(ValueError, match="URI 无效"):
        GoogleProvider._contents(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "http://[bad"}}],
                }
            ],
            None,
            None,
        )


# ── Google 工具调用／工具结果四条形态判据 ──
#
# 这四条在整套测试里从未执行（行覆盖证实；把它们的 ``raise`` 换成 ``pass`` 后
# 全量 3544 条仍全绿）。删除后行为确实改变（实测，均经 ``GoogleProvider._contents``）：
# * 1811 ``function_call`` 缺 name（或 name 非字符串）→ 缺名时 **静默产出**一个
#   ``name=None`` 的 ``FunctionCall``，非字符串时漏出裸
#   ``ValidationError: 1 validation error for FunctionCall``；
# * 1819 ``arguments`` 是非法 JSON 字符串 → 消息从「不是有效 JSON」退化成
#   「必须是 JSON 对象」——把「JSON 语法错了」说成「JSON 顶层不是对象」；
# * 1823 ``arguments`` 是非对象（数组）→ 裸 ``TypeError: cannot convert
#   dictionary update sequence element #0 to a sequence``（``dict(arguments)``）；
# * 1837 ``function_response`` 无可关联的 ``tool_call_id``／``name`` → 缺标识时
#   **静默产出**一个无名的 ``FunctionResponse``（Google 侧无法关联回任何调用），
#   非字符串标识时漏出裸 ``ValidationError``。
#
# 同批的 1795 经甄别是**等价变异**，不补测试：它是 ``role="tool"`` 路径的同一条
# 判据，删掉后落到 1837 那条 ``function_response`` 分支，抛**同一条消息**的同一个
# ``ValueError``，结论不变。


def _google_contents(parts):
    return GoogleProvider._contents([{"role": "user", "content": parts}], None, None)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "function_call", "arguments": {}},
        {"type": "function_call", "name": 5, "arguments": {}},
    ],
    ids=["missing-name", "non-string-name"],
)
def test_google_function_call_requires_string_name(part):
    """``function_call`` 必须带字符串 name。

    删除守卫后缺名静默产出 ``name=None`` 的 ``FunctionCall``，非字符串 name
    漏出裸 pydantic ``ValidationError``。
    """
    with pytest.raises(ValueError, match="function_call 缺少 name"):
        _google_contents([part])


def test_google_function_call_rejects_invalid_json_arguments():
    """``arguments`` 是非法 JSON 字符串时必须报「不是有效 JSON」。"""
    with pytest.raises(ValueError, match="arguments 不是有效 JSON"):
        _google_contents([{"type": "function_call", "name": "f", "arguments": "{bad"}])


def test_google_function_call_rejects_non_object_arguments():
    """``arguments`` 必须是 JSON 对象，数组要报「必须是 JSON 对象」而不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="arguments 必须是 JSON 对象"):
        _google_contents([{"type": "function_call", "name": "f", "arguments": [1, 2]}])


@pytest.mark.parametrize(
    "part",
    [
        {"type": "function_response", "response": {}},
        {"type": "function_response", "response": {}, "tool_call_id": 5},
        {"type": "function_response", "response": {}, "name": 5},
    ],
    ids=["no-identifier", "non-string-id", "non-string-name"],
)
def test_google_function_response_requires_linkable_identifier(part):
    """``function_response`` 必须带可关联的 ``tool_call_id`` 或 ``name``。

    删除守卫后缺标识静默产出无名的 ``FunctionResponse``，Google 侧无法关联回
    任何一次调用；非字符串标识漏出裸 pydantic ``ValidationError``。
    """
    with pytest.raises(ValueError, match="function_response 缺少可关联的 tool_call_id/name"):
        _google_contents([part])


def test_google_tool_parts_still_convert_when_well_formed():
    """反向守卫：合法的 ``function_call`` 与 ``function_response`` 照常转换。"""
    contents = _google_contents(
        [
            {"type": "function_call", "id": "c1", "name": "f", "arguments": {"a": 1}},
            {"type": "function_response", "tool_call_id": "c1", "response": {"ok": True}},
        ]
    )

    calls = [
        part.function_call for content in contents for part in content.parts if part.function_call
    ]
    responses = [
        part.function_response
        for content in contents
        for part in content.parts
        if part.function_response
    ]
    assert [call.name for call in calls] == ["f"]
    assert calls[0].args == {"a": 1}
    assert [response.name for response in responses] == ["f"]


# ── Google 渠道 google.py 覆盖缺口守卫（选项／凭证／资源／媒体／partial_args／
#    分片合并／token 统计／Embedding／流终止／close）──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3638 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * 239／391 ``options`` 非 Mapping → 裸 ``TypeError: 'int' object is not iterable``；
# * 401 options 键非字符串 → 裸 ``TypeError: sequence item 0: expected str instance``；
# * 585 ``http_options`` 键非字符串 → **静默返回 None**、校验整段跳过；
# * 677 ``debug_config`` 键非字符串 → 裸 ``TypeError``；
# * 448 ``_require_resource`` 资源为 None → **静默返回 None**；
# * 490 异步 ``aio`` 节点缺失 → 消息退化成「未提供 Auth Token 资源」，丢掉「异步」；
# * 370 缺 google-auth → 消息退化成「凭证文件无法加载」；
# * 374 凭证文件无法加载 → 裸 ``UnboundLocalError``；
# * 925 ``output_dimensionality`` 非整数 → 裸 ``TypeError``，布尔 → **静默接受 True**；
# * 929 维度越界 → **静默接受 0／10**9；
# * 1571 Vertex 模式非 gs:// HTTPS → 消息退化，丢掉「Vertex 模式」；
# * 1590 ``mime_type`` 非字符串 → 裸 ``AttributeError``，空白 → 消息退化；
# * 1795 工具结果缺标识 → **静默产出**无名 ``FunctionResponse``；
# * 1958／1973／2000 ``parts``／``embeddings``／``candidates`` 非列表 → 裸 ``TypeError``；
# * 2100 ``partial_arg`` 缺值 → **静默返回 None**；
# * 2141 json_path 索引过大 → **静默接受**（会按该下标把列表填充到那么大）；
# * 2153 json_path 不支持的写法 → **死循环**（``rest`` 不推进，实测挂起）；
# * 2164 ``partial_args`` 非列表 → 裸 ``TypeError``；
# * 2172 根值非对象 → 裸 ``ValueError: dictionary update sequence element``；
# * 2180／2192 json_path 无法写入 → 裸 ``AttributeError``／``TypeError``；
# * 2286／2316／2354／2376／2403／2406／2409 工具调用分片合并的七条冲突判据 →
#   **静默合并**两个不同调用（或换成另一条同族消息）；
# * 2781／2786／2828／3016／3021／3059／3065 token 统计的字段支持判据 →
#   消息退化成裸 ``AttributeError``（``models``／``aio`` 不存在）；
# * 2832／3069 compute_tokens 的 tools 判据 → 被 ``_validate_compute_token_fields``
#   先拦（**死代码**）；
# * 3184／3307／4382／4488 流在终止事件前结束 → **静默返回已收片段**；
# * 3556／3659 ``validate_reward`` 双配置 → 裸 ``AttributeError``；
# * 4012 Live 缺 ``aio`` → 裸 ``UnboundLocalError``；4016 connect 非可调用 → 裸 ``TypeError``；
# * 4530／4594 Embedding 覆盖 ``model`` → 消息退化；4554／4620 Vertex 单内容非唯一 →
#   **静默接受多个 embedding**；4564／4632 数量不一致 → **静默返回不匹配的向量**；
# * 4692／4714 ``close``／``aclose`` 首个异常 → **静默吞掉**。
#
# 同批的 1164／1171／1519／1862／1865／1869 经甄别是**等价变异**，不补测试：
# 1164／1171 删掉后落到下游 ``validate_secret_free_payload`` 的同一条消息；
# 1519 删掉后落到调用方 ``_contents`` 的同一条消息；1862／1865／1869 删掉后落到
# 上游 ``normalize_chat_messages`` 的同族消息，结论都不变。


def _google_options_provider(**options):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-1.5-flash"
    provider._provider = "google"
    provider._options = dict(options)
    provider._client = None
    return provider


def _google_contents_vertex(parts):
    """``_contents`` 的 Vertex 模式入口（既有 ``_google_contents`` 不带该开关）。"""
    return GoogleProvider._contents(
        [{"role": "user", "content": parts}], None, None, vertex_mode=True
    )


@pytest.mark.parametrize(
    "options",
    [pytest.param(5, id="int"), pytest.param([1, 2], id="list")],
)
def test_google_constructor_rejects_non_mapping_options(options):
    """``options`` 非 Mapping 必须报 ``ValueError``，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="Google options 必须是对象"):
        GoogleProvider("m", options=options)


def test_google_validate_options_rejects_non_mapping():
    """``_validate_options`` 自身也要收敛非 Mapping。"""
    with pytest.raises(ValueError, match="Google options 必须是对象"):
        GoogleProvider._validate_options(5)


def test_google_options_reject_non_string_key():
    """``options`` 键非字符串必须报错，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="Google options 的键必须是字符串"):
        GoogleProvider("m", options={1: "x"})


def test_google_http_options_reject_non_string_key():
    """``http_options`` 键非字符串必须报错，不能静默跳过校验。"""
    with pytest.raises(ValueError, match=r"options\.http_options 的键必须是字符串"):
        GoogleProvider._validate_options({"http_options": {1: "x"}})


def test_google_debug_config_rejects_non_string_key():
    """``debug_config`` 键非字符串必须报错，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="Google debug_config 的键必须是字符串"):
        GoogleProvider._coerce_debug_config({1: "x"})


def test_google_require_resource_rejects_none_resource():
    """资源属性存在但为 ``None`` 时必须报错，不能静默返回 ``None``。"""
    with pytest.raises(NotImplementedError, match="未提供 Tuned Models资源"):
        GoogleProvider._require_resource(
            SimpleNamespace(tuned_models=None), "tuned_models", "Tuned Models"
        )


def test_google_async_resource_reports_missing_aio_node():
    """异步 ``aio`` 节点缺失时消息必须保留「异步」。"""
    provider = _google_options_provider()
    provider._get_client = lambda: SimpleNamespace()

    with pytest.raises(NotImplementedError, match="异步 Auth Token资源"):
        provider._require_provider_resource("async_create_auth_token")


def test_google_credentials_require_google_auth(monkeypatch):
    """缺 google-auth 时必须报专属消息，不能退化成「文件无法加载」。"""
    import builtins

    original = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "google.auth" or name.startswith("google.auth."):
            raise ImportError("no google-auth")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="缺少 google-auth"):
        GoogleProvider._load_credentials_file("/x")


def test_google_credentials_file_failure_reports_readably():
    """凭证文件无法加载时必须报 ``ValueError``，不是裸 ``UnboundLocalError``。"""
    with pytest.raises(ValueError, match="GOOGLE_APPLICATION_CREDENTIALS 文件无法加载"):
        GoogleProvider._load_credentials_file("/nonexistent/definitely-not-here.json")


@pytest.mark.parametrize(
    "value",
    [pytest.param("x", id="str"), pytest.param(True, id="bool"), pytest.param(1.5, id="float")],
)
def test_google_embedding_dimensions_reject_non_integer(value):
    """``output_dimensionality`` 非整数必须报错，布尔不能静默当 1。"""
    with pytest.raises(ValueError, match="output_dimensionality 必须是整数"):
        GoogleProvider._validate_embedding_dimensions(value)


@pytest.mark.parametrize(
    "value",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative"), pytest.param(10**9, id="huge")],
)
def test_google_embedding_dimensions_reject_out_of_range(value):
    """``output_dimensionality`` 越界必须报错，不能静默接受。"""
    with pytest.raises(ValueError, match="output_dimensionality 必须在 1 到"):
        GoogleProvider._validate_embedding_dimensions(value)


def test_google_vertex_media_rejects_non_gs_https_uri():
    """Vertex 模式仅支持 gs:// 或可公开访问 HTTPS，消息必须点明 Vertex 模式。"""
    with pytest.raises(ValueError, match="Vertex/Enterprise 模式仅支持 gs://"):
        _google_contents_vertex([{"type": "file", "file": "https://user:pass@example.com/x"}])


@pytest.mark.parametrize(
    "mime",
    [pytest.param(5, id="int"), pytest.param("  ", id="blank")],
)
def test_google_media_mime_type_must_be_non_empty_string(mime):
    """媒体块的 ``mime_type`` 必须是非空字符串，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="mime_type 必须是非空字符串"):
        _google_contents(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/x.png", "mime_type": mime},
                }
            ]
        )


def test_google_tool_result_requires_linkable_identifier():
    """工具结果块缺可关联标识时必须报错，不能静默产出无名响应。"""
    with pytest.raises(ValueError, match="工具结果缺少可关联的 tool_call_id/name"):
        _google_contents([{"type": "tool_result", "response": {}}])


@pytest.mark.parametrize(
    "method,attr,source",
    [
        pytest.param("_require_parts", "parts", "响应", id="parts"),
        pytest.param("_require_embeddings", "embeddings", "响应", id="embeddings"),
        pytest.param("_require_candidates", "candidates", "响应", id="candidates"),
    ],
)
def test_google_response_containers_reject_non_list(method, attr, source):
    """响应容器字段非列表必须报 ``RuntimeError``，不是裸 ``TypeError``。"""
    owner = SimpleNamespace(**{attr: 5})

    with pytest.raises(RuntimeError, match="必须是列表"):
        getattr(GoogleProvider, method)(owner, source=source)


def test_google_partial_arg_value_requires_a_value():
    """``PartialArg`` 没有任何标量值时必须报错，不能静默返回 ``None``。"""
    with pytest.raises(ValueError, match="partial_args 缺少有效值"):
        GoogleProvider._partial_arg_value(SimpleNamespace())


def test_google_partial_arg_tokens_reject_oversized_index():
    """json_path 索引过大必须报错，不能静默接受（会按该下标填充列表）。"""
    with pytest.raises(ValueError, match="json_path 索引过大"):
        GoogleProvider._partial_arg_tokens("$[999999]")


def test_google_partial_arg_tokens_reject_unsupported_path():
    """不支持的 json_path 必须报错——删掉守卫会**死循环**（``rest`` 不推进）。"""
    with pytest.raises(ValueError, match="不支持的 json_path"):
        GoogleProvider._partial_arg_tokens("$..x")


def test_google_partial_args_to_mapping_rejects_non_list():
    """``partial_args`` 非列表且非 Mapping 必须报错，不是裸 ``TypeError``。"""
    with pytest.raises(ValueError, match="partial_args 必须是列表"):
        GoogleProvider._partial_args_to_mapping(5)


def test_google_partial_args_root_value_must_be_object():
    """根路径的 partial 值必须是 JSON 对象，不是裸 ``ValueError: dictionary update``。"""
    with pytest.raises(ValueError, match="根值必须是 JSON 对象"):
        GoogleProvider._partial_args_to_mapping([SimpleNamespace(json_path="$", string_value="x")])


def test_google_partial_args_write_into_list_rejects_conflict():
    """向列表元素写入子字段时类型冲突必须报错，不是裸 ``AttributeError``。"""
    with pytest.raises(ValueError, match="json_path 无法写入"):
        GoogleProvider._partial_args_to_mapping(
            [
                SimpleNamespace(json_path="$[0]", string_value="x"),
                SimpleNamespace(json_path="$[0].a", string_value="y"),
            ]
        )


def test_google_partial_args_write_into_scalar_rejects_conflict():
    """向标量写入子字段必须报错，不是裸 ``TypeError: 'str' object does not support``。"""
    with pytest.raises(ValueError, match="json_path 无法写入"):
        GoogleProvider._partial_args_to_mapping(
            [
                SimpleNamespace(json_path="$.a", string_value="x"),
                SimpleNamespace(json_path="$.a.b", string_value="y"),
            ]
        )


def _merge_into(accumulator, tool_call):
    GoogleProvider._merge_tool_call(accumulator, tool_call)
    return accumulator


def _fc(**kwargs):
    return {"type": "function", **kwargs}


def test_google_merge_tool_call_rejects_move_onto_existing_key():
    """把分片并入已存在的 key 时必须报错，不能静默覆盖。"""
    with pytest.raises(RuntimeError, match="id/index 指向多个调用"):
        _merge_into(
            {("index", 5): _fc(index=5, name="other"), ("id", "a"): _fc(id="a", name="f")},
            _fc(index=5, id="a", name="f"),
        )


def test_google_merge_tool_call_index_branch_rejects_multiple_id_matches():
    """index 分支里同一 id 命中多个候选时必须报错。"""
    with pytest.raises(RuntimeError, match="id 指向多个调用"):
        _merge_into(
            {("id", "a"): _fc(id="a", name="f"), ("position", 0): _fc(name="f", id="a")},
            _fc(index=1, id="a", name="f"),
        )


def test_google_merge_tool_call_id_branch_rejects_ambiguous_name_candidates():
    """call_id 分支下同名多候选时必须报错，不能静默挑一个。"""
    with pytest.raises(RuntimeError, match="无法关联并行调用"):
        _merge_into(
            {("position", 0): _fc(name="f"), ("position", 1): _fc(name="f")},
            _fc(call_id="c1", name="f"),
        )


def test_google_merge_tool_call_without_identity_rejects_ambiguous_candidates():
    """既无 id 也无 index 且同名多候选时必须报错。"""
    with pytest.raises(RuntimeError, match="无法关联并行调用"):
        _merge_into(
            {("position", 0): _fc(name="f"), ("position", 1): _fc(name="f")},
            _fc(name="f"),
        )


def test_google_merge_tool_call_rejects_conflicting_id():
    """同一分片键上的 id 冲突必须报错，不能静默保留旧值。"""
    with pytest.raises(RuntimeError, match="id 指向多个调用"):
        _merge_into(
            {("index", 0): _fc(index=0, id="a", name="f")},
            _fc(index=0, call_id="b", name="f"),
        )


def test_google_merge_tool_call_rejects_conflicting_index():
    """同一分片键上的 index 冲突必须报错。"""
    with pytest.raises(RuntimeError, match="index 指向多个调用"):
        _merge_into(
            {("id", "a"): _fc(id="a", index=0, name="f")},
            _fc(index=1, id="a", name="f"),
        )


def test_google_merge_tool_call_rejects_conflicting_name():
    """同一分片键上的 name 冲突必须报错。"""
    with pytest.raises(RuntimeError, match="name 指向多个调用"):
        _merge_into({("id", "a"): _fc(id="a", name="f")}, _fc(call_id="a", name="g"))


def _google_vertex_mode_provider(vertex, **options):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-1.5-flash"
    provider._provider = "google"
    provider._options = {**options, "vertexai": vertex}
    client = SimpleNamespace(vertexai=vertex)
    provider._client = client
    provider._get_client = lambda: client
    provider._get_aclient = lambda: client
    return provider


def _tool_defs():
    return [{"type": "function", "function": {"name": "f"}}]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("field", ["system_prompt", "tools"])
def test_google_developer_count_tokens_rejects_unsupported_fields(async_mode, field):
    """Developer API 的 count_tokens 不支持 system_prompt／tools。"""
    provider = _google_vertex_mode_provider(False)
    kwargs = {field: "sys" if field == "system_prompt" else _tool_defs()}
    request = CompletionRequest(prompt="hi", **kwargs)

    with pytest.raises(ValueError, match="Developer API 的 count_tokens 不支持"):
        if async_mode:
            asyncio.run(provider.async_count_tokens(request))
        else:
            provider.count_tokens(request)


def test_google_compute_tokens_rejects_system_prompt():
    """Vertex compute_tokens 不支持 system_prompt。"""
    provider = _google_vertex_mode_provider(True)

    with pytest.raises(ValueError, match="compute_tokens 不支持 system_prompt"):
        provider.compute_tokens(CompletionRequest(prompt="hi", system_prompt="sys"))


def test_google_async_compute_tokens_requires_vertex_mode():
    """compute_tokens 仅支持 Vertex/Enterprise 模式。"""
    provider = _google_vertex_mode_provider(False)

    with pytest.raises(ValueError, match="仅支持 Vertex/Enterprise 模式"):
        asyncio.run(provider.async_compute_tokens(CompletionRequest(prompt="hi")))


def test_google_async_compute_tokens_rejects_system_prompt():
    """异步 compute_tokens 不支持 system_prompt。"""
    provider = _google_vertex_mode_provider(True)

    with pytest.raises(ValueError, match="compute_tokens 不支持 system_prompt"):
        asyncio.run(
            provider.async_compute_tokens(CompletionRequest(prompt="hi", system_prompt="sys"))
        )


def _google_stream_provider():
    """构造一个「无终止事件」的 Google 流替身。"""
    chunk = SimpleNamespace(
        response_id="r",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hi")]),
                finish_reason=None,
            )
        ],
    )

    class _AsyncIter:
        def __init__(self):
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return chunk

    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini"
    provider._provider = "google"
    provider._options = {}
    aio = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **kwargs: _AsyncIter())
    )
    provider._client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **kwargs: iter([chunk])),
        aio=aio,
    )
    provider._get_client = lambda: provider._client
    provider._get_aclient = lambda: provider._client
    return provider


def test_google_stream_events_rejects_missing_terminal_event():
    """流在终止事件之前结束必须报错，不能静默返回已收片段。"""
    provider = _google_stream_provider()

    with pytest.raises(RuntimeError, match="流在终止事件之前结束"):
        list(provider.stream_events(CompletionRequest(prompt="hi")))


def test_google_async_stream_events_rejects_missing_terminal_event():
    """异步分支同判据。"""
    provider = _google_stream_provider()

    async def _drain():
        return [event async for event in provider.astream_events(CompletionRequest(prompt="hi"))]

    with pytest.raises(RuntimeError, match="异步流在终止事件之前结束"):
        asyncio.run(_drain())


def test_google_invoke_stream_rejects_missing_terminal_event():
    """``invoke(stream=True)`` 同判据。"""
    provider = _google_stream_provider()

    with pytest.raises(RuntimeError, match="流在终止事件之前结束"):
        list(provider.invoke("hi", stream=True))


def test_google_ainvoke_stream_rejects_missing_terminal_event():
    """``ainvoke(stream=True)`` 同判据。"""
    provider = _google_stream_provider()

    async def _drain():
        return [chunk async for chunk in provider.ainvoke("hi", stream=True)]

    with pytest.raises(RuntimeError, match="异步流在终止事件之前结束"):
        asyncio.run(_drain())


@pytest.mark.parametrize("async_mode", [False, True])
def test_google_validate_tuning_reward_rejects_dual_configs(async_mode):
    """``single_reward_config`` 与 ``composite_reward_config`` 不能同时传入。"""
    provider = object.__new__(GoogleProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._client = SimpleNamespace(tunings=object(), aio=SimpleNamespace(tunings=object()))
    provider._get_client = lambda: provider._client

    async def _run():
        return await provider.async_validate_tuning_reward(
            "p",
            sample_response=None,
            example=None,
            single_reward_config=1,
            composite_reward_config=2,
        )

    with pytest.raises(ValueError, match="不能同时传入"):
        if async_mode:
            asyncio.run(_run())
        else:
            provider.validate_tuning_reward(
                "p",
                sample_response=None,
                example=None,
                single_reward_config=1,
                composite_reward_config=2,
            )


def test_google_connect_live_reports_missing_aio_node():
    """缺 ``aio`` 节点时必须报 ``NotImplementedError``，不是裸 ``UnboundLocalError``。"""
    provider = object.__new__(GoogleProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._client = SimpleNamespace()
    provider._get_client = lambda: provider._client

    with pytest.raises(NotImplementedError, match="未提供异步 Live 资源"):
        provider.connect_live()


def test_google_connect_live_rejects_non_callable_connect():
    """``live.connect`` 不可调用时必须报错，不是裸 ``TypeError``。"""
    provider = object.__new__(GoogleProvider)
    provider._model_name = "m"
    provider._options = {}
    provider._client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=5)))
    provider._get_client = lambda: provider._client

    with pytest.raises(NotImplementedError, match="未提供可调用的 Live 连接资源"):
        provider.connect_live()


def _google_embedding_provider(vertex, count):
    provider = object.__new__(GoogleProvider)
    provider._model_name = "gemini-embedding-001"
    provider._provider = "google"
    provider._options = {"vertexai": vertex}
    provider._client = None

    def _mk(**kwargs):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1]) for _ in range(count)])

    async def _amk(**kwargs):
        return _mk(**kwargs)

    models = SimpleNamespace(embed_content=_mk)
    aio = SimpleNamespace(models=SimpleNamespace(embed_content=_amk))
    provider._client = SimpleNamespace(vertexai=vertex, models=models, aio=aio)
    provider._get_client = lambda: provider._client
    return provider


@pytest.mark.parametrize("async_mode", [False, True])
def test_google_embedding_rejects_reserved_field_override(async_mode):
    """Embedding 请求覆盖 ``model`` 必须报「不允许覆盖请求字段」。"""
    provider = _google_embedding_provider(False, 1)

    async def _run():
        return await provider.aembed_documents(["a"], model="x")

    with pytest.raises(ValueError, match="Embedding 不允许覆盖请求字段"):
        if async_mode:
            asyncio.run(_run())
        else:
            provider.embed_documents(["a"], model="x")


@pytest.mark.parametrize("async_mode", [False, True])
def test_google_vertex_embedding_requires_single_embedding(async_mode):
    """Vertex 单内容请求必须返回唯一 embedding，不能静默接受多个。"""
    provider = _google_embedding_provider(True, 2)

    async def _run():
        return await provider.aembed_documents(["a"])

    with pytest.raises(RuntimeError, match="单内容请求未返回唯一 embedding"):
        if async_mode:
            asyncio.run(_run())
        else:
            provider.embed_documents(["a"])


@pytest.mark.parametrize("async_mode", [False, True])
def test_google_embedding_count_must_match_input(async_mode):
    """Embedding 返回数量与输入数量不一致时必须报错。"""
    provider = _google_embedding_provider(False, 2)

    async def _run():
        return await provider.aembed_documents(["a"])

    with pytest.raises(RuntimeError, match="返回数量与输入文本数量不一致"):
        if async_mode:
            asyncio.run(_run())
        else:
            provider.embed_documents(["a"])


@pytest.mark.parametrize("async_mode", [False, True])
def test_google_close_reports_first_error(async_mode):
    """``close``／``aclose`` 首个关闭异常必须向上报告，不能静默吞掉。"""

    class _BadClose:
        def close(self):
            raise RuntimeError("close 失败")

        aio = SimpleNamespace(aclose=lambda: (_ for _ in ()).throw(RuntimeError("close 失败")))

    provider = object.__new__(GoogleProvider)
    provider._client = _BadClose()
    provider._provider = "google"

    with pytest.raises(RuntimeError, match="close 失败"):
        if async_mode:
            asyncio.run(provider.aclose())
        else:
            provider.close()


# ── anthropic / model_provider / resources / local_hash 的覆盖缺口 ──
#
# 下列判据在整套测试里从未执行（行覆盖 + 变异扫描双证；把它们的 ``raise`` 换成
# ``pass`` 后全量 3698 条仍全绿）。删除后行为确实改变（实测，均经真实入口）：
# * anthropic 192 构造期缺 ``ANTHROPIC_API_KEY`` → **静默构造**出无凭证的
#   provider（``logger.error`` 已打，但对象照样返回给调用方）；
# * anthropic 279 ``supports_sampling_option`` 收到未知字段 → **静默返回
#   ``False``**，把「这个字段名本身是非法的」说成「这个模型不支持该字段」；
#   调用方（``chat_service._temperature_kwargs``／``chat.core``）只按真假值
#   分支，于是非法字段名被静默吞掉；
# * anthropic 2204 ``aclose`` 首个关闭异常 → **静默吞掉**（同步 ``close``
#   有覆盖，异步侧没有——sibling 缺口）；
# * anthropic 1419 流错误事件（``type`` 以 ``.error`` 结尾但**无** ``error``
#   子对象，如 ``overloaded.error``）→ ``raise_for_stream_error_event`` 不拦
#   这一形态，本条是唯一的报错点；删除后该事件被**静默返回 None**，错误信号
#   在流里消失；
# * model_provider 1126／1143 Responses 的 ``tool_calls`` 容器／元素形态 →
#   上游 ``normalize_messages``（466／470 行）先以同族消息拦住，两条不可达
#   （等价变异，见下）；
# * resources 236 ``_delegated_resource`` SDK 抛 ``AttributeError`` →
#   **裸 ``UnboundLocalError``**（``value`` 未绑定）；240 资源值为 ``None`` →
#   **静默返回 None**，调用方拿 None 当资源用、在更远处才崩；
# * resources 282 dunder 探针 → 经真实入口仍以 ``AttributeError`` 收场
#   （等价变异，见下）；
# * local_hash 87 Embedding 请求参数 → **静默接受**并原样丢弃。
#
# 同批经甄别不补测试的：model_provider 1771／1785（tenacity 的 ``reraise=True``
# 保证重试耗尽时抛原异常，末尾两条 ``RuntimeError`` 不可达）、2155（``client_getter()``
# 返回 None 时随即被下游以同一条消息拦住）、anthropic 335（三条同消息判据中的
# 第一条：bool／非数值落到 337 的 ``is_double_convertible``，收场同一条消息）、
# resources 282（``__getattr__`` 里的 dunder 探针：本函数的三条 ``AttributeError``
# 消息逐字相同，删除任一条仍以同一条消息收场）。


def _anthropic_provider_with_settings(monkeypatch, **settings):
    """经真实 ``__init__`` 构造 AnthropicProvider。

    不能用字符串形式 ``monkeypatch.setattr("src.providers.anthropic.get_settings", ...)``：
    ``tests/providers/test_factory.py`` 会把该模块从 ``sys.modules`` 删掉再重新导入，
    字符串形式解析到的是**新**模块，而 ``__init__`` 仍从**旧**模块字典取
    ``get_settings``——patch 静默落空。单跑本文件不触发，只有全量运行才暴露。
    """
    monkeypatch.setitem(
        AnthropicProvider.__init__.__globals__,
        "get_settings",
        lambda: SimpleNamespace(**settings),
    )
    return AnthropicProvider("claude-sonnet-4-6")


def test_anthropic_constructor_requires_api_key(monkeypatch):
    """缺 API Key 必须在构造期拒绝，不能静默构造出无凭证的 provider。"""
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        _anthropic_provider_with_settings(monkeypatch, anthropic_api_key=None)


def test_anthropic_supports_sampling_option_rejects_unknown_field(monkeypatch):
    """未知采样字段名必须报错，不能静默返回 ``False``。

    ``False`` 与「模型不支持该字段」在调用方（``chat_service._temperature_kwargs``）
    那里无法区分——非法字段名会被静默吞掉。
    """
    provider = _anthropic_provider_with_settings(monkeypatch, anthropic_api_key="k")

    with pytest.raises(ValueError, match="不支持的采样字段: bogus"):
        provider.supports_sampling_option("bogus")


def test_anthropic_aclose_reports_first_error():
    """``aclose`` 首个关闭异常必须向上报告，不能静默吞掉。

    与同步 ``close`` 对称：Google／Volcengine 的两个入口都有覆盖，Anthropic
    只有同步侧。
    """

    class _BadClose:
        def close(self):
            raise RuntimeError("close 失败")

    provider = object.__new__(AnthropicProvider)
    provider._client = _BadClose()
    provider._aclient = None

    with pytest.raises(RuntimeError, match="close 失败"):
        asyncio.run(provider.aclose())


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({"type": "overloaded.error"}, id="suffix-only"),
        pytest.param({"type": "rate_limit_error.error"}, id="dotted"),
    ],
    ids=lambda e: e["type"],
)
def test_anthropic_stream_error_event_without_embedded_error_is_reported(event):
    """``type`` 以 ``.error`` 结尾的流事件必须报错，不能静默返回 ``None``。

    这类事件**没有** ``error`` 子对象，``raise_for_stream_error_event`` 按
    ``embedded_error is None`` 放行——本条是该形态唯一的报错点。删除后错误信号
    在流里消失：调用方继续消费后续事件，把半截响应当成完整答案。
    """
    with pytest.raises(RuntimeError, match="流返回错误"):
        AnthropicProvider._stream_event(event, "rid")


def _resources_facade(**client_attrs):
    from src.providers import resources as resources_module

    provider = SimpleNamespace(
        _provider="openai",
        sdk_client=SimpleNamespace(**client_attrs),
    )
    return resources_module.OpenAIResources(provider)


def test_resources_delegated_resource_reports_missing_sdk_resource():
    """SDK 没有该资源节点时必须报 ``NotImplementedError``，不是裸 ``UnboundLocalError``。"""
    facade = _resources_facade()

    with pytest.raises(NotImplementedError, match="SDK 未提供 beta 资源"):
        facade._delegated_resource("beta")


def test_resources_delegated_resource_rejects_none_resource():
    """资源节点存在但为 ``None`` 时必须报错，不能静默返回 ``None``。"""
    facade = _resources_facade(beta=None)

    with pytest.raises(NotImplementedError, match="SDK 未提供 beta 资源"):
        facade._delegated_resource("beta")


def test_resources_native_facade_proxies_present_resource():
    """反向守卫：真实存在的资源照常代理。"""
    facade = _resources_facade(beta=SimpleNamespace(create=lambda **kw: kw))

    assert facade._delegated_resource("beta") is not None


def test_resources_native_getattr_rejects_none_resource():
    """``__getattr__`` 的动态资源为 ``None`` 时必须 ``AttributeError``，不能返回 ``None``。

    与 ``_delegated_resource`` 是两条独立路径：本函数在 dunder 探针、非原生渠道
    之后还有一道「值为 ``None``」的判据，删除后 ``getattr(facade, name)`` 返回
    ``None``——调用方按属性访问的惯例会继续用，崩在更远处。
    """
    facade = _resources_facade(some_resource=None)

    with pytest.raises(AttributeError, match="some_resource"):
        _ = facade.some_resource


def test_resources_native_facade_dunder_probe_is_rejected_before_delegation():
    """dunder 探针必须在触达 SDK 之前被拒，否则 ``deepcopy`` 会无限递归。

    ``copy``／``pickle`` 的协议在重建对象时会先 ``hasattr(y, '__setstate__')``
    之类地探测 dunder——此时新实例的属性还没填充，``__getattr__`` 里的
    ``self.provider`` 本身又走 ``__getattr__``，没有这道判据就无限递归
    （实测删掉守卫后 ``copy.deepcopy(facade)`` 抛 ``RecursionError``）。
    """
    import copy

    facade = _resources_facade(beta="REAL")

    copied = copy.deepcopy(facade)

    assert type(copied) is type(facade)


def test_local_hash_embedding_rejects_unsupported_kwargs():
    """LocalHash Embedding 的请求参数必须报错，不能静默接受并丢弃。"""
    from src.providers.local_hash import LocalHashEmbeddingProvider

    provider = LocalHashEmbeddingProvider("local-hash-32")

    with pytest.raises(ValueError, match="不支持请求参数: bogus"):
        provider.embed_documents(["文本"], bogus=1)
