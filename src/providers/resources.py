"""显式的供应商资源能力包装。

这些 Facade 只负责把调用转发到已经初始化的 SDK 客户端，不把文件、批处理
或缓存生命周期混入普通聊天请求。
"""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from src.utils.security import (
    validate_secret_free_resource_args,
    validate_secret_free_resource_kwargs,
)

_FacadeMethod = TypeVar("_FacadeMethod", bound=Callable[..., Any])


_NATIVE_SCALARS = (str, bytes, bytearray, int, float, bool, complex, type(None))
_RESOURCE_TREE_RETURNING_METHODS = frozenset({"with_options", "copy"})


def _allow_resource_business_parameters(*names: str) -> Callable[[_FacadeMethod], _FacadeMethod]:
    """声明官方 SDK 明确定义、但名称类似凭证的业务参数。"""

    allowed = frozenset(names)

    def decorator(method: _FacadeMethod) -> _FacadeMethod:
        method._resource_business_parameters = allowed  # type: ignore[attr-defined]
        return method

    return decorator


class _NativeResourceProxy:
    """为 SDK 原生资源树增加请求扩展校验，同时保持结果对象原样返回。

    `.native` 仍然返回官方客户端本身；只有通过 Facade 动态访问的资源节点
    使用此代理。这样既能兼容 SDK 新增资源，又不会让 `extra_headers`、
    `extra_query` 或 `extra_body` 绕过统一凭证边界。
    """

    __slots__ = ("_children", "_path", "_provider", "_value")

    def __init__(self, value: Any, provider: Any, path: str):
        self._value = value
        self._provider = provider
        self._path = path
        self._children: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name in self._children:
            return self._children[name]
        value = getattr(self._value, name)
        wrapped = _wrap_native_resource_value(
            value,
            self._provider,
            f"{self._path}.{name}",
        )
        self._children[name] = wrapped
        return wrapped

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # 先做凭证校验：它约束的是「参数能不能出现在请求里」，与渠道是否具备
        # 该能力无关，也是更根本的边界。顺序反了会让携带凭证的调用报出能力
        # 错误，把真正的安全问题掩盖成配置问题。
        safe_args = validate_secret_free_resource_args(
            args,
            f"{_resource_provider_label(self._provider)} 原生资源 {self._path}",
        )
        safe_kwargs = validate_secret_free_resource_kwargs(
            kwargs,
            f"{_resource_provider_label(self._provider)} 原生资源 {self._path}",
        )
        # 再做能力门禁：动态访问的资源节点必须和显式 Facade 方法受同一套约束。
        # 只做凭证校验不够：``resources.responses.create(...)`` 这类路径会绕过
        # server_verified_protocols 检查，把本地 SDK 资源直接发往未验证的服务端。
        resource_guard = getattr(self._provider, "_require_provider_resource", None)
        if callable(resource_guard):
            # 传完整路径（``volcengine.responses.create``）而不是末段方法名：
            # ``files.create`` 与 ``responses.create`` 末段相同，只看方法名无法
            # 区分该走哪条能力门禁。
            resource_guard(self._path)
        # Call results are deliberately not wrapped.  They are response models,
        # pagers, streams, or context managers rather than mutable resource
        # trees; preserving their SDK identity keeps normal client code intact.
        result = self._value(*safe_args, **safe_kwargs)
        # ``with_options``/``copy`` return another SDK client (and some SDK
        # resource nodes expose the same helpers). Keep that resource tree
        # behind the proxy so subsequent calls cannot bypass the same guard.
        method_name = self._path.rsplit(".", 1)[-1]
        if method_name in _RESOURCE_TREE_RETURNING_METHODS:
            return _wrap_native_resource_value(result, self._provider, self._path)
        return result

    def __repr__(self) -> str:
        return repr(self._value)


def _wrap_native_resource_value(value: Any, provider: Any, path: str) -> Any:
    if isinstance(value, _NativeResourceProxy):
        return value
    if isinstance(value, _NATIVE_SCALARS):
        return value
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return value
    if inspect.ismodule(value) or inspect.isclass(value):
        return value
    return _NativeResourceProxy(value, provider, path)


def _resource_provider_label(provider: Any) -> str:
    return str(
        getattr(provider, "_provider", None)
        or provider.__class__.__name__
    )


def _guard_resource_method(method: _FacadeMethod) -> _FacadeMethod:
    """在所有显式 Facade 方法前统一校验 SDK 扩展参数。"""

    allowed_business_parameters = frozenset(
        getattr(method, "_resource_business_parameters", ())
    )

    def sanitize_call(
        self: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        label = f"{_resource_provider_label(object.__getattribute__(self, 'provider'))} 资源"
        if not allowed_business_parameters:
            return (
                validate_secret_free_resource_args(args, label),
                validate_secret_free_resource_kwargs(kwargs, label),
            )

        signature = inspect.signature(method)
        bound = signature.bind_partial(self, *args, **kwargs)
        for name, value in list(bound.arguments.items()):
            if name == "self" or name in allowed_business_parameters:
                continue
            parameter = signature.parameters[name]
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                bound.arguments[name] = validate_secret_free_resource_args(value, label)
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                bound.arguments[name] = validate_secret_free_resource_kwargs(value, label)
            else:
                bound.arguments[name] = validate_secret_free_resource_kwargs(
                    {name: value}, label
                )[name]
        return bound.args[1:], dict(bound.kwargs)

    if inspect.iscoroutinefunction(method):

        @wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            safe_args, safe_kwargs = sanitize_call(self, args, kwargs)
            return await method(self, *safe_args, **safe_kwargs)

        return async_wrapper  # type: ignore[return-value]

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        safe_args, safe_kwargs = sanitize_call(self, args, kwargs)
        return method(self, *safe_args, **safe_kwargs)

    return wrapper  # type: ignore[return-value]


class _NativeFacade:
    """保留官方 SDK 的完整资源面，Facade 只负责稳定的常用别名。"""

    _delegate_native = False
    _delegate_client_name = "sdk_client"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """为子类显式资源方法安装统一的请求扩展校验。"""

        super().__init_subclass__(**kwargs)
        for name, method in list(cls.__dict__.items()):
            if name.startswith("_") or not callable(method):
                continue
            setattr(cls, name, _guard_resource_method(method))

    def __init__(self, provider: Any):
        self.provider = provider

    @property
    def native(self) -> Any:
        return self.provider.sdk_client

    @property
    def async_native(self) -> Any:
        return self.provider.async_sdk_client

    def _delegated_resource(self, name: str, *, asynchronous: bool = False) -> Any:
        """读取 SDK 资源并套用请求参数安全代理。

        ``native``/``async_native`` 是有意保留的原生客户端出口；Facade 的
        其它资源节点则必须经过同一个代理，避免显式属性绕过动态资源的
        ``extra_headers``、``extra_query`` 和 ``extra_body`` 校验。
        """
        client_name = "async_sdk_client" if asynchronous else "sdk_client"
        client = getattr(self.provider, client_name)
        try:
            value = getattr(client, name)
        except (AttributeError, NotImplementedError) as exc:
            raise NotImplementedError(
                f"{_resource_provider_label(self.provider)} SDK 未提供 {name} 资源。"
            ) from exc
        if value is None:
            raise NotImplementedError(
                f"{_resource_provider_label(self.provider)} SDK 未提供 {name} 资源。"
            )
        label = _resource_provider_label(self.provider)
        return _wrap_native_resource_value(value, self.provider, f"{label}.{name}")

    @property
    def beta(self) -> Any:
        """访问供应商 SDK 的 beta 资源树（若该 SDK 提供）。"""
        if not self._delegate_native:
            raise AttributeError("beta")
        return self._delegated_resource("beta")

    @property
    def async_beta(self) -> Any:
        """访问异步 SDK 的 beta 资源树（若该 SDK 提供）。"""
        if not self._delegate_native:
            raise AttributeError("async_beta")
        return self._delegated_resource("beta", asynchronous=True)

    def __getattr__(self, name: str) -> Any:
        """Expose the complete official SDK tree without widening compatible channels.

        Explicit facade methods remain the stable cross-provider surface. Official
        providers additionally allow direct access to resources introduced by the
        installed SDK after this module was released (for example Realtime,
        Webhooks, Live, Interactions, or Beta resources). Dynamic nodes are
        proxied so every callable still enforces the credential boundary; use
        `.native` when an unwrapped SDK client is explicitly required.
        """
        # Python introspection, copy/deepcopy and pickle probe dunder names via
        # getattr. Never forward those probes to a third-party SDK resource.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if not self._delegate_native:
            raise AttributeError(name)
        client = getattr(self.provider, self._delegate_client_name)
        try:
            value = getattr(client, name)
        except (AttributeError, NotImplementedError) as exc:
            raise AttributeError(name) from exc
        if value is None:
            raise AttributeError(name)
        return _wrap_native_resource_value(
            value,
            self.provider,
            f"{_resource_provider_label(self.provider)}.{name}",
        )


class OpenAICompatibleResources(_NativeFacade):
    """OpenAI-compatible channels expose only their native client by default."""


class AsyncOpenAICompatibleResources(_NativeFacade):
    """Async native-client access for OpenAI-compatible channels."""


class _AsyncOnlyResourceFacade(_NativeFacade):
    """Expose async-only SDK resources while retaining native delegation."""

    _delegate_native = True
    _delegate_client_name = "async_sdk_client"


class OpenAIResources(OpenAICompatibleResources):
    """OpenAI 官方资源 Facade；兼容渠道不会继承此类能力声明。"""

    _delegate_native = True

    @property
    def skills(self) -> Any:
        """访问 OpenAI Skills 资源树。"""
        return self._delegated_resource("skills")

    @property
    def realtime(self) -> Any:
        """访问 OpenAI Realtime 资源树。"""
        return self._delegated_resource("realtime")

    @property
    def webhooks(self) -> Any:
        """访问 OpenAI Webhooks 资源树。"""
        return self._delegated_resource("webhooks")

    @property
    def admin(self) -> Any:
        """访问 OpenAI Admin 资源树。"""
        return self._delegated_resource("admin")

    @property
    def content_provenance_checks(self) -> Any:
        """访问 OpenAI 内容来源校验资源树。"""
        return self._delegated_resource("content_provenance_checks")

    def upload_file(self, file: Any, purpose: str = "assistants", **kwargs: Any) -> Any:
        return self.provider.upload_file(file, purpose=purpose, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        return self.provider.list_files(**kwargs)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_file(file_id, **kwargs)

    def file_content(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.file_content(file_id, **kwargs)

    def retrieve_file_content(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_file_content(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_file(file_id, **kwargs)

    def wait_for_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.wait_for_file(file_id, **kwargs)

    def create_batch(self, input_file_id: str, endpoint: str | None = None, **kwargs: Any) -> Any:
        return self.provider.create_batch(input_file_id, endpoint=endpoint, **kwargs)

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_batch(batch_id, **kwargs)

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_batch(batch_id, **kwargs)

    def list_batches(self, **kwargs: Any) -> Any:
        return self.provider.list_batches(**kwargs)

    def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_response(response_id, **kwargs)

    def cancel_response(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_response(response_id, **kwargs)

    def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_response(response_id, **kwargs)

    def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.list_response_input_items(response_id, **kwargs)

    def connect_responses(self, **kwargs: Any) -> Any:
        return self.provider.connect_responses(**kwargs)

    def stream_responses(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.stream_responses(request, **kwargs)

    def count_input_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return self.provider.count_input_tokens(request, **kwargs)

    def create_vector_store(self, **kwargs: Any) -> Any:
        return self.provider.create_vector_store(**kwargs)

    def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_vector_store(vector_store_id, **kwargs)

    def list_vector_stores(self, **kwargs: Any) -> Any:
        return self.provider.list_vector_stores(**kwargs)

    def search_vector_store(self, vector_store_id: str, query: Any, **kwargs: Any) -> Any:
        return self.provider.search_vector_store(vector_store_id, query, **kwargs)

    def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_vector_store(vector_store_id, **kwargs)

    def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.update_vector_store(vector_store_id, **kwargs)

    def create_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.create_vector_store_file(vector_store_id, file_id, **kwargs)

    def create_vector_store_file_and_poll(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.create_vector_store_file_and_poll(vector_store_id, file_id, **kwargs)

    def upload_vector_store_file(self, vector_store_id: str, file: Any, **kwargs: Any) -> Any:
        return self.provider.upload_vector_store_file(vector_store_id, file, **kwargs)

    def upload_vector_store_file_and_poll(self, vector_store_id: str, file: Any, **kwargs: Any) -> Any:
        return self.provider.upload_vector_store_file_and_poll(vector_store_id, file, **kwargs)

    def retrieve_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_vector_store_file(vector_store_id, file_id, **kwargs)

    def list_vector_store_files(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.list_vector_store_files(vector_store_id, **kwargs)

    def update_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.update_vector_store_file(vector_store_id, file_id, **kwargs)

    def delete_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_vector_store_file(vector_store_id, file_id, **kwargs)

    def vector_store_file_content(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.vector_store_file_content(vector_store_id, file_id, **kwargs)

    def poll_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.poll_vector_store_file(vector_store_id, file_id, **kwargs)

    def create_vector_store_file_batch(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.create_vector_store_file_batch(vector_store_id, **kwargs)

    def create_vector_store_file_batch_and_poll(self, vector_store_id: str, **kwargs: Any) -> Any:
        return self.provider.create_vector_store_file_batch_and_poll(vector_store_id, **kwargs)

    def retrieve_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    def cancel_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    def poll_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.poll_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    def list_vector_store_file_batch_files(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.list_vector_store_file_batch_files(vector_store_id, batch_id, **kwargs)

    def upload_vector_store_file_batch_and_poll(self, vector_store_id: str, files: Any, **kwargs: Any) -> Any:
        return self.provider.upload_vector_store_file_batch_and_poll(vector_store_id, files, **kwargs)

    def list_models(self, **kwargs: Any) -> Any:
        return self.provider.list_models(**kwargs)

    def retrieve_model(self, model: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_model(model, **kwargs)

    def delete_model(self, model: str, **kwargs: Any) -> Any:
        return self.provider.delete_model(model, **kwargs)

    def create_moderation(self, input: Any, **kwargs: Any) -> Any:
        return self.provider.create_moderation(input, **kwargs)

    def generate_image(self, prompt: str, **kwargs: Any) -> Any:
        return self.provider.generate_image(prompt, **kwargs)

    def edit_image(self, image: Any, prompt: str, **kwargs: Any) -> Any:
        return self.provider.edit_image(image, prompt, **kwargs)

    def create_image_variation(self, image: Any, **kwargs: Any) -> Any:
        return self.provider.create_image_variation(image, **kwargs)

    def text_to_speech(self, text: str, model: str, voice: str, **kwargs: Any) -> Any:
        return self.provider.text_to_speech(text, model, voice, **kwargs)

    def transcribe_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return self.provider.transcribe_audio(file, model, **kwargs)

    def translate_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return self.provider.translate_audio(file, model, **kwargs)

    def create_video(self, **kwargs: Any) -> Any:
        return self.provider.create_video(**kwargs)

    def create_video_and_poll(self, **kwargs: Any) -> Any:
        return self.provider.create_video_and_poll(**kwargs)

    def retrieve_video(self, video_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_video(video_id, **kwargs)

    def list_videos(self, **kwargs: Any) -> Any:
        return self.provider.list_videos(**kwargs)

    def delete_video(self, video_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_video(video_id, **kwargs)

    def download_video(self, video_id: str, **kwargs: Any) -> Any:
        return self.provider.download_video(video_id, **kwargs)

    def create_video_character(self, name: str, video: Any, **kwargs: Any) -> Any:
        return self.provider.create_video_character(name, video, **kwargs)

    def retrieve_video_character(self, character_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_video_character(character_id, **kwargs)

    def edit_video(self, prompt: str, video: Any, **kwargs: Any) -> Any:
        return self.provider.edit_video(prompt, video, **kwargs)

    def extend_video(self, prompt: str, seconds: Any, video: Any, **kwargs: Any) -> Any:
        return self.provider.extend_video(prompt, seconds, video, **kwargs)

    def remix_video(self, video_id: str, prompt: str, **kwargs: Any) -> Any:
        return self.provider.remix_video(video_id, prompt, **kwargs)

    def poll_video(self, video_id: str, **kwargs: Any) -> Any:
        return self.provider.poll_video(video_id, **kwargs)

    def create_upload(self, **kwargs: Any) -> Any:
        return self.provider.create_upload(**kwargs)

    def complete_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return self.provider.complete_upload(upload_id, **kwargs)

    def cancel_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_upload(upload_id, **kwargs)

    def create_upload_part(self, upload_id: str, data: Any, **kwargs: Any) -> Any:
        return self.provider.create_upload_part(upload_id, data, **kwargs)

    def upload_file_chunked(self, **kwargs: Any) -> Any:
        return self.provider.upload_file_chunked(**kwargs)

    def parse(self, text_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.parse(text_format, request, **kwargs)

    def compact_responses(self, **kwargs: Any) -> Any:
        return self.provider.compact_responses(**kwargs)

    def create_conversation(self, **kwargs: Any) -> Any:
        return self.provider.create_conversation(**kwargs)

    def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_conversation(conversation_id, **kwargs)

    def update_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self.provider.update_conversation(conversation_id, **kwargs)

    def delete_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_conversation(conversation_id, **kwargs)

    def list_conversation_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return self.provider.list_conversation_items(conversation_id, **kwargs)

    def create_conversation_items(self, conversation_id: str, items: Any, **kwargs: Any) -> Any:
        return self.provider.create_conversation_items(conversation_id, items, **kwargs)

    def retrieve_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_conversation_item(conversation_id, item_id, **kwargs)

    def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_conversation_item(conversation_id, item_id, **kwargs)

    def create_container(self, **kwargs: Any) -> Any:
        return self.provider.create_container(**kwargs)

    def retrieve_container(self, container_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_container(container_id, **kwargs)

    def list_containers(self, **kwargs: Any) -> Any:
        return self.provider.list_containers(**kwargs)

    def delete_container(self, container_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_container(container_id, **kwargs)

    def create_container_file(self, container_id: str, file: Any = None, file_id: str | None = None, **kwargs: Any) -> Any:
        return self.provider.create_container_file(container_id, file=file, file_id=file_id, **kwargs)

    def list_container_files(self, container_id: str, **kwargs: Any) -> Any:
        return self.provider.list_container_files(container_id, **kwargs)

    def retrieve_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_container_file(container_id, file_id, **kwargs)

    def delete_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_container_file(container_id, file_id, **kwargs)

    def container_file_content(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return self.provider.container_file_content(container_id, file_id, **kwargs)

    def create_fine_tuning_job(self, **kwargs: Any) -> Any:
        return self.provider.create_fine_tuning_job(**kwargs)

    def list_fine_tuning_jobs(self, **kwargs: Any) -> Any:
        return self.provider.list_fine_tuning_jobs(**kwargs)

    def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_fine_tuning_job(job_id, **kwargs)

    def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_fine_tuning_job(job_id, **kwargs)

    def pause_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self.provider.pause_fine_tuning_job(job_id, **kwargs)

    def resume_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return self.provider.resume_fine_tuning_job(job_id, **kwargs)

    def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> Any:
        return self.provider.list_fine_tuning_events(job_id, **kwargs)

    def create_eval(self, **kwargs: Any) -> Any:
        return self.provider.create_eval(**kwargs)

    def list_evals(self, **kwargs: Any) -> Any:
        return self.provider.list_evals(**kwargs)

    def retrieve_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_eval(eval_id, **kwargs)

    def update_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self.provider.update_eval(eval_id, **kwargs)

    def delete_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_eval(eval_id, **kwargs)

    def create_eval_run(self, eval_id: str, **kwargs: Any) -> Any:
        return self.provider.create_eval_run(eval_id, **kwargs)

    def list_eval_runs(self, eval_id: str, **kwargs: Any) -> Any:
        return self.provider.list_eval_runs(eval_id, **kwargs)

    def retrieve_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_eval_run(eval_id, run_id, **kwargs)

    def cancel_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_eval_run(eval_id, run_id, **kwargs)

    def delete_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_eval_run(eval_id, run_id, **kwargs)

    def list_eval_run_output_items(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return self.provider.list_eval_run_output_items(eval_id, run_id, **kwargs)

    def retrieve_eval_run_output_item(
        self, eval_id: str, run_id: str, output_item_id: str, **kwargs: Any
    ) -> Any:
        return self.provider.retrieve_eval_run_output_item(
            eval_id, run_id, output_item_id, **kwargs
        )


class AsyncOpenAIResources(AsyncOpenAICompatibleResources):
    """OpenAI AsyncOpenAI 资源的显式异步 Facade。"""

    _delegate_native = True
    _delegate_client_name = "async_sdk_client"

    @property
    def skills(self) -> Any:
        """访问 AsyncOpenAI Skills 资源树。"""
        return self._delegated_resource("skills", asynchronous=True)

    @property
    def realtime(self) -> Any:
        """访问 AsyncOpenAI Realtime 资源树。"""
        return self._delegated_resource("realtime", asynchronous=True)

    @property
    def webhooks(self) -> Any:
        """访问 AsyncOpenAI Webhooks 资源树。"""
        return self._delegated_resource("webhooks", asynchronous=True)

    @property
    def admin(self) -> Any:
        """访问 AsyncOpenAI Admin 资源树。"""
        return self._delegated_resource("admin", asynchronous=True)

    @property
    def content_provenance_checks(self) -> Any:
        """访问 AsyncOpenAI 内容来源校验资源树。"""
        return self._delegated_resource("content_provenance_checks", asynchronous=True)

    async def upload_file(self, file: Any, purpose: str = "assistants", **kwargs: Any) -> Any:
        return await self.provider.async_upload_file(file, purpose=purpose, **kwargs)

    async def list_files(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_files(**kwargs)

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_file(file_id, **kwargs)

    async def file_content(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_file_content(file_id, **kwargs)

    async def retrieve_file_content(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_file_content(file_id, **kwargs)

    async def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file(file_id, **kwargs)

    async def wait_for_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_wait_for_file(file_id, **kwargs)

    async def create_batch(self, input_file_id: str, endpoint: str | None = None, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch(input_file_id, endpoint=endpoint, **kwargs)

    async def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_batch(batch_id, **kwargs)

    async def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_batch(batch_id, **kwargs)

    async def list_batches(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_batches(**kwargs)

    async def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_response(response_id, **kwargs)

    async def cancel_response(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_response(response_id, **kwargs)

    async def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_response(response_id, **kwargs)

    async def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_response_input_items(response_id, **kwargs)

    def connect_responses(self, **kwargs: Any) -> Any:
        return self.provider.async_connect_responses(**kwargs)

    def stream_responses(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.async_stream_responses(request, **kwargs)

    async def count_input_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return await self.provider.async_count_input_tokens(request, **kwargs)

    async def create_vector_store(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_vector_store(**kwargs)

    async def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_vector_store(vector_store_id, **kwargs)

    async def list_vector_stores(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_vector_stores(**kwargs)

    async def search_vector_store(self, vector_store_id: str, query: Any, **kwargs: Any) -> Any:
        return await self.provider.async_search_vector_store(vector_store_id, query, **kwargs)

    async def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_vector_store(vector_store_id, **kwargs)

    async def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_vector_store(vector_store_id, **kwargs)

    async def create_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_create_vector_store_file(vector_store_id, file_id, **kwargs)

    async def create_vector_store_file_and_poll(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_create_vector_store_file_and_poll(vector_store_id, file_id, **kwargs)

    async def upload_vector_store_file(self, vector_store_id: str, file: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_vector_store_file(vector_store_id, file, **kwargs)

    async def upload_vector_store_file_and_poll(self, vector_store_id: str, file: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_vector_store_file_and_poll(vector_store_id, file, **kwargs)

    async def create_vector_store_file_batch(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_create_vector_store_file_batch(vector_store_id, **kwargs)

    async def create_vector_store_file_batch_and_poll(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_create_vector_store_file_batch_and_poll(vector_store_id, **kwargs)

    async def retrieve_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_vector_store_file(vector_store_id, file_id, **kwargs)

    async def list_vector_store_files(self, vector_store_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_vector_store_files(vector_store_id, **kwargs)

    async def update_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_vector_store_file(vector_store_id, file_id, **kwargs)

    async def delete_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_vector_store_file(vector_store_id, file_id, **kwargs)

    async def vector_store_file_content(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_vector_store_file_content(vector_store_id, file_id, **kwargs)

    async def poll_vector_store_file(self, vector_store_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_poll_vector_store_file(vector_store_id, file_id, **kwargs)

    async def retrieve_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    async def cancel_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    async def poll_vector_store_file_batch(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_poll_vector_store_file_batch(vector_store_id, batch_id, **kwargs)

    async def list_vector_store_file_batch_files(self, vector_store_id: str, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_vector_store_file_batch_files(vector_store_id, batch_id, **kwargs)

    async def upload_vector_store_file_batch_and_poll(self, vector_store_id: str, files: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_vector_store_file_batch_and_poll(vector_store_id, files, **kwargs)

    async def list_models(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_models(**kwargs)

    async def retrieve_model(self, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_model(model, **kwargs)

    async def delete_model(self, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_model(model, **kwargs)

    async def create_moderation(self, input: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_moderation(input, **kwargs)

    async def generate_image(self, prompt: str, **kwargs: Any) -> Any:
        return await self.provider.async_generate_image(prompt, **kwargs)

    async def edit_image(self, image: Any, prompt: str, **kwargs: Any) -> Any:
        return await self.provider.async_edit_image(image, prompt, **kwargs)

    async def create_image_variation(self, image: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_image_variation(image, **kwargs)

    async def text_to_speech(self, text: str, model: str, voice: str, **kwargs: Any) -> Any:
        return await self.provider.async_text_to_speech(text, model, voice, **kwargs)

    async def transcribe_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_transcribe_audio(file, model, **kwargs)

    async def translate_audio(self, file: Any, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_translate_audio(file, model, **kwargs)

    async def create_video(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_video(**kwargs)

    async def create_video_and_poll(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_video_and_poll(**kwargs)

    async def retrieve_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_video(video_id, **kwargs)

    async def list_videos(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_videos(**kwargs)

    async def delete_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_video(video_id, **kwargs)

    async def download_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_download_video(video_id, **kwargs)

    async def create_video_character(self, name: str, video: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_video_character(name, video, **kwargs)

    async def retrieve_video_character(self, character_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_video_character(character_id, **kwargs)

    async def edit_video(self, prompt: str, video: Any, **kwargs: Any) -> Any:
        return await self.provider.async_edit_video(prompt, video, **kwargs)

    async def extend_video(self, prompt: str, seconds: Any, video: Any, **kwargs: Any) -> Any:
        return await self.provider.async_extend_video(prompt, seconds, video, **kwargs)

    async def remix_video(self, video_id: str, prompt: str, **kwargs: Any) -> Any:
        return await self.provider.async_remix_video(video_id, prompt, **kwargs)

    async def poll_video(self, video_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_poll_video(video_id, **kwargs)

    async def create_upload(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_upload(**kwargs)

    async def complete_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_complete_upload(upload_id, **kwargs)

    async def cancel_upload(self, upload_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_upload(upload_id, **kwargs)

    async def create_upload_part(self, upload_id: str, data: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_upload_part(upload_id, data, **kwargs)

    async def upload_file_chunked(self, **kwargs: Any) -> Any:
        return await self.provider.async_upload_file_chunked(**kwargs)

    async def parse(self, text_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_parse(text_format, request, **kwargs)

    async def compact_responses(self, **kwargs: Any) -> Any:
        return await self.provider.async_compact_responses(**kwargs)

    async def create_conversation(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_conversation(**kwargs)

    async def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_conversation(conversation_id, **kwargs)

    async def update_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_conversation(conversation_id, **kwargs)

    async def delete_conversation(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_conversation(conversation_id, **kwargs)

    async def list_conversation_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_conversation_items(conversation_id, **kwargs)

    async def create_conversation_items(self, conversation_id: str, items: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_conversation_items(conversation_id, items, **kwargs)

    async def retrieve_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_conversation_item(conversation_id, item_id, **kwargs)

    async def delete_conversation_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_conversation_item(conversation_id, item_id, **kwargs)

    async def create_container(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_container(**kwargs)

    async def retrieve_container(self, container_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_container(container_id, **kwargs)

    async def list_containers(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_containers(**kwargs)

    async def delete_container(self, container_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_container(container_id, **kwargs)

    async def create_container_file(self, container_id: str, file: Any = None, file_id: str | None = None, **kwargs: Any) -> Any:
        return await self.provider.async_create_container_file(container_id, file=file, file_id=file_id, **kwargs)

    async def list_container_files(self, container_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_container_files(container_id, **kwargs)

    async def retrieve_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_container_file(container_id, file_id, **kwargs)

    async def delete_container_file(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_container_file(container_id, file_id, **kwargs)

    async def container_file_content(self, container_id: str, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_container_file_content(container_id, file_id, **kwargs)

    async def create_fine_tuning_job(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_fine_tuning_job(**kwargs)

    async def list_fine_tuning_jobs(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_fine_tuning_jobs(**kwargs)

    async def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_fine_tuning_job(job_id, **kwargs)

    async def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_fine_tuning_job(job_id, **kwargs)

    async def pause_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_pause_fine_tuning_job(job_id, **kwargs)

    async def resume_fine_tuning_job(self, job_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_resume_fine_tuning_job(job_id, **kwargs)

    async def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_fine_tuning_events(job_id, **kwargs)

    async def create_eval(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_eval(**kwargs)

    async def list_evals(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_evals(**kwargs)

    async def retrieve_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_eval(eval_id, **kwargs)

    async def update_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_eval(eval_id, **kwargs)

    async def delete_eval(self, eval_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_eval(eval_id, **kwargs)

    async def create_eval_run(self, eval_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_create_eval_run(eval_id, **kwargs)

    async def list_eval_runs(self, eval_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_eval_runs(eval_id, **kwargs)

    async def retrieve_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_eval_run(eval_id, run_id, **kwargs)

    async def cancel_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_eval_run(eval_id, run_id, **kwargs)

    async def delete_eval_run(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_eval_run(eval_id, run_id, **kwargs)

    async def list_eval_run_output_items(self, eval_id: str, run_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_eval_run_output_items(eval_id, run_id, **kwargs)

    async def retrieve_eval_run_output_item(
        self, eval_id: str, run_id: str, output_item_id: str, **kwargs: Any
    ) -> Any:
        return await self.provider.async_retrieve_eval_run_output_item(
            eval_id, run_id, output_item_id, **kwargs
        )


class GoogleResources(_NativeFacade):

    _delegate_native = True

    @property
    def auth_tokens(self) -> Any:
        """访问 google-genai 的 API token 管理资源。"""
        return self._delegated_resource("auth_tokens")

    def create_auth_token(self, **kwargs: Any) -> Any:
        return self.provider.create_auth_token(**kwargs)

    def upload_file(self, file: Any, **kwargs: Any) -> Any:
        return self.provider.upload_file(file, **kwargs)

    @_allow_resource_business_parameters("auth")
    def register_files(self, auth: Any, uris: list[str], **kwargs: Any) -> Any:
        return self.provider.register_files(auth=auth, uris=uris, **kwargs)

    def compute_tokens(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.compute_tokens(request, **kwargs)

    def count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return self.provider.count_tokens(request, **kwargs)

    def get_file(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_file(name, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        return self.provider.list_files(**kwargs)

    def download_file(self, file: Any, **kwargs: Any) -> bytes:
        return self.provider.download_file(file, **kwargs)

    def delete_file(self, name: str, **kwargs: Any) -> Any:
        return self.provider.delete_file(name, **kwargs)

    def create_cache(self, **kwargs: Any) -> Any:
        return self.provider.create_cache(**kwargs)

    def get_cache(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_cache(name, **kwargs)

    def list_caches(self, **kwargs: Any) -> Any:
        return self.provider.list_caches(**kwargs)

    def update_cache(self, name: str, **kwargs: Any) -> Any:
        return self.provider.update_cache(name, **kwargs)

    def delete_cache(self, name: str, **kwargs: Any) -> Any:
        return self.provider.delete_cache(name, **kwargs)

    def create_batch(self, src: Any, **kwargs: Any) -> Any:
        return self.provider.create_batch(src, **kwargs)

    def create_embedding_batch(self, src: Any, **kwargs: Any) -> Any:
        return self.provider.create_embedding_batch(src, **kwargs)

    def get_batch(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_batch(name, **kwargs)

    def list_batches(self, **kwargs: Any) -> Any:
        return self.provider.list_batches(**kwargs)

    def cancel_batch(self, name: str, **kwargs: Any) -> Any:
        return self.provider.cancel_batch(name, **kwargs)

    def delete_batch(self, name: str, **kwargs: Any) -> Any:
        return self.provider.delete_batch(name, **kwargs)

    def list_models(self, **kwargs: Any) -> Any:
        return self.provider.list_models(**kwargs)

    def get_model(self, model: str, **kwargs: Any) -> Any:
        return self.provider.get_model(model, **kwargs)

    def delete_model(self, model: str, **kwargs: Any) -> Any:
        return self.provider.delete_model(model, **kwargs)

    def update_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return self.provider.update_model(model, config=config, **kwargs)

    def tune(
        self, base_model: str, training_dataset: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        return self.provider.tune(
            base_model, training_dataset, config=config, **kwargs
        )

    def get_tuning(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_tuning(name, **kwargs)

    def list_tunings(self, **kwargs: Any) -> Any:
        return self.provider.list_tunings(**kwargs)

    def cancel_tuning(self, name: str, **kwargs: Any) -> Any:
        return self.provider.cancel_tuning(name, **kwargs)

    def validate_tuning_reward(
        self,
        parent: str,
        sample_response: Any,
        example: Any,
        single_reward_config: Any = None,
        composite_reward_config: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.provider.validate_tuning_reward(
            parent,
            sample_response,
            example,
            single_reward_config=single_reward_config,
            composite_reward_config=composite_reward_config,
            config=config,
            **kwargs,
        )

    def recontext_image(self, source: Any, **kwargs: Any) -> Any:
        return self.provider.recontext_image(source, **kwargs)

    def generate_images(self, prompt: str, **kwargs: Any) -> Any:
        return self.provider.generate_images(prompt, **kwargs)

    def edit_image(self, prompt: str, reference_images: list[Any], **kwargs: Any) -> Any:
        return self.provider.edit_image(prompt, reference_images, **kwargs)

    def upscale_image(self, image: Any, upscale_factor: str, **kwargs: Any) -> Any:
        return self.provider.upscale_image(image, upscale_factor, **kwargs)

    def segment_image(self, source: Any, **kwargs: Any) -> Any:
        return self.provider.segment_image(source, **kwargs)

    def generate_videos(self, **kwargs: Any) -> Any:
        return self.provider.generate_videos(**kwargs)

    def get_operation(self, operation: Any, **kwargs: Any) -> Any:
        return self.provider.get_operation(operation, **kwargs)

    def list_file_search_documents(self, parent: str, **kwargs: Any) -> Any:
        return self.provider.list_file_search_documents(parent, **kwargs)

    def get_file_search_document(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_file_search_document(name, **kwargs)

    def delete_file_search_document(self, name: str, **kwargs: Any) -> Any:
        return self.provider.delete_file_search_document(name, **kwargs)

    def create_chat(self, **kwargs: Any) -> Any:
        return self.provider.create_chat(**kwargs)

    def create_interaction(self, **kwargs: Any) -> Any:
        return self.provider.create_interaction(**kwargs)

    def get_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return self.provider.get_interaction(interaction_id, **kwargs)

    def cancel_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_interaction(interaction_id, **kwargs)

    def delete_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_interaction(interaction_id, **kwargs)

    def create_agent(self, **kwargs: Any) -> Any:
        return self.provider.create_agent(**kwargs)

    def get_agent(self, agent_id: str, **kwargs: Any) -> Any:
        return self.provider.get_agent(agent_id, **kwargs)

    def list_agents(self, **kwargs: Any) -> Any:
        return self.provider.list_agents(**kwargs)

    def delete_agent(self, agent_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_agent(agent_id, **kwargs)

    def create_webhook(self, **kwargs: Any) -> Any:
        return self.provider.create_webhook(**kwargs)

    def get_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return self.provider.get_webhook(webhook_id, **kwargs)

    def list_webhooks(self, **kwargs: Any) -> Any:
        return self.provider.list_webhooks(**kwargs)

    def update_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return self.provider.update_webhook(webhook_id, **kwargs)

    def delete_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_webhook(webhook_id, **kwargs)

    def ping_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return self.provider.ping_webhook(webhook_id, **kwargs)

    def rotate_webhook_signing_secret(self, webhook_id: str, **kwargs: Any) -> Any:
        return self.provider.rotate_webhook_signing_secret(webhook_id, **kwargs)

    def create_environment(self, **kwargs: Any) -> Any:
        return self.provider.create_environment(**kwargs)

    def get_environment(self, environment_id: str, **kwargs: Any) -> Any:
        return self.provider.get_environment(environment_id, **kwargs)

    def list_environments(self, **kwargs: Any) -> Any:
        return self.provider.list_environments(**kwargs)

    def delete_environment(self, environment_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_environment(environment_id, **kwargs)

    def get_environment_files(self, environment_id: str, path: str, **kwargs: Any) -> Any:
        return self.provider.get_environment_files(environment_id, path, **kwargs)

    def create_trigger(self, **kwargs: Any) -> Any:
        return self.provider.create_trigger(**kwargs)

    def get_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return self.provider.get_trigger(trigger_id, **kwargs)

    def list_triggers(self, **kwargs: Any) -> Any:
        return self.provider.list_triggers(**kwargs)

    def update_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return self.provider.update_trigger(trigger_id, **kwargs)

    def delete_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_trigger(trigger_id, **kwargs)

    def run_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return self.provider.run_trigger(trigger_id, **kwargs)

    def list_trigger_executions(self, trigger_id: str, **kwargs: Any) -> Any:
        return self.provider.list_trigger_executions(trigger_id, **kwargs)

    def connect_live(self, *, model: str | None = None, config: Any = None) -> Any:
        return self.provider.connect_live(model=model, config=config)

    def create_file_search_store(self, **kwargs: Any) -> Any:
        return self.provider.create_file_search_store(**kwargs)

    def get_file_search_store(self, name: str, **kwargs: Any) -> Any:
        return self.provider.get_file_search_store(name, **kwargs)

    def list_file_search_stores(self, **kwargs: Any) -> Any:
        return self.provider.list_file_search_stores(**kwargs)

    def delete_file_search_store(self, name: str, **kwargs: Any) -> Any:
        return self.provider.delete_file_search_store(name, **kwargs)

    def import_file_to_file_search_store(self, file_search_store_name: str, file_name: str, **kwargs: Any) -> Any:
        return self.provider.import_file_to_file_search_store(file_search_store_name, file_name, **kwargs)

    def upload_to_file_search_store(self, file_search_store_name: str, file: Any, **kwargs: Any) -> Any:
        return self.provider.upload_to_file_search_store(file_search_store_name, file, **kwargs)

    def download_file_search_media(self, media_id: str, **kwargs: Any) -> bytes:
        return self.provider.download_file_search_media(media_id, **kwargs)


class AsyncGoogleResources(_NativeFacade):
    """google-genai `Client.aio` 资源的显式异步 Facade。"""

    _delegate_native = True
    _delegate_client_name = "async_sdk_client"

    @property
    def auth_tokens(self) -> Any:
        """访问 google-genai 异步 API token 管理资源。"""
        return self._delegated_resource("auth_tokens", asynchronous=True)

    async def create_auth_token(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_auth_token(**kwargs)

    async def upload_file(self, file: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_file(file, **kwargs)

    @_allow_resource_business_parameters("auth")
    async def register_files(self, auth: Any, uris: list[str], **kwargs: Any) -> Any:
        return await self.provider.async_register_files(auth=auth, uris=uris, **kwargs)

    async def count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return await self.provider.async_count_tokens(request, **kwargs)

    async def compute_tokens(self, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_compute_tokens(request, **kwargs)

    async def get_file(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_file(name, **kwargs)

    async def list_files(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_files(**kwargs)

    async def download_file(self, file: Any, **kwargs: Any) -> bytes:
        return await self.provider.async_download_file(file, **kwargs)

    async def delete_file(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file(name, **kwargs)

    async def create_interaction(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_interaction(**kwargs)

    async def get_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_interaction(interaction_id, **kwargs)

    async def cancel_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_interaction(interaction_id, **kwargs)

    async def delete_interaction(self, interaction_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_interaction(interaction_id, **kwargs)

    async def create_agent(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_agent(**kwargs)

    async def get_agent(self, agent_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_agent(agent_id, **kwargs)

    async def list_agents(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_agents(**kwargs)

    async def delete_agent(self, agent_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_agent(agent_id, **kwargs)

    async def create_webhook(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_webhook(**kwargs)

    async def get_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_webhook(webhook_id, **kwargs)

    async def list_webhooks(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_webhooks(**kwargs)

    async def update_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_webhook(webhook_id, **kwargs)

    async def delete_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_webhook(webhook_id, **kwargs)

    async def ping_webhook(self, webhook_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_ping_webhook(webhook_id, **kwargs)

    async def rotate_webhook_signing_secret(self, webhook_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_rotate_webhook_signing_secret(webhook_id, **kwargs)

    async def create_environment(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_environment(**kwargs)

    async def get_environment(self, environment_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_environment(environment_id, **kwargs)

    async def list_environments(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_environments(**kwargs)

    async def delete_environment(self, environment_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_environment(environment_id, **kwargs)

    async def get_environment_files(self, environment_id: str, path: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_environment_files(environment_id, path, **kwargs)

    async def create_trigger(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_trigger(**kwargs)

    async def get_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_trigger(trigger_id, **kwargs)

    async def list_triggers(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_triggers(**kwargs)

    async def update_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_trigger(trigger_id, **kwargs)

    async def delete_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_trigger(trigger_id, **kwargs)

    async def run_trigger(self, trigger_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_run_trigger(trigger_id, **kwargs)

    async def list_trigger_executions(self, trigger_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_trigger_executions(trigger_id, **kwargs)

    async def create_cache(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_cache(**kwargs)

    async def get_cache(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_cache(name, **kwargs)

    async def list_caches(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_caches(**kwargs)

    async def update_cache(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_update_cache(name, **kwargs)

    async def delete_cache(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_cache(name, **kwargs)

    async def create_batch(self, src: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch(src, **kwargs)

    async def create_embedding_batch(self, src: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_embedding_batch(src, **kwargs)

    async def get_batch(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_batch(name, **kwargs)

    async def list_batches(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_batches(**kwargs)

    async def cancel_batch(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_batch(name, **kwargs)

    async def delete_batch(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_batch(name, **kwargs)

    async def list_models(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_models(**kwargs)

    async def get_model(self, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_model(model, **kwargs)

    async def delete_model(self, model: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_model(model, **kwargs)

    async def update_model(self, model: str, config: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_update_model(model, config=config, **kwargs)

    async def tune(
        self, base_model: str, training_dataset: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        return await self.provider.async_tune(
            base_model, training_dataset, config=config, **kwargs
        )

    async def get_tuning(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_tuning(name, **kwargs)

    async def list_tunings(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_tunings(**kwargs)

    async def cancel_tuning(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_tuning(name, **kwargs)

    async def validate_tuning_reward(
        self,
        parent: str,
        sample_response: Any,
        example: Any,
        single_reward_config: Any = None,
        composite_reward_config: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self.provider.async_validate_tuning_reward(
            parent,
            sample_response,
            example,
            single_reward_config=single_reward_config,
            composite_reward_config=composite_reward_config,
            config=config,
            **kwargs,
        )

    async def recontext_image(self, source: Any, **kwargs: Any) -> Any:
        return await self.provider.async_recontext_image(source, **kwargs)

    async def generate_images(self, prompt: str, **kwargs: Any) -> Any:
        return await self.provider.async_generate_images(prompt, **kwargs)

    async def edit_image(self, prompt: str, reference_images: list[Any], **kwargs: Any) -> Any:
        return await self.provider.async_edit_image(prompt, reference_images, **kwargs)

    async def upscale_image(self, image: Any, upscale_factor: str, **kwargs: Any) -> Any:
        return await self.provider.async_upscale_image(image, upscale_factor, **kwargs)

    async def segment_image(self, source: Any, **kwargs: Any) -> Any:
        return await self.provider.async_segment_image(source, **kwargs)

    async def generate_videos(self, **kwargs: Any) -> Any:
        return await self.provider.async_generate_videos(**kwargs)

    async def get_operation(self, operation: Any, **kwargs: Any) -> Any:
        return await self.provider.async_get_operation(operation, **kwargs)

    async def list_file_search_documents(self, parent: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_file_search_documents(parent, **kwargs)

    async def get_file_search_document(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_file_search_document(name, **kwargs)

    async def delete_file_search_document(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file_search_document(name, **kwargs)

    async def create_chat(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_chat(**kwargs)

    def connect_live(self, *, model: str | None = None, config: Any = None) -> Any:
        """返回异步 Live API 上下文管理器，调用方使用 `async with`。"""
        return self.provider.async_connect_live(model=model, config=config)

    async def create_file_search_store(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_file_search_store(**kwargs)

    async def get_file_search_store(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_file_search_store(name, **kwargs)

    async def list_file_search_stores(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_file_search_stores(**kwargs)

    async def delete_file_search_store(self, name: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file_search_store(name, **kwargs)

    async def import_file_to_file_search_store(self, file_search_store_name: str, file_name: str, **kwargs: Any) -> Any:
        return await self.provider.async_import_file_to_file_search_store(
            file_search_store_name, file_name, **kwargs
        )

    async def upload_to_file_search_store(self, file_search_store_name: str, file: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_to_file_search_store(
            file_search_store_name, file, **kwargs
        )

    async def download_file_search_media(self, media_id: str, **kwargs: Any) -> bytes:
        return await self.provider.async_download_file_search_media(media_id, **kwargs)


class AnthropicResources(_NativeFacade):

    _delegate_native = True

    def _beta_resource(self, name: str) -> Any:
        """返回 Anthropic Beta 资源，保留 SDK 的完整方法树。"""
        return getattr(self.beta, name)

    @property
    def beta_agents(self) -> Any:
        return self._beta_resource("agents")

    @property
    def beta_deployments(self) -> Any:
        return self._beta_resource("deployments")

    @property
    def beta_deployment_runs(self) -> Any:
        return self._beta_resource("deployment_runs")

    @property
    def beta_dreams(self) -> Any:
        return self._beta_resource("dreams")

    @property
    def beta_environments(self) -> Any:
        return self._beta_resource("environments")

    @property
    def beta_files(self) -> Any:
        return self._beta_resource("files")

    @property
    def beta_memory_stores(self) -> Any:
        return self._beta_resource("memory_stores")

    @property
    def beta_models(self) -> Any:
        return self._beta_resource("models")

    @property
    def beta_sessions(self) -> Any:
        return self._beta_resource("sessions")

    @property
    def beta_skills(self) -> Any:
        return self._beta_resource("skills")

    @property
    def beta_tunnels(self) -> Any:
        return self._beta_resource("tunnels")

    @property
    def beta_user_profiles(self) -> Any:
        return self._beta_resource("user_profiles")

    @property
    def beta_vaults(self) -> Any:
        return self._beta_resource("vaults")

    @property
    def beta_webhooks(self) -> Any:
        return self._beta_resource("webhooks")

    def count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return self.provider.count_tokens(request, **kwargs)

    def create_batch(self, requests: Any, **kwargs: Any) -> Any:
        return self.provider.create_batch(requests, **kwargs)

    def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_batch(batch_id, **kwargs)

    def batch_results(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.batch_results(batch_id, **kwargs)

    def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.cancel_batch(batch_id, **kwargs)

    def list_batches(self, **kwargs: Any) -> Any:
        return self.provider.list_batches(**kwargs)

    def delete_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_batch(batch_id, **kwargs)

    def upload_file(self, file: Any, **kwargs: Any) -> Any:
        return self.provider.upload_file(file, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        return self.provider.list_files(**kwargs)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_file(file_id, **kwargs)

    def download_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.download_file(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_file(file_id, **kwargs)

    def list_models(self, **kwargs: Any) -> Any:
        return self.provider.list_models(**kwargs)

    def retrieve_model(self, model_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_model(model_id, **kwargs)

    def parse(self, output_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.parse(output_format, request, **kwargs)

    def beta_create(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.beta_create(request, **kwargs)

    def beta_parse(self, output_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.beta_parse(output_format, request, **kwargs)

    def beta_count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return self.provider.beta_count_tokens(request, **kwargs)

    def beta_stream(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.beta_stream(request, **kwargs)

    def beta_tool_runner(self, tools: Any, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.beta_tool_runner(tools, request, **kwargs)


class AsyncAnthropicResources(_NativeFacade):
    """Anthropic AsyncAnthropic 资源的显式异步 Facade。"""

    _delegate_native = True
    _delegate_client_name = "async_sdk_client"

    def _beta_resource(self, name: str) -> Any:
        """返回 AsyncAnthropic Beta 资源，保留 SDK 的完整方法树。"""
        return getattr(self.async_beta, name)

    @property
    def beta_agents(self) -> Any:
        return self._beta_resource("agents")

    @property
    def beta_deployments(self) -> Any:
        return self._beta_resource("deployments")

    @property
    def beta_deployment_runs(self) -> Any:
        return self._beta_resource("deployment_runs")

    @property
    def beta_dreams(self) -> Any:
        return self._beta_resource("dreams")

    @property
    def beta_environments(self) -> Any:
        return self._beta_resource("environments")

    @property
    def beta_files(self) -> Any:
        return self._beta_resource("files")

    @property
    def beta_memory_stores(self) -> Any:
        return self._beta_resource("memory_stores")

    @property
    def beta_models(self) -> Any:
        return self._beta_resource("models")

    @property
    def beta_sessions(self) -> Any:
        return self._beta_resource("sessions")

    @property
    def beta_skills(self) -> Any:
        return self._beta_resource("skills")

    @property
    def beta_tunnels(self) -> Any:
        return self._beta_resource("tunnels")

    @property
    def beta_user_profiles(self) -> Any:
        return self._beta_resource("user_profiles")

    @property
    def beta_vaults(self) -> Any:
        return self._beta_resource("vaults")

    @property
    def beta_webhooks(self) -> Any:
        return self._beta_resource("webhooks")

    async def count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return await self.provider.async_count_tokens(request, **kwargs)

    async def create_batch(self, requests: Any, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch(requests, **kwargs)

    async def retrieve_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_batch(batch_id, **kwargs)

    async def batch_results(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_batch_results(batch_id, **kwargs)

    async def cancel_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_cancel_batch(batch_id, **kwargs)

    async def list_batches(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_batches(**kwargs)

    async def delete_batch(self, batch_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_batch(batch_id, **kwargs)

    async def upload_file(self, file: Any, **kwargs: Any) -> Any:
        return await self.provider.async_upload_file(file, **kwargs)

    async def list_files(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_files(**kwargs)

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_file(file_id, **kwargs)

    async def download_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_download_file(file_id, **kwargs)

    async def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file(file_id, **kwargs)

    async def list_models(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_models(**kwargs)

    async def retrieve_model(self, model_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_model(model_id, **kwargs)

    async def parse(self, output_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_parse(output_format, request, **kwargs)

    async def beta_create(self, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_beta_create(request, **kwargs)

    async def beta_parse(self, output_format: Any, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_beta_parse(output_format, request, **kwargs)

    async def beta_count_tokens(self, request: Any = None, **kwargs: Any) -> int:
        return await self.provider.async_beta_count_tokens(request, **kwargs)

    def beta_stream(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.async_beta_stream(request, **kwargs)

    def beta_tool_runner(self, tools: Any, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.async_beta_tool_runner(tools, request, **kwargs)


class ArkResources(_NativeFacade):

    _delegate_native = True

    def upload_file(self, file: Any, purpose: str, **kwargs: Any) -> Any:
        return self.provider.upload_file(file, purpose, **kwargs)

    def list_files(self, **kwargs: Any) -> Any:
        return self.provider.list_files(**kwargs)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_file(file_id, **kwargs)

    def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_file(file_id, **kwargs)

    def wait_for_file(self, file_id: str, **kwargs: Any) -> Any:
        return self.provider.wait_for_file(file_id, **kwargs)

    def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.retrieve_response(response_id, **kwargs)

    def create_response(self, request: Any = None, **kwargs: Any) -> Any:
        return self.provider.create_response(request, **kwargs)

    def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_response(response_id, **kwargs)

    def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        return self.provider.list_response_input_items(response_id, **kwargs)

    def list_input_items(self, response_id: str, **kwargs: Any) -> Any:
        """Ark SDK 顶层 input_items 资源的直观别名。"""
        return self.provider.list_input_items(response_id, **kwargs)

    def create_batch_chat(self, **kwargs: Any) -> Any:
        return self.provider.create_batch_chat(**kwargs)

    def generate_batch_chat(self, **kwargs: Any) -> Any:
        return self.provider.generate_batch_chat(**kwargs)

    def create_batch_embedding(self, **kwargs: Any) -> Any:
        return self.provider.create_batch_embedding(**kwargs)

    def create_batch_multimodal_embedding(self, **kwargs: Any) -> Any:
        return self.provider.create_batch_multimodal_embedding(**kwargs)

    def create_context(self, **kwargs: Any) -> Any:
        return self.provider.create_context(**kwargs)

    def context_complete(self, **kwargs: Any) -> Any:
        return self.provider.context_complete(**kwargs)

    def create_content_generation_task(self, **kwargs: Any) -> Any:
        return self.provider.create_content_generation_task(**kwargs)

    def get_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        return self.provider.get_content_generation_task(task_id, **kwargs)

    def list_content_generation_tasks(self, **kwargs: Any) -> Any:
        return self.provider.list_content_generation_tasks(**kwargs)

    def delete_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        return self.provider.delete_content_generation_task(task_id, **kwargs)

    def count_tokens(self, text: str | list[str], **kwargs: Any) -> int:
        return self.provider.count_tokens(text, **kwargs)

    def multimodal_embed(self, inputs: Any, **kwargs: Any) -> Any:
        return self.provider.multimodal_embed(inputs, **kwargs)

    def generate_image(self, **kwargs: Any) -> Any:
        return self.provider.generate_image(**kwargs)

    def beta_chat_parse(self, **kwargs: Any) -> Any:
        return self.provider.beta_chat_parse(**kwargs)

    def beta_chat_stream(self, **kwargs: Any) -> Any:
        return self.provider.beta_chat_stream(**kwargs)

    def bot_chat(self, **kwargs: Any) -> Any:
        return self.provider.bot_chat(**kwargs)

    def classify(
        self, query: str, labels: list[str], model: str | None = None, **kwargs: Any
    ) -> Any:
        return self.provider.classify(query, labels, model=model, **kwargs)


class AsyncArkResources(_NativeFacade):
    """Ark AsyncArk 资源的显式异步 Facade。"""

    _delegate_native = True
    _delegate_client_name = "async_sdk_client"

    async def upload_file(self, file: Any, purpose: str, **kwargs: Any) -> Any:
        return await self.provider.async_upload_file(file, purpose, **kwargs)

    async def list_files(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_files(**kwargs)

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_file(file_id, **kwargs)

    async def delete_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_file(file_id, **kwargs)

    async def wait_for_file(self, file_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_wait_for_file(file_id, **kwargs)

    async def retrieve_response(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_retrieve_response(response_id, **kwargs)

    async def create_response(self, request: Any = None, **kwargs: Any) -> Any:
        return await self.provider.async_create_response(request, **kwargs)

    async def delete_response(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_response(response_id, **kwargs)

    async def list_response_input_items(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_response_input_items(response_id, **kwargs)

    async def list_input_items(self, response_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_list_input_items(response_id, **kwargs)

    async def create_batch_chat(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch_chat(**kwargs)

    async def generate_batch_chat(self, **kwargs: Any) -> Any:
        return await self.provider.async_generate_batch_chat(**kwargs)

    async def create_batch_embedding(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch_embedding(**kwargs)

    async def create_batch_multimodal_embedding(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_batch_multimodal_embedding(**kwargs)

    async def create_context(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_context(**kwargs)

    async def context_complete(self, **kwargs: Any) -> Any:
        return await self.provider.async_context_complete(**kwargs)

    async def create_content_generation_task(self, **kwargs: Any) -> Any:
        return await self.provider.async_create_content_generation_task(**kwargs)

    async def get_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_get_content_generation_task(task_id, **kwargs)

    async def list_content_generation_tasks(self, **kwargs: Any) -> Any:
        return await self.provider.async_list_content_generation_tasks(**kwargs)

    async def delete_content_generation_task(self, task_id: str, **kwargs: Any) -> Any:
        return await self.provider.async_delete_content_generation_task(task_id, **kwargs)

    async def count_tokens(self, text: str | list[str], **kwargs: Any) -> int:
        return await self.provider.async_count_tokens(text, **kwargs)

    async def multimodal_embed(self, inputs: Any, **kwargs: Any) -> Any:
        return await self.provider.async_multimodal_embed(inputs, **kwargs)

    async def generate_image(self, **kwargs: Any) -> Any:
        return await self.provider.async_generate_image(**kwargs)

    async def beta_chat_parse(self, **kwargs: Any) -> Any:
        return await self.provider.async_beta_chat_parse(**kwargs)

    def beta_chat_stream(self, **kwargs: Any) -> Any:
        """返回可直接用于 ``async with`` 的 Ark 异步流上下文管理器。"""
        return self.provider.async_beta_chat_stream(**kwargs)

    async def bot_chat(self, **kwargs: Any) -> Any:
        return await self.provider.async_bot_chat(**kwargs)

    async def classify(
        self, query: str, labels: list[str], model: str | None = None, **kwargs: Any
    ) -> Any:
        return await self.provider.async_classify(query, labels, model=model, **kwargs)
