import asyncio
import sys
import time
from dataclasses import dataclass
from functools import wraps
from inspect import (Signature, isasyncgenfunction, iscoroutinefunction,
                     signature)
from typing import (Any, AsyncGenerator, Callable, Dict, Iterable, List,
                    Optional, Tuple, Type, TypeVar, overload)

DUMMY_TYPE = "__BATCH__"


def extract_self_if_method_call(args: List[Any], func: Callable) -> Optional[object]:
    """Check if this is a method rather than a function.

    Does this by checking to see if `func` is the attribute of the first
    (`self`) argument under `func.__name__`. Unfortunately, this is the most
    robust solution to this I was able to find. It would also be preferable
    to do this check when the decorator runs, rather than when the method is.

    Returns the `self` object if it's a method call, else None.

    Arguments:
        args: arguments to the function/method call.
        func: the unbound function that was called.
    """
    if len(args) > 0:
        method = getattr(args[0], func.__name__, False)
        if method:
            wrapped = getattr(method, "__wrapped__", False)
            if wrapped and wrapped == func:
                return args[0]

    return None


def get_or_create_event_loop() -> asyncio.BaseEventLoop:
    """Get a running async event loop if one exists, otherwise create one.

    This function serves as a proxy for the deprecating get_event_loop().
    It tries to get the running loop first, and if no running loop
    could be retrieved:
    - For python version <3.10: it falls back to the get_event_loop
        call.
    - For python version >= 3.10: it uses the same python implementation
        of _get_event_loop() at asyncio/events.py.

    Ideally, one should use high level APIs like asyncio.run() with python
    version >= 3.7, if not possible, one should create and manage the event
    loops explicitly.
    """
    vers_info = sys.version_info
    if vers_info.major >= 3 and vers_info.minor >= 10:
        # This follows the implementation of the deprecating `get_event_loop`
        # in python3.10's asyncio. See python3.10/asyncio/events.py
        # _get_event_loop()
        loop = None
        try:
            loop = asyncio.get_running_loop()
            assert loop is not None
            return loop
        except RuntimeError as e:
            # No running loop, relying on the error message as for now to
            # differentiate runtime errors.
            assert "no running event loop" in str(e)
            return asyncio.get_event_loop_policy().get_event_loop()

    return asyncio.get_event_loop()


def is_cython(obj):
    """Check if an object is a Cython function or method"""

    # TODO(suo): We could split these into two functions, one for Cython
    # functions and another for Cython methods.
    # TODO(suo): There doesn't appear to be a Cython function 'type' we can
    # check against via isinstance. Please correct me if I'm wrong.
    def check_cython(x):
        return type(x).__name__ == "cython_function_or_method"

    # Check if function or method, respectively
    return check_cython(obj) or (hasattr(obj, "__func__") and check_cython(obj.__func__))


def get_signature(func):
    """Get signature parameters.

    Support Cython functions by grabbing relevant attributes from the Cython
    function and attaching to a no-op function. This is somewhat brittle, since
    inspect may change, but given that inspect is written to a PEP, we hope
    it is relatively stable. Future versions of Python may allow overloading
    the inspect 'isfunction' and 'ismethod' functions / create ABC for Python
    functions. Until then, it appears that Cython won't do anything about
    compatability with the inspect module.

    Args:
        func: The function whose signature should be checked.

    Returns:
        A function signature object, which includes the names of the keyword
            arguments as well as their default values.

    Raises:
        TypeError: A type error if the signature is not supported
    """
    # The first condition for Cython functions, the latter for Cython instance
    # methods
    if is_cython(func):
        attrs = ["__code__", "__annotations__", "__defaults__", "__kwdefaults__"]

        if all(hasattr(func, attr) for attr in attrs):
            original_func = func

            def func():
                return

            for attr in attrs:
                setattr(func, attr, getattr(original_func, attr))
        else:
            raise TypeError(f"{func!r} is not a Python function we can process")

    return signature(func)


def extract_signature(func, ignore_first=False):
    """Extract the function signature from the function.

    Args:
        func: The function whose signature should be extracted.
        ignore_first: True if the first argument should be ignored. This should
            be used when func is a method of a class.

    Returns:
        List of Parameter objects representing the function signature.
    """
    signature_parameters = list(get_signature(func).parameters.values())

    if ignore_first:
        if len(signature_parameters) == 0:
            raise ValueError("Methods must take a 'self' argument, but the " f"method '{func.__name__}' does not have one.")
        signature_parameters = signature_parameters[1:]

    return signature_parameters


def flatten_args(signature_parameters: list, args, kwargs):
    """Validates the arguments against the signature and flattens them.

    The flat list representation is a serializable format for arguments.
    Since the flatbuffer representation of function arguments is a list, we
    combine both keyword arguments and positional arguments. We represent
    this with two entries per argument value - [DUMMY_TYPE, x] for positional
    arguments and [KEY, VALUE] for keyword arguments. See the below example.
    See `recover_args` for logic restoring the flat list back to args/kwargs.

    Args:
        signature_parameters: The list of Parameter objects
            representing the function signature, obtained from
            `extract_signature`.
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.

    Returns:
        List of args and kwargs. Non-keyword arguments are prefixed
            by internal enum DUMMY_TYPE.

    Raises:
        TypeError: Raised if arguments do not fit in the function signature.
    """

    reconstructed_signature = Signature(parameters=signature_parameters)
    try:
        reconstructed_signature.bind(*args, **kwargs)
    except TypeError as exc:  # capture a friendlier stacktrace
        raise TypeError(str(exc)) from None
    list_args = []
    for arg in args:
        list_args += [DUMMY_TYPE, arg]

    for keyword, arg in kwargs.items():
        list_args += [keyword, arg]
    return list_args


def recover_args(flattened_args):
    """Recreates `args` and `kwargs` from the flattened arg list.

    Args:
        flattened_args: List of args and kwargs. This should be the output of
            `flatten_args`.

    Returns:
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.
    """
    assert len(flattened_args) % 2 == 0, "Flattened arguments need to be even-numbered. See `flatten_args`."
    args = []
    kwargs = {}
    for name_index in range(0, len(flattened_args), 2):
        name, arg = flattened_args[name_index], flattened_args[name_index + 1]
        if name == DUMMY_TYPE:
            args.append(arg)
        else:
            kwargs[name] = arg

    return args, kwargs


@dataclass
class _SingleRequest:
    self_arg: Any
    flattened_args: List[Any]
    future: asyncio.Future


@dataclass
class _GeneratorResult:
    result: Any
    next_future: asyncio.Future


def _batch_args_kwargs(
    list_of_flattened_args: List[List[Any]],
) -> Tuple[Tuple[Any], Dict[Any, Any]]:
    """Batch a list of flatten args and returns regular args and kwargs"""
    # Ray's flatten arg format is a list with alternating key and values
    # e.g. args=(1, 2), kwargs={"key": "val"} got turned into
    #      [None, 1, None, 2, "key", "val"]
    arg_lengths = {len(args) for args in list_of_flattened_args}
    assert len(arg_lengths) == 1, "All batch requests should have the same number of parameters."
    arg_length = arg_lengths.pop()

    batched_flattened_args = []
    for idx in range(arg_length):
        if idx % 2 == 0:
            batched_flattened_args.append(list_of_flattened_args[0][idx])
        else:
            batched_flattened_args.append([item[idx] for item in list_of_flattened_args])

    return recover_args(batched_flattened_args)


class _BatchQueue:
    def __init__(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        handle_batch_func: Optional[Callable] = None,
    ) -> None:
        """Async queue that accepts individual items and returns batches.

        Respects max_batch_size and timeout_s; a batch will be returned when
        max_batch_size elements are available or the timeout has passed since
        the previous get.

        If handle_batch_func is passed in, a background coroutine will run to
        poll from the queue and call handle_batch_func on the results.

        Cannot be pickled.

        Arguments:
            max_batch_size: max number of elements to return in a batch.
            timeout_s: time to wait before returning an incomplete
                batch.
            handle_batch_func(Optional[Callable]): callback to run in the
                background to handle batches if provided.
        """
        self.queue: asyncio.Queue[_SingleRequest] = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.queue_put_event = asyncio.Event()

        self._handle_batch_task = None
        if handle_batch_func is not None:
            self._handle_batch_task = get_or_create_event_loop().create_task(self._process_batches(handle_batch_func))

    def put(self, request: Tuple[_SingleRequest, asyncio.Future]) -> None:
        self.queue.put_nowait(request)
        self.queue_put_event.set()

    async def wait_for_batch(self) -> List[Any]:
        """Wait for batch respecting self.max_batch_size and self.timeout_s.

        Returns a batch of up to self.max_batch_size items. Waits for up to
        to self.timeout_s after receiving the first request that will be in
        the next batch. After the timeout, returns as many items as are ready.

        Always returns a batch with at least one item - will block
        indefinitely until an item comes in.
        """

        batch = []
        batch.append(await self.queue.get())

        # Cache current max_batch_size and batch_wait_timeout_s for this batch.
        max_batch_size = self.max_batch_size
        batch_wait_timeout_s = self.batch_wait_timeout_s

        # Wait self.timeout_s seconds for new queue arrivals.
        batch_start_time = time.time()
        while True:
            remaining_batch_time_s = max(batch_wait_timeout_s - (time.time() - batch_start_time), 0)
            try:
                # Wait for new arrivals.
                await asyncio.wait_for(self.queue_put_event.wait(), remaining_batch_time_s)
            except asyncio.TimeoutError:
                pass

            # Add all new arrivals to the batch.
            while len(batch) < max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            self.queue_put_event.clear()

            if time.time() - batch_start_time >= batch_wait_timeout_s or len(batch) >= max_batch_size:
                break

        return batch

    def _validate_results(self, results: Iterable[Any], input_batch_length: int) -> None:
        if len(results) != input_batch_length:
            raise Exception(
                "Batched function doesn't preserve batch size. "
                f"The input list has length {input_batch_length} but the "
                f"returned list has length {len(results)}."
            )

    async def _consume_func_generator(
        self,
        func_generator: AsyncGenerator,
        initial_futures: List[asyncio.Future],
        input_batch_length: int,
    ) -> None:
        """Consumes batch function generator.

        This function only runs if the function decorated with @serve.batch
        is a generator.
        """

        FINISHED_TOKEN = None

        try:
            futures = initial_futures
            async for results in func_generator:
                self._validate_results(results, input_batch_length)
                next_futures = []
                for result, future in zip(results, futures):
                    if future is FINISHED_TOKEN:
                        # This caller has already terminated.
                        next_futures.append(FINISHED_TOKEN)
                    elif result in [StopIteration, StopAsyncIteration]:
                        # User's code returned sentinel. No values left
                        # for caller. Terminate iteration for caller.
                        future.set_exception(StopAsyncIteration)
                        next_futures.append(FINISHED_TOKEN)
                    else:
                        next_future = get_or_create_event_loop().create_future()
                        future.set_result(_GeneratorResult(result, next_future))
                        next_futures.append(next_future)
                futures = next_futures

            for future in futures:
                if future is not FINISHED_TOKEN:
                    future.set_exception(StopAsyncIteration)
        except Exception as e:
            for future in futures:
                if future is not FINISHED_TOKEN:
                    future.set_exception(e)

    async def _process_batches(self, func: Callable) -> None:
        """Loops infinitely and processes queued request batches."""

        while True:
            batch: List[_SingleRequest] = await self.wait_for_batch()
            assert len(batch) > 0
            self_arg = batch[0].self_arg
            args, kwargs = _batch_args_kwargs([item.flattened_args for item in batch])
            futures = [item.future for item in batch]

            # Method call.
            if self_arg is not None:
                func_future_or_generator = func(self_arg, *args, **kwargs)
            # Normal function call.
            else:
                func_future_or_generator = func(*args, **kwargs)

            if isasyncgenfunction(func):
                func_generator = func_future_or_generator
                await self._consume_func_generator(func_generator, futures, len(batch))
            else:
                try:
                    func_future = func_future_or_generator
                    results = await func_future
                    self._validate_results(results, len(batch))
                    for result, future in zip(results, futures):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)

    def __del__(self):
        if self._handle_batch_task is None or not get_or_create_event_loop().is_running():
            return

        # TODO(edoakes): although we try to gracefully shutdown here, it still
        # causes some errors when the process exits due to the asyncio loop
        # already being destroyed.
        self._handle_batch_task.cancel()


class _LazyBatchQueueWrapper:
    """Stores a _BatchQueue and updates its settings.

    _BatchQueue cannot be pickled, you must construct it lazily
    at runtime inside a replica. This class initializes a queue only upon
    first access.
    """

    def __init__(
        self,
        max_batch_size: int = 10,
        batch_wait_timeout_s: float = 0.0,
        handle_batch_func: Optional[Callable] = None,
        batch_queue_cls: Type[_BatchQueue] = _BatchQueue,
    ):
        self._queue: Type[_BatchQueue] = None
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.handle_batch_func = handle_batch_func
        self.batch_queue_cls = batch_queue_cls

    @property
    def queue(self) -> Type[_BatchQueue]:
        """Returns _BatchQueue.

        Initializes queue when called for the first time.
        """
        if self._queue is None:
            self._queue = self.batch_queue_cls(
                self.max_batch_size,
                self.batch_wait_timeout_s,
                self.handle_batch_func,
            )
        return self._queue

    def set_max_batch_size(self, new_max_batch_size: int) -> None:
        """Updates queue's max_batch_size."""

        self.max_batch_size = new_max_batch_size

        if self._queue is not None:
            self._queue.max_batch_size = new_max_batch_size

    def set_batch_wait_timeout_s(self, new_batch_wait_timeout_s: float) -> None:
        self.batch_wait_timeout_s = new_batch_wait_timeout_s

        if self._queue is not None:
            self._queue.batch_wait_timeout_s = new_batch_wait_timeout_s

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def get_batch_wait_timeout_s(self) -> float:
        return self.batch_wait_timeout_s


def _validate_max_batch_size(max_batch_size):
    if not isinstance(max_batch_size, int):
        if isinstance(max_batch_size, float) and max_batch_size.is_integer():
            max_batch_size = int(max_batch_size)
        else:
            raise TypeError(f"max_batch_size must be integer >= 1, got {max_batch_size}")

    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be an integer >= 1, got {max_batch_size}")


def _validate_batch_wait_timeout_s(batch_wait_timeout_s):
    if not isinstance(batch_wait_timeout_s, (float, int)):
        raise TypeError("batch_wait_timeout_s must be a float >= 0, " f"got {batch_wait_timeout_s}")

    if batch_wait_timeout_s < 0:
        raise ValueError("batch_wait_timeout_s must be a float >= 0, " f"got {batch_wait_timeout_s}")


T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F")
G = TypeVar("G", bound=Callable[..., Any])



# Normal decorator use case (called with no arguments).
@overload
def batch(func: F) -> G:
    pass


# "Decorator factory" use case (called with arguments).
@overload
def batch(
    max_batch_size: int = 10,
    batch_wait_timeout_s: float = 0.0,
) -> Callable[[F], G]:
    pass


def batch(
    _func: Optional[Callable] = None,
    max_batch_size: int = 10,
    batch_wait_timeout_s: float = 0.0,
    *,
    batch_queue_cls: Type[_BatchQueue] = _BatchQueue,
):
    """Converts a function to asynchronously handle batches.

    The function can be a standalone function or a class method. In both
    cases, the function must be `async def` and take a list of objects as
    its sole argument and return a list of the same length as a result.

    When invoked, the caller passes a single object. These will be batched
    and executed asynchronously once there is a batch of `max_batch_size`
    or `batch_wait_timeout_s` has elapsed, whichever occurs first.

    `max_batch_size` and `batch_wait_timeout_s` can be updated using setter
    methods from the batch_handler (`set_max_batch_size` and
    `set_batch_wait_timeout_s`).

    Example:

    .. code-block:: python

            from ray import serve
            from starlette.requests import Request

            @serve.deployment
            class BatchedDeployment:
                @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
                async def batch_handler(self, requests: List[Request]) -> List[str]:
                    response_batch = []
                    for r in requests:
                        name = (await requests.json())["name"]
                        response_batch.append(f"Hello {name}!")

                    return response_batch

                def update_batch_params(self, max_batch_size, batch_wait_timeout_s):
                    self.batch_handler.set_max_batch_size(max_batch_size)
                    self.batch_handler.set_batch_wait_timeout_s(batch_wait_timeout_s)

                async def __call__(self, request: Request):
                    return await self.batch_handler(request)

            app = BatchedDeployment.bind()

    Arguments:
        max_batch_size: the maximum batch size that will be executed in
            one call to the underlying function.
        batch_wait_timeout_s: the maximum duration to wait for
            `max_batch_size` elements before running the current batch.
        batch_queue_cls: the class to use for the underlying batch queue.
    """
    # `_func` will be None in the case when the decorator is parametrized.
    # See the comment at the end of this function for a detailed explanation.
    if _func is not None:
        if not callable(_func):
            raise TypeError("@serve.batch can only be used to decorate functions or methods.")

        if not iscoroutinefunction(_func):
            raise TypeError("Functions decorated with @serve.batch must be 'async def'")

    _validate_max_batch_size(max_batch_size)
    _validate_batch_wait_timeout_s(batch_wait_timeout_s)

    def _batch_decorator(_func):
        lazy_batch_queue_wrapper = _LazyBatchQueueWrapper(
            max_batch_size,
            batch_wait_timeout_s,
            _func,
            batch_queue_cls,
        )

        async def batch_handler_generator(
            first_future: asyncio.Future,
        ) -> AsyncGenerator:
            """Generator that handles generator batch functions."""

            future = first_future
            while True:
                try:
                    async_response: _GeneratorResult = await future
                    future = async_response.next_future
                    yield async_response.result
                except StopAsyncIteration:
                    break

        def enqueue_request(args, kwargs) -> asyncio.Future:
            self = extract_self_if_method_call(args, _func)
            flattened_args: List = flatten_args(extract_signature(_func), args, kwargs)

            if self is None:
                # For functions, inject the batch queue as an
                # attribute of the function.
                batch_queue_object = _func
            else:
                # For methods, inject the batch queue as an
                # attribute of the object.
                batch_queue_object = self
                # Trim the self argument from methods
                flattened_args = flattened_args[2:]

            batch_queue = lazy_batch_queue_wrapper.queue

            # Magic batch_queue_object attributes that can be used to change the
            # batch queue attributes on the fly.
            # This is purposefully undocumented for now while we figure out
            # the best API.
            if hasattr(batch_queue_object, "_ray_serve_max_batch_size"):
                new_max_batch_size = getattr(batch_queue_object, "_ray_serve_max_batch_size")
                _validate_max_batch_size(new_max_batch_size)
                batch_queue.max_batch_size = new_max_batch_size

            if hasattr(batch_queue_object, "_ray_serve_batch_wait_timeout_s"):
                new_batch_wait_timeout_s = getattr(batch_queue_object, "_ray_serve_batch_wait_timeout_s")
                _validate_batch_wait_timeout_s(new_batch_wait_timeout_s)
                batch_queue.batch_wait_timeout_s = new_batch_wait_timeout_s

            future = get_or_create_event_loop().create_future()
            batch_queue.put(_SingleRequest(self, flattened_args, future))
            return future

        # TODO (shrekris-anyscale): deprecate batch_queue_cls argument and
        # convert batch_wrapper into a class once `self` argument is no
        # longer needed in `enqueue_request`.
        @wraps(_func)
        def generator_batch_wrapper(*args, **kwargs):
            first_future = enqueue_request(args, kwargs)
            return batch_handler_generator(first_future)

        @wraps(_func)
        async def batch_wrapper(*args, **kwargs):
            # This will raise if the underlying call raised an exception.
            return await enqueue_request(args, kwargs)

        if isasyncgenfunction(_func):
            wrapper = generator_batch_wrapper
        else:
            wrapper = batch_wrapper

        # We store the lazy_batch_queue_wrapper's getters and setters as
        # batch_wrapper attributes, so they can be accessed in user code.
        wrapper._get_max_batch_size = lazy_batch_queue_wrapper.get_max_batch_size
        wrapper._get_batch_wait_timeout_s = lazy_batch_queue_wrapper.get_batch_wait_timeout_s
        wrapper.set_max_batch_size = lazy_batch_queue_wrapper.set_max_batch_size
        wrapper.set_batch_wait_timeout_s = lazy_batch_queue_wrapper.set_batch_wait_timeout_s

        return wrapper

    # Unfortunately, this is required to handle both non-parametrized
    # (@serve.batch) and parametrized (@serve.batch(**kwargs)) usage.
    # In the former case, `serve.batch` will be called with the underlying
    # function as the sole argument. In the latter case, it will first be
    # called with **kwargs, then the result of that call will be called
    # with the underlying function as the sole argument (i.e., it must be a
    # "decorator factory.").
    return _batch_decorator(_func) if callable(_func) else _batch_decorator
