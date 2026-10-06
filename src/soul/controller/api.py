import asyncio
import contextlib
import json
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from soul.protocol import SoulError


def error_body(exc, anthropic=False):
    status = getattr(exc, "status", 500)
    if anthropic:
        kind = (
            "invalid_request_error"
            if status == 400
            else "authentication_error"
            if status == 401
            else "overloaded_error"
            if status == 503
            else "api_error"
        )
        return {"type": "error", "error": {"type": kind, "message": str(exc)}}
    return {
        "error": {
            "message": str(exc),
            "type": "invalid_request_error" if status < 500 else "server_error",
            "param": None,
            "code": getattr(exc, "code", "internal_error"),
        }
    }


def sse(data, event=None):
    return (f"event: {event}\n" if event else "") + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def usage(done):
    return {
        "prompt_tokens": done["input_tokens"],
        "completion_tokens": done["output_tokens"],
        "total_tokens": done["input_tokens"] + done["output_tokens"],
    }


async def respond(engine, body, family, http_request):
    anthropic, completion = family == "anthropic", family == "completion"
    id = ("msg_" if anthropic else "cmpl-" if completion else "chatcmpl-") + uuid.uuid4().hex
    created = int(time.time())
    try:
        request = body.generation()
    except ValueError as exc:
        raise SoulError(str(exc), 400, "invalid_request") from exc
    generator = engine.generate(request, id)

    async def monitor_disconnect():
        while not await http_request.is_disconnected():
            await asyncio.sleep(0.2)

    # Detect disconnects during queueing/downloads and non-streaming generation, too.
    disconnected = asyncio.create_task(monitor_disconnect())

    async def next_event():
        task = asyncio.create_task(anext(generator))
        try:
            done, _ = await asyncio.wait({task, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise asyncio.CancelledError
            return await task
        except BaseException:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            raise

    async def close():
        disconnected.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnected
        await generator.aclose()

    try:
        start = await next_event()  # Return ordinary 4xx/5xx before sending HTTP stream headers.
    except BaseException:
        await close()
        raise

    def chunk(text=None, finish=None, role=False):
        if completion:
            choice = {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish}
        else:
            choice = {
                "index": 0,
                "delta": {"role": "assistant", "content": ""}
                if role
                else {"content": text}
                if text is not None
                else {},
                "logprobs": None,
                "finish_reason": finish,
            }
        result = {
            "id": id,
            "object": "text_completion" if completion else "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [choice],
        }
        if body.stream_options and body.stream_options.include_usage:
            result["usage"] = None
        return result

    def anthropic_message(text, done=None):
        reason = (
            None
            if done is None
            else "max_tokens"
            if done["finish_reason"] == "length"
            else "stop_sequence"
            if done["stop_sequence"]
            else "end_turn"
        )
        return {
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": body.model,
            "content": [{"type": "text", "text": text}] if done else [],
            "stop_reason": reason,
            "stop_sequence": done["stop_sequence"] if done else None,
            "usage": {
                "input_tokens": start["input_tokens"],
                "output_tokens": done["output_tokens"] if done else 0,
            },
        }

    if not body.stream:
        text, done = "", None
        try:
            while True:
                try:
                    event = await next_event()
                except StopAsyncIteration:
                    break
                if event["kind"] == "text":
                    text += event["text"]
                elif event["kind"] == "done":
                    done = event
            if anthropic:
                return JSONResponse(anthropic_message(text, done))
            choice = {"index": 0, "finish_reason": done["finish_reason"], "logprobs": None}
            choice.update(
                {"text": text} if completion else {"message": {"role": "assistant", "content": text}}
            )
            return JSONResponse(
                {
                    "id": id,
                    "object": "text_completion" if completion else "chat.completion",
                    "created": created,
                    "model": body.model,
                    "choices": [choice],
                    "usage": usage(done),
                }
            )
        finally:
            await close()

    # StreamingResponse owns disconnect detection once it takes over the ASGI receive channel.
    disconnected.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await disconnected

    async def stream():
        try:
            if anthropic:
                yield sse({"type": "message_start", "message": anthropic_message("")}, "message_start")
                yield sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
            else:
                yield sse(chunk(role=True))
            async for event in generator:
                if event["kind"] == "text":
                    if anthropic:
                        yield sse(
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": event["text"]},
                            },
                            "content_block_delta",
                        )
                    else:
                        yield sse(chunk(text=event["text"]))
                elif event["kind"] == "done":
                    if anthropic:
                        message = anthropic_message("", event)
                        yield sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
                        yield sse(
                            {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": message["stop_reason"],
                                    "stop_sequence": message["stop_sequence"],
                                },
                                "usage": {"output_tokens": event["output_tokens"]},
                            },
                            "message_delta",
                        )
                        yield sse({"type": "message_stop"}, "message_stop")
                    else:
                        yield sse(chunk(finish=event["finish_reason"]))
                        if body.stream_options and body.stream_options.include_usage:
                            result = chunk()
                            result.update(choices=[], usage=usage(event))
                            yield sse(result)
            if not anthropic:
                yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield sse(error_body(exc, anthropic), "error" if anthropic else None)
        finally:
            await generator.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
