# API coverage

Soul accepts the **text subset** of the following interfaces. Model IDs refer to Hugging Face repositories, not OpenAI/Anthropic hosted models. Unsupported fields produce `400` errors. There are no proxy calls to either vendor.

| Endpoint | Supported |
| --- | --- |
| `GET /healthz` | Unauthenticated liveness/version |
| `GET /v1/models` | Ready local deployments, OpenAI list envelope |
| `POST /v1/chat/completions` | Text messages, nonstream/stream, usage |
| `POST /v1/completions` | One string prompt, nonstream/stream |
| `POST /v1/messages` | Anthropic text messages/system, nonstream/stream |
| `GET /admin/state` | Nodes, model assignments, metrics, recent requests, queue |
| `GET /admin/nodes/{id}/logs` | Recent node events |
| `POST /admin/models/load` | `{"model":"owner/repo"}`; waits for placement/load |
| `POST /admin/models/unload` | `{"model":"owner/repo"}`; releases a ready deployment |
| `WS /internal/nodes` | Authenticated MessagePack node channel |

## OpenAI subset

Authentication: `Authorization: Bearer <SOUL_API_KEY>`.

Chat accepts `model`, `messages`, `max_tokens` **or** `max_completion_tokens`, `temperature` (0–2), `top_p` (0–1, exclusive lower bound), `stream`, `stop` (one string or at most four strings), `seed`, `n=1`, optional `user`, and `stream_options.include_usage`. Roles are system/developer/user/assistant; developer is mapped to system for model templates. Content is a string or a list of `{type:"text", text:"..."}`. `user` is accepted but has no scheduling or storage effect.

Text Completions accepts one string `prompt`, the common sampling/stream fields, `n=1`, `seed`, `stop`, optional `user`, and `stream_options.include_usage`. It does not accept a batch of prompts or pretokenized input. Completion streaming uses `text_completion` envelopes; Chat uses `chat.completion.chunk` with role/content deltas. Both end with a finish reason followed by `data: [DONE]`. When requested, a separate last usage chunk has empty choices.

The first successful HTTP stream bytes are sent **after** placement, loading, tokenization and context validation. Reverse-proxy/client timeouts must permit cold model loading. Error events after headers cannot change the HTTP status code; failed streams do not emit the success `[DONE]` sentinel.

## Anthropic subset

Authentication: `x-api-key: <SOUL_API_KEY>` (Bearer is also accepted). The `anthropic-version` header is accepted but does not negotiate versions.

Accepts `model`, required `max_tokens`, `messages` with user/assistant roles, top-level `system` (string or text blocks), `temperature` (0–1), `top_p`, `stream`, and up to four `stop_sequences`. Other features, including thinking, tools, images, cache-control blocks and metadata, are unsupported. Text block fields are strictly validated.

Streaming emits `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop` in order. Content is represented as a single text block. Finish reasons map to `end_turn`, `max_tokens`, or `stop_sequence`. Output usage is cumulative. Errors use `event: error` and an Anthropic error envelope.

## Constraints

- One active generation, one sequence at a time. No continuous batching, beam search, speculative decoding or multiple choices.
- Default output limit 128 tokens for OpenAI endpoints. The Anthropic endpoint requires `max_tokens`.
- Prompt tokens plus requested output limit must fit the controller's configured model context. No silent truncation.
- Temperature zero selects greedy argmax. Positive temperature uses node-local categorical sampling, with optional nucleus filtering. A seed controls the final node's CPU random generator; different backends/precision may still yield different results.
- Explicit stop text never appears in the response, including when split across token boundaries. Sampled EOS/stop tokens count toward generated token usage.
- Request bodies are bounded at 1 MiB; an individual message is limited to 262,144 characters. RPC frames are bounded at 64 MiB.
- Model loading is on demand, so configure client timeouts for initial downloads. Controller flags separately configure queue, RPC and load timeouts.
- Models without a chat template use a simple line-separated role transcript with a trailing `assistant:` prompt. Prefer an instruction model with an appropriate tokenizer template.

Not implemented: OpenAI Responses, embeddings, audio, image/video input, tool/function calls, JSON schema/structured output, logprobs, frequency/presence penalties, vendor-specific caching, vendor moderation fields, or vendor model aliases. These are intentionally rejected rather than approximated.
