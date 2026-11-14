// The client wires and the three upstream wires, in both directions.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. Each request is
// answered by one provider on its own wire (openai, anthropic or gemini). This file translates the
// request into the provider's wire, parses the provider's answer (JSON or SSE) into one normalized
// shape, and rebuilds the answer in the wire the client used.
//
// The one rule behind every translation: the provider echoes what it parsed (system prompt, user
// text, images, tool names, stop sequences, max_tokens, JSON mode), and the client compares that
// echo with what it sent. A field rewritten or dropped loses the request. So every translation
// below preserves each echoed field exactly:
//
//   - system: OpenAI system messages, Anthropic's top-level system, Gemini's systemInstruction all
//     carry the same text (multiple system messages joined with "\n", as each wire's parser joins
//     them).
//   - user text: the last user message's text blocks, joined with "\n" on every wire.
//   - images: a base64 image keeps its bytes (data URL, inlineData and Anthropic base64 source all
//     hash the same bytes); a URL image keeps its URL string.
//   - tools: the tool names, in order, on every wire (OpenAI function names, Anthropic tool names,
//     Gemini functionDeclarations).
//   - max_tokens: passed through as max_tokens (OpenAI/Anthropic) or maxOutputTokens (Gemini),
//     never invented when the client sent none.
//   - stop sequences: stop (OpenAI), stop_sequences (Anthropic), stopSequences (Gemini), same list.
//   - JSON mode: response_format on the OpenAI wire, responseMimeType application/json on Gemini;
//     the Anthropic wire cannot carry it, so a JSON-mode OpenAI request is never routed to an
//     Anthropic-wire provider (see lib/plan.ts).
//
// Streaming is buffered, not piped: the whole upstream SSE is read, parsed into the normalized
// answer, and re-emitted in the client's event order. That is what lets a streamed answer carry
// the provider's usage (the OpenAI client never sets stream_options, so a piped-through stream
// would carry no usage chunk and fail the usage check) and the Anthropic message_start carry the
// true input tokens instead of 0.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings use
// fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One data URL split into its media type and base64 payload, or null when it is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*)(?:;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

// ---------------------------------------------------------------------------
// Request builders: the client's body rewritten into a provider's wire.
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions request naming `model`, with usage requested on streams. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** An Anthropic Messages request naming `model`: the client's body as the provider takes it. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  return { ...body, model };
}

function openAIToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = dataUrlOf(part.image_url.url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url: part.image_url.url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini content part: text or an inline image; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = dataUrlOf(part.image_url.url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    return { fileData: { fileUri: part.image_url.url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part: text, image or tool result; null for the rest. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_result') {
    return {
      functionResponse: {
        name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool_result',
        response: { content: typeof block.content === 'string' ? block.content : JSON.stringify(block.content ?? '') },
      },
    };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** The stop list of an OpenAI body: its `stop`, as an array of strings. */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** The stop list of an Anthropic body: its `stop_sequences`, as an array of strings. */
function anthropicStops(body: Json): string[] {
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the system
 * messages become the system string, image_url parts become image blocks, tool_calls become
 * tool_use blocks, tool messages become tool_result blocks, and the output cap, temperature,
 * top_p, stop sequences and streaming pass through. (No response_format: the Anthropic wire
 * cannot carry JSON mode, so the router never sends a JSON-mode request here.)
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          content.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
    } else if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((b) => (isRecord(b) ? [openAIPartToAnthropic(b)].flatMap((x) => (x === null ? [] : [x])) : []));
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsOf(body);
  if (tools.length > 0) {
    out.tools = tools.map((t) => {
      const fn = isRecord(t.function) ? t.function : {};
      return { name: fn.name, description: fn.description, input_schema: fn.parameters };
    });
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI tools as Gemini functionDeclarations, in order. */
function openAIToolsToGemini(body: Json): Json[] {
  return openAIToolsOf(body).map((t) => {
    const fn = isRecord(t.function) ? t.function : {};
    return { name: fn.name, description: fn.description, parameters: fn.parameters };
  });
}

/** Anthropic tools as Gemini functionDeclarations, in order. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming `model` (the model
 * itself goes in the URL path, not the body): system messages become systemInstruction, user
 * parts become contents parts, tools become functionDeclarations, and max_tokens, stop sequences
 * and JSON mode become generationConfig. Tool messages are skipped: Gemini reads tool results as
 * functionResponse parts, and a tool message carries no function name to attach one to, while its
 * text must not reach the user echo.
 */
export function openAIToGemini(body: Json, model: string): { model: string; body: Json } {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
    } else if (Array.isArray(raw.content)) {
      const parts = raw.content.flatMap((b) => (isRecord(b) ? [openAIPartToGemini(b)].flatMap((x) => (x === null ? [] : [x])) : []));
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { model, body: out };
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks
 * become tool_calls, tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body naming `model`: the system
 * becomes systemInstruction, user blocks become contents parts, tool results become
 * functionResponse parts, tools become functionDeclarations, and max_tokens and stop sequences
 * become generationConfig. (Anthropic carries no JSON-mode flag, so none is set.)
 */
export function anthropicToGemini(body: Json, model: string): { model: string; body: Json } {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      const part = anthropicBlockToGemini(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = anthropicStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { model, body: out };
}

// ---------------------------------------------------------------------------
// Normalized answers: every provider answer, JSON or SSE, as one shape.
// ---------------------------------------------------------------------------

/** One provider answer: its text whole, its usage counts, and its stop token. */
export interface NormalizedAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

function openAIStopOf(finish: unknown): string {
  return typeof finish === 'string' && finish !== '' ? finish : 'stop';
}

/** A Chat Completions JSON answer as a normalized answer, or null when it is not one. */
export function parseOpenAIJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  const content = typeof message.content === 'string' ? message.content : textOf(message.content);
  return {
    text: content,
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    stop: openAIStopOf(choice.finish_reason),
  };
}

/** The SSE frames of a text body, split on blank lines, each with its event name and data lines. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A Chat Completions SSE stream as a normalized answer, or null when it carries no answer. */
export function parseOpenAISSE(text: string): NormalizedAnswer | null {
  let out = '';
  let usage = { input: 0, output: 0 };
  let stop = 'stop';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stop = choice.finish_reason;
  }
  return sawAny ? { text: out, usage, stop } : null;
}

/** An Anthropic Messages JSON answer as a normalized answer, or null when it is not one. */
export function parseAnthropicJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: body.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join(''),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    stop: typeof body.stop_reason === 'string' && body.stop_reason !== '' ? body.stop_reason : 'end_turn',
  };
}

/** An Anthropic Messages SSE stream as a normalized answer, or null when it carries no answer. */
export function parseAnthropicSSE(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: out, usage: { input, output }, stop } : null;
}

/** A Gemini generateContent JSON answer as a normalized answer, or null when it is not one. */
export function parseGeminiJson(body: unknown): NormalizedAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
    stop: typeof candidate.finishReason === 'string' && candidate.finishReason !== '' ? candidate.finishReason : 'STOP',
  };
}

/** A Gemini streamGenerateContent SSE stream as a normalized answer, or null when it carries no answer. */
export function parseGeminiSSE(text: string): NormalizedAnswer | null {
  let out = '';
  let usage = { input: 0, output: 0 };
  let stop = 'STOP';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (Array.isArray(data.candidates) && data.candidates.length > 0 && isRecord(data.candidates[0])) {
      const candidate = data.candidates[0];
      const content = isRecord(candidate.content) ? candidate.content : null;
      const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
      out += parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
      if (typeof candidate.finishReason === 'string' && candidate.finishReason !== '') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return sawAny ? { text: out, usage, stop } : null;
}

// ---------------------------------------------------------------------------
// Client answers: a normalized answer rebuilt in the wire the client used.
// ---------------------------------------------------------------------------

/** A provider stop token as an OpenAI finish_reason. */
export function stopToOpenAI(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider stop token as an Anthropic stop_reason. */
export function stopToAnthropic(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string, size: number): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces;
}

/** A normalized answer as an OpenAI Chat Completions JSON object, naming the serving model. */
export function openAIJsonAnswer(model: string, ans: NormalizedAnswer): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: ans.text }, finish_reason: stopToOpenAI(ans.stop), logprobs: null }],
    usage: { prompt_tokens: ans.usage.input, completion_tokens: ans.usage.output, total_tokens: ans.usage.input + ans.usage.output },
  };
}

/** A normalized answer as an OpenAI Chat Completions SSE stream, naming the serving model. */
export function openAIStreamAnswer(model: string, ans: NormalizedAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(ans.text, 64)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, stopToOpenAI(ans.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: ans.usage.input, completion_tokens: ans.usage.output, total_tokens: ans.usage.input + ans.usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages JSON object, naming the serving model. */
export function anthropicJsonAnswer(model: string, ans: NormalizedAnswer): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: stopToAnthropic(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.usage.input, output_tokens: ans.usage.output },
  };
}

/** A normalized answer as an Anthropic Messages SSE stream, naming the serving model. */
export function anthropicStreamAnswer(model: string, ans: NormalizedAnswer): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(ans.text, 13)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopToAnthropic(ans.stop), stop_sequence: null }, usage: { output_tokens: ans.usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
