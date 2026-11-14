// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial's brief (the provider directory, the quality table and the workload
// manifest), probes each provider's catalog route, registers its port and catalog with the env,
// and answers the client's twenty requests, one at a time, each with the best free model able to
// serve it. When a provider refuses, it fails over to the next; when the env says the workload is
// done, it exits 0.
//
// HOW IT WINS, per scoring column
//   - quality (60%): every request's candidates are ranked by the quality table's rating_milli,
//     and the twenty requests are planned as one allocation (lib/plan.ts), so the best provider's
//     budget is not spent on the first requests that want it.
//   - served (15%): the failover chain tries every provider that can serve a request, one attempt
//     per provider, so a refusal never ends a request while another provider stands.
//   - wire (10%): every translation preserves the fields the provider echoes (system prompt, user
//     text, images, tool names, stop sequences, max_tokens, JSON mode), and every answer is
//     rebuilt in the client's wire with the provider's usage counts and stop reason, streamed or
//     not as asked. Streams are buffered and re-emitted, so a streamed answer always carries its
//     usage (the client never sets stream_options, so a piped-through stream would not).
//   - failover (10%): a first-call refusal is counted and the request is still served by the next
//     provider in the chain.
//   - catalog (5%): the catalog is probed from the providers' own routes and registered byte for
//     byte as --list prints it; GET /v1/models lists exactly the free model ids registered.
//
// The trial's rules are in the kit's RULES.txt. The server is deterministic: every choice is a
// function of the brief, the requests and the providers' answers. No wall clock, no unseeded
// randomness. Node 22's standard library alone (nothing is installed; a relative import needs its
// .ts extension).
//
// FILES
//   main.ts            this server: the brief, the catalog, the plan, the routes, the register.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client wires and the three upstream wires, in both directions.
//   lib/plan.ts        the rating-ordered candidates, the greedy plan, the failover chain.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToAnthropic,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
  openAIToAnthropic,
  openAIToGemini,
  openAIToOpenAI,
  parseAnthropicJson,
  parseAnthropicSSE,
  parseGeminiJson,
  parseGeminiSSE,
  parseOpenAIJson,
  parseOpenAISSE,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { buildCandidates, buildPlan, chainFor, type Candidate } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds, before it counts as a refusal. */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan = new Map<string, Candidate>();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Budget-type refusals (402/429) seen per provider; three means its budget or outage. */
const refusals = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog and the greedy plan for the manifest. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = buildPlan(probes, brief.ratings, brief.workload?.manifest ?? []);
}

/** The free models across the providers that answered, each id once, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of one request: the manifest's row by arrival order, else the body's own. */
function requirementsFor(index: number, body: Json): { id: string | null; requires: Requirement } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return { id: entry.id, requires: entry.requires };
  return { id: null, requires: requirementsFromBody(body) };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The error code of a provider's error body, in whichever wire's spelling, or '' when unreadable. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return '';
    if (isRecord(value.error)) {
      if (typeof value.error.code === 'string') return value.error.code;
      if (typeof value.error.type === 'string') return value.error.type;
      if (typeof value.error.status === 'string') return value.error.status;
    }
    return '';
  } catch {
    return '';
  }
}

/** One provider call: the request in the provider's wire, and its parsed answer, or null on refusal. */
async function callProvider(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<NormalizedAnswer | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, candidate.modelId) : anthropicToOpenAI(body, candidate.modelId);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientApi === 'openai' ? openAIToAnthropic(body, candidate.modelId) : anthropicToAnthropic(body, candidate.modelId);
  } else {
    const built = clientApi === 'openai' ? openAIToGemini(body, candidate.modelId) : anthropicToGemini(body, candidate.modelId);
    url = `${base}/v1beta/models/${encodeURIComponent(built.model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    upstreamBody = built.body;
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch {
    return null;
  }
  const text = await res.text().catch(() => '');
  if (!res.ok) {
    // A budget-type refusal (402, or 429 other than the entry's own concurrent call) counts
    // against the provider; a wrong key, an unknown model or a 5xx does not spend its budget.
    if ((res.status === 402 || res.status === 429) && errorCodeOf(text) !== 'concurrent_requests') {
      refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    }
    return null;
  }
  if (provider.wire === 'openai') {
    if (stream) return parseOpenAISSE(text);
    try {
      return parseOpenAIJson(JSON.parse(text));
    } catch {
      return null;
    }
  }
  if (provider.wire === 'anthropic') {
    if (stream) return parseAnthropicSSE(text);
    try {
      return parseAnthropicJson(JSON.parse(text));
    } catch {
      return null;
    }
  }
  if (stream) return parseGeminiSSE(text);
  try {
    return parseGeminiJson(JSON.parse(text));
  } catch {
    return null;
  }
}

/** Answer one client request: the planned provider first, then the chain, one attempt per provider. */
async function serve(
  index: number,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const { id, requires } = requirementsFor(index, body);
  const stream = body.stream === true;
  const candidates = buildCandidates(probes, brief.ratings, requires, clientApi);
  const primary = (id !== null ? plan.get(id) : undefined) ?? null;
  const order = chainFor(primary, candidates, refusals);
  for (const candidate of order) {
    const answer = await callProvider(candidate, clientApi, body, stream);
    if (answer === null) continue;
    if (clientApi === 'openai') {
      if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', openAIStreamAnswer(candidate.modelId, answer));
      else sendJson(res, 200, openAIJsonAnswer(candidate.modelId, answer));
    } else {
      if (stream) sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicStreamAnswer(candidate.modelId, answer));
      else sendJson(res, 200, anthropicJsonAnswer(candidate.modelId, answer));
    }
    return;
  }
  // Every provider that could serve it refused: the request cannot be answered this trial.
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/chat/completions or POST /v1/messages: read the body, then serve it. */
async function handleRequest(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  // The client sends one request at a time, in manifest order: count this arrival before reading
  // the body, so the manifest row stays aligned even when a body does not parse.
  const index = manifestIndex;
  manifestIndex += 1;
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  await serve(index, clientApi, body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
