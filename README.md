# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Ruiii-newbie/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Ruiii-newbie/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One OpenAI- and Anthropic-compatible server that routes each request to the best free model across ten providers, failing over when a provider's free tier refuses, and reporting the catalog it found.

## Run it

```sh
node main.ts --list
```

With a trial's env, `node main.ts` listens on a port it picks, registers with `$MAKEFUN_ENV_URL`, serves `POST /v1/chat/completions` and `POST /v1/messages` (JSON and streaming) plus `GET /v1/models`, and exits 0 when the workload is done.

## How it routes

- Every free model that meets a request's requirements is ranked by the brief's quality table (`rating_milli`); a paid model is never a pick, and a JSON-mode OpenAI request never goes to an Anthropic-wire provider, which cannot carry `response_format`.
- The twenty requests are planned as one allocation: each takes the best candidate whose provider still has planned room (smallest published budget minus one call held back for the failure schedule), so the best provider's budget is not spent on the first requests that want it.
- On a 402/429 the request fails over to the next provider, one attempt per provider; a provider that has refused three times is treated as spent.
- Every translation preserves the fields the provider echoes (system prompt, user text, images, tool names, stop sequences, `max_tokens`, JSON mode), and every answer is rebuilt in the client's wire with the provider's usage counts and stop reason. Streams are buffered and re-emitted, so a streamed answer always carries its usage.

---

<sub>Entered on [make.fun](https://make.fun/?ref=575ehchc&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
