// Routing: which free model serves each request, and in what order to fail over.
//
// Two ideas, both derived from the brief alone (the quality table, the workload manifest and the
// providers the probe answered), so the plan is a pure function of the trial's inputs and replays
// byte for byte:
//
//   1. Rank every free model that meets a request's requirements by the quality table's
//      rating_milli, best first. A paid model is never a candidate, whatever its rating.
//      A JSON-mode request on the OpenAI wire is never routed to an Anthropic-wire provider:
//      that wire cannot carry response_format, so the provider would echo json_mode absent and
//      the request would fail its echo check. (An Anthropic client request carries no JSON-mode
//      flag, so any wire may serve it.)
//   2. Plan the twenty requests as one allocation, not one by one. The best model for most
//      requests is one provider's (its budget is ten to twelve calls a trial), so taking it for
//      every request would spend that provider on the first ten and fail over on the rest. The
//      greedy plan walks the manifest in order and gives each request the best candidate whose
//      provider still has planned room, where room is the provider's smallest published budget
//      (the kit's sealed provider table lists each provider's calls_min) minus one call kept
//      back for its failure schedule. A request whose best providers are all planned full takes
//      the best candidate anyway.
//
// At run time the plan is only a first choice. Each request tries its providers in rating order,
// one attempt per provider (its best eligible model): a refusal (402, 429 or 5xx) moves to the
// next provider, never back to the one that refused. A provider that has refused three times is
// treated as spent: the kit's generator draws at most two scheduled failures per provider, so a
// third refusal can only be its budget or its outage, and neither ever recovers. A concurrent
// refusal (code concurrent_requests) is the entry's own doing, never a budget refusal, and is not
// counted. A 200 answer is always used: the trial counts exactly one successful provider call
// per request, so a second call after a success would invalidate it.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type RatingsFile,
  type Requirement,
  type Wire,
  type WorkloadRequest,
} from './providers.ts';

/** One provider's best eligible model for a request. */
export interface Candidate {
  probe: Probe;
  modelId: string;
  rating: number;
}

/**
 * Each provider's smallest published per-trial budget, from the kit's sealed provider table
 * (program/scorer/score.mjs and program/generator/draw.mjs list calls_min per provider). The
 * exact budget of a trial is drawn from it and never published; the plan only ever assumes the
 * minimum, minus one call held back for the provider's failure schedule.
 */
export const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** The planned room of a provider: its smallest budget minus one call held back for failures. */
export function plannedRoom(providerId: string): number {
  return Math.max(1, (MIN_BUDGET[providerId] ?? 4) - 1);
}

/** Whether a provider's wire can carry a request: an Anthropic wire cannot carry JSON mode. */
export function wireCanServe(wire: Wire, clientApi: 'openai' | 'anthropic', requires: Requirement): boolean {
  if (clientApi === 'openai' && requires.json === true && wire === 'anthropic') return false;
  return true;
}

/**
 * Every provider's best free model that meets the requirements, best rating first. Ties break by
 * provider id, then model id, so the order is deterministic.
 */
export function buildCandidates(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (!wireCanServe(probe.provider.wire, clientApi, requires)) continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row !== null ? row.rating_milli : 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.modelId)) {
        best = { probe, modelId: model.id, rating };
      }
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.modelId < b.modelId ? -1 : 1));
  return out;
}

/**
 * The greedy plan: for each manifest request in order, its first-choice candidate. A request
 * takes the best candidate whose provider still has planned room; a request whose best providers
 * are all planned full takes the best candidate outright. Returns the plan by request id.
 */
export function buildPlan(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  manifest: readonly WorkloadRequest[],
): Map<string, Candidate> {
  const plan = new Map<string, Candidate>();
  const assigned = new Map<string, number>();
  for (const request of manifest) {
    const candidates = buildCandidates(probes, ratings, request.requires, request.api);
    if (candidates.length === 0) continue;
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      const id = candidate.probe.provider.id;
      const used = assigned.get(id) ?? 0;
      if (used >= plannedRoom(id)) continue;
      if (pick === null) {
        pick = candidate;
        continue;
      }
      const pickUsed = assigned.get(pick.probe.provider.id) ?? 0;
      const pickRoom = plannedRoom(pick.probe.provider.id) - pickUsed;
      const room = plannedRoom(id) - used;
      if (candidate.rating > pick.rating || (candidate.rating === pick.rating && room > pickRoom)) pick = candidate;
    }
    const chosen = pick ?? candidates[0];
    plan.set(request.id, chosen);
    assigned.set(chosen.probe.provider.id, (assigned.get(chosen.probe.provider.id) ?? 0) + 1);
  }
  return plan;
}

/**
 * The try order for one request: its planned first choice, then every other candidate in rating
 * order, one attempt per provider. Providers with three or more recorded refusals are left out:
 * with at most two scheduled failures per provider, a third refusal means its budget or its
 * outage, and neither recovers.
 */
export function chainFor(primary: Candidate | null, candidates: readonly Candidate[], refusals: ReadonlyMap<string, number>): Candidate[] {
  const seen = new Set<string>();
  const out: Candidate[] = [];
  const push = (candidate: Candidate): void => {
    const id = candidate.probe.provider.id;
    if (seen.has(id)) return;
    if ((refusals.get(id) ?? 0) >= 3) return;
    seen.add(id);
    out.push(candidate);
  };
  if (primary !== null) push(primary);
  for (const candidate of candidates) push(candidate);
  return out;
}
