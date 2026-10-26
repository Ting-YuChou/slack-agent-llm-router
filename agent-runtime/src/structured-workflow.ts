import { createHash, randomBytes, randomUUID } from 'node:crypto';
import path from 'node:path';
import type { Candidate } from './quality-gate.js';
import type { ControlStore } from './ci-control.js';
import { renderPrompt } from './prompts.js';
import { failureCause, HarnessError, type CauseCode } from './error-taxonomy.js';

export type WorkflowMode = 'single' | 'structured';
export type InvestigationRole = 'code_explorer' | 'ci_investigator' | 'test_analyst';
export interface InvestigationTask { id: string; role: InvestigationRole; purpose: string; paths: string[]; dependencies: string[]; expected_output: string }
export interface InvestigationResult { status: 'succeeded' | 'unknown'; findings: Array<{ path: string; line: number; reason: string }>; evidence: string[]; unresolved: string[]; suggested_tasks: string[] }
export interface InvestigationPlan { tasks: InvestigationTask[]; summary?: string }
export interface WorkflowNode extends Omit<InvestigationTask, 'role'> { role: InvestigationRole | 'planner'; workflow: string; wave: number; fixed_dependencies?: string[]; state: 'pending' | 'running' | 'succeeded' | 'unresolved' | 'unknown' | 'interrupted'; degraded_cause?: CauseCode; attempt: number; execution?: string; result?: InvestigationResult | InvestigationPlan; queued_at: number; started_at?: number; ended_at?: number; duration_ms?: number; queue_ms?: number; agent_work_ms?: number; attempt_queue_ms?: number }
export interface DiscoveryInput { run_id: string; owner: string; requirement: string; baseline: string; snapshot: Candidate; fingerprint: string; initial_active_ms?: number; guidance?: RepoGuidance }
/** A required investigation that degraded to an unresolved gap instead of failing the run. */
export interface InvestigationDegraded { type: 'investigation_degraded'; node_id: string; role: InvestigationRole; wave: number; attempt: number; cause: CauseCode }
export interface DiscoveryHooks { degraded?(event: InvestigationDegraded): void }
export interface InvestigationRequest { input: DiscoveryInput; node: WorkflowNode; results: WorkflowNode[]; signal: AbortSignal }
export function workflowMode(value: unknown): WorkflowMode {
  if (value === undefined || value === 'single') return 'single';
  if (value === 'structured') return value;
  throw new Error('Invalid workflow_mode');
}
const relative = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 512 && !/^[\\/]|^[a-z]:|[\x00-\x1f]/i.test(value) && !value.split(/[\\/]/).includes('..');
/** `./src/`, `src//a` and `src` name the same scope; compare them in one form. */
const normalize = (value: string) => path.posix.normalize(value.replace(/\\/g, '/')).replace(/\/+$/, '') || '.';
const text = (value: unknown, max = 4000): value is string => typeof value === 'string' && value.trim().length > 0 && value.length <= max;
export function validatePlan(raw: unknown, previous: InvestigationTask[], required = false): InvestigationPlan {
  const plan = raw as InvestigationPlan;
  if (!plan || !Array.isArray(plan.tasks) || previous.length + plan.tasks.length > 8 || (plan.summary !== undefined && !text(plan.summary, 8000))) throw new Error('Invalid investigation plan');
  if (required && !plan.tasks.length) throw new Error("First wave requires investigation tasks");
  const all = [...previous, ...plan.tasks]; const ids = new Set(previous.map(task => task.id));
  for (const task of plan.tasks) {
    if (!task || !/^[a-z][\w-]{0,63}$/.test(task.id) || task.id.startsWith('planner-') || ids.has(task.id) || !['code_explorer', 'ci_investigator', 'test_analyst'].includes(task.role) || !text(task.purpose) || !text(task.expected_output) || !Array.isArray(task.paths) || !task.paths.length || task.paths.length > 32 || !task.paths.every(relative) || !Array.isArray(task.dependencies) || task.dependencies.some(id => typeof id !== 'string')) throw new Error('Invalid investigation task');
    task.paths = task.paths.map(normalize);
    ids.add(task.id);
  }
  const visiting = new Set<string>(), visited = new Set<string>();
  function visit(id: string) {
    if (visiting.has(id) || !ids.has(id)) throw new Error('Invalid workflow dependency/cycle');
    if (visited.has(id)) return;
    visiting.add(id); const node = all.find(task => task.id === id)!;
    for (const dependency of node.dependencies) visit(dependency);
    visiting.delete(id); visited.add(id);
  }
  for (const task of all) visit(task.id);
  return plan;
}
export function validateInvestigation(raw: unknown, task: Pick<InvestigationTask, 'paths'>): InvestigationResult {
  const r = raw as InvestigationResult;
  if (!r || !['succeeded', 'unknown'].includes(r.status) || !Array.isArray(r.findings) || r.findings.length > 40 || ![r.evidence, r.unresolved, r.suggested_tasks].every(a => Array.isArray(a) && a.length <= 40 && a.every(v => text(v)))) throw new Error('Invalid investigation output');
  const scopes = task.paths.map(normalize);
  for (const finding of r.findings) {
    if (!finding || !relative(finding.path) || !Number.isInteger(finding.line) || finding.line < 1 || !text(finding.reason)) throw new Error('Invalid investigation evidence');
    finding.path = normalize(finding.path);
    if (!scopes.some(root => root === '.' || finding.path === root || finding.path.startsWith(root + '/'))) throw new Error('Invalid investigation evidence');
  }
  if (r.status === 'succeeded' && !r.evidence.length) throw new Error('Missing investigation evidence');
  if (JSON.stringify(r).length > 64_000) throw new Error('Investigation output too large');
  return r;
}

const settled = (node: WorkflowNode) => node.state === 'succeeded' || node.state === 'unresolved';

export const IMPLEMENTATION_BRIEF_MAX_BYTES = 8192;
const BRIEF_ITEM_MAX = 400;
/** What the writer gets from discovery: validated claims only, never the nodes' raw JSON. */
export interface ImplementationBrief { goal: string[]; findings: Array<{ path: string; line: number; claim: string }>; files: string[]; tests: string[]; open: string[] }
const TEST_PATH = /(^|\/)(tests?|__tests__|specs?)\/|\.(test|spec)\.[^/]+$|(^|\/)test_[^/]+$/;
/** One line of untrusted text: no newlines to forge a section with, and no longer than one item. */
const clip = (value: string) => { const line = value.replace(/\s+/g, ' ').trim(); return line.length > BRIEF_ITEM_MAX ? `${line.slice(0, BRIEF_ITEM_MAX - 1)}…` : line; };
const unique = (values: string[]) => [...new Set(values.map(clip).filter(Boolean))];

export function buildImplementationBrief(requirement: string, nodes: WorkflowNode[]): ImplementationBrief {
  const summary = nodes.filter(node => node.role === 'planner' && node.state === 'succeeded').sort((a, b) => b.wave - a.wave)
    .map(node => (node.result as InvestigationPlan | undefined)?.summary).find(Boolean);
  const findings: ImplementationBrief['findings'] = [], files: string[] = [], tests: string[] = [], testFiles: string[] = [], gaps: string[] = [], open: string[] = [], followUps: string[] = [];
  const seen = new Set<string>();
  for (const node of nodes) {
    if (node.role === 'planner') continue;
    if (node.state === 'unresolved') gaps.push(`${node.id} (${node.role}) did not complete (${node.degraded_cause ?? 'investigation_reported_unknown'}); its gap is yours to check: ${node.purpose}`);
    const result = node.result as InvestigationResult | undefined;
    if (!result) continue;
    for (const finding of result.findings) {
      const claim = clip(finding.reason), key = `${finding.path}:${finding.line} ${claim}`;
      if (seen.has(key)) continue;
      seen.add(key); findings.push({ path: finding.path, line: finding.line, claim });
      (TEST_PATH.test(finding.path) ? testFiles : files).push(finding.path);
    }
    open.push(...result.unresolved);
    (node.role === 'test_analyst' ? tests : followUps).push(...result.suggested_tasks);
  }
  return { goal: unique([requirement, ...(summary ? [`Investigation summary: ${summary}`] : [])]), findings,
    files: unique(files), tests: unique([...tests, ...testFiles.map(file => `Extend or add tests in ${file}`)]),
    open: unique([...gaps, ...open, ...followUps.map(task => `Suggested follow-up: ${task}`)]) };
}

/** Renders the brief, dropping the tail of the longest list until it fits IMPLEMENTATION_BRIEF_MAX_BYTES. */
export function renderImplementationBrief(brief: ImplementationBrief): string {
  const all = { findings: brief.findings.map(finding => `${finding.path}:${finding.line} ${finding.claim}`), files: brief.files, tests: brief.tests, open: brief.open };
  // No list can fill the brief with more than this many items, so the trimming below stays cheap.
  const lists = { findings: all.findings.slice(0, 40), files: all.files.slice(0, 20), tests: all.tests.slice(0, 20), open: all.open.slice(0, 20) };
  const omitted = { findings: all.findings.length - lists.findings.length, files: all.files.length - lists.files.length, tests: all.tests.length - lists.tests.length, open: all.open.length - lists.open.length };
  const bullets = (items: string[], dropped = 0) => [...(items.length ? items : dropped ? [] : ['none reported']), ...(dropped ? [`… ${dropped} more omitted`] : [])].map(item => `- ${item}`).join('\n');
  const render = () => renderPrompt('investigation-results', { goal: bullets(brief.goal), findings: bullets(lists.findings, omitted.findings), files: bullets(lists.files, omitted.files), tests: bullets(lists.tests, omitted.tests), open: bullets(lists.open, omitted.open) });
  for (let text = render(); ; text = render()) {
    if (Buffer.byteLength(text) <= IMPLEMENTATION_BRIEF_MAX_BYTES) return text;
    const longest = (Object.keys(lists) as Array<keyof typeof lists>).filter(key => lists[key].length)
      .sort((a, b) => Buffer.byteLength(lists[b].join('\n')) - Buffer.byteLength(lists[a].join('\n')))[0];
    if (!longest) return text;
    lists[longest].pop(); omitted[longest]++;
  }
}
export const REPO_GUIDANCE_MAX_BYTES = 8192;
/** The repository's AGENTS.md as the baseline commit recorded it: at most REPO_GUIDANCE_MAX_BYTES of it, and its full size. */
export interface RepoGuidance { text: string; bytes: number; truncated: boolean }

/**
 * Keeps the whole characters of `head` that fit the cap. Read REPO_GUIDANCE_MAX_BYTES + 4 bytes of the
 * file: a character cut at the end of the read then starts past the cap and is never kept half-decoded.
 */
export function repoGuidance(head: Buffer, bytes: number): RepoGuidance {
  let text = '', size = 0, cut = head.length < bytes;
  for (const char of head.toString('utf8')) {
    // Invalid bytes decode to the three-byte U+FFFD, so measure what is kept, not what was read.
    const width = Buffer.byteLength(char);
    if (size + width > REPO_GUIDANCE_MAX_BYTES) { cut = true; break; }
    text += char; size += width;
  }
  return { text, bytes, truncated: cut };
}

/** The planner and investigator section for AGENTS.md, fenced like other untrusted text; empty when there is none. */
export function renderRepoGuidance(guidance: RepoGuidance | undefined): string {
  if (!guidance?.text.trim()) return '';
  let boundary: string;
  do boundary = `UNTRUSTED-${randomBytes(12).toString('hex')}`; while (guidance.text.includes(boundary));
  const truncated = guidance.truncated ? ` It is truncated to fit ${REPO_GUIDANCE_MAX_BYTES} bytes; the whole file is ${guidance.bytes} bytes.` : '';
  return renderPrompt('repo-guidance', { boundary, body: guidance.text, truncated });
}
const degradeCause = (error: unknown): CauseCode => failureCause(error) ?? 'investigation_execution_failed';

/** Host-owned two-wave graph. Only execution results are reusable, never permissions. */
export class WorkflowCoordinator {
  private readonly running = new Set<string>();
  constructor(private readonly options: { store: ControlStore; concurrency: number; degradeUnknown?: boolean; execute(request: InvestigationRequest): Promise<unknown>; record?(node: WorkflowNode): Promise<void> }) {}
  tasks(id: string): WorkflowNode[] { return this.options.store.workflowNodes<WorkflowNode>(id); }
  canResume(id: string) { const state = this.options.store.get<{ writer_started: boolean; expired?: boolean }>('structured',id); return Boolean(state && !state.writer_started && !state.expired); }
  elapsed(id: string) { return this.options.store.get<{ elapsed_ms: number }>('structured',id)?.elapsed_ms ?? 0; }
  async discover(input: DiscoveryInput, signal: AbortSignal, hooks: DiscoveryHooks = {}): Promise<string> {
    if (this.running.has(input.run_id)) throw new Error('workflow_busy');
    const identity = createHash('sha256').update(JSON.stringify([input.requirement, input.baseline, input.snapshot.hash, input.fingerprint])).digest('hex');
    const prior = this.options.store.get<{ identity: string; writer_started: boolean; elapsed_ms?: number }>('structured', input.run_id);
    if (prior && (prior.identity !== identity || prior.writer_started)) throw new Error('workflow_identity_changed_or_writer_started');
    const clockStart = Date.now(), consumed = prior?.elapsed_ms ?? input.initial_active_ms ?? 0;
    const checkpoint = () => this.options.store.put('structured', input.run_id, { identity, writer_started: false, input, updated_at: Date.now(), elapsed_ms: consumed + Date.now() - clockStart });
    checkpoint();
    const heartbeat = setInterval(() => { try { checkpoint(); } catch (error) { controller.abort(error); } }, 1000); heartbeat.unref();
    this.running.add(input.run_id);
    const controller = new AbortController(); const bounded = AbortSignal.any([signal, controller.signal]);
    try {
      for (let wave = 1; wave <= 2; wave++) {
        bounded.throwIfAborted();
        let planner = this.tasks(input.run_id).find(node => node.id === `planner-${wave}`);
        if (!planner) { planner = { id: `planner-${wave}`, role: 'planner', purpose: wave === 1 ? 'Plan focused read-only investigation' : 'Synthesize evidence; add missing investigations only if needed', expected_output: 'Validated investigation graph', paths: ['.'], dependencies: [], fixed_dependencies: wave === 1 ? [] : this.tasks(input.run_id).filter(node => node.wave === 1 && node.role !== 'planner').map(node => node.id), workflow: input.run_id, wave, state: 'pending', attempt: 0, queued_at: Date.now() }; this.options.store.saveWorkflowNode(input.run_id, planner.id, planner); this.options.store.saveDependencies(input.run_id, planner.id, planner.fixed_dependencies ?? []); }
        await this.execute(input, planner, bounded, hooks);
        const plan = planner.result as InvestigationPlan;
        for (const task of plan.tasks) if (!this.tasks(input.run_id).some(node => node.id === task.id)) {
          const node: WorkflowNode = { ...task, fixed_dependencies: [planner.id], workflow: input.run_id, wave, state: 'pending', attempt: 0, queued_at: Date.now() };
          this.options.store.saveWorkflowNode(input.run_id, task.id, node);
          this.options.store.saveDependencies(input.run_id, task.id, [...task.dependencies, ...node.fixed_dependencies!]);
          await this.options.record?.(node);
        }
        while (true) {
          const nodes = this.tasks(input.run_id).filter(node => node.role !== 'planner' && node.wave === wave && !settled(node));
          if (!nodes.length) break;
          const completed = new Set(this.tasks(input.run_id).filter(settled).map(node => node.id));
          const ready = nodes.filter(node => [...node.dependencies, ...(node.fixed_dependencies ?? [])].every(id => completed.has(id))).slice(0, this.options.concurrency);
          if (!ready.length) throw new Error('workflow_dependencies_unavailable');
          const results = await Promise.allSettled(ready.map(node => this.execute(input, node, bounded, hooks).catch(error => { controller.abort(error); throw error; })));
          const failure = results.find(result => result.status === 'rejected'); if (failure?.status === 'rejected') throw failure.reason;
        }
      }
      signal.throwIfAborted();
      const nodes = this.tasks(input.run_id), investigations = nodes.filter(node => node.role !== 'planner');
      // With nothing resolved the writer would have nothing to go on.
      const gap = investigations.find(node => node.state === 'unresolved');
      if (gap && investigations.every(node => node.state === 'unresolved')) throw new HarnessError(gap.degraded_cause ?? 'investigation_reported_unknown', 'investigation_all_unresolved');
      return renderImplementationBrief(buildImplementationBrief(input.requirement, nodes));
    } finally { clearInterval(heartbeat); try { checkpoint(); } finally { this.running.delete(input.run_id); } }
  }
  private async execute(input: DiscoveryInput, node: WorkflowNode, signal: AbortSignal, hooks: DiscoveryHooks) {
    if (settled(node)) return;
    while (node.attempt < 2) {
      signal.throwIfAborted(); const execution = randomUUID();
      node = Object.assign(node, { state: 'running', execution, attempt: node.attempt + 1, started_at: Date.now() });
      delete node.result; delete node.degraded_cause;
      node.queue_ms ??= node.started_at! - node.queued_at; node.attempt_queue_ms = 0;
      this.options.store.claimWorkflowNode(input.run_id, node.id, execution, node);
      try {
        await this.options.record?.(node);
        const raw = await this.options.execute({ input, node, results: this.tasks(input.run_id).filter(settled), signal });
        signal.throwIfAborted();
        node.result = node.role === 'planner' ? validatePlan(raw, this.tasks(input.run_id).filter(item => item.role !== 'planner' && item.wave < node.wave) as InvestigationTask[],node.wave === 1) : validateInvestigation(raw, node);
        if ('status' in node.result && node.result.status !== 'succeeded') throw new Error('investigation_unknown');
        node.state = 'succeeded';
        node.ended_at = Date.now(); node.duration_ms = node.ended_at - node.started_at!;
        node.agent_work_ms = (node.agent_work_ms ?? 0) + Math.max(0,node.duration_ms - node.attempt_queue_ms!);
        this.options.store.finishWorkflowNode(input.run_id, node.id, execution, node);
        await this.options.record?.(node); return;
      } catch (error) {
        const fatal = signal.aborted || (error instanceof Error && /budget|stop_unconfirmed|control_store/.test(error.message));
        // An investigator that still fails after its retry may continue as an unresolved gap the writer sees; a planner never does.
        const degrade = !fatal && node.attempt >= 2 && this.options.degradeUnknown === true && node.role !== 'planner';
        node.state = degrade ? 'unresolved' : signal.aborted ? 'interrupted' : 'unknown'; node.ended_at = Date.now(); node.duration_ms = node.ended_at - node.started_at!;
        if (degrade) node.degraded_cause = degradeCause(error);
        node.agent_work_ms = (node.agent_work_ms ?? 0) + Math.max(0,node.duration_ms - node.attempt_queue_ms!);
        this.options.store.finishWorkflowNode(input.run_id, node.id, execution, node); await this.options.record?.(node);
        if (degrade) { hooks.degraded?.({ type: 'investigation_degraded', node_id: node.id, role: node.role as InvestigationRole, wave: node.wave, attempt: node.attempt, cause: node.degraded_cause! }); return; }
        if (fatal || node.attempt >= 2) throw error;
      }
    }
    throw new Error('investigation_attempts_exhausted');
  }
  writerStarted(id: string) { const state = this.options.store.get<object>('structured', id); if (!state) throw new Error('workflow_missing'); this.options.store.put('structured', id, { ...state, writer_started: true }); }
}
