import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { DockerPiProcess, type AgentContainerOptions } from './container-runtime.js';
import { HarnessError } from './error-taxonomy.js';
import type { PublicRunEvent } from './pi-rpc.js';
import { collectSubmission } from './submission-collector.js';
import { renderRepoGuidance, validateInvestigation, validatePlan, type InvestigationRequest, type InvestigationTask } from './structured-workflow.js';
import type { AgentTelemetry } from './telemetry.js';
import type { AgentTracing } from './tracing.js';
import type { RouteDecision } from './jev-router.js';
import { MCP_SERVERS, namespacedMcpToolNames } from './mcp-config.js';
import { renderPrompt } from './prompts.js';

export function investigationDockerOptions(request: InvestigationRequest, options: { image: string; network: string; gatewayUrl: string; statePath: string }): AgentContainerOptions {
  const planner = request.node.role === 'planner';
  return { ...options, name: `pi-investigation-${request.node.execution!.slice(0, 8)}`, worktreePath: request.input.snapshot.path, gitMetadataPath: '', sessionStatePath: options.statePath,
    workflowKey: request.input.run_id, modelRef: planner ? 'openai/gpt-5.6-sol' : 'openai/gpt-5.6-luna', reasoningEffort: planner ? 'high' : 'max',
    extensionPaths: ['/opt/pi/extensions/model-gateway.ts', '/opt/pi/extensions/workflow-agent.ts'], pluginPaths: [], skillPaths: [],
    toolNames: ['read', 'grep', 'find', 'ls', planner ? 'submit_plan' : 'submit_investigation'], readOnlyWorkspace: true, continueSession: false,
    user: `${process.getuid?.()}:${process.getgid?.()}`, workflowContext: { role: request.node.role, wave: request.node.wave, paths: request.node.paths, previousTasks: request.results.filter(node => node.role !== 'planner').map(({ id, role, purpose, paths, dependencies, expected_output }) => ({ id, role, purpose, paths, dependencies, expected_output })) },
  };
}

/** The slice of a Pi process an investigation drives; tests inject a scripted one. */
export interface InvestigationProcess { prompt: DockerPiProcess['prompt']; close(): Promise<void>; queueMs?: number }
export type InvestigationProcessFactory = (
  options: ReturnType<typeof investigationDockerOptions>,
  onEvent: (event: PublicRunEvent) => void,
  onRecord: (record: Record<string, unknown>) => void,
) => InvestigationProcess;

/** Runs one isolated execution and confirms shutdown. It has no delivery authority. */
export class AgentExecutionRunner {
  private readonly active = new Map<string, InvestigationProcess>();
  constructor(private readonly options: { image: string; network: string; gatewayUrl: string; stateRoot: string; telemetry?: AgentTelemetry; tracing?: AgentTracing;
    token(id: string, route: RouteDecision): string; prompt?(id: string): void;
    execution(id: string, parent: string, role: string, attempt: number, phase: 'start' | 'end', cost?: number, queueMs?: number): Promise<void>;
    /** `role` is the node's (planner, code_explorer, ...), so the run can attribute the spend. */
    cost(parent: string, cost: number, role: string): Promise<void>;
    githubMcp?: { gatewayUrl: string; token(id: string, owner: string): string };
    createProcess?: InvestigationProcessFactory;
  }) {}
  async stop(parent: string) {
    for (const [key, process] of this.active) if (key.startsWith(parent + ':')) { await process.close(); this.active.delete(key); }
  }
  async run(request: InvestigationRequest): Promise<unknown> {
    const { node, input, signal } = request; signal.throwIfAborted();
    const id = node.execution!, statePath = path.join(this.options.stateRoot, id), key = `${input.run_id}:${id}`;
    await mkdir(statePath, { recursive: true, mode: 0o700 });
    await this.options.execution(id, input.run_id, node.role, node.attempt, 'start');
    let cost = 0, process: InvestigationProcess | undefined;
    const config = investigationDockerOptions(request, { ...this.options, statePath });
    const github = node.role === 'ci_investigator' ? this.options.githubMcp : undefined;
    if (github) {
      config.directMcpOnly = true;
      config.mcp = { mode: 'github_read_only', servers: [{ server: 'github', gatewayUrl: github.gatewayUrl, token: '' }] };
      config.workflowContext!.mcpTools = namespacedMcpToolNames('github',MCP_SERVERS.github.directTools);
    }
    const route: RouteDecision = { modelRef: config.modelRef, effort: config.reasoningEffort!, source: node.role };
    const tool = node.role === 'planner' ? 'submit_plan' : 'submit_investigation';
    const guidance = renderRepoGuidance(input.guidance);
    const validate = (args: unknown) => node.role === 'planner'
      ? validatePlan(args, request.results.filter(item => item.role !== 'planner') as InvestigationTask[], node.wave === 1)
      : validateInvestigation(args, node);
    try {
      const { value } = await collectSubmission({
        process: ({ onEvent, onRecord }) => {
          process = this.options.createProcess?.(config, onEvent, onRecord)
            ?? new DockerPiProcess(config, (execution, selected) => this.options.token(execution, selected), onEvent,
              github ? execution => ({ github: github.token(execution, input.owner) }) : undefined, undefined, this.options.telemetry, id, this.options.tracing, onRecord, this.options.prompt);
          this.active.set(key, process);
          process.prompt(id, renderPrompt(node.role === 'planner' ? 'planner' : 'investigator', {
            ...(node.role === 'planner' ? {} : { role: node.role }),
            context: JSON.stringify({ requirement: input.requirement, baseline: input.baseline, snapshot_root: input.snapshot.path, task: node, prior_results: request.results.map(({ id, role, result }) => ({ id, role, result })) }),
            guidance: guidance && `\n${guidance}`,
          }), route);
        },
        toolName: tool, validate, timeoutMs: node.role === 'planner' ? 120_000 : 180_000, signal, usage: spent => { cost += spent; },
        errors: {
          invalid: reason => new HarnessError('investigation_invalid_submission', `investigation_invalid_submission: ${reason}`),
          missing: () => new HarnessError('investigation_missing_structured_result', 'investigation_missing_structured_result'),
          timeout: () => new HarnessError('investigation_timeout', 'investigation_timeout'),
          failed: () => new HarnessError('investigation_execution_failed', 'investigation_execution_failed'),
          aborted: () => new Error('investigation_aborted'),
        },
      });
      return value;
    } finally {
      try { await process?.close(); } catch { throw new Error('investigation_stop_unconfirmed'); }
      const queueMs = process?.queueMs ?? 0;
      node.attempt_queue_ms = queueMs; node.queue_ms = (node.queue_ms ?? 0) + queueMs;
      this.active.delete(key);
      await this.options.execution(id, input.run_id, node.role, node.attempt, 'end', cost, queueMs);
      await this.options.cost(input.run_id, cost, node.role);
    }
  }
}
