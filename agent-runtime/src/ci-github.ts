import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { redact } from "./telemetry.js";
import { ControlStore } from "./ci-control.js";
import type { RepairTarget } from "./ci-repair.js";
const exec = promisify(execFile);
/** Only the host calls this client; credentials never enter prompts or containers. */
export class CiGitHub {
  constructor(private readonly options: { repository: string; workflows: string[]; checkout?: "head" | "merge"; repoPath: string; token: () => Promise<string>; publisherToken?: () => Promise<string>; store: ControlStore; fetch?: typeof fetch }) {}
  private async api(endpoint: string, init?: RequestInit, publish = false): Promise<any> {
    const token = publish ? await this.options.publisherToken?.() : await this.options.token();
    if (!token) throw new Error("publisher_credentials_missing");
    const response = await (this.options.fetch ?? fetch)(`https://api.github.com/repos/${this.options.repository}/${endpoint}`, { ...init, redirect: "error", signal: AbortSignal.timeout(15_000), headers: { accept: "application/vnd.github+json", authorization: `Bearer ${token}`, "x-github-api-version": "2022-11-28", "content-type": "application/json" } });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`github_request_failed_${response.status}`);
    return response.json();
  }
  async investigate(input: string): Promise<RepairTarget> {
    const prUrl = /^https:\/\/github\.com\/([^/]+\/[^/]+)\/pull\/(\d+)\/?$/.exec(input);
    let run: any, pr: any;
    if (prUrl) {
      if (prUrl[1] !== this.options.repository) throw new Error("repository_not_allowlisted");
      pr = await this.api(`pulls/${prUrl[2]}`);
      if (!pr) throw new Error("pr_not_found");
      const runs = await this.api(`actions/runs?head_sha=${encodeURIComponent(pr.head.sha)}&status=failure&per_page=100`);
      run = runs?.workflow_runs?.find((candidate: any) => this.options.workflows.includes(candidate.name) && candidate.conclusion === "failure");
      if (!run) throw new Error("no_allowlisted_failed_workflow_for_head");
    } else {
      if (!/^\d+$/.test(input)) throw new Error("Expected same-repository PR URL or workflow run ID");
      run = await this.api(`actions/runs/${input}`);
      const prs = run?.pull_requests;
      if (!Array.isArray(prs) || prs.length !== 1) throw new Error("workflow_must_identify_one_pr");
      pr = await this.api(`pulls/${prs[0].number}`);
    }
    if (!pr || pr.state !== "open" || pr.merged || pr.head?.repo?.full_name !== this.options.repository || pr.base?.repo?.full_name !== this.options.repository || run.status !== "completed" || run.conclusion !== "failure" || !this.options.workflows.includes(run.name) || run.head_branch !== pr.head.ref || run.head_sha !== pr.head.sha || !/^[a-f0-9]{40}$/.test(run.head_sha)) throw new Error("stale_fork_or_unsupported_ci_identity");
    let tested = run.head_sha;
    if (run.event === "pull_request" && this.options.checkout !== "head") {
      tested = pr.merge_commit_sha;
      if (!/^[a-f0-9]{40}$/.test(tested)) throw new Error("merge_snapshot_unavailable");
      const merge = await this.api(`git/commits/${tested}`);
      if (!merge?.parents?.some((parent: any) => parent.sha === pr.head.sha) || !merge.parents.some((parent: any) => parent.sha === pr.base.sha)) throw new Error("merge_snapshot_identity_mismatch");
    }
    const jobs = await this.api(`actions/runs/${run.id}/attempts/${run.run_attempt}/jobs?per_page=100`);
    const failed = jobs.jobs.filter((job: any) => ["failure", "timed_out"].includes(job.conclusion));
    const logs = await Promise.all(failed.slice(0, 5).map(async (job: any) => ({ job: job.id, text: await this.jobLog(job.id) })));
    const diagnostic = JSON.stringify(redact({ logs, jobs: failed.map((job: any) => ({ id: job.id, name: job.name, conclusion: job.conclusion, steps: job.steps?.filter((step: any) => step.conclusion === "failure").map((step: any) => ({ name: step.name, number: step.number })) })) }));
    return { repository: this.options.repository, pr: pr.number, head: pr.head.sha, tested, branch: pr.head.ref, failed_jobs: failed.map((job: any) => job.name), workflow: run.name, run: run.id, attempt: run.run_attempt, diagnostic };
  }
  private async jobLog(id: number): Promise<string> {
    const fetcher = this.options.fetch ?? fetch;
    const response = await fetcher(`https://api.github.com/repos/${this.options.repository}/actions/jobs/${id}/logs`, { redirect: "manual", signal: AbortSignal.timeout(15_000), headers: { authorization: `Bearer ${await this.options.token()}`, accept: "application/vnd.github+json" } });
    let log = response;
    if (response.status === 302) {
      const location = new URL(response.headers.get("location") ?? "");
      if (location.protocol !== "https:") throw new Error("Invalid log redirect");
      log = await fetcher(location, { redirect: "error", signal: AbortSignal.timeout(15_000) });
    }
    if (!log.ok || !log.body) return "Logs unavailable";
    const reader = log.body.getReader(); const chunks: Buffer[] = []; let bytes = 0;
    try { while (bytes < 128_000) { const { done, value } = await reader.read(); if (done) break; const chunk = Buffer.from(value).subarray(0, 128_000 - bytes); chunks.push(chunk); bytes += chunk.length; } }
    finally { await reader.cancel(); }
    return Buffer.concat(chunks).toString("utf8");
  }
  async fetchCommit(sha: string) {
    if (!/^[a-f0-9]{40}$/.test(sha)) throw new Error("Invalid commit");
    await this.git(["fetch", "--no-tags", `https://github.com/${this.options.repository}.git`, sha], await this.options.token());
  }
  async publish(target: RepairTarget, commit: string, _localBranch: string, id: string): Promise<string> {
    if (!/^[a-f0-9]{40}$/.test(commit) || !/^[\w-]+$/.test(id)) throw new Error("Invalid delivery identity");
    const branch = `pi-agent/repair-${id}`;
    const saved = this.options.store.get<{ commit: string; branch: string; url?: string }>("delivery", id);
    if (saved && (saved.commit !== commit || saved.branch !== branch)) throw new Error("delivery_identity_conflict");
    const record = saved ?? { commit, branch }; this.options.store.put("delivery", id, record);
    const current = await this.api(`pulls/${target.pr}`);
    if (!current || current.state !== "open" || current.merged || current.head.sha !== target.head || current.head.ref !== target.branch) throw new Error("publication_stale");
    const remote = await this.api(`git/ref/heads/${branch}`, undefined, true);
    if (remote && remote.object?.sha !== commit) throw new Error("repair_branch_conflict");
    if (!remote) {
      const token = await this.options.publisherToken?.(); if (!token) throw new Error("publisher_credentials_missing");
      await this.git(["-c", "core.hooksPath=/dev/null", "push", "--no-verify", `https://github.com/${this.options.repository}.git`, `${commit}:refs/heads/${branch}`], token);
    }
    const head = `${this.options.repository.split('/')[0]}:${branch}`;
    const existing = await this.api(`pulls?state=all&head=${encodeURIComponent(head)}&base=${encodeURIComponent(target.branch)}`, undefined, true);
    let pr = existing?.find((item: any) => item.head?.ref === branch && item.base?.ref === target.branch);
    if (!pr) pr = await this.api("pulls", { method: "POST", body: JSON.stringify({ title: `Fix CI for #${target.pr}`, head: branch, base: target.branch, draft: true, body: `Repairs CI run ${target.run}, attempt ${target.attempt}, for commit ${target.head}.\n\nVerification and independent grader passed.\nDelivery: ${id}` }) }, true);
    if (!pr?.html_url) throw new Error("publication_result_unknown");
    this.options.store.put("delivery", id, { ...record, url: pr.html_url }); return pr.html_url;
  }
  private async git(args: string[], token: string) {
    await exec("git", args, { cwd: this.options.repoPath, timeout: 60_000, maxBuffer: 1024 * 1024, env: { PATH: process.env.PATH, GIT_TERMINAL_PROMPT: "0", GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_SYSTEM: "/dev/null", GIT_CONFIG_COUNT: "1", GIT_CONFIG_KEY_0: "http.https://github.com/.extraHeader", GIT_CONFIG_VALUE_0: `Authorization: Basic ${Buffer.from(`x-access-token:${token}`).toString('base64')}` } });
  }
}
