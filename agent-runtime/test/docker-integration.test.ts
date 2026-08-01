import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { DockerPiProcess } from "../src/container-runtime.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { WorktreeManager } from "../src/worktree-manager.js";

const exec = promisify(execFile);
const enabled = process.env.PI_AGENT_DOCKER_INTEGRATION === "1";

async function command(command: string, args: string[], cwd?: string): Promise<string> {
  return (await exec(command, args, { cwd, encoding: "utf8" })).stdout.trim();
}

test("real Pi expands an allowlisted skill, requests approval, and writes only in the isolated worktree", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-integration-${suffix}`;
  const gateway = `model-gateway-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-integration-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-integration-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-integration-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`integration-${suffix}`);
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

  const fakeGatewayScript = String.raw`
const http = require("node:http"); let call = 0;
const usage = {input_tokens:1,output_tokens:1,total_tokens:2,input_tokens_details:{cached_tokens:0}};
function send(res, events) { res.writeHead(200,{"content-type":"text/event-stream"}); for(const e of events) res.write("data: "+JSON.stringify(e)+"\n\n"); res.end("data: [DONE]\n\n"); }
http.createServer((req,res)=>{ let body=""; req.on("data",c=>body+=c); req.on("end",()=>{
  call++;
  if(call===1){
    if(!body.includes("Close one concrete coverage gap")){ res.writeHead(400); res.end("skill was not expanded"); return; }
    const item={type:"function_call",id:"fc_1",call_id:"call_1",name:"write",arguments:'{"path":"agent.txt","content":"written by pi\\n"}',status:"completed"}; send(res,[
    {type:"response.created",response:{id:"resp_1"}},
    {type:"response.output_item.added",output_index:0,item},
    {type:"response.function_call_arguments.done",output_index:0,arguments:item.arguments},
    {type:"response.output_item.done",output_index:0,item},
    {type:"response.completed",response:{id:"resp_1",status:"completed",output:[item],usage}}
  ]); return; }
  const item={type:"message",id:"msg_1",role:"assistant",status:"completed",content:[{type:"output_text",text:"Implemented the requested file.",annotations:[]}],phase:"final_answer"}; send(res,[
    {type:"response.created",response:{id:"resp_2"}},
    {type:"response.output_item.added",output_index:0,item},
    {type:"response.output_text.delta",output_index:0,content_index:0,delta:"Implemented the requested file."},
    {type:"response.output_item.done",output_index:0,item},
    {type:"response.completed",response:{id:"resp_2",status:"completed",output:[item],usage}}
  ]);
});}).listen(8080,"0.0.0.0");`;

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", fakeGatewayScript]);
    await new Promise((resolve) => setTimeout(resolve, 500));
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess(
      {
        name: `pi-integration-${suffix}`,
        image: "slack-pi-agent:0.83.0",
        network,
        worktreePath: isolated.path,
        gitMetadataPath: path.join(repo, ".git"),
        sessionStatePath,
        gatewayUrl: "http://model-gateway:8080/v1",
        extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts"],
        pluginPaths: ["/opt/pi/plugins/workspace-summary.ts"],
        skillPaths: ["/opt/pi/skills/test-gap/SKILL.md"],
        toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls", "workspace_summary"],
        user: `${hostUid}:${hostGid}`,
      },
      () => "fake-run-token",
      (event) => {
        events.push(event);
        if (event.type === "approval") piProcess!.decide(event.approval_id, true);
        if (event.type === "settled") resolveSettled();
        if (event.type === "error") resolveSettled();
      },
    );
    piProcess.prompt("run-1", "/skill:test-gap Create agent.txt with the requested content.");
    await settled;

    assert.equal(await readFile(path.join(isolated.path, "agent.txt"), "utf8"), "written by pi\n");
    assert.ok(events.some((event) => event.type === "approval"));
    assert.ok(events.some((event) => event.type === "tool" && event.tool === "write"));
    assert.doesNotMatch(JSON.stringify(events), /thinking|chain.of.thought/i);
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});
