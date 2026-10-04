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

test("real Pi preserves its session while switching from Luna max to Sol high", { skip: !enabled, timeout: 60_000 }, async () => {
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
  const request = JSON.parse(body);
  const expected = call < 2
    ? {model:"gpt-5.6-luna",effort:"max"}
    : {model:"gpt-5.6-sol",effort:"high"};
  if(request.model!==expected.model || request.reasoning?.effort!==expected.effort){
    res.writeHead(400); res.end("unexpected model or reasoning effort"); return;
  }
  if(call >= 2 && !body.includes("written by pi")){ res.writeHead(400); res.end("session history was not restored"); return; }
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
  const text=call===2 ? "Implemented the requested file." : "agent.txt contains written by pi.";
  const item={type:"message",id:"msg_1",role:"assistant",status:"completed",content:[{type:"output_text",text,annotations:[]}],phase:"final_answer"}; send(res,[
    {type:"response.created",response:{id:"resp_2"}},
    {type:"response.output_item.added",output_index:0,item},
    {type:"response.output_text.delta",output_index:0,content_index:0,delta:text},
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
    let settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess(
      {
        name: `pi-integration-${suffix}`,
        image: "slack-pi-agent:1.0.1",
        network,
        worktreePath: isolated.path,
        gitMetadataPath: path.join(repo, ".git"),
        sessionStatePath,
        gatewayUrl: "http://model-gateway:8080",
        modelRef: "openai/gpt-5.6-luna",
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

    settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess.prompt("run-2", "Read agent.txt and confirm its content.", {
      modelRef: "openai/gpt-5.6-sol",
      effort: "high",
      source: "jev",
    });
    await settled;

    assert.ok(
      events.some((event) => event.type === "answer" && /written by pi/i.test(event.text)),
      JSON.stringify(events),
    );
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("Pi 1.0.1 resumes a session persisted by Pi 0.83", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-legacy-${suffix}`;
  const gateway = `model-gateway-legacy-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-legacy-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-legacy-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-legacy-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`legacy-${suffix}`);
  const fixture = await readFile(new URL("../../test/fixtures/pi-0.83-session.jsonl", import.meta.url), "utf8");
  await writeFile(
    path.join(sessionStatePath, "legacy-session.jsonl"),
    fixture.replace("__WORKTREE__", isolated.path),
  );
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

  const fakeGatewayScript = String.raw`
const http = require("node:http");
const usage = {input_tokens:1,output_tokens:1,total_tokens:2,input_tokens_details:{cached_tokens:0}};
http.createServer((req,res)=>{ let body=""; req.on("data",c=>body+=c); req.on("end",()=>{
  if(!body.includes("legacy-session-marker-083")){ res.writeHead(400); res.end("legacy session history was not restored"); return; }
  const text="Restored legacy-session-marker-083.";
  const item={type:"message",id:"msg_restored",role:"assistant",status:"completed",content:[{type:"output_text",text,annotations:[]}],phase:"final_answer"};
  res.writeHead(200,{"content-type":"text/event-stream"});
  for(const event of [
    {type:"response.created",response:{id:"resp_restored"}},
    {type:"response.output_item.added",output_index:0,item},
    {type:"response.output_text.delta",output_index:0,content_index:0,delta:text},
    {type:"response.output_item.done",output_index:0,item},
    {type:"response.completed",response:{id:"resp_restored",status:"completed",output:[item],usage}}
  ]) res.write("data: "+JSON.stringify(event)+"\n\n");
  res.end("data: [DONE]\n\n");
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
        name: `pi-legacy-${suffix}`,
        image: "slack-pi-agent:1.0.1",
        network,
        worktreePath: isolated.path,
        gitMetadataPath: path.join(repo, ".git"),
        sessionStatePath,
        gatewayUrl: "http://model-gateway:8080",
        modelRef: "openai/gpt-5.6-luna",
        extensionPaths: ["/opt/pi/extensions/model-gateway.ts"],
        pluginPaths: [],
        skillPaths: [],
        toolNames: ["read"],
        continueSession: true,
        user: `${hostUid}:${hostGid}`,
      },
      () => "fake-run-token",
      (event) => {
        events.push(event);
        if (event.type === "settled" || event.type === "error") resolveSettled();
      },
    );
    piProcess.prompt("legacy-follow-up", "Repeat the marker from the previous turn.");
    await settled;
    assert.ok(
      events.some((event) => event.type === "answer" && /legacy-session-marker-083/.test(event.text)),
      JSON.stringify(events),
    );
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("Pi builtin MCP uses only the trusted GitHub registration and ignores project mcp.json", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-mcp-${suffix}`;
  const modelGateway = `model-gateway-mcp-${suffix}`;
  const mcpGateway = `mcp-gateway-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-mcp-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-mcp-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-mcp-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await mkdir(path.join(repo, ".pi"));
  await writeFile(path.join(repo, ".pi", "mcp.json"), JSON.stringify({ mcpServers: { evil: { url: "http://evil.invalid/mcp", exposure: "direct" } } }));
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "."], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`mcp-${suffix}`);
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

const fakeMcpScript = String.raw`
const http=require("node:http");
http.createServer((req,res)=>{let body="";req.on("data",c=>body+=c);req.on("end",()=>{
  if(req.method==="GET"){res.writeHead(405);res.end();return;}
  if(req.method==="DELETE"){res.writeHead(200);res.end();return;}
  const msg=JSON.parse(body||"{}");
  if(msg.method==="notifications/initialized"){res.writeHead(202);res.end();return;}
  let result;
  if(msg.method==="initialize") result={protocolVersion:"2025-06-18",capabilities:{tools:{}},serverInfo:{name:"github",version:"test"}};
  else if(msg.method==="tools/list") result={tools:[{name:"get_file_contents",description:"Read a repository file",inputSchema:{type:"object",properties:{owner:{type:"string"},repo:{type:"string"},path:{type:"string"}},required:["owner","repo","path"]}}]};
  else if(msg.method==="tools/call"&&msg.params?.name==="get_file_contents"&&msg.params?.arguments?.owner==="acme"&&msg.params?.arguments?.repo==="widgets") result={content:[{type:"text",text:"trusted-mcp-marker"}]};
  else {res.writeHead(400);res.end("unexpected MCP request");return;}
  res.writeHead(200,{"content-type":"application/json","mcp-session-id":"fake-session"});res.end(JSON.stringify({jsonrpc:"2.0",id:msg.id,result}));
});}).listen(8090,"0.0.0.0");`;
  const fakeModelScript = String.raw`
const http=require("node:http");let call=0;const usage={input_tokens:1,output_tokens:1,total_tokens:2,input_tokens_details:{cached_tokens:0}};
function send(res,events){res.writeHead(200,{"content-type":"text/event-stream"});for(const e of events)res.write("data: "+JSON.stringify(e)+"\n\n");res.end("data: [DONE]\n\n");}
http.createServer((req,res)=>{let body="";req.on("data",c=>body+=c);req.on("end",()=>{
  if(body.includes("mcp__evil__")){res.writeHead(400);res.end("project MCP was loaded");return;}
  call++;
  if(call===1){
    if(!body.includes("mcp__github__get_file_contents")){res.writeHead(400);res.end("trusted MCP tool missing");return;}
    const item={type:"function_call",id:"fc_mcp",call_id:"call_mcp",name:"mcp__github__get_file_contents",arguments:'{"owner":"acme","repo":"widgets","path":"README.md"}',status:"completed"};send(res,[
      {type:"response.created",response:{id:"resp_mcp_1"}},{type:"response.output_item.added",output_index:0,item},
      {type:"response.function_call_arguments.done",output_index:0,arguments:item.arguments},{type:"response.output_item.done",output_index:0,item},
      {type:"response.completed",response:{id:"resp_mcp_1",status:"completed",output:[item],usage}}]);return;
  }
  if(!body.includes("trusted-mcp-marker")){res.writeHead(400);res.end("MCP result missing");return;}
  const text="Read trusted-mcp-marker through GitHub MCP.";const item={type:"message",id:"msg_mcp",role:"assistant",status:"completed",content:[{type:"output_text",text,annotations:[]}],phase:"final_answer"};send(res,[
    {type:"response.created",response:{id:"resp_mcp_2"}},{type:"response.output_item.added",output_index:0,item},
    {type:"response.output_text.delta",output_index:0,content_index:0,delta:text},{type:"response.output_item.done",output_index:0,item},
    {type:"response.completed",response:{id:"resp_mcp_2",status:"completed",output:[item],usage}}]);
});}).listen(8080,"0.0.0.0");`;

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", modelGateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", fakeModelScript]);
    await command("docker", ["run", "--detach", "--rm", "--name", mcpGateway, "--network", network, "--network-alias", "mcp-gateway", "node:22.22.0-bookworm-slim", "node", "-e", fakeMcpScript]);
    await new Promise((resolve) => setTimeout(resolve, 500));
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess({
      name: `pi-mcp-${suffix}`, image: "slack-pi-agent:1.0.1", network,
      worktreePath: isolated.path, gitMetadataPath: path.join(repo, ".git"), sessionStatePath,
      gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
      extensionPaths: ["/opt/pi/extensions/model-gateway.ts"], pluginPaths: [], skillPaths: [], toolNames: ["read"],
      user: `${hostUid}:${hostGid}`,
      mcp: { mode: "github_read_only", gatewayUrl: "http://mcp-gateway:8090/mcp", tools: ["get_file_contents"] },
    }, () => "fake-model-token", (event) => {
      events.push(event);
      if (event.type === "settled" || event.type === "error") resolveSettled();
    }, () => "fake-mcp-token");
    piProcess.prompt("mcp-run", "Read README.md from acme/widgets with GitHub MCP.");
    await settled;
    const diagnostic = JSON.stringify({ events, model: await command("docker", ["logs", modelGateway]), mcp: await command("docker", ["logs", mcpGateway]) });
    assert.ok(events.some((event) => event.type === "tool" && event.tool === "mcp__github__get_file_contents"), diagnostic);
    assert.ok(events.some((event) => event.type === "answer" && event.text.includes("trusted-mcp-marker")), diagnostic);
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", modelGateway]).catch(() => "");
    await command("docker", ["stop", mcpGateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});
