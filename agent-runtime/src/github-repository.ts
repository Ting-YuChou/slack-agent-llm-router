export function parseGitHubRepositories(value: string | undefined): string[] {
  if (!value?.trim()) return [];
  const repositories = value.split(",").map((item) => item.trim()).filter(Boolean);
  if (repositories.some((item) => !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(item))) {
    throw new Error("PI_AGENT_GITHUB_REPOSITORIES must contain owner/repository entries");
  }
  const normalized = repositories.map((item) => item.toLowerCase());
  if (new Set(normalized).size !== repositories.length) throw new Error("PI_AGENT_GITHUB_REPOSITORIES contains a duplicate repository");
  return repositories;
}

export function repositoryFromRemote(remote: string): string | null {
  const trimmed = remote.trim();
  const scp = /^git@github\.com:([^/]+)\/([^/]+?)(?:\.git)?$/.exec(trimmed);
  if (scp) return `${scp[1]}/${scp[2]}`;
  let url: URL;
  try { url = new URL(trimmed); } catch { return null; }
  if (url.hostname.toLowerCase() !== "github.com") return null;
  const parts = url.pathname.replace(/^\/+|\/+$/g, "").replace(/\.git$/, "").split("/");
  return parts.length === 2 && parts.every(Boolean) ? `${parts[0]}/${parts[1]}` : null;
}
