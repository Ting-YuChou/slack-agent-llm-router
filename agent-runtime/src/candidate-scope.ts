/** Files a candidate changed that fall outside the trusted profile's allowed roots. */
export function filesOutsideScope(files: string[], allowedPaths: string[]): string[] {
  return files.filter(file => !allowedPaths.some(root => root === "." || file === root || file.startsWith(root.replace(/\/$/, "") + "/")));
}

export function assertCandidateScope(files: string[], allowedPaths: string[]): void {
  if (filesOutsideScope(files, allowedPaths).length) throw new Error("candidate_outside_allowed_scope");
}
