/** A candidate touched files outside the trusted profile's allowed roots; repairable by reverting them. */
export class ScopeViolationError extends Error {
  constructor(readonly files: string[]) {
    super("candidate_outside_allowed_scope");
  }
}

/** Files a candidate changed that fall outside the trusted profile's allowed roots. */
export function filesOutsideScope(files: string[], allowedPaths: string[]): string[] {
  return files.filter(file => !allowedPaths.some(root => {
    const normalized = root.replace(/\/+$/, "");
    return root === "." || normalized === "." || file === normalized || file.startsWith(`${normalized}/`);
  }));
}

export function assertCandidateScope(files: string[], allowedPaths: string[]): void {
  const outside = filesOutsideScope(files, allowedPaths);
  if (outside.length) throw new ScopeViolationError(outside);
}
