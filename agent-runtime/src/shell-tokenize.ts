/**
 * A deliberately small shell tokenizer for policy decisions.
 *
 * It understands words, quotes, escapes, the list operators `| || && ;` and newlines, and plain
 * redirects. Anything that makes the executed text differ from the written text (expansion,
 * substitution, subshells, groups, heredocs, background jobs) is refused instead of modelled, so the
 * caller can fall back to asking a human.
 */

export interface ShellRedirect {
  /** `write` creates or truncates a file, `read` reads one, `dup` copies a file descriptor. */
  kind: "write" | "read" | "dup";
  target: string;
}

export interface ShellSegment {
  argv: string[];
  redirects: ShellRedirect[];
  /** True when a word contains `* ? [` outside quotes, so the shell may expand it to other paths. */
  unquotedGlob: boolean;
}

export type ShellTokenizeResult =
  | { ok: true; segments: ShellSegment[] }
  | { ok: false; reason: string };

type Token =
  | { type: "word"; value: string; glob: boolean }
  | { type: "op"; value: "|" | "||" | "&&" | ";" }
  | { type: "redirect"; kind: ShellRedirect["kind"]; target?: string };

class Refused extends Error {}

function refuse(reason: string): never {
  throw new Refused(reason);
}

const WORD_BREAK = new Set([" ", "\t", "\n", ";", "|", "&", "<", ">"]);

function lex(input: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  const peek = (offset = 0) => input[index + offset] ?? "";

  const readWord = (): { value: string; glob: boolean } => {
    let value = "";
    let glob = false;
    while (index < input.length && !WORD_BREAK.has(peek())) {
      const char = peek();
      if (char === "'") {
        const end = input.indexOf("'", index + 1);
        if (end < 0) refuse("unterminated_quote");
        value += input.slice(index + 1, end);
        index = end + 1;
      } else if (char === "\"") {
        index += 1;
        for (;;) {
          if (index >= input.length) refuse("unterminated_quote");
          const inner = peek();
          if (inner === "\"") { index += 1; break; }
          if (inner === "$") refuse("expansion");
          if (inner === "`") refuse("command_substitution");
          if (inner === "\\" && ["\"", "\\", "$", "`", "\n"].includes(peek(1))) {
            if (peek(1) !== "\n") value += peek(1);
            index += 2;
            continue;
          }
          value += inner;
          index += 1;
        }
      } else if (char === "\\") {
        if (index + 1 >= input.length) refuse("trailing_escape");
        if (peek(1) !== "\n") value += peek(1);
        index += 2;
      } else if (char === "$") {
        refuse(peek(1) === "(" ? "command_substitution" : "expansion");
      } else if (char === "`") {
        refuse("command_substitution");
      } else if ("(){}".includes(char)) {
        refuse("group_or_subshell");
      } else {
        if ("*?[".includes(char)) glob = true;
        value += char;
        index += 1;
      }
    }
    return { value, glob };
  };

  const readRedirect = (fdPrefix: string): Token => {
    const char = peek();
    if (char === "<") {
      if (peek(1) === "<") refuse("heredoc");
      if (peek(1) === "(") refuse("process_substitution");
      if (peek(1) === ">") refuse("read_write_redirect");
      if (peek(1) === "&") { index += 2; return { type: "redirect", kind: "dup", target: readDupTarget() }; }
      index += 1;
      return { type: "redirect", kind: "read" };
    }
    // char === ">"
    if (peek(1) === "(") refuse("process_substitution");
    if (peek(1) === "&") {
      index += 2;
      const target = readDupTarget();
      // `>&file` without a descriptor number sends both streams to a file.
      if (/^(?:\d+|-)$/.test(target)) return { type: "redirect", kind: "dup", target };
      if (fdPrefix) refuse("redirect_syntax");
      return { type: "redirect", kind: "write", target };
    }
    index += peek(1) === ">" || peek(1) === "|" ? 2 : 1;
    return { type: "redirect", kind: "write" };
  };

  const readDupTarget = (): string => {
    while (peek() === " " || peek() === "\t") index += 1;
    const word = readWord();
    if (!word.value) refuse("redirect_syntax");
    return word.value;
  };

  while (index < input.length) {
    const char = peek();
    if (char === " " || char === "\t") { index += 1; continue; }
    if (char === "\n") { tokens.push({ type: "op", value: ";" }); index += 1; continue; }
    if (char === "#" ) {
      while (index < input.length && peek() !== "\n") index += 1;
      continue;
    }
    if (char === ";") {
      if (peek(1) === ";" || peek(1) === "&") refuse("case_syntax");
      tokens.push({ type: "op", value: ";" });
      index += 1;
      continue;
    }
    if (char === "|") {
      if (peek(1) === "&") refuse("pipe_stderr");
      if (peek(1) === "|") { tokens.push({ type: "op", value: "||" }); index += 2; continue; }
      tokens.push({ type: "op", value: "|" });
      index += 1;
      continue;
    }
    if (char === "&") {
      if (peek(1) === "&") { tokens.push({ type: "op", value: "&&" }); index += 2; continue; }
      if (peek(1) === ">") {
        index += peek(2) === ">" ? 3 : 2;
        tokens.push({ type: "redirect", kind: "write" });
        continue;
      }
      refuse("background");
    }
    if (char === "<" || char === ">") { tokens.push(readRedirect("")); continue; }
    const fd = /^\d+(?=[<>])/.exec(input.slice(index));
    if (fd) {
      index += fd[0].length;
      tokens.push(readRedirect(fd[0]));
      continue;
    }
    const word = readWord();
    tokens.push({ type: "word", value: word.value, glob: word.glob });
  }
  return tokens;
}

function parse(tokens: Token[]): ShellSegment[] {
  const segments: ShellSegment[] = [];
  let current: ShellSegment = { argv: [], redirects: [], unquotedGlob: false };
  let pendingOperator: Token | undefined;
  const close = (operator: Token) => {
    if (!current.argv.length) refuse("empty_command");
    segments.push(current);
    current = { argv: [], redirects: [], unquotedGlob: false };
    pendingOperator = operator;
  };
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token.type === "op") {
      // Blank lines and repeated newlines are harmless; other empty commands are syntax errors.
      if (token.value === ";" && !current.argv.length && !current.redirects.length && (!segments.length || pendingOperator?.type === "op" && pendingOperator.value === ";")) continue;
      close(token);
      continue;
    }
    pendingOperator = undefined;
    if (token.type === "redirect") {
      let target = token.target;
      if (target === undefined) {
        const next = tokens[index + 1];
        if (!next || next.type !== "word" || !next.value) refuse("redirect_syntax");
        if (next.glob) refuse("redirect_glob");
        target = next.value;
        index += 1;
      }
      current.redirects.push({ kind: token.kind, target });
      continue;
    }
    current.argv.push(token.value);
    if (token.glob) current.unquotedGlob = true;
  }
  if (current.argv.length) segments.push(current);
  else if (current.redirects.length) refuse("empty_command");
  else if (pendingOperator?.type === "op" && pendingOperator.value !== ";") refuse("dangling_operator");
  if (!segments.length) refuse("empty_command");
  return segments;
}

export function tokenizeShell(command: string): ShellTokenizeResult {
  try {
    return { ok: true, segments: parse(lex(command)) };
  } catch (error) {
    if (error instanceof Refused) return { ok: false, reason: error.message };
    throw error;
  }
}
