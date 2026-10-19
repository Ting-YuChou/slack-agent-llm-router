import type { IncomingMessage, ServerResponse } from "node:http";

type Handler = (request: IncomingMessage, response: ServerResponse) => Promise<void>;

/**
 * node:http drops the promise an async handler returns, so an error escaping one becomes an
 * unhandled rejection and terminates the process. This answers 500 instead (or ends a response
 * that already started) and logs only the server name: the error may carry request secrets.
 */
export function guardedHandler(server: string, handler: Handler): (request: IncomingMessage, response: ServerResponse) => void {
  return (request, response) => {
    handler(request, response).catch(() => {
      console.error(JSON.stringify({ event: "http_handler_failed", server }));
      if (response.headersSent) { response.destroy(); return; }
      response.writeHead(500, { "content-type": "application/json" }).end(JSON.stringify({ error: { code: "internal_error" } }));
    });
  };
}
