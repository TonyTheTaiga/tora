import type { RequestHandler } from './$types';

const listeners = new Map<string, Set<ReadableStreamDefaultController>>();

export const GET: RequestHandler = ({ params, request, setHeaders }) => {
  const id = params.experimentId;
  setHeaders({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive'
  });
  const stream = new ReadableStream({
    start(controller) {
      const set = listeners.get(id) ?? new Set();
      set.add(controller);
      listeners.set(id, set);
      request.signal.addEventListener('abort', () => {
        set.delete(controller);
        controller.close();
      });
    }
  });
  return new Response(stream);
};

export const POST: RequestHandler = async ({ params, request }) => {
  const id = params.experimentId;
  const set = listeners.get(id);
  if (set && set.size > 0) {
    const body = await request.text();
    for (const controller of set) {
      controller.enqueue(`data: ${body}\n\n`);
    }
  }
  return new Response(null, { status: 204 });
};
