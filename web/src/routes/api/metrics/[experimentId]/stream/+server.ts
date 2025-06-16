import type { RequestHandler } from './$types';
import { addListener, removeListener } from '$lib/server/broadcast';

export const GET: RequestHandler = ({ params, request, setHeaders }) => {
  const id = params.experimentId;
  setHeaders({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive'
  });
  const stream = new ReadableStream({
    start(controller) {
      addListener(id, controller);
      request.signal.addEventListener('abort', () => {
        removeListener(id, controller);
        controller.close();
      });
    }
  });
  return new Response(stream);
};
