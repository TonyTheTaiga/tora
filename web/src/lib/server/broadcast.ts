const listeners = new Map<string, Set<ReadableStreamDefaultController>>();
const queues = new Map<string, string[]>();
const MAX_QUEUE = 50;

export function addListener(id: string, controller: ReadableStreamDefaultController) {
  const set = listeners.get(id) ?? new Set();
  set.add(controller);
  listeners.set(id, set);
  const queue = queues.get(id);
  if (queue) {
    for (const m of queue) {
      controller.enqueue(`data: ${m}\n\n`);
    }
  }
}

export function removeListener(id: string, controller: ReadableStreamDefaultController) {
  listeners.get(id)?.delete(controller);
}

export function broadcastMetric(id: string, body: string) {
  const queue = queues.get(id) ?? [];
  queue.push(body);
  if (queue.length > MAX_QUEUE) queue.shift();
  queues.set(id, queue);
  const set = listeners.get(id);
  if (!set) return;
  for (const controller of set) {
    controller.enqueue(`data: ${body}\n\n`);
  }
}
