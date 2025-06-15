import { handler } from './build/handler.js';
import http from 'http';

const clients = new Set();

const server = http.createServer((req, res) => {
  if (req.url === '/metrics' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    });
    res.write('\n');
    clients.add(res);
    req.on('close', () => {
      clients.delete(res);
    });
  } else if (req.url === '/metrics' && req.method === 'POST') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });
    req.on('end', () => {
      for (const client of clients) {
        client.write(`data: ${body}\n\n`);
      }
      res.writeHead(204);
      res.end();
    });
  } else {
    handler(req, res);
  }
});

server.listen(process.env.PORT || 3000);
