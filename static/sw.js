// ZenZone AI Panel — Service Worker
// Caches the app shell; always fetches API/SSE calls from the network.

const CACHE = 'zenzone-v1';
const SHELL = [
  '/',
  '/static/index.html',
  '/static/manifest.json',
  '/static/favicon.ico',
  '/static/icon-128.png',
  '/static/icon-512.png',
  '/static/apple-touch-icon.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Always use network for API and SSE calls
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/chat')) {
    return;
  }
  // Cache-first for everything else
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
