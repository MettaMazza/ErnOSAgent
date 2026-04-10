// ErnOS Agent — Service Worker
// Caches static assets for offline-first PWA experience
const CACHE_NAME = 'ernosagent-v1';
const STATIC_ASSETS = [
    '/',
    '/app.js',
    '/app.css',
    '/manifest.json',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Skip WebSocket and API requests
    if (url.pathname.startsWith('/ws') || url.pathname.startsWith('/api/')) {
        return;
    }

    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) {
                // Return cache, update in background
                fetch(event.request).then(fresh => {
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, fresh));
                }).catch(() => {});
                return cached;
            }
            return fetch(event.request).then(resp => {
                const clone = resp.clone();
                caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                return resp;
            });
        })
    );
});
