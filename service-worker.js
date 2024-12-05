// const CACHE_NAME = 'text-recognition-app-v1';
// const urlsToCache = [
//   '/',
//   '/index.html',
//   '/styles.css',
//   '/app.js',
//   'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs',
//   'https://cdn.jsdelivr.net/npm/@tensorflow-models/text-recognition'
// ];

// self.addEventListener('install', event => {
//   event.waitUntil(
//     caches.open(CACHE_NAME)
//       .then(cache => cache.addAll(urlsToCache))
//   );
// });

// self.addEventListener('fetch', event => {
//   event.respondWith(
//     caches.match(event.request)
//       .then(response => response || fetch(event.request))
//   );
// });


const CACHE_NAME = 'text-recognition-app-v1';
const urlsToCache = [
  './', // Use relative paths for local resources
  './index.html',
  './styles.css',
  './app.js',
  './worker.js' // Add the new worker file to cache
];

// Separate third-party URLs that might need different caching strategies
const thirdPartyUrls = [
  'https://docs.opencv.org/4.5.2/opencv.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      
      // Cache local resources first
      console.log('Caching local resources...');
      try {
        await cache.addAll(urlsToCache);
        console.log('Local resources cached successfully');
      } catch (error) {
        console.error('Failed to cache local resources:', error);
      }
      
      // Cache third-party resources individually with fallback
      console.log('Caching third-party resources...');
      await Promise.all(
        thirdPartyUrls.map(async url => {
          try {
            const response = await fetch(url, { mode: 'no-cors' });
            await cache.put(url, response);
            console.log(`Cached: ${url}`);
          } catch (error) {
            console.error(`Failed to cache: ${url}`, error);
          }
        })
      );
    })()
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    (async () => {
      try {
        // Try to get from cache first
        const cachedResponse = await caches.match(event.request);
        if (cachedResponse) {
          return cachedResponse;
        }

        // If not in cache, fetch from network
        const response = await fetch(event.request);
        
        // Only cache successful responses
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        // Clone the response since it can only be consumed once
        const responseToCache = response.clone();
        const cache = await caches.open(CACHE_NAME);
        await cache.put(event.request, responseToCache);

        return response;
      } catch (error) {
        console.error('Fetch handler error:', error);
        // You could return a custom offline page here
        return new Response('Network error occurred', { status: 503 });
      }
    })()
  );
});

// Add cache cleanup on activation
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});