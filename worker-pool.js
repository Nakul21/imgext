class WorkerPool {
    constructor(workerScript, poolSize = null) {
        if (typeof Worker === 'undefined') {
            throw new Error('Web Workers are not supported in this environment');
        }

        this.workers = [];
        this.available = [];
        this.queue = [];
        this.poolSize = poolSize || navigator.hardwareConcurrency || 4;
        this.workerScript = workerScript;
        
        this.initializeWorkers();
    }
    
    initializeWorkers() {
        for (let i = 0; i < this.poolSize; i++) {
            const worker = new Worker(this.workerScript, { type: 'module' });
            worker.onmessage = this.handleWorkerMessage.bind(this);
            this.workers.push(worker);
            this.available.push(i);
        }
    }
    
    handleWorkerMessage(event) {
        console.log('Worker message received:', event.data);
    }
    
    async initialize() {
        const initPromises = this.workers.map((worker) => {
            return new Promise((resolve) => {
                const handler = (e) => {
                    if (e.data.type === 'initialized') {
                        worker.removeEventListener('message', handler);
                        resolve();
                    }
                };
                worker.addEventListener('message', handler);
                worker.postMessage({ type: 'init' });
            });
        });
        
        await Promise.all(initPromises);
    }
    
    async processTask(task) {
        return new Promise((resolve, reject) => {
            const workerIndex = this.available.shift();
            
            if (workerIndex !== undefined) {
                const worker = this.workers[workerIndex];
                
                const handler = (e) => {
                    if (e.data.type === task.responseType) {
                        worker.removeEventListener('message', handler);
                        this.available.push(workerIndex);
                        this.processNextTask();
                        resolve(e.data);
                    } else if (e.data.type === 'error') {
                        worker.removeEventListener('message', handler);
                        this.available.push(workerIndex);
                        this.processNextTask();
                        reject(new Error(e.data.error));
                    }
                };
                
                worker.addEventListener('message', handler);
                worker.postMessage(task.message);
            } else {
                this.queue.push({ task, resolve, reject });
            }
        });
    }
    
    processNextTask() {
        if (this.queue.length > 0 && this.available.length > 0) {
            const { task, resolve, reject } = this.queue.shift();
            this.processTask(task).then(resolve).catch(reject);
        }
    }
    
    terminate() {
        this.workers.forEach(worker => worker.terminate());
        this.workers = [];
        this.available = [];
        this.queue = [];
    }
}

export default WorkerPool;
