#!/usr/bin/env python3
"""GPU MR diagnosis only; each attempted registration is logged, never a transfer claim."""
import gc, json, os, socket, time
import torch, bonsai

torch.cuda.set_device(0)
endpoint = bonsai.FabricEndpoint(0, 1)
print(json.dumps({'hostname': socket.gethostname(), 'gpu': torch.cuda.get_device_name(), 'probe': 'MR allocation/init size sweep', 'environment': {k: os.getenv(k) for k in ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_MODULE_LOADING']}}), flush=True)
for n in [16 << 20, 512 << 20, 1 << 30, (2 << 30) - 65536, (2 << 30) + 65536, 2395865088, 3 << 30]:
    for init in ['empty', 'zero_sync']:
        t = torch.empty(n, device='cuda', dtype=torch.uint8)
        if init == 'zero_sync':
            t.zero_(); torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            handle, desc = endpoint.register_sink(t.data_ptr(), n)
            endpoint.unregister(handle)
            result = 'PASS'
        except Exception as exc:
            result = repr(exc)
        print(json.dumps({'bytes': n, 'init': init, 'ptr': hex(t.data_ptr()), 'seconds': time.perf_counter()-start, 'result': result}), flush=True)
        del t; gc.collect(); torch.cuda.empty_cache()
print('REGISTER_PROBE_COMPLETE', flush=True)
