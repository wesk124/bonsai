#!/usr/bin/env python3
import gc, json, os, socket, time
import torch, bonsai
print(json.dumps({'hostname': socket.gethostname(), 'environment': {k:os.getenv(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','CUDA_MODULE_LOADING']}}), flush=True)
endpoint = bonsai.FabricEndpoint(0, 1)
for size in [16 << 20, 2395865088]:
    for offset, trim in [(0,0),(512,0),(0,256),(512,256)]:
        backing = torch.empty(size+offset, device='cuda', dtype=torch.uint8)
        t=backing[offset:offset+size-trim]
        for init in ['empty','zero_sync']:
            if init == 'zero_sync':
                t.zero_(); torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                handle, desc = endpoint.register_sink(t.data_ptr(), t.numel()); endpoint.unregister(handle)
                result='PASS'
            except Exception as exc:
                result=repr(exc)
            print(json.dumps({'bytes':t.numel(),'offset':offset,'trim':trim,'init':init,'ptr':hex(t.data_ptr()),'result':result,'seconds':time.perf_counter()-start}),flush=True)
        del t,backing; gc.collect(); torch.cuda.empty_cache()
print('ALIGNMENT_PROBE_COMPLETE',flush=True)
