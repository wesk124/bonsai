// Correctness-first native CUDA implementation; no full-cache temporary.
// One thread per quantization group. INT4 nibble writes use atomicOr on the
// aligned frame word, including odd head/row/layer tails.
#include "bonsai_codec.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <initializer_list>

static __device__ uint64_t ceildiv(uint64_t x,uint64_t y){return (x+y-1)/y;}
static __device__ uint64_t address(const BonsaiChunkArgs& a,uint64_t h,uint64_t t,uint64_t d){return h*a.stride_head+(t+a.token_begin)*a.stride_token+d*a.stride_dim;}
static __device__ float load(const BonsaiChunkArgs& a,bool value,uint64_t h,uint64_t t,uint64_t d){return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(value?a.v_ptr:a.k_ptr)[address(a,h,t,d)]);}
static __device__ uint64_t groups_per_head(const BonsaiChunkArgs& a,bool v){uint64_t t=a.token_end-a.token_begin,g=v?a.v_group:a.k_group;return (v?a.v_axis:a.k_axis)==0?t*ceildiv(a.head_dim,g):ceildiv(t,g)*a.head_dim;}
static __device__ void group_coords(const BonsaiChunkArgs& a,bool v,uint64_t gi,uint64_t& h,uint64_t& t,uint64_t& d,uint64_t& count){
    uint64_t g=v?a.v_group:a.k_group,nt=a.token_end-a.token_begin,per=groups_per_head(a,v);h=gi/per;gi%=per;
    if((v?a.v_axis:a.k_axis)==0){uint64_t ng=ceildiv(a.head_dim,g);t=gi/ng;d=(gi%ng)*g;count=g<uint64_t(a.head_dim)-d?g:uint64_t(a.head_dim)-d;}else{t=(gi/a.head_dim)*g;d=gi%a.head_dim;count=g<nt-t?g:nt-t;}
}
static __global__ void quant_groups(BonsaiChunkArgs a,bool v,bool validation,int* error){
    uint64_t nt=a.token_end-a.token_begin,per=groups_per_head(a,v),ng=per*a.num_kv_heads;
    for(uint64_t gi=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;gi<ng;gi+=uint64_t(gridDim.x)*blockDim.x){
        uint64_t h,t,d,count;group_coords(a,v,gi,h,t,d,count);bool along_token=(v?a.v_axis:a.k_axis)!=0,sym=v?a.v_symmetric:a.k_symmetric;
        float lo=INFINITY,hi=-INFINITY;bool bad=false;
        for(uint64_t j=0;j<count;++j){float x=load(a,v,h,t+(along_token?j:0),d+(along_token?0:j));if(!isfinite(x)){bad=true;}lo=fminf(lo,x);hi=fmaxf(hi,x);}
        if(bad){if(validation)atomicExch(error,1);continue;}
        float qmax=float((1u<<(sym?a.bits-1:a.bits))-1);
        float raw_s=sym?fmaxf(fabsf(lo),fabsf(hi))/qmax:(hi-lo)/qmax;
        __half hs=__float2half_rn(raw_s),hz=__float2half_rn(sym?0.0f:lo);
        float s=__half2float(hs),z=__half2float(hz);
        if(!isfinite(raw_s)||!isfinite(s)||!isfinite(z)){if(validation)atomicExch(error,2);continue;}
        if(validation)continue;
        uint64_t meta_index=(uint64_t(a.layer_begin)*ng+gi)*(sym?1:2);
        __half* meta=reinterpret_cast<__half*>(a.frame+(v?a.v_meta:a.k_meta));meta[meta_index]=hs;if(!sym)meta[meta_index+1]=hz;
        uint8_t* payload=reinterpret_cast<uint8_t*>(a.frame+(v?a.v_payload:a.k_payload));
        for(uint64_t j=0;j<count;++j){uint64_t tj=t+(along_token?j:0),dj=d+(along_token?0:j);float x=load(a,v,h,tj,dj);float q=s==0?0.0f:nearbyintf((x-z)/s);q=fminf(qmax,fmaxf(sym?-qmax:0.0f,q));int code=int(q);
            uint64_t idx=((uint64_t(a.layer_begin)*a.num_kv_heads+h)*nt+tj)*a.head_dim+dj;
            if(a.bits==8){payload[idx]=uint8_t(code);}else{uint64_t byte=idx/2;uint32_t shift=uint32_t(byte%4)*8+uint32_t(idx%2)*4;atomicOr(reinterpret_cast<unsigned int*>(payload+(byte/4)*4),(uint32_t(code)&15u)<<shift);}
        }
    }
}
static __global__ void bf16_copy(BonsaiChunkArgs a,bool v,bool decode){
    uint64_t nt=a.token_end-a.token_begin,n=uint64_t(a.num_kv_heads)*nt*a.head_dim;
    auto* tensor=reinterpret_cast<uint16_t*>(v?a.v_ptr:a.k_ptr);auto* payload=reinterpret_cast<uint16_t*>(a.frame+(v?a.v_payload:a.k_payload));
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x){uint64_t d=i%a.head_dim,t=(i/a.head_dim)%nt,h=i/(nt*a.head_dim),at=address(a,h,t,d),pi=uint64_t(a.layer_begin)*n+i;if(decode)tensor[at]=payload[pi];else payload[pi]=tensor[at];}
}
static __global__ void decode_quant(BonsaiChunkArgs a,bool v){
    uint64_t nt=a.token_end-a.token_begin,n=uint64_t(a.num_kv_heads)*nt*a.head_dim,g=v?a.v_group:a.k_group;bool sym=v?a.v_symmetric:a.k_symmetric,along_token=(v?a.v_axis:a.k_axis)!=0;
    auto* tensor=reinterpret_cast<__nv_bfloat16*>(v?a.v_ptr:a.k_ptr);auto* payload=reinterpret_cast<const uint8_t*>(a.frame+(v?a.v_payload:a.k_payload));auto* meta=reinterpret_cast<const __half*>(a.frame+(v?a.v_meta:a.k_meta));
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*blockDim.x){uint64_t d=i%a.head_dim,t=(i/a.head_dim)%nt,h=i/(nt*a.head_dim),lh=uint64_t(a.layer_begin)*a.num_kv_heads+h;
        uint64_t gi=along_token?(lh*ceildiv(nt,g)+t/g)*a.head_dim+d:(lh*nt+t)*ceildiv(a.head_dim,g)+d/g;
        uint64_t pi=uint64_t(a.layer_begin)*n+i;int q;if(a.bits==8){q=sym?int(reinterpret_cast<const int8_t*>(payload)[pi]):int(payload[pi]);}else{q=(payload[pi/2]>>((pi%2)*4))&15;if(sym&&q>=8)q-=16;}
        float s=__half2float(meta[gi*(sym?1:2)]),z=sym?0.0f:__half2float(meta[gi*2+1]);tensor[address(a,h,t,d)]=__float2bfloat16_rn(float(q)*s+z);
    }
}
static __global__ void validate_metadata(BonsaiChunkArgs a,bool v,int* error){
    bool sym=v?a.v_symmetric:a.k_symmetric;uint64_t ng=groups_per_head(a,v)*a.num_kv_heads;
    auto* meta=reinterpret_cast<const __half*>(a.frame+(v?a.v_meta:a.k_meta));
    for(uint64_t gi=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;gi<ng;gi+=uint64_t(gridDim.x)*blockDim.x){
        uint64_t at=(uint64_t(a.layer_begin)*ng+gi)*(sym?1:2);float s=__half2float(meta[at]);float z=sym?0.0f:__half2float(meta[at+1]);
        if(!isfinite(s)||s<0.0f||!isfinite(z))atomicExch(error,3);
    }
}
static unsigned blocks(uint64_t n){uint64_t b=(n+127)/128;return unsigned(b<65535?b:65535);}
extern "C" int bonsai_zero_frame(uint64_t frame,uint64_t bytes,uint64_t stream){return int(cudaMemsetAsync(reinterpret_cast<void*>(frame),0,bytes,reinterpret_cast<cudaStream_t>(stream)));}
extern "C" int bonsai_validate_chunk(const BonsaiChunkArgs* args,uint64_t stream){
    if(args->bits==16)return 0;
    cudaStream_t s=reinterpret_cast<cudaStream_t>(stream);int* flag=nullptr;cudaError_t status=cudaMalloc(reinterpret_cast<void**>(&flag),sizeof(int));if(status!=cudaSuccess)return int(status);
    status=cudaMemsetAsync(flag,0,sizeof(int),s);
    if(status==cudaSuccess){for(bool v:{false,true}){uint64_t nt=args->token_end-args->token_begin,g=v?args->v_group:args->k_group;uint64_t ng=uint64_t(args->num_kv_heads)*((v?args->v_axis:args->k_axis)==0?nt*((args->head_dim+g-1)/g):((nt+g-1)/g)*args->head_dim);quant_groups<<<blocks(ng),128,0,s>>>(*args,v,true,flag);}status=cudaGetLastError();}
    int host=0;if(status==cudaSuccess)status=cudaMemcpyAsync(&host,flag,sizeof(int),cudaMemcpyDeviceToHost,s);if(status==cudaSuccess)status=cudaStreamSynchronize(s);
    cudaError_t freed=cudaFree(flag);if(status!=cudaSuccess)return int(status);if(freed!=cudaSuccess)return int(freed);return -host;
}
extern "C" int bonsai_validate_metadata(const BonsaiChunkArgs* args,uint64_t stream){
    if(args->bits==16)return 0;
    cudaStream_t s=reinterpret_cast<cudaStream_t>(stream);int* flag=nullptr;cudaError_t status=cudaMalloc(reinterpret_cast<void**>(&flag),sizeof(int));if(status!=cudaSuccess)return int(status);
    status=cudaMemsetAsync(flag,0,sizeof(int),s);
    if(status==cudaSuccess){for(bool v:{false,true}){uint64_t nt=args->token_end-args->token_begin,g=v?args->v_group:args->k_group;uint64_t ng=uint64_t(args->num_kv_heads)*((v?args->v_axis:args->k_axis)==0?nt*((args->head_dim+g-1)/g):((nt+g-1)/g)*args->head_dim);validate_metadata<<<blocks(ng),128,0,s>>>(*args,v,flag);}status=cudaGetLastError();}
    int host=0;if(status==cudaSuccess)status=cudaMemcpyAsync(&host,flag,sizeof(int),cudaMemcpyDeviceToHost,s);if(status==cudaSuccess)status=cudaStreamSynchronize(s);
    cudaError_t freed=cudaFree(flag);if(status!=cudaSuccess)return int(status);if(freed!=cudaSuccess)return int(freed);return -host;
}
static int launch(const BonsaiChunkArgs* a,uint64_t stream,bool decode){
    cudaStream_t s=reinterpret_cast<cudaStream_t>(stream);uint64_t n=uint64_t(a->num_kv_heads)*(a->token_end-a->token_begin)*a->head_dim;
    for(bool v:{false,true}){if(a->bits==16)bf16_copy<<<blocks(n),128,0,s>>>(*a,v,decode);else if(decode)decode_quant<<<blocks(n),128,0,s>>>(*a,v);else{uint64_t nt=a->token_end-a->token_begin,g=v?a->v_group:a->k_group;uint64_t ng=uint64_t(a->num_kv_heads)*((v?a->v_axis:a->k_axis)==0?nt*((a->head_dim+g-1)/g):((nt+g-1)/g)*a->head_dim);quant_groups<<<blocks(ng),128,0,s>>>(*a,v,false,nullptr);}cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return int(e);}return 0;
}
extern "C" int bonsai_encode_chunk(const BonsaiChunkArgs* a,uint64_t s){return launch(a,s,false);}
extern "C" int bonsai_decode_chunk(const BonsaiChunkArgs* a,uint64_t s){return launch(a,s,true);}
