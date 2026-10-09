// External, version-pinned diagnostic. No engine source is patched.
// Passive mode logs pipeline selection and command-buffer GPU intervals.
// Stage mode puts each existing fused op in its own timestamped compute encoder.
// Stage boundaries serialize op groups and can change cache behavior and latency.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ggml-impl.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "llama-batch.h"
#include <dlfcn.h>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <cstdio>
#include <unistd.h>

// The pinned .m wrappers each contain one Objective-C object pointer.
struct ggml_metal_encoder { id<MTLComputeCommandEncoder> obj; };
struct ggml_metal_pipeline { id<MTLComputePipelineState> obj; };
struct OpPrefix { void *dev; void *lib; ggml_metal_encoder_t enc; };
struct Sample { int index; std::string op, name, source, type, kernels; int64_t columns; size_t bytes; };
struct Graph { int id, tokens; bool main, active, stage; int call; };
struct Encoder {
    Graph graph;
    id<MTLCommandBuffer> cb;
    id<MTLCounterSampleBuffer> counters = nil;
    NSMutableArray * counterBuffers = nil;
    MTLTimestamp cpu0 = 0, gpu0 = 0;
    std::vector<Sample> samples;
};
struct Op { std::shared_ptr<Encoder> encoder; std::vector<ggml_tensor *> nodes; };
static std::mutex mutex;
static std::map<void *, std::shared_ptr<Op>> ops;
static std::map<void *, std::shared_ptr<Encoder>> encoders;
static std::map<void *, std::string> pipelines;
static std::atomic<int> graph_id{0};
static Graph current_graph;
static std::atomic<int> call_id{0};
static int current_call = 0, current_tokens = 0;
static bool current_main = false;
static thread_local Sample * current_sample = nullptr;
static FILE * output = nullptr;
__attribute__((constructor)) static void loaded() { fprintf(stderr,"phase07 external diagnostic loaded\n"); }

static bool enabled() {
    const char * path = getenv("PHASE07_CONTROL");
    return path && access(path,F_OK)==0;
}
static void write(const std::string & text) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!output) {
        const char * path = getenv("PHASE07_OUTPUT");
        if (!path || !(output=fopen(path,"w"))) abort();
    }
    fputs(text.c_str(),output); fflush(output);
}
static std::string clean(const char * text) {
    std::string s=text?text:"";
    for (char & c:s) if (c=='\t' || c=='\n') c=' ';
    return s;
}
static enum ggml_status probe_graph(ggml_metal_t ctx, ggml_cgraph * gf) {
    static auto real=&ggml_metal_graph_compute;
    Graph g{++graph_id,0,false,enabled(),false,current_call};
    for(int i=0;i<gf->n_nodes;++i) {
        auto t=gf->nodes[i];
        if(t->src[0] && t->src[1] && strstr(t->src[0]->name,"ffn_gate_inp.weight")) {
            g.tokens=(int)t->src[1]->ne[1];
            if(strstr(t->src[0]->name,"blk.0.")) g.main=true;
        }
        if(t->src[0] && strstr(t->src[0]->name,"blk.2.")) g.main=true;
    }
    if(current_call) g.tokens=current_tokens;
    g.stage=g.active && g.tokens>0 && g.tokens<=4 && getenv("PHASE07_STAGE");
    current_graph=g;
    if(g.active) {
        char line[256];snprintf(line,sizeof(line),"graph\t%d\t%d\t%d\t%d\t%d\t%d\n",g.id,g.tokens,g.main,g.stage,gf->n_nodes,g.call);write(line);
    }
    uint64_t begin=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    auto rc=real(ctx,gf);
    if(g.active) {char line[256];snprintf(line,sizeof(line),"graph_cpu\t%d\t%llu\t%llu\n",g.id,begin,clock_gettime_nsec_np(CLOCK_UPTIME_RAW));write(line);}
    return rc;
}
static ggml_metal_encoder_t probe_encoder(ggml_metal_cmd_buf_t raw, bool concurrent) {
    static auto real=&ggml_metal_encoder_init;
    auto result=real(raw,concurrent);
    auto e=std::make_shared<Encoder>();e->graph=current_graph;e->cb=(id<MTLCommandBuffer>)raw;
    if(e->graph.active) {
        e->cb.label=[NSString stringWithFormat:@"phase07 g%d m%d main%d",e->graph.id,e->graph.tokens,e->graph.main];
        if(e->graph.stage) {
            id<MTLDevice> device=e->cb.device;
            MTLCounterSampleBufferDescriptor * desc=[MTLCounterSampleBufferDescriptor new];
            for(id<MTLCounterSet> set in device.counterSets) if([set.name isEqualToString:MTLCommonCounterSetTimestamp]) desc.counterSet=set;
            desc.storageMode=MTLStorageModeShared;desc.sampleCount=4096;
            NSError * error=nil;e->counters=[device newCounterSampleBufferWithDescriptor:desc error:&error];[desc release];
            if(!e->counters) {fprintf(stderr,"phase07 counters: %s\n",error.description.UTF8String);abort();}
            e->counterBuffers=[NSMutableArray new];[e->counterBuffers addObject:e->counters];[e->counters release];
            [device sampleTimestamps:&e->cpu0 gpuTimestamp:&e->gpu0];
        }
        [e->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            @autoreleasepool {
                char line[512];
                snprintf(line,sizeof(line),"cb\t%d\t%d\t%d\t%.9f\t%.9f\t%lu\n",e->graph.id,e->graph.tokens,e->graph.main,cb.GPUStartTime,cb.GPUEndTime,(unsigned long)cb.status);write(line);
                if(e->counters) {
                    MTLTimestamp cpu1,gpu1;[cb.device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
                    // Apple's sampleTimestamps CPU clock is in nanoseconds; calibrate GPU clock scale.
                    double scale=(double)(cpu1-e->cpu0)/(double)(gpu1-e->gpu0);
                    snprintf(line,sizeof(line),"calibration\t%d\t%llu\t%llu\t%llu\t%llu\t%.12f\n",e->graph.id,e->cpu0,e->gpu0,cpu1,gpu1,scale);write(line);
                    NSMutableData * data=[NSMutableData data];
                    for(NSUInteger i=0;i<e->counterBuffers.count;++i) {
                        auto count=std::min((size_t)2048,e->samples.size()-i*2048);
                        [data appendData:[(id<MTLCounterSampleBuffer>)e->counterBuffers[i] resolveCounterRange:NSMakeRange(0,count*2)]];
                    }
                    if(data.length!=e->samples.size()*2*sizeof(MTLCounterResultTimestamp)) abort();
                    auto values=(const MTLCounterResultTimestamp *)data.bytes;
                    std::string payload;
                    for(auto & s:e->samples) {
                        auto begin=values[s.index*2].timestamp,end=values[s.index*2+1].timestamp;
                        if(s.kernels.empty()) continue; // Graph nodes with no GPU dispatch have no samples.
                        if(!begin || end<begin || begin==MTLCounterErrorValue || end==MTLCounterErrorValue) {fprintf(stderr,"phase07 invalid timestamp\n");abort();}
                        char prefix[512];snprintf(prefix,sizeof(prefix),"op\t%d\t%d\t%d\t%llu\t%llu\t%.3f\t%zu\t%lld\t",e->graph.id,e->graph.tokens,e->graph.main,begin,end,(end-begin)*scale,s.bytes,s.columns);
                        payload+=std::string(prefix)+s.op+'\t'+s.name+'\t'+s.source+'\t'+s.type+'\t'+s.kernels+'\n';
                    }
                    write(payload);
                    [e->counterBuffers release];e->counters=nil;
                } else {
                    std::string payload;
                    for(auto & s:e->samples) {
                        char prefix[256];snprintf(prefix,sizeof(prefix),"dispatch\t%d\t%d\t%d\t%zu\t%lld\t",e->graph.id,e->graph.tokens,e->graph.main,s.bytes,s.columns);
                        payload+=std::string(prefix)+s.op+'\t'+s.name+'\t'+s.source+'\t'+s.type+'\t'+s.kernels+'\n';
                    }
                    write(payload);
                }
            }
        }];
    }
    {std::lock_guard<std::mutex> lock(mutex);encoders[result]=e;}
    return result;
}
static ggml_metal_op_t probe_init(ggml_metal_device_t dev, ggml_metal_cmd_buf_t cb,ggml_cgraph *gf,ggml_metal_fusion_info *fi,int start,int end,bool concurrency,bool capture,int debug) {
    static auto real=&ggml_metal_op_init;
    auto ctx=real(dev,cb,gf,fi,start,end,concurrency,capture,debug);
    auto enc=((OpPrefix *)ctx)->enc;
    std::lock_guard<std::mutex> lock(mutex);auto o=std::make_shared<Op>();o->encoder=encoders.at(enc);
    for(int i=start;i<end;++i) if(!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) o->nodes.push_back(gf->nodes[i]);
    ops[ctx]=o;return ctx;
}
static int probe_op(ggml_metal_op_t ctx,int idx) {
    static auto real=&ggml_metal_op_encode;
    std::shared_ptr<Op> o;{std::lock_guard<std::mutex> lock(mutex);o=ops.at(ctx);}
    auto e=o->encoder;if(!e->graph.active) return real(ctx,idx);
    auto t=o->nodes.at(idx);auto src=t->src[0];
    Sample s{(int)e->samples.size(),clean(ggml_op_desc(t)),clean(t->name),src?clean(src->name):"",src?ggml_type_name(src->type):"","",t->src[1]?t->src[1]->ne[1]:0,src?ggml_nbytes(src):0};
    auto enc=((OpPrefix *)ctx)->enc;
    if(e->graph.stage) {
        if(s.index>0 && s.index%2048==0) {
            MTLCounterSampleBufferDescriptor * desc=[MTLCounterSampleBufferDescriptor new];
            for(id<MTLCounterSet> set in e->cb.device.counterSets) if([set.name isEqualToString:MTLCommonCounterSetTimestamp]) desc.counterSet=set;
            desc.storageMode=MTLStorageModeShared;desc.sampleCount=4096;
            NSError * error=nil;e->counters=[e->cb.device newCounterSampleBufferWithDescriptor:desc error:&error];[desc release];
            if(!e->counters) abort();[e->counterBuffers addObject:e->counters];[e->counters release];
        }
        [enc->obj endEncoding];[enc->obj release];
        MTLComputePassDescriptor * desc=[MTLComputePassDescriptor computePassDescriptor];
        desc.dispatchType=MTLDispatchTypeConcurrent;
        auto att=desc.sampleBufferAttachments[0];att.sampleBuffer=e->counters;
        att.startOfEncoderSampleIndex=(s.index%2048)*2;att.endOfEncoderSampleIndex=(s.index%2048)*2+1;
        enc->obj=[[e->cb computeCommandEncoderWithDescriptor:desc] retain];
        enc->obj.label=[NSString stringWithUTF8String:s.name.c_str()];
    }
    current_sample=&s;int n=real(ctx,idx);current_sample=nullptr;
    e->samples.push_back(std::move(s));
    if(e->graph.stage) {
        [enc->obj endEncoding];[enc->obj release];
        enc->obj=[[e->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] retain];
    }
    return n;
}
static void probe_pipeline(ggml_metal_encoder_t enc,ggml_metal_pipeline_with_params p) {
    static auto real=&ggml_metal_encoder_set_pipeline;
    if(current_sample) {
        std::lock_guard<std::mutex> lock(mutex);
        auto name=pipelines.find(p.pipeline);
        current_sample->kernels+=(current_sample->kernels.empty()?"":";")+(name==pipelines.end()?clean(p.pipeline->obj.label.UTF8String):name->second);
    }
    real(enc,p);
}
static void probe_add(ggml_metal_pipelines_t ps,const char *name,ggml_metal_pipeline_t p) {
    static auto real=&ggml_metal_pipelines_add;
    {std::lock_guard<std::mutex> lock(mutex);pipelines[p]=clean(name);}real(ps,name,p);
}
static void probe_free(ggml_metal_op_t ctx) {
    static auto real=&ggml_metal_op_free;
    {std::lock_guard<std::mutex> lock(mutex);auto it=ops.find(ctx);if(it!=ops.end()) {encoders.erase(((OpPrefix *)ctx)->enc);ops.erase(it);}}
    real(ctx);
}
static int32_t call_start(llama_context * ctx,int count,bool main) {
    current_call=++call_id;current_tokens=count;current_main=main;
    if(enabled()) {
        char line[256];snprintf(line,sizeof(line),"call_begin\t%d\t%d\t%d\t%llu\t%p\n",current_call,count,main,clock_gettime_nsec_np(CLOCK_UPTIME_RAW),ctx);write(line);
    }
    return current_call;
}
static void call_end(int id,int rc) {
    if(enabled()) {char line[256];snprintf(line,sizeof(line),"call_end\t%d\t%d\t%llu\n",id,rc,clock_gettime_nsec_np(CLOCK_UPTIME_RAW));write(line);}
    current_call=0;
}
static int32_t probe_decode(llama_context *ctx,llama_batch batch) {
    auto real=&llama_decode;
    if(current_call) return real(ctx,batch);
    int id=call_start(ctx,batch.n_tokens,true);auto rc=real(ctx,batch);call_end(id,rc);return rc;
}
static int32_t probe_process(llama_context *ctx,llama_process_type type,llama_batch_ext * batch) {
    auto real=&llama_process;
    if(current_call) return real(ctx,type,batch);
    int id=call_start(ctx,(int)batch->tokens.size(),false);auto rc=real(ctx,type,batch);call_end(id,rc);return rc;
}
static ggml_status probe_cpu(ggml_cgraph * gf,ggml_cplan * plan) {
    auto real=&ggml_graph_compute;
    bool active=enabled();uint64_t begin=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    auto rc=real(gf,plan);
    uint64_t end=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if(active) {
        size_t ple=0,embedding=0;
        for(int i=0;i<gf->n_nodes;++i) {
            auto t=gf->nodes[i];auto src=t->src[0];
            if(t->op==GGML_OP_GET_ROWS && src) {
                size_t bytes=ggml_row_size(src->type,src->ne[0])*ggml_nelements(t->src[1]);
                if(t->flags & GGML_TENSOR_FLAG_COMPUTE) {
                    if(strstr(src->name,"per_layer_token_embd")) ple+=bytes;
                    else if(strstr(src->name,"token_embd")) embedding+=bytes;
                }
                char row[512];snprintf(row,sizeof(row),"cpu_rows\t%d\t%d\t%d\t%lld\t%lld\t%zu\t%s\t%s\t%s\n",current_call,current_tokens,t->flags,src->ne[0],ggml_nelements(t->src[1]),bytes,ggml_type_name(src->type),t->name,src->name);write(row);
            }
        }
        char line[512];snprintf(line,sizeof(line),"cpu\t%d\t%d\t%llu\t%llu\t%zu\t%zu\t%d\t%d\n",current_call,current_tokens,begin,end,ple,embedding,gf->n_nodes,rc);write(line);
    }
    return rc;
}
#define INTERPOSE(replacement,replacee) __attribute__((used)) static struct { const void *a;const void *b; } i_##replacement __attribute__((section("__DATA,__interpose"))) = {(const void *)&replacement,(const void *)&replacee}
INTERPOSE(probe_graph,ggml_metal_graph_compute);
INTERPOSE(probe_encoder,ggml_metal_encoder_init);
INTERPOSE(probe_init,ggml_metal_op_init);
INTERPOSE(probe_op,ggml_metal_op_encode);
INTERPOSE(probe_pipeline,ggml_metal_encoder_set_pipeline);
INTERPOSE(probe_add,ggml_metal_pipelines_add);
INTERPOSE(probe_free,ggml_metal_op_free);
INTERPOSE(probe_decode,llama_decode);
INTERPOSE(probe_process,llama_process);
INTERPOSE(probe_cpu,ggml_graph_compute);
