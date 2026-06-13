import json
import os
from dataclasses import dataclass


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    char_count: int


PYTORCH_KNOWLEDGE_BASE = [
    {
        "title": "DataLoader Performance and num_workers",
        "url": "https://pytorch.org/docs/stable/data.html",
        "text": """
DataLoader num_workers Best Practices

The DataLoader class supports loading data in parallel using multiple worker processes.
Setting num_workers=0 (the default) means data loading happens in the main process,
which creates a bottleneck during training because the GPU sits idle while waiting for data.

Recommended settings:
- Start with num_workers=2 or num_workers=4
- A common rule of thumb is num_workers = 4 * num_GPUs
- Use pin_memory=True when training on GPU to speed up host-to-device transfers

Example:
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

persistent_workers=True keeps worker processes alive between epochs, avoiding
the overhead of spawning new processes each epoch.

prefetch_factor controls how many batches are prefetched per worker.
Default is 2. Increasing it can help when data loading is slow.

Common DataLoader bottleneck symptoms:
- GPU utilization fluctuates between 0% and high values
- Training step time is much longer than forward+backward pass alone
- CPU usage is low during training

To diagnose: use torch.profiler and look for long gaps between training steps.
        """
    },
    {
        "title": "torch.no_grad and inference mode",
        "url": "https://pytorch.org/docs/stable/generated/torch.no_grad.html",
        "text": """
torch.no_grad Context Manager

torch.no_grad() is a context manager that disables gradient computation.
It should always be used during inference and evaluation.

When gradients are enabled (the default), PyTorch builds a computation graph
for every operation to support backpropagation. This uses extra memory and
compute that is completely wasted during inference.

Usage:
    model.eval()
    with torch.no_grad():
        output = model(input)

Benefits:
- Reduces memory consumption by not storing intermediate activations
- Speeds up computation by skipping gradient tracking
- Prevents accidental gradient accumulation

torch.inference_mode() is a stronger version introduced in PyTorch 1.9:
    with torch.inference_mode():
        output = model(input)

inference_mode is faster than no_grad because it additionally prevents
the result tensors from being used in gradient computation later.

Common mistake: calling model.eval() without torch.no_grad().
model.eval() only changes behavior of dropout and batchnorm layers.
It does NOT disable gradient computation. You need both.

   
    model.eval()
    output = model(input)

   
    model.eval()
    with torch.no_grad():
        output = model(input)
        """
    },
    {
        "title": "GPU Memory Management and OOM Errors",
        "url": "https://pytorch.org/docs/stable/notes/cuda.html",
        "text": """
GPU Memory Management in PyTorch

PyTorch uses a caching memory allocator to speed up memory allocations.
Memory is not immediately released back to the OS when tensors are deleted.
Instead it is kept in a cache and reused for future allocations.

Checking memory usage:
    torch.cuda.memory_allocated()    
    torch.cuda.memory_reserved()   
    torch.cuda.max_memory_allocated() 

Freeing memory:
    del tensor
    torch.cuda.empty_cache()  

Out of Memory (OOM) errors:
Common causes:
1. Batch size too large
2. Gradients accumulating during inference (missing no_grad)
3. Storing references to intermediate tensors
4. Memory fragmentation

Solutions:
- Reduce batch size
- Use gradient checkpointing: torch.utils.checkpoint.checkpoint()
- Use mixed precision training with torch.cuda.amp.autocast()
- Clear cache between epochs

Gradient Checkpointing:
Trades compute for memory by recomputing activations during backward pass
instead of storing them. Reduces memory by sqrt(n) for n layers.

    from torch.utils.checkpoint import checkpoint
    output = checkpoint(model_segment, input)

Mixed Precision (AMP):
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        output = model(input)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
        """
    },
    {
        "title": "PyTorch Profiler Usage and Bottleneck Detection",
        "url": "https://pytorch.org/docs/stable/profiler.html",
        "text": """
torch.profiler - Performance Profiling

The PyTorch profiler helps identify performance bottlenecks in your model.

Basic usage:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        model(input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

Key metrics to look at:
- self_cpu_time_total: time spent in this op on CPU (excluding children)
- self_cuda_time_total: time spent in this op on GPU
- cpu_memory_usage: memory allocated by this op on CPU
- cuda_memory_usage: memory allocated by this op on GPU

Identifying bottlenecks:
- If self_cpu_time_total >> self_cuda_time_total: CPU bottleneck
- If self_cuda_time_total >> self_cpu_time_total: GPU compute bound
- Large memory usage ops: candidates for gradient checkpointing

Common slow operations:
- aten::_slow_conv2d_forward: use cudnn convolution instead (move to CUDA)
- aten::copy_: excessive host-device transfers, check tensor devices
- aten::item: GPU-CPU sync, avoid in loops

Exporting traces:
    prof.export_chrome_trace("trace.json")
Open in Chrome at chrome://tracing for visual timeline.

Using record_function to label sections:
    with record_function("data_loading"):
        batch = next(dataloader_iter)
    with record_function("forward_pass"):
        output = model(batch)
        """
    },
    {
        "title": "Tensor Device Management and CUDA Semantics",
        "url": "https://pytorch.org/docs/stable/notes/cuda.html",
        "text": """
CUDA Semantics and Tensor Device Management

Every tensor in PyTorch lives on a specific device (CPU or a GPU).
Operations between tensors on different devices will raise a RuntimeError.

Best practice - define device once and use everywhere:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

   
    x = torch.zeros(10, device=device)
    x = torch.tensor([1, 2, 3], device=device)

Common mistake - creating tensors without specifying device:
   
    mask = torch.zeros(batch_size, seq_len)

   
    mask = torch.zeros(batch_size, seq_len, device=device)

Moving tensors between devices:
    tensor_gpu = tensor_cpu.to(device)
    tensor_gpu = tensor_cpu.cuda()
    tensor_cpu = tensor_gpu.cpu()

Host-to-device transfers are expensive. Minimize them by:
- Creating tensors directly on the target device
- Using pin_memory=True in DataLoader for faster CPU->GPU transfers
- Batching transfers instead of transferring one item at a time

Checking device:
    tensor.device         
    tensor.is_cuda        
    torch.cuda.current_device()  
        """
    },
    {
        "title": "Model Performance Optimization Tips",
        "url": "https://pytorch.org/docs/stable/notes/performance.html",
        "text": """
PyTorch Performance Tuning Guide

General Tips:

1. Use asynchronous data loading
   DataLoader with num_workers > 0 overlaps data loading with GPU compute.
   This hides I/O latency and keeps the GPU busy.

2. Enable cuDNN benchmark mode
   torch.backends.cudnn.benchmark = True
   Finds the best convolution algorithm for your input size.
   Only use when input size is fixed across batches.

3. Avoid unnecessary CPU-GPU synchronization
   Operations that cause sync: .item(), .numpy(), print(tensor)
   These force the CPU to wait for GPU to finish, killing parallelism.

4. Use in-place operations carefully
   In-place ops save memory but can cause issues with autograd.
   Prefer out-of-place when in doubt.

5. Batch matrix operations
   Instead of looping over batch items, use batched ops:
   torch.bmm() for batched matrix multiply
   torch.vmap() for vectorized operations

6. Gradient accumulation for large effective batch sizes
   optimizer.zero_grad()
   for i, batch in enumerate(loader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i+1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

7. Use channels_last memory format for CNNs
   model = model.to(memory_format=torch.channels_last)
   input = input.to(memory_format=torch.channels_last)
   This can improve performance on modern GPUs with NHWC kernels.

8. Profile before optimizing
   Always profile first to find the actual bottleneck.
   Optimizing the wrong thing wastes time.
        """
    },
    {
        "title": "torch.nn.Conv2d and Convolution Performance",
        "url": "https://pytorch.org/docs/stable/nn.html",
        "text": """
Convolution Layer Performance

torch.nn.Conv2d is often the most compute-intensive layer in CNNs.

Constructor:
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1, bias=True)

Performance tips:

1. Use GPU - convolutions are massively parallelized on CUDA
   CPU convolution (aten::_slow_conv2d_forward) is much slower.
   Always move model to GPU for training.

2. Enable cuDNN
   torch.backends.cudnn.enabled = True  (default)
   torch.backends.cudnn.benchmark = True  (for fixed input sizes)

3. Depthwise convolutions for efficiency
   groups=in_channels makes it a depthwise convolution
   Much fewer parameters and FLOPs than standard convolution

4. Kernel size matters
   3x3 convolutions are highly optimized in cuDNN
   Large kernels (7x7, 11x11) are much slower per parameter

5. Channels last format
   input = input.contiguous(memory_format=torch.channels_last)
   Can give 20-30% speedup on NVIDIA GPUs

Memory usage:
    output_size = batch * out_channels * H_out * W_out * 4 bytes (float32)
    H_out = (H_in + 2*padding - kernel_size) / stride + 1

Slow conv warning:
If you see aten::_slow_conv2d_forward in profiler output, your
convolution is running on CPU. Move model and inputs to CUDA.
        """
    },
    {
        "title": "Autograd and Gradient Computation",
        "url": "https://pytorch.org/docs/stable/autograd.html",
        "text": """
Autograd - Automatic Differentiation

PyTorch's autograd engine automatically computes gradients.
Every tensor operation is recorded in a computation graph when
requires_grad=True.

Disabling gradient tracking:
    
    with torch.no_grad():
        y = model(x)

    
    tensor = tensor.detach()

   
    for param in model.base_layers.parameters():
        param.requires_grad = False

Memory implications:
Autograd stores intermediate activations for backward pass.
This is the main reason training uses more memory than inference.
For a model with N layers, roughly N activation tensors are stored.

Gradient accumulation patterns:
    optimizer.zero_grad()       
    loss.backward()          
    optimizer.step()     

retain_graph=True:
Normally the computation graph is freed after backward().
Use retain_graph=True only when you need to call backward() multiple times:
    loss.backward(retain_graph=True)

create_graph=True:
For higher-order gradients (gradient of gradient).
Very memory intensive, use only when needed.

Detecting gradient issues:
    torch.autograd.set_detect_anomaly(True)
Raises an error with stack trace when NaN/Inf appears in gradients.
Slow - only use for debugging.
        """
    },
    {
        "title": "Mixed Precision Training with AMP",
        "url": "https://pytorch.org/docs/stable/amp.html",
        "text": """
Automatic Mixed Precision (AMP)

AMP uses float16 for most operations and float32 for numerically sensitive ones.
This gives 2-3x speedup on modern GPUs with Tensor Cores (V100, A100, RTX series).

Basic usage:
    scaler = torch.cuda.amp.GradScaler()

    for batch in loader:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

Why GradScaler?
float16 has limited range. Small gradients underflow to zero.
GradScaler multiplies loss by a scale factor before backward(),
then divides before optimizer step. Automatically adjusts scale factor.

autocast rules:
- Matrix multiplies and convolutions: float16
- Reductions, batch norm, softmax: float32
- You don't need to manually cast anything inside autocast

Memory savings:
- float16 uses 2 bytes vs float32's 4 bytes
- Allows ~2x larger batch sizes
- Activations stored in float16 during forward pass

Supported GPU architectures:
- NVIDIA Volta (V100) and newer for float16
- NVIDIA Ampere (A100) and newer for bfloat16
- bfloat16 preferred for training stability: dtype=torch.bfloat16

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(input)
        """
    },
    {
        "title": "PyTorch Optimization and Learning Rate Scheduling",
        "url": "https://pytorch.org/docs/stable/optim.html",
        "text": """
torch.optim - Optimization Algorithms

Common optimizers and when to use them:

SGD with momentum:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
Best for: image classification, when you have time to tune lr schedule.

Adam:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )
Best for: NLP, transformers, when you want faster convergence.

AdamW (Adam with decoupled weight decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
Preferred over Adam for transformers. Fixes weight decay implementation.

Performance tips:
1. zero_grad(set_to_none=True) is faster than zero_grad()
   Sets gradients to None instead of zeros, saves a memset.

2. Fused optimizer kernels (PyTorch 2.0+)
   optimizer = torch.optim.Adam(params, lr=1e-3, fused=True)
   Runs optimizer step in a single CUDA kernel. Faster on GPU.

3. Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    scheduler.step()

4. Gradient clipping prevents exploding gradients:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        """
    },
    {
        "title": "Multiprocessing and Shared Memory",
        "url": "https://pytorch.org/docs/stable/multiprocessing.html",
        "text": """
PyTorch Multiprocessing

PyTorch uses multiprocessing for parallel data loading (DataLoader workers)
and distributed training.

DataLoader workers share tensors via shared memory (not copying).
This is why pin_memory and shared memory matter for performance.

Starting method:
    torch.multiprocessing.set_start_method('spawn') 
    torch.multiprocessing.set_start_method('fork')   

CUDA and multiprocessing:
- CUDA tensors cannot be passed between processes with 'fork'
- Use 'spawn' or 'forkserver' when using CUDA in worker processes
- DataLoader workers should not use CUDA directly

Sharing tensors between processes:
    tensor = torch.zeros(10)
    tensor.share_memory_() 
    

Common multiprocessing errors:
1. "RuntimeError: CUDA error: initialization error"
   Cause: CUDA initialized before fork()
   Fix: Use spawn start method

2. DataLoader worker crashes
   Cause: Exception in worker not propagated properly
   Fix: Set num_workers=0 temporarily to debug in main process

3. Memory leak with num_workers > 0
   Cause: Workers holding references to dataset
   Fix: Use persistent_workers=True or ensure dataset is picklable
        """
    },
    {
        "title": "Tensor Memory Layout and Contiguous Tensors",
        "url": "https://pytorch.org/docs/stable/tensors.html",
        "text": """
Tensor Memory Layout

PyTorch tensors are stored in contiguous blocks of memory by default.
Operations like transpose, permute, and slice create views with non-contiguous
memory layout, which can hurt performance.

Checking contiguity:
    tensor.is_contiguous() 

Making contiguous:
    tensor = tensor.contiguous() 

Why contiguity matters:
- Many CUDA kernels require contiguous input
- Non-contiguous tensors may trigger implicit copies
- view() requires contiguous tensor; reshape() works on either

Memory formats:
- torch.contiguous_format: default NCHW for 4D tensors (batch, channel, H, W)
- torch.channels_last: NHWC format, faster for CNNs on modern GPUs

Converting to channels_last:
    tensor = tensor.to(memory_format=torch.channels_last)
    model = model.to(memory_format=torch.channels_last)

Strides:
Strides define how many elements to skip to move one step in each dimension.
Contiguous tensor strides decrease from left to right.
    tensor = torch.zeros(3, 4)
    tensor.stride()  

.item() performance:
Calling .item() on a CUDA tensor forces synchronization between CPU and GPU.
The CPU must wait for all pending GPU operations to complete.
Avoid in hot loops. Accumulate tensors, call .item() once at the end.
        """
    },
]


def scrape_all_pages() -> list[ScrapedPage]:
    """
    Return curated PyTorch knowledge base as ScrapedPage objects.
    No network requests needed.
    """
    pages = []
    for i, entry in enumerate(PYTORCH_KNOWLEDGE_BASE):
        text = entry["text"].strip()
        page = ScrapedPage(
            url=entry["url"],
            title=entry["title"],
            text=text,
            char_count=len(text),
        )
        pages.append(page)
        print(f"  [{i+1}/{len(PYTORCH_KNOWLEDGE_BASE)}] ✓ '{page.title}' — {page.char_count:,} chars")

    return pages