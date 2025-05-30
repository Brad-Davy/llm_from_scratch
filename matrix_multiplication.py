import torch
import time

def run_matrix_multiplication(size=20000):
    print(f"🔢 Matrix size: {size} x {size}")

    # Generate random matrices on CPU
    A_cpu = torch.randn(size, size)
    B_cpu = torch.randn(size, size)

    # CPU multiplication
    start_cpu = time.time()
    C_cpu = torch.matmul(A_cpu, B_cpu)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"🖥️  CPU time: {cpu_time:.4f} seconds")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🟢 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

        # Move tensors to GPU
        A_gpu = A_cpu.to(device)
        B_gpu = B_cpu.to(device)

        # Warm up GPU
        _ = torch.matmul(A_gpu, B_gpu)

        # GPU multiplication
        torch.cuda.synchronize()
        start_gpu = time.time()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        end_gpu = time.time()
        gpu_time = end_gpu - start_gpu

        print(f"⚡ GPU time: {gpu_time:.4f} seconds")
        print(f"🚀 Speedup: {cpu_time / gpu_time:.2f}x")

        # Check for correctness
        C_gpu_cpu = C_gpu.cpu()
        max_diff = (C_cpu - C_gpu_cpu).abs().max().item()
        print(f"✅ Max difference between CPU and GPU result: {max_diff:.6e}")
    else:
        print("❌ CUDA is not available. Only CPU benchmark was run.")

if __name__ == "__main__":
    run_matrix_multiplication()

