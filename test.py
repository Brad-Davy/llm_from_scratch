import torch

def check_gpu():
    print("Checking for GPU availability...\n")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"  - GPU device count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Simple tensor test on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
            z = x + y
            print(f"  - Tensor addition successful: {z}")
            print("🎉 PyTorch can run on the GPU correctly.")
        except Exception as e:
            print(f"⚠️ Error during tensor operation on GPU: {e}")
    else:
        print("❌ CUDA is not available. PyTorch is using CPU.")

if __name__ == "__main__":
    check_gpu()

