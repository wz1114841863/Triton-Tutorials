import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,  # Pointer of input vector x
    y_ptr,  # Pointer of input vector y
    output_ptr,  # Pointer of output vector
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # There are multiple programs in the grid
    # each program has a unique program id
    pid = tl.program_id(axis=0)  # Use a 1D launch grid so axis is 0

    # Compute the starting index for this program
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Create a mask to handle out-of-bounds

    # Load x and y from DRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y

    # Store the result to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert (
        x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    ), "Tensors must be on the correct device"
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="vector_addition_performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )

    benchmark.run(
        print_data=True,
        show_plots=True,
        save_path="./tutorials/results/vector_addition/",
    )
