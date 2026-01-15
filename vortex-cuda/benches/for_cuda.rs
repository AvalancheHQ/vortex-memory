#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use cudarc::driver::PushKernelArg;
use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::has_nvcc;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;
use vortex_session::VortexSession;

fn main() {
    warm_up_vtables();
    divan::main();
}

const BENCH_ARGS: &[usize] = &[10_000_000, 100_000_000];

#[divan::bench(args = BENCH_ARGS)]
fn frame_of_reference_kernel(bencher: Bencher, len: usize) {
    if !has_nvcc() {
        return;
    }

    bencher
        .with_inputs(|| {
            let primitive_array = PrimitiveArray::new(
                Buffer::from((0u32..len as u32).collect::<Vec<u32>>()),
                Validity::NonNullable,
            )
            .into_array();

            let for_offset = 10u32;

            let for_array = FoRArray::try_new(primitive_array, for_offset.into())
                .vortex_expect("failed to create FoR array");

            let encoded = for_array.encoded();
            let unpacked_array = encoded.to_primitive();
            let unpacked_slice = unpacked_array.as_slice::<u32>();

            let cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
                .vortex_expect("failed to create execution context");
            let device_data = cuda_ctx
                .to_device(unpacked_slice)
                .vortex_expect("Failed to copy to device");

            (for_array, for_offset, device_data, cuda_ctx)
        })
        .bench_refs(
            |&mut (ref for_array, ref reference, ref mut device_data, ref mut cuda_ctx)| {
                launch_kernel(for_array, *reference, device_data, cuda_ctx).unwrap();
            },
        );
}

fn launch_kernel(
    for_array: &FoRArray,
    reference: u32,
    device_data: &mut cudarc::driver::CudaSlice<u32>,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<()> {
    let array_len = for_array.len() as u64;
    vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype()],
        launch_args: [*device_data, reference, array_len],
        array_len: for_array.len()
    );

    cuda_ctx.synchronize()?;

    Ok(())
}
