// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::mem;
use std::sync::{Arc, OnceLock};

use cudarc::cufile::{Cufile, FileHandle};
use cudarc::driver::{CudaSlice, CudaStream, CudaView};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_err};
use vortex_layout::segments::{GpuSegmentFuture, GpuSegmentSource, SegmentId};

use crate::SegmentSpec;

pub struct FileGpuSegmentSource {
    segments: Arc<[SegmentSpec]>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    cu_file: Arc<Cufile>,
    file_handle: Arc<FileHandle>,
    contents: OnceLock<CudaSlice<u8>>,
    length: u64,
}

impl FileGpuSegmentSource {
    pub fn new(segments: Arc<[SegmentSpec]>, stream: Arc<CudaStream>, file: File) -> Self {
        let len = file.metadata().unwrap().len();
        let cu_file = Cufile::new()
            .map_err(|e| vortex_err!("cu file {e}"))
            .vortex_expect("Failed to create cufile");

        let file_handle = cu_file
            .register(file)
            .map_err(|e| vortex_err!("cu file register {e}"))
            .vortex_unwrap();

        FileGpuSegmentSource {
            segments,
            stream,
            cu_file,
            file_handle: Arc::new(file_handle),
            contents: OnceLock::new(),
            length: len,
        }
    }

    fn contents(&self) -> CudaView<'static, u8> {
        let contents_view = self
            .contents
            .get_or_init(|| {
                let mut cu_slice = unsafe { self.stream.alloc::<u8>(self.length as usize) }
                    .map_err(|e| vortex_err!("cu slice {e}"))
                    .vortex_expect("Failed to allocate cu slice");
                self.file_handle
                    .sync_read(0, &mut cu_slice)
                    .map_err(|e| vortex_err!("sync read {e}"))
                    .vortex_unwrap();
                cu_slice
            })
            .as_view();
        // SAFETY: This is only fine because the callers of this function will be dropped before this segment source is dropped
        unsafe { mem::transmute::<CudaView<'_, u8>, CudaView<'static, u8>>(contents_view) }
    }
}

impl GpuSegmentSource for FileGpuSegmentSource {
    fn request(&self, id: SegmentId) -> GpuSegmentFuture {
        let spec = self
            .segments
            .get(*id as usize)
            .vortex_expect("missing segment id")
            .clone();

        let off_usize = usize::try_from(spec.offset).vortex_expect("offset must fit usize");
        let len_usize = usize::try_from(spec.length).vortex_expect("length must fit usize");

        Ok(self.contents().slice(off_usize..len_usize))
    }
}
