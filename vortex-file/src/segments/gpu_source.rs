// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaView};
use vortex_error::VortexExpect;
use vortex_layout::segments::{GpuSegmentFuture, GpuSegmentSource, SegmentId};

use crate::SegmentSpec;

pub struct FileGpuSegmentSource {
    segments: Arc<[SegmentSpec]>,
    contents: CudaSlice<u8>,
}

impl FileGpuSegmentSource {
    pub fn new(segments: Arc<[SegmentSpec]>, contents: CudaSlice<u8>) -> Self {
        FileGpuSegmentSource { segments, contents }
    }

    #[inline]
    fn contents(&self) -> CudaView<'static, u8> {
        // SAFETY: This is only fine because the callers of this function will be dropped before this segment source is dropped
        unsafe {
            mem::transmute::<CudaView<'_, u8>, CudaView<'static, u8>>(self.contents.as_view())
        }
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

        Ok(self.contents().slice(off_usize..(off_usize + len_usize)))
    }
}
