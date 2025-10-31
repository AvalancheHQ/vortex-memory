// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaView;
use vortex_error::VortexExpect;
use vortex_layout::segments::{GpuSegmentFuture, GpuSegmentSource, SegmentId};

use crate::SegmentSpec;

pub struct FileGpuSegmentSource {
    segments: Arc<[SegmentSpec]>,
    contents: CudaView<'static, u8>,
}

impl FileGpuSegmentSource {
    pub fn new(segments: Arc<[SegmentSpec]>, contents: CudaView<'static, u8>) -> Self {
        FileGpuSegmentSource { segments, contents }
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

        Ok(self.contents.slice(off_usize..(off_usize + len_usize)))
    }
}
