// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use bitvec::macros::internal::funty::Fundamental;
use num_traits::AsPrimitive;
use parking_lot::Mutex;
use vortex::array::Array;
use vortex::array::ToCanonical;
use vortex::array::VectorExecutor;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::mask::MaskExecutor;
use vortex::array::vectors::VectorIntoArray;
use vortex::compute;
use vortex::compute2::take::Take;
use vortex::dtype::IntegerPType;
use vortex::dtype::PTypeDowncastExt;
use vortex::dtype::match_each_integer_ptype;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::session::VortexSession;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PVector;

use crate::duckdb::SelectionVector;
use crate::duckdb::Vector;
use crate::duckdb::{LogicalType, ReusableDict};
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::cache::ConversionCache;
use crate::exporter::constant;
use crate::exporter::new_array_exporter;
use crate::exporter::new_vector_array_exporter;

struct DictExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
    cache_id: u64,
    value_id: usize,
}

pub(crate) fn new_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    // Whether to return a duckdb flat vector or not.
    mut flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_type: LogicalType = values.dtype().try_into()?;
    if let Some(constant) = values.as_opt::<ConstantVTable>() {
        return constant::new_exporter_with_mask(
            &ConstantArray::new(constant.scalar().clone(), array.codes().len()),
            array.codes().validity_mask(),
            cache,
        );
    }

    let codes_mask = array.codes().validity_mask();

    match codes_mask {
        Mask::AllTrue(_) => {}
        Mask::AllFalse(len) => return Ok(all_invalid::new_exporter(len, &values_type)),
        Mask::Values(_) => {
            // duckdb cannot have a dictionary with validity in the codes, so flatten the array and
            // apply the validity mask there.
            flatten = true;
        }
    }

    let values_key = Arc::as_ptr(values).addr();
    let codes = array.codes().to_primitive();

    let reusable_dict = if flatten {
        let canonical = cache
            .canonical_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());
        let canonical = match canonical {
            Some(c) => c,
            None => {
                let canonical = values.to_canonical();
                cache
                    .canonical_cache
                    .insert(values_key, (values.clone(), canonical.clone()));
                canonical
            }
        };
        return new_array_exporter(&compute::take(canonical.as_ref(), codes.as_ref())?, cache);
    } else {
        // Check if we have a cached vector and extract it if we do.
        let reusable_dict = cache
            .values_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());

        match reusable_dict {
            Some(reusable_dict) => reusable_dict,
            None => {
                // Create a new reusable dictionary for the values.
                let reusable_dict = ReusableDict::new(values.dtype().try_into()?, values.len());
                let mut dict_vector = reusable_dict.vector();
                new_array_exporter(values, cache)?.export(0, values.len(), &mut dict_vector)?;

                cache
                    .values_cache
                    .insert(values_key, (values.clone(), reusable_dict.clone()));

                reusable_dict
            }
        }
    };

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values: reusable_dict,
            codes,
            codes_type: PhantomData::<I>,
            cache_id: cache.instance_id(),
            value_id: values_key,
        }))
    })
}

impl<I: IntegerPType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Create a selection vector from the codes.
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut_sel_vec = unsafe { sel_vec.as_slice_mut(len) };
        for (dst, src) in mut_sel_vec.iter_mut().zip(
            self.codes.as_slice::<I>()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        ) {
            *dst = src
        }

        vector.reuse_dictionary(&self.values, &sel_vec);

        Ok(())
    }
}

struct DictVectorExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PVector<I>,
    cache_id: u64,
    value_id: usize,
}

pub(crate) fn new_vector_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    session: &VortexSession,
    // Whether to return a duckdb flat vector or not.
    mut flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_type: LogicalType = values.dtype().try_into()?;
    if let Some(constant) = values.as_opt::<ConstantVTable>() {
        return constant::new_exporter_with_mask(
            &ConstantArray::new(constant.scalar().clone(), array.codes().len()),
            array.codes().is_null()?.not()?.execute_mask(session)?,
            cache,
        );
    }

    let codes = array.codes().execute_vector(session)?.into_primitive();

    match codes.validity() {
        Mask::AllTrue(_) => {}
        Mask::AllFalse(len) => {
            return Ok(all_invalid::new_exporter(*len, &values_type));
        }
        Mask::Values(_) => {
            // duckdb cannot have a dictionary with validity in the codes, so flatten the array and
            // apply the validity mask there.
            flatten = true;
        }
    }

    let values_key = Arc::as_ptr(values).addr();

    let reusable_dict = if flatten {
        let values = cache.vector_cache.get(&values_key);
        let values = match values {
            Some(c) => c.value().1.clone(),
            None => {
                let value_vector = array.values().execute_vector(session)?;
                cache
                    .vector_cache
                    .insert(values_key, (array.values().clone(), value_vector.clone()));
                value_vector
            }
        };
        return new_vector_array_exporter(
            Take::take(values, &codes).into_array(array.dtype()),
            cache,
            session,
        );
    } else {
        // Check if we have a cached vector and extract it if we do.
        let reusable_dict = cache
            .values_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());

        match reusable_dict {
            Some(reusable_dict) => reusable_dict,
            None => {
                // Create a new reusable dictionary for the values.
                let reusable_dict = ReusableDict::new(values.dtype().try_into()?, values.len());
                let mut dict_vector = reusable_dict.vector();
                new_array_exporter(values, cache)?.export(0, values.len(), &mut dict_vector)?;

                cache
                    .values_cache
                    .insert(values_key, (values.clone(), reusable_dict.clone()));

                reusable_dict
            }
        }
    };

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictVectorExporter {
            values: reusable_dict,
            codes: codes.downcast::<I>(),
            cache_id: cache.instance_id(),
            value_id: values_key,
        }))
    })
}

impl<I: IntegerPType + AsPrimitive<u32>> ColumnExporter for DictVectorExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Create a selection vector from the codes.
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut_sel_vec = unsafe { sel_vec.as_slice_mut(len) };
        for (dst, src) in mut_sel_vec.iter_mut().zip(
            self.codes.as_ref()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        ) {
            *dst = src
        }

        vector.reuse_dictionary(&self.values, &sel_vec);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::buffer::Buffer;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ColumnExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::dict::new_exporter_with_flatten;
    use crate::exporter::dict::new_vector_exporter_with_flatten;
    use crate::exporter::new_array_exporter;

    pub(crate) fn new_exporter(
        array: &DictArray,
        cache: &ConversionCache,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        new_exporter_with_flatten(array, cache, false)
    }

    #[test]
    fn test_constant_dict() {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );
    }

    #[test]
    fn test_constant_dict_vector() {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        let session = VortexSession::default();
        new_vector_exporter_with_flatten(&arr, &ConversionCache::default(), &session, false)
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );
    }

    #[test]
    fn test_constant_dict_vector_null() {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None::<u32>, None]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        let session = VortexSession::default();
        new_vector_exporter_with_flatten(&arr, &ConversionCache::default(), &session, false)
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 2 = [ NULL]
"#
        );
    }

    #[test]
    fn test_nullable_dict() {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(1)]).into_array(),
            PrimitiveArray::from_option_iter([Some(10), None]).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        // some-invalid codes cannot be exported as a dictionary.
        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
        );

        let mut flat_chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_array_exporter(arr.to_canonical().as_ref(), &ConversionCache::default())
            .unwrap()
            .export(0, 3, &mut flat_chunk.get_vector(0))
            .unwrap();
        flat_chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&flat_chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
        )
    }

    #[ignore = "TODO(connor)[4809]: Exporters do not correctly handle empty vectors"]
    #[test]
    fn test_export_empty_dict() {
        let arr = DictArray::new(
            Buffer::<u32>::empty().into_array(),
            Buffer::<u32>::empty().into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 0, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(0);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 0 = [ ]
"#
        );
    }
}
