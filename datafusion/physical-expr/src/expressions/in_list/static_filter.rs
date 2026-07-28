// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, BooleanArray, DictionaryArray, types::ArrowDictionaryKeyType};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::ArrowNativeType;
use datafusion_common::Result;

/// Maps predicate results for dictionary values back to the dictionary keys.
///
/// This is equivalent to `take(values_contains, dictionary.keys())`, but avoids
/// the setup and allocation overhead of the generic Arrow `take` kernel for the
/// common Boolean predicate case.
pub(super) fn remap_dictionary_predicate<K>(
    dictionary: &DictionaryArray<K>,
    values_contains: &BooleanArray,
) -> BooleanArray
where
    K: ArrowDictionaryKeyType,
{
    let keys = dictionary.keys();
    let values = BooleanBuffer::collect_bool(keys.len(), |index| {
        keys.is_valid(index) && values_contains.value(keys.value(index).as_usize())
    });

    let nulls = match (keys.null_count(), values_contains.null_count()) {
        (0, 0) => None,
        (_, 0) => keys.nulls().cloned(),
        _ => Some(NullBuffer::new(BooleanBuffer::collect_bool(
            keys.len(),
            |index| {
                keys.is_valid(index)
                    && values_contains.is_valid(keys.value(index).as_usize())
            },
        ))),
    };

    BooleanArray::new(values, nulls)
}

/// Trait for InList static filters.
///
/// Static filters store a pre-computed set of values (the haystack) and check
/// whether needle values are contained in that set. The haystack is always
/// represented in its non-dictionary (value) type. Dictionary haystacks are
/// flattened via `cast()` before construction.
///
/// Dictionary-encoded needles are unwrapped inside `contains()` and
/// evaluated against the dictionary's values.
pub(super) trait StaticFilter {
    fn null_count(&self) -> usize;

    /// Checks if values in `v` (needle) are contained in this filter's
    /// haystack. `v` may be dictionary-encoded, in which case the
    /// implementation unwraps the dictionary and operates on its values.
    fn contains(&self, v: &dyn Array, negated: bool) -> Result<BooleanArray>;
}
