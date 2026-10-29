// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion between protobuf `Struct`/`Value` and `serde_json::Value`, used
//! for the free-form JSON fields such as KV transfer parameters.

/// Largest integer exactly representable as `f64` (2^53 - 1). Integral
/// numbers within this bound are emitted as JSON integers so consumers
/// expecting ints (e.g. `image_grid_thw`) don't see `16.0`.
const MAX_SAFE_INTEGER_F64: f64 = ((1u64 << 53) - 1) as f64;

/// Convert a protobuf `Struct` into a JSON object.
pub fn proto_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields.iter().map(|(k, v)| (k.clone(), proto_value_to_json(v))).collect(),
    )
}

/// Convert a protobuf `Value` into JSON.
pub fn proto_value_to_json(v: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match v.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) if n.fract() == 0.0 && n.abs() <= MAX_SAFE_INTEGER_F64 => {
            serde_json::Value::Number(serde_json::Number::from(*n as i64))
        }
        Some(Kind::NumberValue(n)) => serde_json::json!(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => proto_struct_to_json(s),
    }
}

/// Convert a JSON object into a protobuf `Struct`; `None` for any other JSON value.
pub fn json_to_proto_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_proto_value(v))).collect(),
        }),
        _ => None,
    }
}

/// Convert JSON into a protobuf `Value`.
pub fn json_to_proto_value(v: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_proto_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_proto_value(v))).collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}
