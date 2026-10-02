//! Payload implementation with setter and comparison support
use crate::utils::errors::DBError;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayloadValue {
    Int(i64),
    Float(OrderedFloat<f64>),
    Str(String),
    Bool(bool),
    ListInt(Vec<i64>),
    ListFloat(Vec<OrderedFloat<f64>>),
    ListStr(Vec<String>),
    ListBool(Vec<bool>),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Payload(pub HashMap<String, PayloadValue>);

impl Payload {
    pub fn set(&mut self, key: &str, value: PayloadValue) {
        self.0.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&PayloadValue> {
        self.0.get(key)
    }

    pub fn compare_field(
        &self,
        field: &str,
        op: ScalarComparisonOp,
        other: &PayloadValue,
    ) -> Result<bool, DBError> {
        let Some(value) = self.get(field) else {
            log::debug!(target: "payload", "Field '{}' is missing from the payload.", field);
            return Ok(false);
        };
        match (value, other, op) {
            (
                PayloadValue::ListStr(values),
                PayloadValue::ListStr(expected),
                ScalarComparisonOp::Eq,
            ) => Ok(values == expected),
            (
                PayloadValue::ListStr(values),
                PayloadValue::ListStr(expected),
                ScalarComparisonOp::Neq,
            ) => Ok(values != expected),
            (
                PayloadValue::ListStr(values),
                PayloadValue::Str(expected),
                ScalarComparisonOp::Eq,
            ) => Ok(values.contains(expected)),
            (
                PayloadValue::ListStr(values),
                PayloadValue::Str(expected),
                ScalarComparisonOp::Neq,
            ) => Ok(!values.contains(expected)),
            (PayloadValue::ListStr(_), _, _) => Err(DBError::InvalidPayload(
                "Invalid operation for ListStr".into(),
            )),
            _ => value.compare_scalar(op, other).ok_or_else(|| {
                DBError::InvalidPayload(format!("Type mismatch for field: {field}"))
            }),
        }
    }

    pub fn evaluate_list_field(&self, field: &str, op: ListQueryOp) -> Result<bool, DBError> {
        match self.get(field) {
            Some(value) => value.evaluate_list_query(op).ok_or_else(|| {
                DBError::InvalidPayload(format!("Invalid list operation on field: {field}"))
            }),
            None => Err(DBError::InvalidPayload(format!("Missing field: {field}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarComparisonOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone)]
pub enum ListQueryOp<'a> {
    Contains(&'a PayloadValue),
    Equals(&'a PayloadValue),
    Length(ScalarComparisonOp, usize),
    ElementCompare(usize, ScalarComparisonOp, &'a PayloadValue),
}

impl PayloadValue {
    pub fn compare_scalar(&self, op: ScalarComparisonOp, other: &PayloadValue) -> Option<bool> {
        use PayloadValue::*;
        use ScalarComparisonOp::*;

        match (self, other) {
            (Int(a), Int(b)) => Some(match op {
                Eq => a == b,
                Neq => a != b,
                Lt => a < b,
                Lte => a <= b,
                Gt => a > b,
                Gte => a >= b,
            }),
            (Float(a), Float(b)) => Some(match op {
                Eq => a == b,
                Neq => a != b,
                Lt => a < b,
                Lte => a <= b,
                Gt => a > b,
                Gte => a >= b,
            }),
            (Str(a), Str(b)) => Some(match op {
                Eq => a == b,
                Neq => a != b,
                Lt => a < b,
                Lte => a <= b,
                Gt => a > b,
                Gte => a >= b,
            }),
            (Bool(a), Bool(b)) => Some(match op {
                Eq => a == b,
                Neq => a != b,
                _ => return None,
            }),
            _ => None,
        }
    }

    pub fn evaluate_list_query(&self, op: ListQueryOp) -> Option<bool> {
        use ListQueryOp::*;
        use PayloadValue::*;

        match op {
            Contains(val) => match (self, val) {
                (ListInt(vec), Int(x)) => Some(vec.contains(x)),
                (ListFloat(vec), Float(x)) => Some(vec.contains(x)),
                (ListStr(vec), Str(x)) => Some(vec.contains(x)),
                (ListBool(vec), Bool(x)) => Some(vec.contains(x)),
                _ => None,
            },
            Equals(val) => {
                if std::mem::discriminant(self) == std::mem::discriminant(val) {
                    Some(self == val)
                } else {
                    None
                }
            }
            Length(cmp_op, len) => match self {
                ListInt(vec) => Some(Self::compare_len(vec.len(), cmp_op, len)),
                ListFloat(vec) => Some(Self::compare_len(vec.len(), cmp_op, len)),
                ListStr(vec) => Some(Self::compare_len(vec.len(), cmp_op, len)),
                ListBool(vec) => Some(Self::compare_len(vec.len(), cmp_op, len)),
                _ => None,
            },
            ElementCompare(index, cmp_op, val) => match (self, val) {
                (ListInt(vec), Int(x)) => vec
                    .get(index)
                    .map(|v| Self::compare_scalar_static(v, cmp_op, x)),
                (ListFloat(vec), Float(x)) => vec
                    .get(index)
                    .map(|v| Self::compare_scalar_static(v, cmp_op, x)),
                (ListStr(vec), Str(x)) => vec
                    .get(index)
                    .map(|v| Self::compare_scalar_static(v, cmp_op, x)),
                (ListBool(vec), Bool(x)) => {
                    if matches!(cmp_op, ScalarComparisonOp::Eq | ScalarComparisonOp::Neq) {
                        vec.get(index)
                            .map(|v| Self::compare_scalar_static(v, cmp_op, x))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }

    fn compare_len(actual: usize, op: ScalarComparisonOp, expected: usize) -> bool {
        match op {
            ScalarComparisonOp::Eq => actual == expected,
            ScalarComparisonOp::Neq => actual != expected,
            ScalarComparisonOp::Lt => actual < expected,
            ScalarComparisonOp::Lte => actual <= expected,
            ScalarComparisonOp::Gt => actual > expected,
            ScalarComparisonOp::Gte => actual >= expected,
        }
    }
    fn compare_scalar_static<T: PartialOrd + PartialEq>(
        a: &T,
        op: ScalarComparisonOp,
        b: &T,
    ) -> bool {
        match op {
            ScalarComparisonOp::Eq => a == b,
            ScalarComparisonOp::Neq => a != b,
            ScalarComparisonOp::Lt => a < b,
            ScalarComparisonOp::Lte => a <= b,
            ScalarComparisonOp::Gt => a > b,
            ScalarComparisonOp::Gte => a >= b,
        }
    }
}
