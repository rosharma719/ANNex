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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Payload(pub HashMap<String, PayloadValue>);

//Wrapper around a HashMap<String, PayloadValue>
impl Payload {
    //Setter for key and value
    pub fn set(&mut self, key: &str, value: PayloadValue) {
        self.0.insert(key.to_string(), value);
    }

    //Getter for value given key
    pub fn get(&self, key: &str) -> Option<&PayloadValue> {
        self.0.get(key)
    }

    pub fn compare_field(
        &self,
        field: &str,
        op: ScalarComparisonOp,
        other: &PayloadValue,
    ) -> Result<bool, DBError> {
        //println!("Evaluating comparison for field: '{}', operation: {:?}, against value: {:?}", field, op, other);

        match self.get(field) {
            Some(value) => {
                //println!("Payload contains key '{}'. Value: {:?}", field, value);

                match (value, other) {
                    // Handle ListStr comparison
                    (PayloadValue::ListStr(l), PayloadValue::ListStr(o)) => {
                        //println!("Comparing ListStr with ListStr: left = {:?}, right = {:?}", l, o);
                        match op {
                            ScalarComparisonOp::Eq => {
                                //println!("Performing Eq comparison between lists.");
                                let result = l == o;
                                log::debug!(target: "payload", "Result of Eq comparison: {}", result);
                                Ok(result)
                            }
                            ScalarComparisonOp::Neq => {
                                //println!("Performing Neq comparison between lists.");
                                let result = l != o;
                                //println!("Result of Neq comparison: {}", result);
                                Ok(result)
                            }
                            _ => {
                                log::warn!(target: "payload", "Invalid operation for ListStr.");
                                Err(DBError::InvalidPayload("Invalid operation for ListStr".into()))
                            }
                        }
                    },(PayloadValue::ListStr(l), PayloadValue::Str(s)) => {
                        //println!("Comparing ListStr with Str: list = {:?}, string = {:?}", l, s);
                        match op {
                            ScalarComparisonOp::Eq => {
                                //println!("Checking if list contains the string '{}'", s);
                                let result = l.contains(s);
                                //println!("Result of containment check: {}", result);
                                Ok(result)
                            },
                            ScalarComparisonOp::Neq => {
                                //println!("Checking if list does NOT contain the string '{}'", s);
                                let result = !l.contains(s);
                                //println!("Result of non-containment check: {}", result);
                                Ok(result)
                            },
                            _ => {
                                //println!("Invalid operation for ListStr and Str.");
                                Err(DBError::InvalidPayload("Invalid operation for ListStr and Str".into()))
                            }
                        }
                    }
                    ,
                    // Handle other types like Int, Float, etc.
                    _ => {
                        //println!("Performing scalar comparison for field: '{}'", field);
                        let result = value.compare_scalar(op, other);
                        match result {
                            Some(res) => {
                                //println!("Result of scalar comparison: {}", res);
                                Ok(res)
                            },
                            None => {
                                //println!("Type mismatch for field '{}'.", field);
                                Err(DBError::InvalidPayload(format!("Type mismatch for field: {field}")))
                            }
                        }
                    }
                }
            }
            None => {
                log::debug!(target: "payload", "Field '{}' is missing from the payload.", field);
                Ok(false)
            }
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

impl Default for Payload {
    fn default() -> Self {
        Payload(HashMap::new())
    }
}
