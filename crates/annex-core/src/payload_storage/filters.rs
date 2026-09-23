use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Filter {
    Match {
        key: String,
        value: PayloadValue,
    },
    Compare {
        key: String,
        op: ScalarComparisonOp,
        value: PayloadValue,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Evaluates whether a given payload satisfies the filter condition.
pub fn evaluate_filter(filter: &Filter, payload: &Payload) -> Result<bool, DBError> {
    match filter {
        Filter::Match { key, value } => {
            //println!("Evaluating Match filter: key = {}, value = {:?}", key, value);
            match payload.get(key) {
                Some(actual) => {
                    //println!("Payload value for key '{}': {:?}", key, actual);
                    Ok(actual == value)
                }
                None => {
                    log::debug!(target: "filter", "No payload found for key '{}'. Returning false.", key);
                    Ok(false)
                }
            }
        }

        Filter::Compare { key, op, value } => {
            //println!("Evaluating Compare filter: key = {}, op = {:?}, value = {:?}", key, op, value);
            payload.compare_field(key, *op, value)
        }

        Filter::And(conditions) => {
            //println!("Evaluating AND filter with {} conditions.", conditions.len());
            for cond in conditions {
                if !evaluate_filter(cond, payload)? {
                    //println!("Condition failed in AND filter. Returning false.");
                    return Ok(false);
                }
            }
            //println!("All conditions in AND filter passed. Returning true.");
            Ok(true)
        }

        Filter::Or(conditions) => {
            //println!("Evaluating OR filter with {} conditions.", conditions.len());
            for cond in conditions {
                if evaluate_filter(cond, payload)? {
                    log::debug!(target: "filter", "Condition passed in OR filter. Returning true.");
                    return Ok(true);
                }
            }
            //println!("No conditions in OR filter passed. Returning false.");
            Ok(false)
        }

        Filter::Not(inner) => {
            //println!("Evaluating NOT filter.");
            let result = evaluate_filter(inner, payload)?;
            //println!("NOT filter result: {}", result);
            Ok(!result)
        }
    }
}
