use annex::utils::errors::DBError;
use annex::utils::payload::*;
use ordered_float::OrderedFloat;

#[test]
fn test_scalar_comparisons() {
    let a = PayloadValue::Int(10);
    let b = PayloadValue::Int(20);
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Lt, &b), Some(true));

    let a = PayloadValue::Float(OrderedFloat(1.0));
    let b = PayloadValue::Float(OrderedFloat(1.0));
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Eq, &b), Some(true));

    let a = PayloadValue::Str("abc".into());
    let b = PayloadValue::Str("xyz".into());
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Lt, &b), Some(true));

    let a = PayloadValue::Bool(true);
    let b = PayloadValue::Bool(false);
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Neq, &b), Some(true));

    assert_eq!(
        a.compare_scalar(ScalarComparisonOp::Eq, &PayloadValue::Str("true".into())),
        None
    );
}

#[test]
fn test_list_contains() {
    let ints = PayloadValue::ListInt(vec![1, 2, 3]);
    let floats = PayloadValue::ListFloat(vec![OrderedFloat(0.1), OrderedFloat(0.2)]);
    let strs = PayloadValue::ListStr(vec!["a".into(), "b".into()]);
    let bools = PayloadValue::ListBool(vec![true, false]);

    assert_eq!(
        ints.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Int(2))),
        Some(true)
    );
    assert_eq!(
        floats.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Float(OrderedFloat(
            0.3
        )))),
        Some(false)
    );
    assert_eq!(
        strs.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Str("a".into()))),
        Some(true)
    );
    assert_eq!(
        bools.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Bool(false))),
        Some(true)
    );
}

#[test]
fn test_list_element_compare() {
    let ints = PayloadValue::ListInt(vec![5, 10, 15]);
    let floats = PayloadValue::ListFloat(vec![OrderedFloat(0.1), OrderedFloat(0.5)]);
    let strs = PayloadValue::ListStr(vec!["a".into(), "b".into()]);
    let bools = PayloadValue::ListBool(vec![true, false]);

    assert_eq!(
        ints.evaluate_list_query(ListQueryOp::ElementCompare(
            2,
            ScalarComparisonOp::Gt,
            &PayloadValue::Int(10)
        )),
        Some(true)
    );
    assert_eq!(
        floats.evaluate_list_query(ListQueryOp::ElementCompare(
            1,
            ScalarComparisonOp::Lte,
            &PayloadValue::Float(OrderedFloat(0.5))
        )),
        Some(true)
    );
    assert_eq!(
        strs.evaluate_list_query(ListQueryOp::ElementCompare(
            0,
            ScalarComparisonOp::Eq,
            &PayloadValue::Str("a".into())
        )),
        Some(true)
    );
    assert_eq!(
        bools.evaluate_list_query(ListQueryOp::ElementCompare(
            1,
            ScalarComparisonOp::Eq,
            &PayloadValue::Bool(false)
        )),
        Some(true)
    );
}

#[test]
fn test_list_length() {
    let vec = PayloadValue::ListStr(vec!["a".into(), "b".into(), "c".into()]);

    assert_eq!(
        vec.evaluate_list_query(ListQueryOp::Length(ScalarComparisonOp::Eq, 3)),
        Some(true)
    );
    assert_eq!(
        vec.evaluate_list_query(ListQueryOp::Length(ScalarComparisonOp::Gt, 1)),
        Some(true)
    );
}

#[test]
fn test_list_equality() {
    let a = PayloadValue::ListInt(vec![1, 2, 3]);
    let b = PayloadValue::ListInt(vec![1, 2, 3]);
    let c = PayloadValue::ListInt(vec![1, 2]);

    assert_eq!(a.evaluate_list_query(ListQueryOp::Equals(&b)), Some(true));
    assert_eq!(a.evaluate_list_query(ListQueryOp::Equals(&c)), Some(false));

    let wrong_type = PayloadValue::ListFloat(vec![
        OrderedFloat(1.0),
        OrderedFloat(2.0),
        OrderedFloat(3.0),
    ]);
    assert_eq!(
        a.evaluate_list_query(ListQueryOp::Equals(&wrong_type)),
        None
    );
}

#[test]
fn test_payload_set_and_get() {
    let mut payload = Payload::default();
    payload.set("flag", PayloadValue::Bool(true));
    assert_eq!(payload.get("flag"), Some(&PayloadValue::Bool(true)));
}

#[test]
fn test_payload_compare_field() {
    let mut payload = Payload::default();
    payload.set("x", PayloadValue::Int(42));

    let result = payload.compare_field("x", ScalarComparisonOp::Gte, &PayloadValue::Int(40));
    assert!(result.is_ok() && result.unwrap());

    // Missing fields evaluate to false rather than erroring, matching filter semantics.
    let missing = payload.compare_field("y", ScalarComparisonOp::Eq, &PayloadValue::Int(1));
    assert!(matches!(missing, Ok(false)));

    let wrong_type = payload.compare_field(
        "x",
        ScalarComparisonOp::Eq,
        &PayloadValue::Str("forty-two".into()),
    );
    assert!(matches!(wrong_type, Err(DBError::InvalidPayload(_))));
}

#[test]
fn test_payload_evaluate_list_field_errors() {
    let mut payload = Payload::default();
    payload.set("tags", PayloadValue::ListStr(vec!["a".into(), "b".into()]));

    let ok = payload.evaluate_list_field(
        "tags",
        ListQueryOp::Contains(&PayloadValue::Str("a".into())),
    );
    assert!(ok.is_ok() && ok.unwrap());

    let missing = payload.evaluate_list_field(
        "nonexistent",
        ListQueryOp::Contains(&PayloadValue::Str("a".into())),
    );
    assert!(matches!(missing, Err(DBError::InvalidPayload(_))));

    let wrong_type = payload.evaluate_list_field(
        "tags",
        ListQueryOp::Contains(&PayloadValue::Float(OrderedFloat(1.0))),
    );
    assert!(matches!(wrong_type, Err(DBError::InvalidPayload(_))));
}

#[test]
fn test_payload_overwrite() {
    let mut payload = Payload::default();
    payload.set("val", PayloadValue::Int(10));
    payload.set("val", PayloadValue::Int(20));

    assert_eq!(payload.get("val"), Some(&PayloadValue::Int(20)));
}

#[test]
fn test_payload_default_behavior() {
    let payload = Payload::default();
    assert!(payload.get("missing").is_none());
}

#[test]
fn test_list_element_compare_out_of_bounds() {
    let list = PayloadValue::ListInt(vec![1, 2]);

    let result = list.evaluate_list_query(ListQueryOp::ElementCompare(
        5,
        ScalarComparisonOp::Eq,
        &PayloadValue::Int(1),
    ));
    assert_eq!(result, None);
}

#[test]
fn test_list_element_compare_wrong_type() {
    let list = PayloadValue::ListStr(vec!["a".into()]);

    let result = list.evaluate_list_query(ListQueryOp::ElementCompare(
        0,
        ScalarComparisonOp::Eq,
        &PayloadValue::Bool(true),
    ));
    assert_eq!(result, None);
}
