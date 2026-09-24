use std::collections::HashSet;

use annex::payload_storage::stores::PayloadIndex;
use annex::utils::payload::{Payload, PayloadValue};
use ordered_float::OrderedFloat;

#[test]
fn test_index_insert_and_query() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("category", PayloadValue::Str("fruit".into()));
    payload.set("rank", PayloadValue::Int(1));
    payload.set("confidence", PayloadValue::Float(OrderedFloat(0.95)));
    payload.set("active", PayloadValue::Bool(true));

    index.insert(42, &payload);
    index.insert(43, &payload);

    assert_eq!(
        index
            .query_exact("category", &PayloadValue::Str("fruit".into()))
            .unwrap(),
        &HashSet::from([42, 43])
    );
    assert_eq!(
        index.query_exact("rank", &PayloadValue::Int(1)).unwrap(),
        &HashSet::from([42, 43])
    );
    assert_eq!(
        index
            .query_exact("confidence", &PayloadValue::Float(OrderedFloat(0.95)))
            .unwrap(),
        &HashSet::from([42, 43])
    );
    assert_eq!(
        index
            .query_exact("active", &PayloadValue::Bool(true))
            .unwrap(),
        &HashSet::from([42, 43])
    );
}

#[test]
fn test_index_removal() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("rank", PayloadValue::Int(99));

    index.insert(1, &payload);
    index.insert(2, &payload);

    assert_eq!(
        index
            .query_exact("rank", &PayloadValue::Int(99))
            .unwrap()
            .len(),
        2
    );

    index.remove(1, &payload);
    assert_eq!(
        index.query_exact("rank", &PayloadValue::Int(99)).unwrap(),
        &HashSet::from([2])
    );

    index.remove(2, &payload);
    assert!(index.query_exact("rank", &PayloadValue::Int(99)).is_none());
}

#[test]
fn test_non_indexed_types() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("list", PayloadValue::ListStr(vec!["a".into(), "b".into()]));
    payload.set("numbers", PayloadValue::ListInt(vec![1, 2, 3]));

    index.insert(99, &payload);

    assert!(
        index
            .query_exact("list", &PayloadValue::Str("a".into()))
            .is_none()
    );
    assert!(
        index
            .query_exact("numbers", &PayloadValue::Int(1))
            .is_none()
    );

    assert!(index.all_for_key("list").is_none());
    assert!(index.all_for_key("numbers").is_none());
}

#[test]
fn test_all_for_key() {
    let mut index = PayloadIndex::new();

    let mut p1 = Payload::default();
    p1.set("color", PayloadValue::Str("red".into()));

    let mut p2 = Payload::default();
    p2.set("color", PayloadValue::Str("blue".into()));

    let mut p3 = Payload::default();
    p3.set("color", PayloadValue::Str("red".into()));

    index.insert(1, &p1);
    index.insert(2, &p2);
    index.insert(3, &p3);

    let all = index.all_for_key("color").unwrap();
    assert_eq!(all, HashSet::from([1, 2, 3]));
}

#[test]
fn test_duplicate_inserts_are_idempotent() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("apple".into()));

    index.insert(1, &payload);
    index.insert(1, &payload); // Same point inserted again

    let result = index.query_exact("kind", &PayloadValue::Str("apple".into()));
    assert_eq!(result.unwrap(), &HashSet::from([1]));
}

#[test]
fn test_insert_same_key_different_values() {
    let mut index = PayloadIndex::new();

    let mut p1 = Payload::default();
    p1.set("group", PayloadValue::Str("A".into()));

    let mut p2 = Payload::default();
    p2.set("group", PayloadValue::Str("B".into()));

    index.insert(1, &p1);
    index.insert(2, &p2);

    assert_eq!(
        index
            .query_exact("group", &PayloadValue::Str("A".into()))
            .unwrap(),
        &HashSet::from([1])
    );
    assert_eq!(
        index
            .query_exact("group", &PayloadValue::Str("B".into()))
            .unwrap(),
        &HashSet::from([2])
    );
}

#[test]
fn test_query_nonexistent_key_or_value() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("status", PayloadValue::Str("ok".into()));
    index.insert(1, &payload);

    assert!(
        index
            .query_exact("nonexistent", &PayloadValue::Str("nope".into()))
            .is_none()
    );
    assert!(
        index
            .query_exact("status", &PayloadValue::Str("error".into()))
            .is_none()
    );
}
