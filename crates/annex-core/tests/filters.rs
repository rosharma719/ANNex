use annex::payload_storage::filters::{Filter, evaluate_filter};
use annex::utils::payload::{Payload, PayloadValue};

#[test]
fn test_match_filter_true() {
    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("robot".into()));

    let filter = Filter::Match {
        key: "kind".into(),
        value: PayloadValue::Str("robot".into()),
    };

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}

#[test]
fn test_match_filter_false() {
    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("robot".into()));

    let filter = Filter::Match {
        key: "kind".into(),
        value: PayloadValue::Str("animal".into()),
    };

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(!result);
}

#[test]
fn test_not_filter() {
    let mut payload = Payload::default();
    payload.set("status", PayloadValue::Str("inactive".into()));

    let filter = Filter::Not(Box::new(Filter::Match {
        key: "status".into(),
        value: PayloadValue::Str("active".into()),
    }));

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}

#[test]
fn test_and_or_filter() {
    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("robot".into()));
    payload.set("active", PayloadValue::Bool(true));

    let filter = Filter::And(vec![
        Filter::Or(vec![
            Filter::Match {
                key: "kind".into(),
                value: PayloadValue::Str("robot".into()),
            },
            Filter::Match {
                key: "kind".into(),
                value: PayloadValue::Str("animal".into()),
            },
        ]),
        Filter::Match {
            key: "active".into(),
            value: PayloadValue::Bool(true),
        },
    ]);

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}
