//! Explicit, persisted text analysis for the lexical field.
//!
//! The default policy reproduces the historical tokenizer exactly: split on
//! non-alphanumeric characters and lowercase. English adds ASCII folding,
//! a stop-word list, Snowball stemming, and a 40-character token limit,
//! mirroring the analyzer defaults of comparable full-text engines. Policies
//! are part of the persisted collection configuration: the vocabulary is
//! built from analyzed text, so changing the analyzer requires a new
//! collection rather than silently mixing analysis regimes.
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

/// Stop-word removal policy. Only a built-in English list is offered today;
/// custom lists require persisted vocabularies and are future work.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Stopwords {
    #[default]
    None,
    English,
}

/// Tantivy-compatible English list: tokens are produced by splitting on
/// non-alphanumerics, so contractions never reach this filter intact.
const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct TextAnalyzer {
    /// Snowball English stemming.
    pub stem: bool,
    /// Stop-word removal policy.
    pub stopwords: Stopwords,
    /// Fold accents via NFKD decomposition before lowercasing.
    pub ascii_folding: bool,
    /// Truncate tokens longer than this many characters.
    pub max_token_length: Option<usize>,
}

impl TextAnalyzer {
    /// The historical ANNex policy: lowercase alphanumeric tokens only.
    pub fn plain() -> Self {
        Self::default()
    }
    /// Stemming, English stop words, ASCII folding, and a 40-character limit.
    pub fn english() -> Self {
        Self {
            stem: true,
            stopwords: Stopwords::English,
            ascii_folding: true,
            max_token_length: Some(40),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.max_token_length.is_some_and(|n| n == 0 || n > 256) {
            return Err("analyzer max_token_length must be in 1..=256".into());
        }
        Ok(())
    }
    /// Stable identity for provenance records.
    pub fn description(&self) -> String {
        if *self == Self::plain() {
            "unicode-alphanumeric-lowercase-v1".into()
        } else if *self == Self::english() {
            "unicode-alphanumeric-english-fold-stop-stem-40-v1".into()
        } else {
            serde_json::to_string(self).unwrap_or_else(|_| "custom".into())
        }
    }
}

/// Compiled analyzer. Cheap to clone; shared by ingest and query paths so
/// both sides of the lexical index always agree.
pub struct Analyzer {
    config: TextAnalyzer,
    stemmer: Option<Stemmer>,
}

impl Clone for Analyzer {
    fn clone(&self) -> Self {
        Self::new(self.config.clone())
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new(TextAnalyzer::plain())
    }
}

impl Analyzer {
    pub fn new(config: TextAnalyzer) -> Self {
        let stemmer = config.stem.then(|| Stemmer::create(Algorithm::English));
        Self { config, stemmer }
    }
    /// Analyze into term counts. Filter order mirrors comparable engines:
    /// tokenize, fold, lowercase, truncate, stop words, stem.
    pub fn analyze(&self, text: &str) -> BTreeMap<String, f32> {
        let mut counts = BTreeMap::new();
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            let mut token: Cow<str> = if self.config.ascii_folding {
                word.nfkd()
                    .filter(|c| !is_combining_mark(*c))
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
                    .into()
            } else {
                word.to_lowercase().into()
            };
            if let Some(limit) = self.config.max_token_length
                && token.chars().nth(limit).is_some()
            {
                token = token.chars().take(limit).collect::<String>().into();
            }
            if ENGLISH_STOPWORDS.binary_search(&token.as_ref()).is_ok()
                && self.config.stopwords == Stopwords::English
            {
                continue;
            }
            if let Some(stemmer) = &self.stemmer {
                token = stemmer.stem(token.as_ref()).into_owned().into();
            }
            *counts.entry(token.into_owned()).or_default() += 1.0;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(analyzer: &Analyzer, text: &str) -> Vec<String> {
        analyzer.analyze(text).into_keys().collect()
    }

    #[test]
    fn plain_reproduces_the_historical_tokenizer() {
        let analyzer = Analyzer::default();
        assert_eq!(
            keys(&analyzer, "Don't STOP. café's E123 — repair!"),
            ["café", "don", "e123", "repair", "s", "stop", "t"]
        );
    }

    #[test]
    fn plain_counts_repeated_terms() {
        let analyzer = Analyzer::default();
        let counts = analyzer.analyze("cat Cat dog cat");
        assert_eq!(counts["cat"], 3.0);
        assert_eq!(counts["dog"], 1.0);
    }

    #[test]
    fn english_folds_stems_and_removes_stop_words() {
        let analyzer = Analyzer::new(TextAnalyzer::english());
        let counts = analyzer.analyze("The runners are running toward the CAFÉs");
        assert!(counts.contains_key("run"));
        assert!(!counts.contains_key("runners"));
        assert!(counts.contains_key("cafe"));
        for stop in ["the", "are"] {
            assert!(!counts.contains_key(stop), "{stop} should be filtered");
        }
        assert!(counts.contains_key("toward"), "non-list words survive");
    }

    #[test]
    fn english_truncates_long_tokens_at_the_limit() {
        let analyzer = Analyzer::new(TextAnalyzer::english());
        let long = "x".repeat(41);
        assert_eq!(keys(&analyzer, &long), ["x".repeat(40)]);
    }

    #[test]
    fn folding_preserves_non_ascii_that_has_no_decomposition() {
        let analyzer = Analyzer::new(TextAnalyzer {
            ascii_folding: true,
            ..TextAnalyzer::plain()
        });
        assert_eq!(keys(&analyzer, "Ünïcödé ﬁ 東京"), ["fi", "unicode", "東京"]);
    }

    #[test]
    fn individual_options_compose() {
        let stem_only = Analyzer::new(TextAnalyzer {
            stem: true,
            ..TextAnalyzer::plain()
        });
        assert_eq!(keys(&stem_only, "the running"), ["run", "the"]);
        let stop_only = Analyzer::new(TextAnalyzer {
            stopwords: Stopwords::English,
            ..TextAnalyzer::plain()
        });
        assert_eq!(keys(&stop_only, "the running"), ["running"]);
    }

    #[test]
    fn stopword_list_is_sorted_for_binary_search() {
        assert!(ENGLISH_STOPWORDS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn config_validation_and_description() {
        assert!(TextAnalyzer::plain().validate().is_ok());
        assert!(TextAnalyzer::english().validate().is_ok());
        for bad in [
            TextAnalyzer {
                max_token_length: Some(0),
                ..TextAnalyzer::plain()
            },
            TextAnalyzer {
                max_token_length: Some(257),
                ..TextAnalyzer::plain()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert_eq!(
            TextAnalyzer::plain().description(),
            "unicode-alphanumeric-lowercase-v1"
        );
        assert!(
            TextAnalyzer::english()
                .description()
                .starts_with("unicode-alphanumeric-english")
        );
    }

    #[test]
    fn serde_defaults_to_plain_and_round_trips() {
        let parsed: TextAnalyzer = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, TextAnalyzer::plain());
        let english = TextAnalyzer::english();
        let json = serde_json::to_string(&english).unwrap();
        assert_eq!(
            serde_json::from_str::<TextAnalyzer>(&json).unwrap(),
            english
        );
        assert!(serde_json::from_str::<TextAnalyzer>(r#"{"stem":false,"unknown":1}"#).is_err());
    }
}
