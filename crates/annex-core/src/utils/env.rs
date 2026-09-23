pub fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

pub fn env_string_first(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| env_string(key))
}

pub fn env_bool(key: &str) -> Option<bool> {
    env_string(key).map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

pub fn env_bool_first(keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| env_bool(key))
}

pub fn env_usize(key: &str) -> Option<usize> {
    env_string(key).and_then(|value| value.replace('_', "").parse::<usize>().ok())
}

pub fn env_usize_first(keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| env_usize(key))
}

pub fn env_usize_nonzero(key: &str) -> Option<usize> {
    env_usize(key).filter(|value| *value > 0)
}

pub fn env_usize_list_first(keys: &[&str]) -> Option<Vec<usize>> {
    keys.iter().find_map(|key| {
        env_string(key)
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| part.trim().replace('_', "").parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
    })
}

pub fn env_u64(key: &str) -> Option<u64> {
    env_string(key).and_then(|value| value.replace('_', "").parse::<u64>().ok())
}

pub fn env_f32(key: &str) -> Option<f32> {
    env_string(key).and_then(|value| value.replace('_', "").parse::<f32>().ok())
}
