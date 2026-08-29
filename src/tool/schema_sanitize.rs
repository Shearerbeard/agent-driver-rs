//! JSON Schema rewrites for OpenAI strict mode (`schema-sanitize` feature).
//!
//! Ported from the `aura` crate in <https://github.com/mezmo/aura>
//! (Apache-2.0), file `crates/aura/src/schema_sanitize.rs` at commit
//! `8b114b9f0f4cea589a1c50b868d06fe04fb96042`:
//! <https://github.com/mezmo/aura/blob/8b114b9f0f4c/crates/aura/src/schema_sanitize.rs>
//! Copyright Mezmo and the aura contributors. Licensed under the Apache
//! License, Version 2.0; see `LICENSE-APACHE` and `NOTICE` at the crate root.
//!
//! Changes from the upstream file: only `fix_empty_root_required` and
//! `recursive_set_additional_properties_false` (with the private helpers they
//! call) are carried over; the doc examples are dropped; bindings use
//! `if let Value::Object(map)` instead of `&mut ... ref mut` patterns;
//! `is_some_and` replaces `map(..).unwrap_or(false)`; `to_owned` replaces
//! `to_string` on literals; `make_nullable` moves the map with `mem::take`
//! instead of cloning it.

use serde_json::{Value, json};
use std::collections::HashSet;

/// Sets `additionalProperties: false` on every object level that OpenAI's
/// strict mode requires it on, recursing through `anyOf`, `properties`, and
/// `items`. Also normalises `const: null` to `type: "null"` and gives
/// `type: "object"` schemas with no `properties` an empty one.
pub(super) fn recursive_set_additional_properties_false(schema: &mut Value) -> &mut Value {
    normalize_const_null(schema);

    if let Value::Object(map) = schema {
        // Zero-argument MCP tools commonly omit `properties` entirely.
        let is_object_without_properties = map
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t == "object")
            && !map.contains_key("properties");

        if is_object_without_properties {
            map.insert(
                "properties".to_owned(),
                Value::Object(serde_json::Map::new()),
            );
        }

        // Pydantic 2.11+ emits `additionalProperties: true` for arbitrary
        // dictionaries; strict mode needs it overridden to false.
        let should_add_additional_properties = map.contains_key("required")
            || map
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(serde_json::Map::is_empty)
            || map.contains_key("additionalProperties");

        if should_add_additional_properties {
            map.insert("additionalProperties".to_owned(), Value::Bool(false));
        }

        if let Some(Value::Array(any_of_array)) = map.get_mut("anyOf") {
            for sub_schema in any_of_array.iter_mut() {
                recursive_set_additional_properties_false(sub_schema);
            }
        }

        if let Some(Value::Object(properties)) = map.get_mut("properties") {
            for sub_schema in properties.values_mut() {
                recursive_set_additional_properties_false(sub_schema);
            }
        }

        if let Some(items) = map.get_mut("items") {
            recursive_set_additional_properties_false(items);
        }
    }

    schema
}

/// Makes every property required at every nesting level, as OpenAI's strict
/// mode demands, by wrapping each newly required property in `anyOf` with
/// `null` (unless it is already nullable) and listing it in `required`.
pub(super) fn fix_empty_root_required(schema: &mut Value) -> &mut Value {
    fix_required_recursive(schema);
    schema
}

fn fix_required_recursive(schema: &mut Value) {
    if let Value::Object(map) = schema {
        if let Some(Value::Object(properties)) = map.get("properties") {
            let all_property_names: HashSet<String> = properties.keys().cloned().collect();

            let existing_required: HashSet<String> = map
                .get("required")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();

            let missing_properties: Vec<String> = all_property_names
                .difference(&existing_required)
                .cloned()
                .collect();

            if !missing_properties.is_empty()
                && let Some(Value::Object(properties)) = map.get_mut("properties")
            {
                for prop_name in &missing_properties {
                    if let Some(prop_schema) = properties.get_mut(prop_name)
                        && !is_nullable(prop_schema)
                    {
                        make_nullable(prop_schema);
                    }
                }

                let mut new_required: Vec<Value> = existing_required
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();

                for prop_name in missing_properties {
                    new_required.push(Value::String(prop_name));
                }

                map.insert("required".to_owned(), Value::Array(new_required));
            }
        }

        if let Some(Value::Object(properties)) = map.get_mut("properties") {
            for prop_schema in properties.values_mut() {
                fix_required_recursive(prop_schema);
            }
        }

        if let Some(items_schema) = map.get_mut("items") {
            fix_required_recursive(items_schema);
        }

        for key in ["anyOf", "allOf", "oneOf"] {
            if let Some(Value::Array(branches)) = map.get_mut(key) {
                for item in branches.iter_mut() {
                    fix_required_recursive(item);
                }
            }
        }
    }
}

fn is_nullable(schema: &Value) -> bool {
    if let Value::Object(map) = schema {
        if let Some(Value::Array(any_of)) = map.get("anyOf") {
            return any_of.iter().any(is_null_type);
        }

        if let Some(Value::Array(types)) = map.get("type") {
            return types.iter().any(|t| t.as_str() == Some("null"));
        }
    }

    false
}

fn is_null_type(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("null")
}

/// MCP servers emit `const: null` where OpenAI expects `type: "null"`.
fn normalize_const_null(schema: &mut Value) {
    if let Value::Object(map) = schema {
        if map.get("const").is_some_and(Value::is_null) {
            map.remove("const");
            map.remove("nullable");
            map.insert("type".to_owned(), Value::String("null".to_owned()));
        }

        for key in ["anyOf", "allOf", "oneOf"] {
            if let Some(Value::Array(branches)) = map.get_mut(key) {
                for item in branches.iter_mut() {
                    normalize_const_null(item);
                }
            }
        }
        if let Some(Value::Object(properties)) = map.get_mut("properties") {
            for prop in properties.values_mut() {
                normalize_const_null(prop);
            }
        }
        if let Some(items) = map.get_mut("items") {
            normalize_const_null(items);
        }
    }
}

/// Wraps the schema in `anyOf: [<schema>, {type: null}]`, keeping
/// `description`, `default`, and `title` at the top level.
fn make_nullable(schema: &mut Value) {
    if let Value::Object(map) = schema {
        let description = map.remove("description");
        let default = map.remove("default");
        let title = map.remove("title");

        if let Some(Value::Array(any_of)) = map.get_mut("anyOf") {
            if !any_of.iter().any(is_null_type) {
                any_of.push(json!({"type": "null"}));
            }
        } else {
            let current_schema = Value::Object(std::mem::take(map));
            map.insert(
                "anyOf".to_owned(),
                Value::Array(vec![current_schema, json!({"type": "null"})]),
            );
        }

        if let Some(desc) = description {
            map.insert("description".to_owned(), desc);
        }
        if let Some(def) = default {
            map.insert("default".to_owned(), def);
        }
        if let Some(t) = title {
            map.insert("title".to_owned(), t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_is_filled_and_missing_properties_become_nullable() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "context": {"type": "string", "default": "", "description": "ctx"}
            },
            "required": []
        });

        fix_empty_root_required(&mut schema);

        assert_eq!(
            schema["required"],
            json!(["context"]),
            "all properties required"
        );
        assert_eq!(
            schema["properties"]["context"],
            json!({
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": "",
                "description": "ctx"
            }),
            "wrapped in anyOf with metadata kept at the top level"
        );
    }

    #[test]
    fn already_nullable_properties_are_left_alone() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "other": {"type": ["integer", "null"]}
            },
            "required": ["other"]
        });

        fix_empty_root_required(&mut schema);

        let required = schema["required"].as_array().map_or(0, Vec::len);
        assert_eq!(required, 2, "both properties listed in required");
        assert_eq!(
            schema["properties"]["field"]["anyOf"]
                .as_array()
                .map_or(0, Vec::len),
            2,
            "existing anyOf not extended"
        );
        assert_eq!(
            schema["properties"]["other"],
            json!({"type": ["integer", "null"]}),
            "type-array nullability recognised"
        );
    }

    #[test]
    fn required_is_fixed_in_nested_objects_items_and_branches() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}}
                },
                "list": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"b": {"type": "integer"}}}
                }
            },
            "oneOf": [{"type": "object", "properties": {"c": {"type": "boolean"}}}]
        });

        fix_empty_root_required(&mut schema);

        assert_eq!(
            schema["properties"]["nested"]["anyOf"][0]["required"],
            json!(["a"]),
            "nested object"
        );
        assert_eq!(
            schema["properties"]["list"]["anyOf"][0]["items"]["required"],
            json!(["b"]),
            "array items"
        );
        assert_eq!(schema["oneOf"][0]["required"], json!(["c"]), "oneOf branch");
    }

    #[test]
    fn additional_properties_false_where_strict_mode_needs_it() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "opts": {"type": "object", "additionalProperties": true},
                "tags": {"type": "array", "items": {"type": "object", "properties": {}}}
            },
            "required": ["name"]
        });

        recursive_set_additional_properties_false(&mut schema);

        assert_eq!(
            schema["additionalProperties"],
            json!(false),
            "root has required"
        );
        assert_eq!(
            schema["properties"]["opts"]["additionalProperties"],
            json!(false),
            "existing true is overridden"
        );
        assert_eq!(
            schema["properties"]["tags"]["items"]["additionalProperties"],
            json!(false),
            "empty properties in items"
        );
        assert!(
            schema["properties"]["name"]
                .get("additionalProperties")
                .is_none(),
            "a plain string schema is untouched"
        );
    }

    #[test]
    fn object_without_properties_gets_an_empty_one() {
        let mut schema = json!({"type": "object"});

        recursive_set_additional_properties_false(&mut schema);

        assert_eq!(
            schema,
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            "zero-argument tool shape"
        );
    }

    #[test]
    fn const_null_becomes_type_null() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "maybe": {"anyOf": [{"type": "string"}, {"const": null, "nullable": true}]}
            },
            "required": ["maybe"]
        });

        recursive_set_additional_properties_false(&mut schema);

        assert_eq!(
            schema["properties"]["maybe"]["anyOf"][1],
            json!({"type": "null"}),
            "const: null rewritten and nullable flag dropped"
        );
    }

    #[test]
    fn non_object_schemas_are_unchanged() {
        let mut schema = json!({"type": "string"});
        recursive_set_additional_properties_false(&mut schema);
        fix_empty_root_required(&mut schema);
        assert_eq!(
            schema,
            json!({"type": "string"}),
            "no object level to rewrite"
        );
    }
}
