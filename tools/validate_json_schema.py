filter_fact_json_schema = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "string"  # All items in the inner array must be strings
                },
                "minItems": 3,
                "maxItems": 3,
                "additionalItems": False  # Block extra items
            },
        }
    },
    "required": ["fact"]
}

lkg_keyword_json_schema = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1,
        }
    },
    "required": ["keywords"]
}

triple_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Head": {
                "type": "string"
            },
            "Relation": {
                "type": "string"
            },
            "Tail": {
                "type": "string"
            }
        },
        "required": ["Head", "Relation", "Tail"]
    },
}

event_relation_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Head": {
                "type": "string"
            },
            "Relation": {
                "type": "string",
            },
            "Tail": {
                "type": "string"
            }
        },
        "required": ["Head", "Relation", "Tail"]
    },
}
event_entity_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Event": {
                "type": "string"
            },
            "Entity": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1
            }
        },
        "required": ["Event", "Entity"]
    },
}
stage_to_schema = {
    1: triple_json_schema,
    2: event_entity_json_schema,
    3: event_relation_json_schema,
    4: triple_json_schema,
    5: triple_json_schema,
    6: triple_json_schema
}

