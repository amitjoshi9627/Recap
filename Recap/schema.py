input_schema = {
    "type": "object",
    "properties": {
        "input_texts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    "required": ["input_texts"],
}
