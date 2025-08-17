functions_definition = [
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Capture an image from a camera endpoint or return an existing file by timestamped name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "YYYY", "optional": True},
                    "month": {"type": "integer", "description": "MM", "optional": True},
                    "day": {"type": "integer", "description": "DD", "optional": True},
                    "hour": {"type": "integer", "description": "hh (24h)", "optional": True},
                    "minute": {"type": "integer", "description": "mm", "optional": True},
                    "camera_url": {"type": "string", "description": "Override camera URL", "optional": True}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "monitor_equipment_status",
            "description": "Retrieve the closest timestamped image from a folder like 'liquid handler' or 'fumehood'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_type": {
                        "type": "string",
                        "description": "Equipment folder name",
                        "default": "liquid handler"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "DuckDuckGo search results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Retrieve the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_tecan_csv",
            "description": "Create a Tecan pick list CSV mapping reagents to wells for multi-plate runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_position": {"type": "integer", "default": 1},
                    "plate_positions": {"type": "array", "items": {"type": "integer"}},
                    "cat_A": {"type": "array", "items": {"type": "string"}},
                    "cat_B": {"type": "array", "items": {"type": "string"}},
                    "cat_C": {"type": "array", "items": {"type": "string"}},
                    "cat_D": {"type": "array", "items": {"type": "string"}},
                    "cat_E": {"type": "array", "items": {"type": "string"}},
                    "amounts": {"type": "array", "items": {"type": "integer"}},
                    "replicate_mode": {"type": "boolean", "default": False},
                    "out_path": {"type": "string", "default": "tecan_operations.csv"}
                },
                "required": ["plate_positions"]
            }
        }
    }
]
