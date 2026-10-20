TOOLS = [
    {
        "type": "function",
        "name": "list_files",
        "description": (
            "List files and directories directly inside the selected workspace; "
            "directory names end with '/'. Skip symlinks."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read bounded UTF-8 text from a permitted file inside the "
            "selected workspace. Supply a workspace-relative file path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path, for example src/main.py."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_text",
        "description": (
            "Find literal, case-sensitive text in one permitted workspace file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path, for example src/main.py."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Nonempty literal substring to find within each line."
                    ),
                },
            },
            "required": ["path", "query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "find_symbols",
        "description": (
            "List up to twenty top-level functions, async functions, and "
            "classes in one permitted Python file, with line numbers. "
            "Excludes nested definitions and class methods."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative Python file path, for example src/main.py."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
