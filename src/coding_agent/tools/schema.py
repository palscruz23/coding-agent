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
]
