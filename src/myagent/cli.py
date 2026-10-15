"""Minimal CLI; grow one behavior at a time."""

import argparse
import asyncio
from pathlib import Path

from openai import APIError
from prompt_toolkit import PromptSession

from myagent.agent import respond


async def interactive_loop(workspace: Path):
    """Run the interactive prompt and preserve successful conversation turns.

    Args:
        workspace (Path): Selected directory supplied to the agent and displayed
            by the /status command.

    Returns:
       None: Completes on /exit, EOF (Ctrl+D), or Ctrl+C during input.
            Replies, command results, and handled errors are printed.
    """

    session = PromptSession()

    print("MyAgent v0.1.0")
    print("Type /help for commands.")

    previous_response_id = None

    while True:
        try:
            messages = await session.prompt_async("you> ")
            message = messages.strip()

            if message == "/exit":
                break
            if not message:
                continue

            if message == "/help":
                print("/help   Show commands")
                print("/status Show workspace")
                print("/exit   Exit MyAgent")
                continue

            if message == "/status":
                print(f"Workspace: {workspace}")
                continue

            if message.startswith("/"):
                print(f"Unknown command: {message}. Type /help for commands.")
                continue

        except (EOFError, KeyboardInterrupt):
            break

        try:
            response, response_id = await respond(
                message=message,
                previous_response_id=previous_response_id,
                workspace=workspace,
            )

            previous_response_id = response_id
            print(f"\nagent> {response}\n")

        except APIError as exc:
            print(f"API error: {type(exc).__name__}")

        except RuntimeError as exc:
            print(f"Agent error: {exc}")


def main() -> int:
    """Parse CLI arguments, validate the workspace, and run the prompt.

    Args:
        None: Arguments are read from the process command line by argparse.

    Returns:
        int: Exit status 0 after the interactive loop finishes successfully.

    Raises:
        SystemExit: argparse handles help, version, invalid arguments, or a
            workspace that is not an existing directory.
    """
    parser = argparse.ArgumentParser(
        prog="myagent",
        description="A coding agent built one module at a time.",
    )

    parser.add_argument("--version", action="version", version="myagent 0.1.0")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if workspace.is_dir():
        print(workspace)
    else:
        parser.error("Workspace must be an existing directory")

    asyncio.run(interactive_loop(workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
