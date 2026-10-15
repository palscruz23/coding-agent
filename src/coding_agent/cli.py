"""Minimal CLI; grow one behavior at a time."""

import argparse
import asyncio
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI
from prompt_toolkit import PromptSession
from rich.console import Console

from coding_agent.agent import respond

load_dotenv()


def banner() -> None:
    """Print the colored banner, tagline, version, and help hint.

    Args:
        None: No inputs are required.

    Returns:
        None: Writes startup information to the console.
    """
    console = Console()

    BANNER = """
    
 ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗    █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║     ██║   ██║██║  ██║██║██╔██╗ ██║██║  ███╗  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██║     ██║   ██║██║  ██║██║██║╚██╗██║██║   ██║  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
╚██████╗╚██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝  ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
""".strip("\n")

    model = getenv("MODEL", "gpt-5-mini")

    console.print(BANNER, style="bold #3355bb", markup=False)
    console.print(
        "Your agent. Your design. No surprises.\n",
        style="italic bright_white",
    )
    console.print(f"v0.1.0 | Model: {model}", style="dim")
    console.print("Type /help for commands.\n", style="dim")


async def interactive_loop(workspace: Path):
    """Run an interactive session using one shared API client.

    Args:
        workspace (Path): Selected directory passed to the agent and displayed
            by the /status command.

    Returns:
        None: Completes on /exit, EOF (Ctrl+D), or Ctrl+C during input.
            Prints replies, command results, and handled request errors.

    Notes:
        Creates one API client for the session and closes it when the session
        block exits. Only successful replies advance the conversation ID.
    """

    session = PromptSession()

    banner()

    previous_response_id = None
    async with AsyncOpenAI() as client:
        while True:
            try:
                messages = await session.prompt_async("> ")
                message = messages.strip()

                if message == "/exit":
                    break
                if not message:
                    continue

                if message == "/help":
                    print("/help   Show commands")
                    print("/status Show workspace")
                    print("/exit   Exit Coding Agent")
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
                    client=client,
                )

                previous_response_id = response_id

                BLUE = "\033[94m"
                RESET = "\033[0m"

                robot = r"""
.-----.
| ◉_◉ |
'-----'  
> """

                print(f"{BLUE}{robot}{RESET} {response}\n")
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
        prog="coding-agent",
        description="A coding agent built one module at a time.",
    )

    parser.add_argument("--version", action="version", version="coding-agent 0.1.0")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if not workspace.is_dir():
        parser.error("Workspace must be an existing directory")

    asyncio.run(interactive_loop(workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
