# ============================================
# Code Execution Tool
# ============================================
# Provides a sandboxed environment for running
# user-generated or LLM-generated Python code.
#
# The LLM itself never executes code. Instead,
# it invokes this tool which is responsible for:
#   • Executing code in a controlled subprocess
#   • Capturing standard output (stdout)
#   • Capturing standard error (stderr)
#   • Enforcing a timeout to prevent runaway code
# ============================================

import subprocess
import sys
import tempfile
import os
from config.settings import CODE_EXECUTION_TIMEOUT


def execute_code(code: str, timeout: int = None) -> dict:
    if timeout is None:
        timeout = CODE_EXECUTION_TIMEOUT

    # Create a temporary file to hold the code
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        tmp_file.write(code)
        tmp_file.close()

        # Execute the code in a subprocess
        result = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None if result.returncode == 0 else result.stderr,
            "timeout": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": (
                f"⏱️ Execution timed out after {timeout} seconds. "
                f"The code may contain an infinite loop or a very "
                f"long-running operation."
            ),
            "timeout": True,
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": f"Failed to execute code: {str(e)}",
            "timeout": False,
        }

    finally:
        # Clean up the temporary file
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


def extract_code_blocks(text: str) -> list[str]:
    blocks = []
    lines = text.split("\n")
    in_block = False
    current_block = []
    block_lang = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") and not in_block:
            # Start of a code block
            lang = stripped[3:].strip().lower()
            if lang in ("python", "py", "python3", ""):
                in_block = True
                block_lang = lang
                current_block = []
        elif stripped == "```" and in_block:
            # End of a code block
            if current_block:
                blocks.append("\n".join(current_block))
            in_block = False
            current_block = []
            block_lang = None
        elif in_block:
            current_block.append(line)

    return blocks
