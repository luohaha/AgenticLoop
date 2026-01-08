"""Advanced file operation tools inspired by Claude Code."""
import re
from pathlib import Path
from typing import Any, Dict

from tools.base import BaseTool


class GlobTool(BaseTool):
    """Fast file pattern matching tool."""

    @property
    def name(self) -> str:
        return "glob_files"

    @property
    def description(self) -> str:
        return """Fast file pattern matching tool.

Use this to find files by patterns like:
- "**/*.py" - all Python files recursively
- "src/**/*.js" - JavaScript files in src/
- "*.txt" - text files in current directory

Much faster than reading directories recursively.
Returns sorted list of matching file paths."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g., '**/*.py', 'src/**/*.js')"
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: current directory)"
            }
        }

    def execute(self, pattern: str, path: str = ".") -> str:
        """Find files matching glob pattern."""
        try:
            base_path = Path(path)
            if not base_path.exists():
                return f"Error: Path does not exist: {path}"

            matches = sorted(base_path.glob(pattern))
            if not matches:
                return f"No files found matching pattern: {pattern} in {path}"

            # Limit to 100 results to avoid overwhelming output
            if len(matches) > 100:
                result_lines = [str(m) for m in matches[:100]]
                result_lines.append(f"\n... and {len(matches) - 100} more files")
                return "\n".join(result_lines)

            return "\n".join(str(m) for m in matches)
        except Exception as e:
            return f"Error executing glob: {str(e)}"


class GrepTool(BaseTool):
    """Search file contents using regex patterns."""

    @property
    def name(self) -> str:
        return "grep_content"

    @property
    def description(self) -> str:
        return """Search file contents using regex patterns.

Output modes:
- files_only: Just list files containing matches (default)
- with_context: Show matching lines with line numbers
- count: Count matches per file

Much more efficient than reading all files and searching manually.

Examples:
- Find function definitions: "def\\s+\\w+"
- Find imports: "^import\\s+"
- Find TODOs: "TODO|FIXME"

IMPORTANT: Use this instead of reading every file individually."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            },
            "mode": {
                "type": "string",
                "description": "Output mode: files_only, with_context, or count (default: files_only)"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether search is case sensitive (default: true)"
            }
        }

    def execute(
        self,
        pattern: str,
        path: str = ".",
        mode: str = "files_only",
        case_sensitive: bool = True
    ) -> str:
        """Search for pattern in files."""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {str(e)}"

        try:
            base_path = Path(path)
            if not base_path.exists():
                return f"Error: Path does not exist: {path}"

            results = []
            files_searched = 0

            for file_path in base_path.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip common non-text files
                if file_path.suffix in ['.pyc', '.so', '.dylib', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz']:
                    continue

                files_searched += 1

                try:
                    content = file_path.read_text(encoding='utf-8')
                    matches = list(regex.finditer(content))

                    if not matches:
                        continue

                    if mode == "files_only":
                        results.append(str(file_path))
                    elif mode == "count":
                        results.append(f"{file_path}: {len(matches)} matches")
                    elif mode == "with_context":
                        lines = content.splitlines()
                        for match in matches[:5]:  # Limit to 5 matches per file
                            line_no = content[:match.start()].count('\n') + 1
                            if line_no <= len(lines):
                                results.append(f"{file_path}:{line_no}: {lines[line_no-1].strip()}")
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue

                # Limit total results to 50 to avoid overwhelming output
                if len(results) >= 50:
                    break

            if not results:
                return f"No matches found for pattern '{pattern}' in {files_searched} files searched"

            return "\n".join(results)
        except Exception as e:
            return f"Error executing grep: {str(e)}"


class EditTool(BaseTool):
    """Edit files surgically without reading entire contents."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return """Edit files surgically without reading entire contents.

Operations:
- replace: Find and replace text exactly (old_text and new_text required)
- append: Add to end of file (text required)
- insert_at_line: Insert text at specific line number (line_number and text required)

More efficient than reading full file, modifying, and writing back.

IMPORTANT: Use this for small, targeted edits to save tokens."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit"
            },
            "operation": {
                "type": "string",
                "description": "Operation to perform: replace, append, or insert_at_line"
            },
            "old_text": {
                "type": "string",
                "description": "Text to find and replace (for replace operation)"
            },
            "new_text": {
                "type": "string",
                "description": "New text to insert (for replace operation)"
            },
            "text": {
                "type": "string",
                "description": "Text to add (for append or insert_at_line operations)"
            },
            "line_number": {
                "type": "integer",
                "description": "Line number to insert at (for insert_at_line operation, 1-indexed)"
            }
        }

    def execute(
        self,
        file_path: str,
        operation: str,
        old_text: str = "",
        new_text: str = "",
        text: str = "",
        line_number: int = 0,
        **kwargs
    ) -> str:
        """Perform surgical file edit."""
        try:
            path = Path(file_path)

            if not path.exists():
                return f"Error: File does not exist: {file_path}"

            if operation == "replace":
                if not old_text:
                    return "Error: old_text parameter is required for replace operation"

                content = path.read_text(encoding='utf-8')

                if old_text not in content:
                    return f"Error: Text not found in {file_path}"

                # Replace only the first occurrence
                content = content.replace(old_text, new_text, 1)
                path.write_text(content, encoding='utf-8')
                return f"Successfully replaced text in {file_path}"

            elif operation == "append":
                if not text:
                    return "Error: text parameter is required for append operation"

                with path.open("a", encoding='utf-8') as f:
                    f.write(text)
                return f"Successfully appended to {file_path}"

            elif operation == "insert_at_line":
                if not text:
                    return "Error: text parameter is required for insert_at_line operation"
                if line_number <= 0:
                    return "Error: line_number must be positive (1-indexed)"

                lines = path.read_text(encoding='utf-8').splitlines(keepends=True)

                # Insert at the specified line (1-indexed)
                if line_number > len(lines) + 1:
                    return f"Error: line_number {line_number} is beyond file length {len(lines)}"

                # Ensure text ends with newline if inserting in middle
                insert_text = text if text.endswith('\n') else text + '\n'
                lines.insert(line_number - 1, insert_text)

                path.write_text(''.join(lines), encoding='utf-8')
                return f"Successfully inserted text at line {line_number} in {file_path}"

            else:
                return f"Error: Unknown operation '{operation}'. Supported: replace, append, insert_at_line"

        except Exception as e:
            return f"Error executing edit: {str(e)}"
