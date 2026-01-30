"""Interactive multi-turn conversation mode for the agent."""

import json
import shlex
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os
from rich.table import Table

from config import Config
from llm import ModelManager
from memory.store import MemoryStore
from utils import get_log_file_path, terminal_ui
from utils.runtime import get_exports_dir, get_history_file
from utils.tui.input_handler import InputHandler
from utils.tui.status_bar import StatusBar
from utils.tui.theme import Theme, set_theme


class InteractiveSession:
    """Manages an interactive conversation session with the agent."""

    def __init__(self, agent):
        """Initialize interactive session.

        Args:
            agent: The agent instance
        """
        self.agent = agent
        self.conversation_count = 0
        self.show_thinking = Config.TUI_SHOW_THINKING
        self.compact_mode = Config.TUI_COMPACT_MODE

        # Use the agent's model manager to avoid divergence
        self.model_manager = getattr(agent, "model_manager", None) or ModelManager()

        # Initialize TUI components
        self.input_handler = InputHandler(
            history_file=get_history_file(),
            commands=[
                "help",
                "clear",
                "stats",
                "history",
                "dump-memory",
                "theme",
                "verbose",
                "compact",
                "model",
                "exit",
                "quit",
            ],
        )

        # Set up keyboard shortcut callbacks
        self.input_handler.set_callbacks(
            on_clear_screen=self._on_clear_screen,
            on_toggle_thinking=self._on_toggle_thinking,
            on_show_stats=self._on_show_stats,
        )

        # Initialize status bar
        self.status_bar = StatusBar(terminal_ui.console)
        self.status_bar.update(mode="REACT")

    def _on_clear_screen(self) -> None:
        """Handle Ctrl+L - clear screen."""
        terminal_ui.console.clear()

    def _on_toggle_thinking(self) -> None:
        """Handle Ctrl+T - toggle thinking display."""
        self.show_thinking = not self.show_thinking
        status = "enabled" if self.show_thinking else "disabled"
        terminal_ui.print_info(f"Thinking display {status}")

    def _on_show_stats(self) -> None:
        """Handle Ctrl+S - show quick stats."""
        self._show_stats()

    def _show_help(self) -> None:
        """Display help message with available commands."""
        colors = Theme.get_colors()
        terminal_ui.console.print(
            f"\n[bold {colors.primary}]Available Commands:[/bold {colors.primary}]"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/help[/{colors.primary}]             - Show this help message"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/clear[/{colors.primary}]            - Clear conversation memory and start fresh"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/stats[/{colors.primary}]            - Show memory and token usage statistics"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/history[/{colors.primary}]          - List all saved conversation sessions"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/dump-memory <id>[/{colors.primary}] - Export a session's memory to a JSON file"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/theme[/{colors.primary}]            - Toggle between dark and light theme"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/verbose[/{colors.primary}]          - Toggle verbose thinking display"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/compact[/{colors.primary}]          - Toggle compact output mode"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/model[/{colors.primary}]            - Manage models (list/switch/add/edit/remove/default/show)"
        )
        terminal_ui.console.print(
            f"    [{colors.text_muted}]/model                        - List models\n"
            f"    /model <model_id>              - Switch model\n"
            f"    /model add <model_id> key=value...    - Add model (e.g. name=... api_key=...)\n"
            f"    /model edit <model_id> key=value...   - Edit model\n"
            f"    /model remove <model_id>       - Remove model (not current)\n"
            f"    /model default <model_id>      - Set default\n"
            f"    /model show <model_id>         - Show model details\n"
            f"    /model reload                  - Reload .aloop/models.yaml[/{colors.text_muted}]"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/exit[/{colors.primary}]             - Exit interactive mode"
        )
        terminal_ui.console.print(
            f"  [{colors.primary}]/quit[/{colors.primary}]             - Same as /exit"
        )

        terminal_ui.console.print(
            f"\n[bold {colors.primary}]Keyboard Shortcuts:[/bold {colors.primary}]"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Tab[/{colors.secondary}]        - Auto-complete commands"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Ctrl+C[/{colors.secondary}]     - Cancel current operation"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Ctrl+L[/{colors.secondary}]     - Clear screen"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Ctrl+T[/{colors.secondary}]     - Toggle thinking display"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Ctrl+S[/{colors.secondary}]     - Show quick stats"
        )
        terminal_ui.console.print(
            f"  [{colors.secondary}]Up/Down[/{colors.secondary}]    - Navigate command history\n"
        )

    def _show_stats(self) -> None:
        """Display current memory and token statistics."""
        terminal_ui.console.print()
        stats = self.agent.memory.get_stats()
        terminal_ui.print_memory_stats(stats)
        terminal_ui.console.print()

    async def _show_history(self) -> None:
        """Display all saved conversation sessions."""
        try:
            store = MemoryStore()
            sessions = await store.list_sessions(limit=20)

            if not sessions:
                colors = Theme.get_colors()
                terminal_ui.console.print(
                    f"\n[{colors.warning}]No saved sessions found.[/{colors.warning}]"
                )
                terminal_ui.console.print(
                    f"[{colors.text_muted}]Sessions will be saved when using persistent memory mode.[/{colors.text_muted}]\n"
                )
                return

            colors = Theme.get_colors()
            terminal_ui.console.print(
                f"\n[bold {colors.primary}]Saved Sessions (showing most recent 20):[/bold {colors.primary}]\n"
            )

            table = Table(show_header=True, header_style=f"bold {colors.primary}", box=None)
            table.add_column("ID", style=colors.text_muted, width=38)
            table.add_column("Created", width=20)
            table.add_column("Messages", justify="right", width=10)
            table.add_column("Summaries", justify="right", width=10)

            for session in sessions:
                session_id = session["id"]
                created = session["created_at"][:19]
                msg_count = str(session["message_count"])
                summary_count = str(session["summary_count"])
                table.add_row(session_id, created, msg_count, summary_count)

            terminal_ui.console.print(table)
            terminal_ui.console.print()
            terminal_ui.console.print(
                f"[{colors.text_muted}]Tip: Use /dump-memory <session_id> to export a session's memory[/{colors.text_muted}]\n"
            )

        except Exception as e:
            terminal_ui.print_error(str(e), title="Error loading sessions")

    async def _dump_memory(self, session_id: str) -> None:
        """Export a session's memory to a JSON file.

        Args:
            session_id: Session ID to export
        """
        try:
            store = MemoryStore()
            session_data = await store.load_session(session_id)

            if not session_data:
                terminal_ui.print_error(f"Session {session_id} not found")
                return

            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "stats": session_data["stats"],
                "system_messages": [msg.to_dict() for msg in session_data["system_messages"]],
                "messages": [msg.to_dict() for msg in session_data["messages"]],
                "summaries": [
                    {
                        "summary": s.summary,
                        "original_message_count": s.original_message_count,
                        "original_tokens": s.original_tokens,
                        "compressed_tokens": s.compressed_tokens,
                        "compression_ratio": s.compression_ratio,
                        "token_savings": s.token_savings,
                        "preserved_messages": [msg.to_dict() for msg in s.preserved_messages],
                        "metadata": s.metadata,
                    }
                    for s in session_data["summaries"]
                ],
            }

            output_dir = Path(get_exports_dir())
            await aiofiles.os.makedirs(str(output_dir), exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = session_id[:8]
            filename = f"memory_dump_{short_id}_{timestamp}.json"
            output_path = output_dir / filename

            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                payload = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                await f.write(payload)

            terminal_ui.print_success("Memory dumped successfully!")
            colors = Theme.get_colors()
            terminal_ui.console.print(
                f"[{colors.text_muted}]Location:[/{colors.text_muted}] {output_path}"
            )

            terminal_ui.console.print(f"\n[bold {colors.primary}]Summary:[/bold {colors.primary}]")
            terminal_ui.console.print(f"  Session ID: {session_id}")
            terminal_ui.console.print(f"  Messages: {len(export_data['messages'])}")
            terminal_ui.console.print(f"  System Messages: {len(export_data['system_messages'])}")
            terminal_ui.console.print(f"  Summaries: {len(export_data['summaries'])}")
            terminal_ui.console.print()

        except Exception as e:
            terminal_ui.print_error(str(e), title="Error dumping memory")

    def _toggle_theme(self) -> None:
        """Toggle between dark and light theme."""
        current = Theme.get_theme_name()
        new_theme = "light" if current == "dark" else "dark"
        set_theme(new_theme)
        terminal_ui.print_success(f"Switched to {new_theme} theme")

    def _toggle_verbose(self) -> None:
        """Toggle verbose thinking display."""
        self.show_thinking = not self.show_thinking
        status = "enabled" if self.show_thinking else "disabled"
        terminal_ui.print_info(f"Verbose thinking display {status}")

    def _toggle_compact(self) -> None:
        """Toggle compact output mode."""
        self.compact_mode = not self.compact_mode
        status = "enabled" if self.compact_mode else "disabled"
        terminal_ui.print_info(f"Compact mode {status}")

    def _update_status_bar(self) -> None:
        """Update status bar with current stats."""
        stats = self.agent.memory.get_stats()
        model_info = self.agent.get_current_model_info()
        model_name = model_info["name"] if model_info else ""
        self.status_bar.update(
            input_tokens=stats.get("total_input_tokens", 0),
            output_tokens=stats.get("total_output_tokens", 0),
            context_tokens=stats.get("current_tokens", 0),
            cost=stats.get("total_cost", 0),
            model_name=model_name,
        )

    async def _handle_command(self, user_input: str) -> bool:
        """Handle a slash command.

        Args:
            user_input: User input starting with /

        Returns:
            True if should continue loop, False if should exit
        """
        command_parts = user_input.split()
        command = command_parts[0].lower()

        if command in ("/exit", "/quit"):
            colors = Theme.get_colors()
            terminal_ui.console.print(
                f"\n[bold {colors.warning}]Exiting interactive mode. Goodbye![/bold {colors.warning}]"
            )
            return False

        elif command == "/help":
            self._show_help()

        elif command == "/clear":
            self.agent.memory.reset()
            self.conversation_count = 0
            self._update_status_bar()
            terminal_ui.print_success("Memory cleared. Starting fresh conversation.")
            terminal_ui.console.print()

        elif command == "/stats":
            self._show_stats()

        elif command == "/history":
            await self._show_history()

        elif command == "/dump-memory":
            if len(command_parts) < 2:
                terminal_ui.print_error("Please provide a session ID")
                colors = Theme.get_colors()
                terminal_ui.console.print(
                    f"[{colors.text_muted}]Usage: /dump-memory <session_id>[/{colors.text_muted}]\n"
                )
            else:
                await self._dump_memory(command_parts[1])

        elif command == "/theme":
            self._toggle_theme()

        elif command == "/verbose":
            self._toggle_verbose()

        elif command == "/compact":
            self._toggle_compact()

        elif command == "/model":
            await self._handle_model_command(user_input)

        else:
            colors = Theme.get_colors()
            terminal_ui.console.print(
                f"[bold {colors.error}]Unknown command: {command}[/bold {colors.error}]"
            )
            terminal_ui.console.print(
                f"[{colors.text_muted}]Type /help to see available commands[/{colors.text_muted}]\n"
            )

        return True

    def _show_models(self) -> None:
        """Display available models and current selection."""
        colors = Theme.get_colors()
        profiles = self.model_manager.list_models()
        current = self.model_manager.get_current_model()
        default_model_id = self.model_manager.get_default_model_id()

        terminal_ui.console.print(
            f"\n[bold {colors.primary}]Available Models:[/bold {colors.primary}]\n"
        )

        if not profiles:
            terminal_ui.print_error("No models configured.")
            terminal_ui.console.print(
                f"[{colors.text_muted}]Edit `.aloop/models.yaml` to add models and set `default`.[/{colors.text_muted}]\n"
            )
            return

        for profile in profiles:
            markers: list[str] = []
            if current and profile.model_id == current.model_id:
                markers.append(f"[{colors.success}]CURRENT[/{colors.success}]")
            if default_model_id and profile.model_id == default_model_id:
                markers.append(f"[{colors.primary}]DEFAULT[/{colors.primary}]")
            marker = (
                " ".join(markers)
                if markers
                else f"[{colors.text_muted}]      [/{colors.text_muted}]"
            )

            label = profile.display_name
            if current and profile.model_id == current.model_id:
                terminal_ui.console.print(f"  {marker} {label} - {profile.model_id}")
            else:
                terminal_ui.console.print(f"  {marker} {label} - {profile.model_id}")

        terminal_ui.console.print(
            f"\n[{colors.text_muted}]Use /model <model_id> to switch models[/{colors.text_muted}]\n"
        )

    def _switch_model(self, model_id: str) -> None:
        """Switch to a different model.

        Args:
            model_id: LiteLLM model ID to switch to
        """
        colors = Theme.get_colors()

        # Validate the profile
        profile = self.model_manager.get_model(model_id)
        if profile is None:
            terminal_ui.print_error(f"Model '{model_id}' not found")
            available = ", ".join(self.model_manager.get_model_ids())
            if available:
                terminal_ui.console.print(
                    f"[{colors.text_muted}]Available: {available}[/{colors.text_muted}]\n"
                )
            return

        is_valid, error_msg = self.model_manager.validate_model(profile)
        if not is_valid:
            terminal_ui.print_error(error_msg)
            return

        # Perform the switch
        if self.agent.switch_model(model_id):
            new_profile = self.model_manager.get_current_model()
            if new_profile:
                terminal_ui.print_success(
                    f"Switched to model: {new_profile.display_name} ({new_profile.model_id})"
                )
                self._update_status_bar()
            else:
                terminal_ui.print_error("Failed to get current model after switch")
        else:
            terminal_ui.print_error(f"Failed to switch to model '{model_id}'")

    def _parse_kv_args(self, tokens: list[str]) -> tuple[dict[str, str], list[str]]:
        kv: dict[str, str] = {}
        rest: list[str] = []
        for token in tokens:
            if "=" in token:
                k, _, v = token.partition("=")
                kv[k.strip()] = v
            else:
                rest.append(token)
        return kv, rest

    def _mask_secret(self, value: str | None) -> str:
        if not value:
            return "(not set)"
        v = value.strip()
        if len(v) <= 8:
            return "*" * len(v)
        return f"{v[:4]}…{v[-4:]}"

    async def _handle_model_command(self, user_input: str) -> None:
        """Handle the /model command.

        Args:
            user_input: Full user input string
        """
        colors = Theme.get_colors()

        try:
            parts = shlex.split(user_input)
        except ValueError as e:
            terminal_ui.print_error(str(e), title="Invalid /model command")
            return

        if len(parts) == 1:
            self._show_models()
            return

        sub = parts[1]

        if sub == "reload":
            self.model_manager.reload()
            terminal_ui.print_success("Reloaded `.aloop/models.yaml`")
            current_after = self.model_manager.get_current_model()
            if not current_after:
                terminal_ui.print_error(
                    "No models configured after reload. Edit `.aloop/models.yaml` and set `default`."
                )
                return

            # Reinitialize LLM adapter to pick up updated api_key/api_base/timeout/drop_params.
            self.agent.switch_model(current_after.model_id)
            terminal_ui.print_info(f"Reload applied (current: {current_after.model_id}).")
            return

        if sub == "add":
            if len(parts) < 3:
                terminal_ui.print_error("Usage: /model add <model_id> [name=...] [api_key=...] ...")
                return
            model_id = parts[2]
            kv, _ = self._parse_kv_args(parts[3:])
            name = kv.pop("name", "")
            api_key = kv.pop("api_key", None)
            api_base = kv.pop("api_base", None)
            timeout = kv.pop("timeout", None)
            drop_params = kv.pop("drop_params", None)

            timeout_value = 600
            if timeout is not None:
                try:
                    timeout_value = int(str(timeout).strip())
                except ValueError:
                    terminal_ui.print_error("timeout must be an integer (seconds)")
                    return

            drop_params_value = True
            if drop_params is not None:
                v = str(drop_params).strip().lower()
                if v in {"true", "1", "yes", "y", "on"}:
                    drop_params_value = True
                elif v in {"false", "0", "no", "n", "off"}:
                    drop_params_value = False
                else:
                    terminal_ui.print_error("drop_params must be true/false")
                    return

            ok = self.model_manager.add_model(
                model_id=model_id,
                name=name,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout_value,
                drop_params=drop_params_value,
                **kv,
            )
            if not ok:
                terminal_ui.print_error(f"Model '{model_id}' already exists (or invalid).")
                return
            terminal_ui.print_success(f"Added model: {model_id}")
            return

        if sub == "edit":
            if len(parts) < 4:
                terminal_ui.print_error("Usage: /model edit <model_id> <field=value> ...")
                return
            model_id = parts[2]
            kv, rest = self._parse_kv_args(parts[3:])
            if rest:
                terminal_ui.print_error(f"Invalid args (expected key=value): {', '.join(rest)}")
                return
            ok = self.model_manager.edit_model(model_id, **kv)
            if not ok:
                terminal_ui.print_error(f"Model '{model_id}' not found")
                return
            terminal_ui.print_success(f"Updated model: {model_id}")
            current = self.model_manager.get_current_model()
            if current and current.model_id == model_id:
                self.agent.switch_model(model_id)
                terminal_ui.print_info("Updated current model configuration applied.")
            return

        if sub == "remove":
            if len(parts) != 3:
                terminal_ui.print_error("Usage: /model remove <model_id>")
                return
            model_id = parts[2]
            current = self.model_manager.get_current_model()
            if current and current.model_id == model_id:
                terminal_ui.print_error("Cannot remove the current active model. Switch first.")
                return
            ok = self.model_manager.remove_model(model_id)
            if not ok:
                terminal_ui.print_error(f"Model '{model_id}' not found (or is current).")
                return
            terminal_ui.print_success(f"Removed model: {model_id}")
            return

        if sub == "default":
            if len(parts) != 3:
                terminal_ui.print_error("Usage: /model default <model_id>")
                return
            model_id = parts[2]
            ok = self.model_manager.set_default(model_id)
            if not ok:
                terminal_ui.print_error(f"Model '{model_id}' not found")
                return
            terminal_ui.print_success(f"Set default model: {model_id}")
            return

        if sub == "show":
            if len(parts) != 3:
                terminal_ui.print_error("Usage: /model show <model_id>")
                return
            model_id = parts[2]
            profile = self.model_manager.get_model(model_id)
            if not profile:
                terminal_ui.print_error(f"Model '{model_id}' not found")
                return
            terminal_ui.console.print(
                f"\n[bold {colors.primary}]Model:[/bold {colors.primary}] {model_id}"
            )
            terminal_ui.console.print(f"  Name: {profile.name or '(not set)'}")
            terminal_ui.console.print(f"  Provider: {profile.provider}")
            terminal_ui.console.print(f"  API key: {self._mask_secret(profile.api_key)}")
            terminal_ui.console.print(f"  API base: {profile.api_base or '(not set)'}")
            terminal_ui.console.print(f"  Timeout: {profile.timeout}")
            terminal_ui.console.print(f"  Drop params: {profile.drop_params}\n")
            return

        # Otherwise treat as a model_id switch
        self._switch_model(sub)

    async def run(self) -> None:
        """Run the interactive session loop."""
        # Print header
        terminal_ui.print_header(
            "Agentic Loop - Interactive Mode",
            subtitle="Multi-turn conversation with AI Agent",
        )

        # Display configuration
        current = self.model_manager.get_current_model()
        config_dict = {
            "Model": current.model_id if current else "NOT CONFIGURED",
            "Theme": Theme.get_theme_name(),
            "Commands": "/help for all commands",
        }
        terminal_ui.print_config(config_dict)

        colors = Theme.get_colors()
        terminal_ui.console.print(
            f"\n[bold {colors.success}]Interactive mode started. Type your message or use commands.[/bold {colors.success}]"
        )
        terminal_ui.console.print(
            f"[{colors.text_muted}]Tip: Press Tab for auto-complete, Ctrl+T to toggle thinking display[/{colors.text_muted}]\n"
        )

        # Show initial status bar
        if Config.TUI_STATUS_BAR:
            self.status_bar.show()

        while True:
            try:
                # Get user input
                user_input = await self.input_handler.prompt_async("> ")

                # Handle empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process user message
                self.conversation_count += 1

                # Show turn divider
                terminal_ui.print_turn_divider(self.conversation_count)

                # Echo user input in Claude Code style
                terminal_ui.print_user_message(user_input)

                # Update status bar to show processing
                if Config.TUI_STATUS_BAR:
                    self.status_bar.update(is_processing=True)

                try:
                    result = await self.agent.run(user_input)

                    # Display agent response
                    terminal_ui.console.print(
                        f"[bold {colors.secondary}]Assistant:[/bold {colors.secondary}]"
                    )
                    terminal_ui.print_assistant_message(result)

                    # Update status bar
                    self._update_status_bar()
                    if Config.TUI_STATUS_BAR:
                        self.status_bar.update(is_processing=False)
                        self.status_bar.show()

                except KeyboardInterrupt:
                    terminal_ui.console.print(
                        f"\n[bold {colors.warning}]Task interrupted by user.[/bold {colors.warning}]\n"
                    )
                    if Config.TUI_STATUS_BAR:
                        self.status_bar.update(is_processing=False)
                    continue
                except Exception as e:
                    terminal_ui.print_error(str(e))
                    if Config.TUI_STATUS_BAR:
                        self.status_bar.update(is_processing=False)
                    continue

            except KeyboardInterrupt:
                terminal_ui.console.print(
                    f"\n\n[bold {colors.warning}]Interrupted. Type /exit to quit or continue chatting.[/bold {colors.warning}]\n"
                )
                continue
            except EOFError:
                terminal_ui.console.print(
                    f"\n[bold {colors.warning}]Exiting interactive mode. Goodbye![/bold {colors.warning}]"
                )
                break

        # Show final statistics
        terminal_ui.console.print(
            f"\n[bold {colors.primary}]Final Session Statistics:[/bold {colors.primary}]"
        )
        stats = self.agent.memory.get_stats()
        terminal_ui.print_memory_stats(stats)

        # Show log file location
        log_file = get_log_file_path()
        if log_file:
            terminal_ui.print_log_location(log_file)


async def run_interactive_mode(agent) -> None:
    """Run agent in interactive multi-turn conversation mode.

    Args:
        agent: The agent instance
    """
    session = InteractiveSession(agent)
    await session.run()
