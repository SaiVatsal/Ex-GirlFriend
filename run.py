# run.py
"""Parhi AI — Unified Master Launcher & Feature Control Center.

Provides a single interactive menu to launch all Parhi subsystems:
  [1] 🌐 Launch Neural Web GUI (8K Brainbow Connectome & Desktop HUD)
  [2] 💬 Interactive Terminal Chat (Emotion Engine, Memory, Hybrid Brain)
  [3] 🎮 Free Fire / Game Mode (Anti-AFK Autonomous Survivor Loop)
  [4] 📁 Batch Folder Creator (Create 100 friend folders on Desktop)
  [5] 🛡️ Autonomous Laptop Control ("Take Control" Master Mode)
  [6] ⚡ Start Model Context Protocol (MCP) Server (stdio JSON-RPC)
  [7] 🎙️ Always-On Background JARVIS Voice Service (Wake-word "Parhi")
  [8] 🧪 Run All System Diagnostics & Test Suites (59 pytest tests)
  [9] 🔄 Sync & Push All Code to GitHub
  [0] ❌ Exit
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


def print_banner() -> None:
    print(r"""
 =========================================================================
   🧠 PARHI AI — ALL-IN-ONE MASTER CONTROL LAUNCHER v3.5
   Intelligence: Gemini 3.8 Flash + Parhi-GPT Hybrid Engine
   Master Access: Vatsal (PIN: 1327) | Platform: Windows
 =========================================================================
""")


def display_menu() -> None:
    print(" Select which feature you want to run:\n")
    print("  [1] 🌐 Neural Web GUI Interface     (8K Biological Brainbow HUD at port 8000)")
    print("  [2] 💬 Interactive Terminal Chat    (Full CLI with Emotion, Voice & Hybrid Brain)")
    print("  [3] 🎮 Free Fire / Game Mode         (Autonomous Survivor, Anti-AFK, Auto-Loot)")
    print("  [4] 📁 Batch Folder Generator        (Create 100 friend folders on Desktop)")
    print("  [5] 🛡️ Take Control / Laptop Copilot (Authenticated Autonomous Operator Mode)")
    print("  [6] ⚡ Model Context Protocol (MCP)  (Start stdio JSON-RPC MCP Server)")
    print("  [7] 🎙️ Background JARVIS Service    (Always-On Wake-Word 'Parhi' Voice Assistant)")
    print("  [8] 🧪 Run Complete Test Suite      (Verify all 59 unit & integration tests)")
    print("  [9] 🔄 Git Sync & Push to GitHub    (Stage, commit, and push updates to remote)")
    print("  [0] ❌ Exit\n")


def run_gui() -> None:
    print("\n[🚀] Launching Parhi Neural Web GUI Interface...")
    print("     Server: http://127.0.0.1:8000")
    print("     Press Ctrl+C to stop the GUI server.\n")
    try:
        from parhi_gui_server import run_gui as start_server
        start_server()
    except KeyboardInterrupt:
        print("\n[✓] GUI server stopped.")
    except Exception as e:
        print(f"[✗] Error starting GUI: {e}")


def run_chat() -> None:
    print("\n[🚀] Initializing Parhi Interactive Terminal Chat...")
    try:
        subprocess.run([sys.executable, "chat.py"], check=False)
    except Exception as e:
        print(f"[✗] Error running chat: {e}")


def run_game_mode() -> None:
    print("\n=======================================================")
    print("  🎮 FREE FIRE AUTONOMOUS SURVIVOR ENGINE")
    print("=======================================================")
    from game_automation import GameAutomationEngine
    engine = GameAutomationEngine()

    win = engine.find_game_window()
    if win:
        print(f"  [✓] Detected Game/Emulator Window: '{win[1]}'")
    else:
        print("  [!] No emulator detected yet (supports BlueStacks, LDPlayer, MEmu, Nox).")
        print("      Starting survivor routine in active foreground window...")

    dur_str = input("\n  Enter runtime duration in minutes [default 30]: ").strip()
    dur = float(dur_str) if dur_str else 30.0

    print("\n  [SAFETY WARNING]:")
    print("  - Survivor loop includes WASD patrol, sprint, jump, auto-loot ('F'), and heal ('4').")
    print("  - Moving your physical mouse or pressing ESC will immediately stop automation.")
    
    confirm = input("\n  Start Free Fire Autonomous Play now? (Y/n): ").strip().lower()
    if confirm in ("n", "no"):
        print("  [!] Game automation cancelled.")
        return

    ok, msg = engine.start_game_routine(max_duration_minutes=dur)
    print(f"\n  {msg}\n")
    print("  Autonomous loop is running in background. Press Enter here to stop it anytime...")
    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = engine.stop_game_routine("User stopped from launcher")
        print(f"  [✓] {stop_msg}")


def run_batch_folders() -> None:
    print("\n=======================================================")
    print("  📁 BATCH FOLDER GENERATOR")
    print("=======================================================")
    from batch_executor import BatchExecutor
    executor = BatchExecutor()

    count_str = input("  How many folders do you want to create? [default 100]: ").strip()
    count = int(count_str) if count_str.isdigit() else 100

    base_name = input("  Base name prefix? [default 'Friend']: ").strip()
    if not base_name:
        base_name = "Friend"

    print(f"\n  Creating {count} folders on your Desktop (prefix: '{base_name}')...")
    res = executor.create_batch_folders(count=count, base_name=base_name)
    print(f"  [✓] {res.display_text}")

    undo = input("\n  Do you want to undo and remove these created test folders? (y/N): ").strip().lower()
    if undo in ("y", "yes"):
        removed, errs = executor.undo_batch(res.undo_token)
        print(f"  [✓] Undone: Removed {removed} folders cleanly.")


def run_take_control() -> None:
    print("\n=======================================================")
    print("  🛡️ AUTONOMOUS LAPTOP CONTROL ('With My Access Only')")
    print("=======================================================")
    from autonomous_controller import AutonomousController
    controller = AutonomousController()

    print("  Autonomous laptop control allows executing system commands,")
    print("  batch folder generation, and game routines.\n")
    
    pin = input("  Enter Master Access PIN (default: 1327): ").strip()
    ok, msg = controller.authorize(pin)
    print(f"  {msg}")
    if not ok:
        return

    print("\n  [✓] Parhi has taken Autonomous Control of your laptop.")
    print("  Type any command (e.g. 'create 50 folders named Client', 'play free fire', 'open chrome', 'battery').")
    print("  Type 'release control' or 'exit' to return to menu.\n")

    while True:
        try:
            cmd = input("  [Parhi-Control] > ").strip()
            if not cmd or cmd.lower() in ("exit", "quit", "release control", "back"):
                controller.revoke_access()
                print("  [✓] Autonomous control released.")
                break
            
            handled, resp = controller.process_command(cmd)
            if handled and resp:
                print(f"  {resp}")
            else:
                # Fallback to system command
                from system_commands import SystemCommands, detect_system_command
                sc = SystemCommands()
                det = detect_system_command(cmd)
                if det:
                    c_type, targ = det
                    if c_type == "open_app":
                        r = sc.open_app(targ)
                    elif c_type == "close_app":
                        r = sc.close_app(targ)
                    elif c_type == "open_website":
                        r = sc.open_website(targ)
                    elif c_type == "screenshot":
                        r = sc.screenshot()
                    elif c_type == "battery":
                        r = sc.battery_status()
                    else:
                        r = None
                    print(f"  {r.display_text if r else 'Command executed.'}")
                else:
                    print("  Command processed in autonomous mode.")
        except KeyboardInterrupt:
            print("\n  [✓] Autonomous control released.")
            break


def run_mcp_server() -> None:
    print("\n[⚡] Launching Model Context Protocol (MCP) Server over stdio...")
    print("     Clients can connect via parhi_mcp_config.json")
    print("     Press Ctrl+C to terminate server.\n")
    try:
        from parhi_mcp_server import ParhiMCPServer
        server = ParhiMCPServer()
        server.run_stdio()
    except KeyboardInterrupt:
        print("\n[✓] MCP server terminated.")


def run_background_service() -> None:
    print("\n[🎙️] Launching Background JARVIS Voice Service...")
    print("     Wake word: 'Parhi' (voice activation, ambient sound log)")
    print("     Press Ctrl+C to stop service.\n")
    try:
        subprocess.run([sys.executable, "parhi_service.py"], check=False)
    except KeyboardInterrupt:
        print("\n[✓] Background service stopped.")


def run_tests() -> None:
    print("\n[🧪] Executing complete Parhi test suite (pytest)...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=False)
    except Exception as e:
        print(f"[✗] Test execution error: {e}")


def run_git_sync() -> None:
    print("\n[🔄] Staging and pushing latest updates to GitHub...")
    try:
        subprocess.run(["C:\\Program Files\\Git\\cmd\\git.exe", "add", "-A"], check=True)
        commit_msg = input("  Enter commit message [default: 'feat: update Parhi AI components']: ").strip()
        if not commit_msg:
            commit_msg = "feat: update Parhi AI components"
        subprocess.run(["C:\\Program Files\\Git\\cmd\\git.exe", "commit", "-m", commit_msg], check=False)
        print("  Pushing to origin main...")
        subprocess.run(["C:\\Program Files\\Git\\cmd\\git.exe", "push", "origin", "main"], check=True)
        print("  [✓] Successfully pushed to GitHub!")
    except Exception as e:
        print(f"  [✗] Git sync error: {e}")


def main() -> None:
    # Check if a command argument was passed
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip()
        mapping = {
            "gui": run_gui,
            "web": run_gui,
            "chat": run_chat,
            "cli": run_chat,
            "game": run_game_mode,
            "freefire": run_game_mode,
            "batch": run_batch_folders,
            "folders": run_batch_folders,
            "control": run_take_control,
            "mcp": run_mcp_server,
            "service": run_background_service,
            "test": run_tests,
            "tests": run_tests,
            "push": run_git_sync,
            "git": run_git_sync,
        }
        if arg in mapping:
            print_banner()
            mapping[arg]()
            return

    # Interactive Loop
    while True:
        print_banner()
        display_menu()
        try:
            choice = input(" Enter your choice [0-9]: ").strip()
            if choice == "1":
                run_gui()
            elif choice == "2":
                run_chat()
            elif choice == "3":
                run_game_mode()
            elif choice == "4":
                run_batch_folders()
            elif choice == "5":
                run_take_control()
            elif choice == "6":
                run_mcp_server()
            elif choice == "7":
                run_background_service()
            elif choice == "8":
                run_tests()
            elif choice == "9":
                run_git_sync()
            elif choice == "0" or choice.lower() in ("exit", "q", "quit"):
                print("\n Thank you for using Parhi AI. Goodbye! 👋\n")
                break
            else:
                print("\n [!] Invalid choice. Please enter a number between 0 and 9.")
            
            input("\n Press Enter to return to main menu...")
        except (KeyboardInterrupt, EOFError):
            print("\n\n Exiting Parhi Master Launcher. Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
