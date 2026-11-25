import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "skill-distill" / "scripts"
ADAPTERS = ROOT / "skills" / "skill-distill" / "adapters"
sys.path.insert(0, str(SCRIPTS))

import extract_session  # noqa: E402


def write_pi_session(path: Path, cwd: str, user_text: str = "hello", tool_name: str = "bash") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "session", "version": 3, "id": "sess-1", "timestamp": "2026-11-25T00:00:00Z", "cwd": cwd},
        {"type": "model_change", "provider": "spark-tf", "modelId": "qwen3.8-flash-next-tensorfold"},
        {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": user_text}]}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "checking"},
                    {"type": "toolCall", "id": "tc1", "name": tool_name, "arguments": {"command": "echo hi"}},
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "tc1",
                "content": [{"type": "text", "text": "hi"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "DONE"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def make_fake_pi(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_pi = bin_dir / "pi"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
session_dir = None
i = 0
while i < len(args):
    if args[i] == "--session-dir":
        session_dir = args[i + 1]
        i += 2
    else:
        i += 1

Path("argv.json").write_text(json.dumps(args))
Path("result.txt").write_text("created-by-fake-pi\\n")
if session_dir:
    sdir = Path(session_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "session", "version": 3, "id": "fake-1", "cwd": os.getcwd()},
        {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": args[-1]}]}},
        {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "OK"}]}},
    ]
    (sdir / "sess.jsonl").write_text("\\n".join(json.dumps(r) for r in rows) + "\\n")
print("fake-pi ok")
sys.exit(int(os.environ.get("FAKE_PI_RC", "0")))
"""
    )
    fake_pi.chmod(fake_pi.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fake_pi


class PiSupportTests(unittest.TestCase):
    def test_extract_session_current_pi_ignores_dangling_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            fake_home = td_path / "home"
            proj = td_path / "repo"
            proj.mkdir()
            sdir = fake_home / ".pi" / "agent" / "sessions" / "--repo--"
            sdir.mkdir(parents=True)

            older = sdir / "older.jsonl"
            newer = sdir / "newer.jsonl"
            other = sdir / "other.jsonl"
            write_pi_session(older, str(proj), "old task")
            time.sleep(0.02)
            write_pi_session(newer, str(proj), "new task")
            write_pi_session(other, str(td_path / "other"), "other repo")

            dangling = sdir / "dangling.jsonl"
            os.symlink("/nonexistent/target.jsonl", dangling)

            r = subprocess.run(
                [sys.executable, str(SCRIPTS / "extract_session.py"), "--current", "pi", "--cwd", str(proj), "--json", str(td_path / "ref.json")],
                env={**os.environ, "HOME": str(fake_home)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("# Session reference (omp)", r.stdout)
            ref = json.loads((td_path / "ref.json").read_text())
            self.assertEqual(ref["source"], str(newer))
            self.assertEqual(ref["task"], "new task")
            self.assertEqual(ref["tool_calls"], 1)
            self.assertEqual(ref["final"], "DONE")

            # Also verify summarize works directly on our test fixture
            s = extract_session.summarize(newer)
            self.assertEqual(s["user_turns"], 1)

    def test_pi_adapter_flags_prompt_safety_and_trace_copy(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bin_dir = td_path / "bin"
            make_fake_pi(bin_dir)
            ws = td_path / "ws"
            ws.mkdir()
            trace = td_path / "trace"
            trace.mkdir()
            prompt = td_path / "prompt.md"
            prompt.write_text("-leading-dash prompt body\nsecond line")

            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "WORKSPACE": str(ws),
                "PROMPT_FILE": str(prompt),
                "MODEL": "spark-tf/qwen3.8-flash-next-tensorfold",
                "TRACE_DIR": str(trace),
                "STUDENT_ARGS": "--thinking off",
            }
            r = subprocess.run(["bash", str(ADAPTERS / "pi.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            argv = json.loads((ws / "argv.json").read_text())
            self.assertIn("-p", argv)
            self.assertIn("--no-skills", argv)
            self.assertIn("--no-context-files", argv)
            self.assertIn("--no-extensions", argv)
            self.assertIn("--thinking", argv)
            self.assertIn("off", argv)
            self.assertIn("--", argv)
            self.assertEqual(argv[argv.index("--") + 1], "-leading-dash prompt body\nsecond line")
            self.assertTrue((trace / "session.jsonl").exists())

            # PI_KEEP_CONTEXT=1 & PI_KEEP_EXTENSIONS=1 omit both flags and propagate non-zero rc
            env2 = {**env, "PI_KEEP_CONTEXT": "1", "PI_KEEP_EXTENSIONS": "1", "FAKE_PI_RC": "7"}
            r2 = subprocess.run(["bash", str(ADAPTERS / "pi.sh")], env=env2, capture_output=True, text=True)
            self.assertEqual(r2.returncode, 7)
            argv2 = json.loads((ws / "argv.json").read_text())
            self.assertNotIn("--no-context-files", argv2)
            self.assertNotIn("--no-extensions", argv2)

    def test_distill_trial_sync_runs_without_tmux_and_collects_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bin_dir = td_path / "bin"
            make_fake_pi(bin_dir)

            repo = td_path / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "README.md").write_text("hi\n")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-m", "init"],
                cwd=repo,
                check=True,
            )

            prompt = td_path / "prompt.md"
            prompt.write_text("Create result.txt\n")
            run_dir = td_path / "run"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "distill.py"),
                    "init",
                    str(run_dir),
                    "--prompt",
                    str(prompt),
                    "--student",
                    "pi:spark-tf/qwen3.8-flash-next-tensorfold",
                    "--base",
                    str(repo),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            checks = run_dir / "checks.sh"
            checks.write_text('#!/usr/bin/env bash\ntest -f "$WORKSPACE/result.txt"\n')
            checks.chmod(0o755)

            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
            rt = subprocess.run(
                [sys.executable, str(SCRIPTS / "distill.py"), "trial", str(run_dir), "-k", "2", "--sync"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rt.returncode, 0, rt.stdout + rt.stderr)
            self.assertIn("started sync i01-t1", rt.stdout)
            self.assertIn("started sync i01-t2", rt.stdout)

            for k in (1, 2):
                t = run_dir / "iters" / "01" / f"t{k}"
                self.assertEqual((t / "exit_code").read_text().strip(), "0")
                self.assertTrue((t / "trace" / "session.jsonl").exists())
                self.assertTrue((t / "workspace" / "result.txt").exists())

            ev = subprocess.run(
                [sys.executable, str(SCRIPTS / "distill.py"), "evidence", str(run_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("checks exit 0", ev.stdout)
            for k in (1, 2):
                t = run_dir / "iters" / "01" / f"t{k}"
                summary = json.loads((t / "evidence" / "summary.json").read_text())
                self.assertEqual(summary["checks_exit"], 0)
                self.assertTrue((t / "evidence" / "trace.md").exists())


if __name__ == "__main__":
    unittest.main()
