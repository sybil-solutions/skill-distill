import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from distill import cmd_trial, collect
from extract_session import summarize, to_markdown


class ReviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_opencode_summary_bounds_output_and_surfaces_errors(self):
        events = [
            {"type": "tool_use", "part": {"tool": "write", "state": {
                "input": {"filePath": f"file-{i}.txt", "content": "x" * 1000},
                "status": "completed", "output": "unrelated output " * 100,
            }}}
            for i in range(300)
        ]
        events[12]["part"]["state"].update(status="error", error="Cannot write file")
        events.extend([
            {"type": "text", "part": {"text": "Finished"}},
            {"type": "error", "error": {"type": "provider.free-limit", "message": "Quota reached"}},
        ])
        for event in events:
            event["sessionID"] = "synthetic"
        trace = self.root / "session.jsonl"
        trace.write_text("".join(json.dumps(event) + "\n" for event in events))
        summary = summarize(trace)
        self.assertEqual((summary["format"], summary["tool_calls"], summary["tool_errors"]),
                         ("opencode", 300, 1))
        self.assertEqual(len(summary["files_written"]), 300)
        compact = to_markdown(summary, compact=True)
        self.assertLess(len(compact), 4000)
        self.assertIn("Step 13: Cannot write file", compact)
        self.assertIn("Quota reached", compact)
        self.assertIn("Finished", compact)
        self.assertNotIn("unrelated output", compact)
        self.assertIn("unrelated output", to_markdown(summary))

    def test_evidence_lists_only_changes_from_starting_commit(self):
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.root, text=True)

        git("init", "-q")
        (self.root / "stable.txt").write_text("stable")
        (self.root / "edited.txt").write_text("old")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "start")
        base = git("rev-parse", "HEAD").strip()
        (self.root / "edited.txt").write_text("new")
        git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qam", "edit")
        (self.root / "new.txt").write_text("new")
        (self.root / ".skill").mkdir()
        (self.root / ".skill/injected.md").write_text("skill")
        out = self.root / "evidence"
        collect(self.root, out, self.root / "no-checks.sh", base=base)
        self.assertEqual((out / "files.txt").read_text(), "M\tedited.txt\n??\tnew.txt\n")

    def test_extend_keeps_frozen_skill(self):
        run = self.root
        (run / "run.json").write_text(json.dumps({
            "adapter": "opencode", "model": "test", "name": "test", "student_args": "",
            "env": {}, "base": {"kind": "empty"}, "tmux": "test", "timeout_min": 1,
        }))
        (run / "prompt.md").write_text("Original prompt")
        (run / "skill").mkdir()
        (run / "skill/SKILL.md").write_text("first")
        from distill import sh

        def without_tmux(command, **kwargs):
            if command[0] == "tmux":
                return SimpleNamespace(returncode=0)
            return sh(command, **kwargs)

        with patch("distill.sh", side_effect=without_tmux), contextlib.redirect_stdout(io.StringIO()):
            cmd_trial(SimpleNamespace(run=run, iter=None, extend=False, k=1))
            (run / "skill/SKILL.md").write_text("revised")
            cmd_trial(SimpleNamespace(run=run, iter=1, extend=True, k=2))
        for k in range(1, 4):
            trial = run / f"iters/01/t{k}"
            self.assertEqual((trial / "workspace/.skill/test/SKILL.md").read_text(), "first")

    def test_opencode_adapter_runs_in_workspace_and_reports_provider_error(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        runner = bin_dir / "opencode"
        runner.write_text(
            "#!/usr/bin/env bash\n"
            '[[ "$PWD" == "$WORKSPACE" && "$*" != *--dir* ]] || exit 2\n'
            'if read -r _; then exit 3; fi\n'
            'printf \'%s\\n\' \'{"type":"error","sessionID":"test","error":{"message":"Quota reached"}}\'\n'
        )
        runner.chmod(0o755)
        prompt = self.root / "prompt.md"
        prompt.write_text("Create a test file.")
        result = subprocess.run(
            ["bash", str(SKILL / "adapters/opencode.sh")],
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MODEL": "test",
                 "WORKSPACE": str(self.root), "TRACE_DIR": str(self.root), "PROMPT_FILE": str(prompt)},
            input="unexpected input\n", capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Quota reached", (self.root / "session.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
