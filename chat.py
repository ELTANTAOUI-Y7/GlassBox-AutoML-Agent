"""
Test agent.py with your own CSV file.
Usage:
    python3 chat.py <path_to_csv> <target_column>

Example:
    python3 chat.py ~/mydata.csv Churn
"""
import os
import sys
import importlib.util
import webbrowser
from datetime import datetime

_root = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("_agent", os.path.join(_root, "agent.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run = _mod.run


class FakeEnv:
    def __init__(self):
        self._messages = []
        self._files    = {}

    def list_messages(self):
        return self._messages

    def list_files(self):
        return list(self._files.keys())

    def read_file(self, name):
        return self._files.get(name, "")

    def add_reply(self, text):
        print(text)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 chat.py <path_to_csv> <target_column>")
        print("Example: python3 chat.py ~/mydata.csv Churn")
        sys.exit(1)

    csv_path   = os.path.expanduser(sys.argv[1])
    target_col = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    with open(csv_path) as f:
        csv_content = f.read()

    fname = os.path.basename(csv_path)

    env = FakeEnv()
    env._files[fname] = csv_content
    env._messages.append({"role": "user", "content": f"Predict {target_col}\n\n[file: {fname}]"})

    run(env)

    # Generate HTML report if the pipeline succeeded
    report  = getattr(env, "_autofit_report",  None)
    source  = getattr(env, "_autofit_source",  fname)
    n_total = getattr(env, "_autofit_n_total", 0)

    if report is not None:
        try:
            import importlib.util as _ilu
            _rspec = _ilu.spec_from_file_location("report_html", os.path.join(_root, "report_html.py"))
            _rmod  = _ilu.module_from_spec(_rspec)
            _rspec.loader.exec_module(_rmod)

            html = _rmod.generate(report, source, n_total)

            stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = os.path.join(_root, f"report_{stamp}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"\nHTML report saved: {html_path}")
            webbrowser.open(f"file://{html_path}")

        except Exception as exc:
            print(f"\n(HTML report failed: {exc})")
