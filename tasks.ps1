# Windows-first task runner. No make.
#
# Usage:
#   .\tasks.ps1              # defaults to "test"
#   .\tasks.ps1 test         # python -m pytest -q
#   .\tasks.ps1 quick        # python -m sentinel.redteam --quick
#   .\tasks.ps1 redteam      # the full declared red-team run, writes results/
#   .\tasks.ps1 props        # just the hypothesis property tests
#
# The interpreter path is repository-relative ($PSScriptRoot\.venv), never a
# path hardcoded to any one machine, so this runs the same way from any clone
# that has created .venv in the repository root per the README's quick start.

param([Parameter(Position = 0)][string]$Task = "test")

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

switch ($Task) {
    "test"    { & $py -m pytest -q }
    "quick"   { & $py -m sentinel.redteam --quick }
    "redteam" { & $py -m sentinel.redteam --out results\redteam.csv --journal results\redteam.jsonl }
    "props"   { & $py -m pytest tests\test_properties.py -q }
    default   { Write-Host "usage: .\tasks.ps1 [test|quick|redteam|props]"; exit 2 }
}

exit $LASTEXITCODE
