Param([string]$PyroRoot=".")
Write-Host "[INFO] Copying compressible_lagrangian into $PyroRoot\pyro"
$src = Join-Path (Get-Location) "pyro\compressible_lagrangian"
$dst = Join-Path $PyroRoot "pyro\compressible_lagrangian"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
Copy-Item $src $dst -Recurse -Force
$pfile = Join-Path $PyroRoot "pyro\pyro.py"
if (-Not (Test-Path $pfile)) { Write-Warning "[WARN] Could not find $pfile."; exit 0 }
$content = Get-Content $pfile -Raw
if ($content -notmatch "compressible_lagrangian") {
  if ($content -notmatch "import importlib") { $content = "import importlib`r`n" + $content }
  $ins = @"
elif solver == "compressible_lagrangian":
    mod = importlib.import_module("pyro.compressible_lagrangian.simulation")
    self.simulation = mod.Simulation(self.rp)
"@
  $content = $content + "`r`n" + $ins
  Set-Content -Path $pfile -Value $content -NoNewline
  Write-Host "[OK] Registered compressible_lagrangian"
} else { Write-Host "[SKIP] Entry already present" }
Write-Host "[DONE] Lagrangian solver added. Commit your changes."
