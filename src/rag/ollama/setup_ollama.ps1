# Downloads the DeepCNCFQuantized GGUF (~5.3 GB) from Hugging Face and registers it
# in Ollama as the model "deepcncf". Requires Ollama (https://ollama.com/download)
# and the RAG virtualenv (.venv-rag) at the repository root.
#
#   powershell -ExecutionPolicy Bypass -File src\rag\ollama\setup_ollama.ps1

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $here "..\..\..")
$python = Join-Path $repoRoot ".venv-rag\Scripts\python.exe"
$gguf = Join-Path $here "ggml-model-Q4_K_M.gguf"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or not on PATH. Install it from https://ollama.com/download"
}

if (-not (Test-Path $gguf)) {
    Write-Host "Downloading Kubermatic/DeepCNCFQuantized (~5.3 GB) ..."
    & $python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Kubermatic/DeepCNCFQuantized', 'ggml-model-Q4_K_M.gguf', local_dir=r'$here')"
    if ($LASTEXITCODE -ne 0) { throw "Download failed" }
} else {
    Write-Host "GGUF already present: $gguf"
}

Push-Location $here
try {
    ollama create deepcncf -f Modelfile
    if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
} finally {
    Pop-Location
}
Write-Host "Done. Test with:  ollama run deepcncf `"What is Cilium?`""
