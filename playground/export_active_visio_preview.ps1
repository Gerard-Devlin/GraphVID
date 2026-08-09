param(
    [string]$OutputPath = "$(Split-Path -Parent $PSScriptRoot)\artifacts\certvid_visio_architecture\certvid_architecture.png"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$visio = [Runtime.InteropServices.Marshal]::GetActiveObject("Visio.Application")
$doc = $visio.ActiveDocument
$page = $visio.ActivePage

# Selecting through the active drawing window is more reliable across Visio
# versions than CreateSelection enum values.
$window = $visio.ActiveWindow
$window.SelectAll()
$selection = $window.Selection
$selection.Copy()
Start-Sleep -Milliseconds 800

$image = [System.Windows.Forms.Clipboard]::GetImage()
if ($null -eq $image) {
    throw "Visio did not place a rasterizable image on the clipboard"
}

$parent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$image.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$image.Dispose()

try { $doc.Save() } catch { }
try { $doc.Close() } catch { }
try { $visio.Quit() } catch { }

Write-Output "PNG=$OutputPath"
