param(
    [string]$OutputDir = "$(Split-Path -Parent $PSScriptRoot)\artifacts\certvid_visio_architecture"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path
$VsdxPath = Join-Path $OutputDir "certvid_architecture.vsdx"
$PngPath = Join-Path $OutputDir "certvid_architecture.png"
$SvgPath = Join-Path $OutputDir "certvid_architecture.svg"
foreach ($path in @($VsdxPath, $PngPath, $SvgPath)) {
    if (Test-Path $path) { Remove-Item -LiteralPath $path -Force }
}

function Formula-Rgb([string]$hex) {
    $hex = $hex.TrimStart('#')
    $r = [Convert]::ToInt32($hex.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($hex.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($hex.Substring(4, 2), 16)
    return "RGB($r,$g,$b)"
}

function Set-Cell($shape, [string]$cell, [string]$formula) {
    try { $shape.CellsU($cell).FormulaU = $formula } catch { }
}

function Add-Rect {
    param($page, [double]$x, [double]$y, [double]$w, [double]$h,
          [string]$fill = "#FFFFFF", [string]$line = "#000000",
          [double]$radius = 0.0, [double]$lineWeight = 1.0,
          [double]$fillTransparency = 0.0)
    $s = $page.DrawRectangle($x-$w/2, $y-$h/2, $x+$w/2, $y+$h/2)
    Set-Cell $s "FillPattern" "1"
    Set-Cell $s "FillForegnd" (Formula-Rgb $fill)
    Set-Cell $s "FillBkgnd" (Formula-Rgb $fill)
    Set-Cell $s "FillForegndTrans" "${fillTransparency}%"
    if ($line -eq "none") {
        Set-Cell $s "LinePattern" "0"
    } else {
        Set-Cell $s "LineColor" (Formula-Rgb $line)
        Set-Cell $s "LineWeight" "${lineWeight} pt"
    }
    if ($radius -gt 0) { Set-Cell $s "Rounding" "$radius in" }
    return $s
}

function Add-Text {
    param($page, [double]$x, [double]$y, [double]$w, [double]$h,
          [string]$text, [double]$size = 10.0, [string]$color = "#111111",
          [int]$align = 1, [bool]$bold = $false, [string]$font = "Arial")
    $s = Add-Rect $page $x $y $w $h "#FFFFFF" "none" 0 0 100
    $s.Text = $text
    Set-Cell $s "Char.Size" "$size pt"
    Set-Cell $s "Char.Color" (Formula-Rgb $color)
    Set-Cell $s "Para.HorzAlign" "$align"
    Set-Cell $s "VerticalAlign" "1"
    Set-Cell $s "Char.Style" $(if ($bold) { "1" } else { "0" })
    try {
        Set-Cell $s "Char.Font" "FONT(`"$font`")"
    } catch { }
    return $s
}

function Add-Circle {
    param($page, [double]$x, [double]$y, [double]$d,
          [string]$fill, [string]$line = "#1B1B1B", [double]$lineWeight = 1.0,
          [double]$transparency = 0.0, [bool]$gloss = $false)
    $s = $page.DrawOval($x-$d/2, $y-$d/2, $x+$d/2, $y+$d/2)
    Set-Cell $s "FillPattern" "1"
    Set-Cell $s "FillForegnd" (Formula-Rgb $fill)
    Set-Cell $s "FillBkgnd" "RGB(255,255,255)"
    Set-Cell $s "FillForegndTrans" "${transparency}%"
    Set-Cell $s "LineColor" (Formula-Rgb $line)
    Set-Cell $s "LineWeight" "${lineWeight} pt"
    if ($gloss) {
        $h = $page.DrawOval($x-$d*0.27, $y+$d*0.08, $x-$d*0.05, $y+$d*0.30)
        Set-Cell $h "FillPattern" "1"
        Set-Cell $h "FillForegnd" "RGB(255,255,255)"
        Set-Cell $h "FillForegndTrans" "38%"
        Set-Cell $h "LinePattern" "0"
    }
    return $s
}

function Add-Line {
    param($page, [double]$x1, [double]$y1, [double]$x2, [double]$y2,
          [string]$color = "#111111", [double]$weight = 1.2,
          [bool]$arrow = $false, [bool]$dashed = $false)
    $s = $page.DrawLine($x1, $y1, $x2, $y2)
    Set-Cell $s "LineColor" (Formula-Rgb $color)
    Set-Cell $s "LineWeight" "${weight} pt"
    if ($arrow) { Set-Cell $s "EndArrow" "13"; Set-Cell $s "EndArrowSize" "2" }
    if ($dashed) { Set-Cell $s "LinePattern" "2" }
    return $s
}

function Add-Arrow {
    param($page, [double]$x1, [double]$y1, [double]$x2, [double]$y2,
          [string]$color = "#123C82", [double]$weight = 2.0)
    return Add-Line $page $x1 $y1 $x2 $y2 $color $weight $true $false
}

function Add-Lock {
    param($page, [double]$x, [double]$y, [double]$scale = 1.0)
    $body = Add-Rect $page $x ($y-0.015*$scale) (0.18*$scale) (0.19*$scale) "#111111" "#111111" 0.02 0.7
    $arc = $page.DrawOval($x-0.055*$scale, $y+0.04*$scale, $x+0.055*$scale, $y+0.18*$scale)
    Set-Cell $arc "FillPattern" "0"
    Set-Cell $arc "LineColor" "RGB(17,17,17)"
    Set-Cell $arc "LineWeight" "1.1 pt"
    Add-Text $page $x ($y-0.015*$scale) (0.11*$scale) (0.11*$scale) "." (7*$scale) "#FFFFFF" 1 $true | Out-Null
}

function Add-Panel {
    param($page, [double]$x, [double]$w, [string]$fill)
    return Add-Rect $page ($x+$w/2) 2.65 $w 4.80 $fill "none" 0.16 0 0
}

function Add-FilmIcon {
    param($page, [string]$type, [double]$x, [double]$y)
    $navy = "#0A2C98"
    switch ($type) {
        "walk" {
            Add-Circle $page $x ($y+0.18) 0.09 $navy $navy 0.4 | Out-Null
            Add-Line $page $x ($y+0.13) ($x-0.02) ($y-0.05) $navy 1.6 | Out-Null
            Add-Line $page ($x-0.01) ($y+0.06) ($x-0.12) ($y-0.02) $navy 1.4 | Out-Null
            Add-Line $page ($x-0.01) ($y+0.05) ($x+0.10) ($y+0.01) $navy 1.4 | Out-Null
            Add-Line $page ($x-0.02) ($y-0.05) ($x-0.12) ($y-0.20) $navy 1.5 | Out-Null
            Add-Line $page ($x-0.02) ($y-0.05) ($x+0.09) ($y-0.18) $navy 1.5 | Out-Null
        }
        "car" {
            Add-Rect $page $x ($y-0.02) 0.34 0.15 $navy $navy 0.03 0.6 | Out-Null
            Add-Line $page ($x-0.10) ($y+0.06) ($x-0.04) ($y+0.14) $navy 1.4 | Out-Null
            Add-Line $page ($x-0.04) ($y+0.14) ($x+0.10) ($y+0.14) $navy 1.4 | Out-Null
            Add-Line $page ($x+0.10) ($y+0.14) ($x+0.15) ($y+0.06) $navy 1.4 | Out-Null
            Add-Circle $page ($x-0.11) ($y-0.12) 0.08 "#FFFFFF" $navy 1.0 | Out-Null
            Add-Circle $page ($x+0.11) ($y-0.12) 0.08 "#FFFFFF" $navy 1.0 | Out-Null
        }
        "run" {
            Add-Circle $page ($x+0.05) ($y+0.17) 0.09 $navy $navy 0.4 | Out-Null
            Add-Line $page ($x+0.02) ($y+0.12) ($x-0.06) ($y-0.02) $navy 1.6 | Out-Null
            Add-Line $page $x ($y+0.08) ($x-0.14) ($y+0.03) $navy 1.4 | Out-Null
            Add-Line $page ($x-0.06) ($y-0.02) ($x+0.10) ($y-0.05) $navy 1.6 | Out-Null
            Add-Line $page ($x-0.06) ($y-0.02) ($x-0.18) ($y-0.15) $navy 1.6 | Out-Null
            Add-Line $page ($x+0.10) ($y-0.05) ($x+0.18) ($y-0.17) $navy 1.6 | Out-Null
        }
        "mountain" {
            Add-Line $page ($x-0.20) ($y-0.17) ($x-0.06) ($y+0.15) $navy 1.5 | Out-Null
            Add-Line $page ($x-0.06) ($y+0.15) ($x+0.04) ($y-0.04) $navy 1.5 | Out-Null
            Add-Line $page ($x+0.04) ($y-0.04) ($x+0.12) ($y+0.10) $navy 1.5 | Out-Null
            Add-Line $page ($x+0.12) ($y+0.10) ($x+0.22) ($y-0.17) $navy 1.5 | Out-Null
            Add-Line $page ($x-0.20) ($y-0.17) ($x+0.22) ($y-0.17) $navy 1.5 | Out-Null
        }
        "bike" {
            Add-Circle $page ($x-0.13) ($y-0.10) 0.18 "#FFFFFF" $navy 1.1 | Out-Null
            Add-Circle $page ($x+0.14) ($y-0.10) 0.18 "#FFFFFF" $navy 1.1 | Out-Null
            Add-Line $page ($x-0.13) ($y-0.10) $x ($y+0.07) $navy 1.1 | Out-Null
            Add-Line $page $x ($y+0.07) ($x+0.14) ($y-0.10) $navy 1.1 | Out-Null
            Add-Line $page ($x-0.13) ($y-0.10) ($x+0.05) ($y-0.10) $navy 1.1 | Out-Null
            Add-Line $page ($x+0.05) ($y-0.10) $x ($y+0.07) $navy 1.1 | Out-Null
            Add-Line $page $x ($y+0.07) ($x+0.04) ($y+0.16) $navy 1.1 | Out-Null
        }
        "sail" {
            Add-Line $page $x ($y-0.17) $x ($y+0.20) $navy 1.3 | Out-Null
            Add-Line $page $x ($y+0.18) ($x-0.16) ($y-0.08) $navy 1.3 | Out-Null
            Add-Line $page ($x-0.16) ($y-0.08) $x ($y-0.08) $navy 1.3 | Out-Null
            Add-Line $page ($x+0.02) ($y+0.13) ($x+0.14) ($y-0.08) $navy 1.3 | Out-Null
            Add-Line $page ($x+0.14) ($y-0.08) ($x+0.02) ($y-0.08) $navy 1.3 | Out-Null
            Add-Line $page ($x-0.20) ($y-0.17) ($x+0.20) ($y-0.17) $navy 1.5 | Out-Null
        }
    }
}

function Add-Filmstrip {
    param($page)
    $x0 = 0.18; $x1 = 3.30; $y0 = 3.82; $y1 = 5.00
    Add-Rect $page (($x0+$x1)/2) (($y0+$y1)/2) ($x1-$x0) ($y1-$y0) "#FFFFFF" "#111111" 0 1.5 | Out-Null
    for ($i=0; $i -lt 24; $i++) {
        $px = $x0 + 0.08 + $i*0.127
        Add-Rect $page $px 4.94 0.075 0.075 "#FFFFFF" "#111111" 0 0.5 | Out-Null
        Add-Rect $page $px 3.88 0.075 0.075 "#FFFFFF" "#111111" 0 0.5 | Out-Null
    }
    $icons = @("walk", "car", "run", "mountain", "bike", "sail")
    for ($i=0; $i -lt 6; $i++) {
        $cx = 0.45 + $i*0.50
        Add-Line $page ($cx+0.22) 3.93 ($cx+0.22) 4.88 "#111111" 0.7 | Out-Null
        Add-FilmIcon $page $icons[$i] $cx 4.41
    }
    Add-Text $page 3.52 4.40 0.30 0.40 "..." 16 "#111111" 1 $true | Out-Null
}

function Add-DenseTokens {
    param($page)
    $palette = @("#2477EA", "#00B7B7", "#17A899", "#6A45D6", "#FF7318")
    for ($r=0; $r -lt 11; $r++) {
        for ($c=0; $c -lt 15; $c++) {
            $idx = if ($r -lt 4) { [Math]::Min(3, [int]($c/4)) } elseif ($r -gt 7) { ($c+$r)%5 } else { ($c + [int]($r/2))%5 }
            $fill = $palette[$idx]
            $x = 0.28 + $c*0.19
            $y = 3.48 - $r*0.26
            Add-Circle $page $x $y 0.105 $fill "#FFFFFF" 0.2 0 $true | Out-Null
        }
    }
    Add-Text $page 1.70 0.28 2.60 0.35 "Dense Visual Tokens" 12 "#161616" 1 $false | Out-Null
}

function Add-SemanticPanel {
    param($page)
    $x = 3.82; $w = 3.76
    Add-Panel $page $x $w "#EAF7FF" | Out-Null
    $hub = Add-Circle $page 4.06 2.62 0.46 "#F4C542" "#222222" 1.0 0 $true

    $items = @(
        @{y=4.55; label="Visual`nSemantics"; icon="V"; color="#3B8EFF"},
        @{y=3.56; label="Temporal-`nSpatial"; icon="T"; color="#00B5BC"},
        @{y=2.68; label="Motion &`nEvents"; icon="M"; color="#55B92F"},
        @{y=1.71; label="Token`nQuality"; icon="Q"; color="#FF7A00"},
        @{y=0.73; label="Query`nRelevance"; icon="?"; color="#9C5DE5"}
    )
    foreach ($item in $items) {
        Add-Arrow $page 4.20 2.62 4.72 $item.y $item.color 1.5 | Out-Null
        Add-Text $page 4.93 $item.y 0.36 0.45 $item.icon 24 $item.color 1 $true "Segoe UI Symbol" | Out-Null
        Add-Text $page 5.55 $item.y 0.90 0.55 $item.label 10.5 "#151515" 0 $false | Out-Null
    }
    $targetY = @(4.15,3.40,2.68,1.82,1.03)
    for ($i=0; $i -lt 5; $i++) {
        Add-Arrow $page 6.15 $items[$i].y 6.88 $targetY[$i] $items[$i].color 1.5 | Out-Null
    }
    $vecColors = @("#4B8FEA","#12AFB4","#67BC42","#F7B94F","#A56BCB")
    for ($i=0; $i -lt 5; $i++) {
        Add-Rect $page 7.11 (3.85-$i*0.34) 0.28 0.34 $vecColors[$i] "#222222" 0 0.8 | Out-Null
    }
    Add-Text $page 7.05 1.22 0.70 0.58 "Design`nvector" 10 "#222222" 1 $false | Out-Null
}

function Add-CertificatePanel {
    param($page)
    $x = 7.78; $w = 3.52
    Add-Panel $page $x $w "#FFF0EA" | Out-Null
    Add-Text $page 8.50 4.60 0.46 0.44 "F" 25 "#111111" 1 $true "Arial" | Out-Null
    Add-Circle $page 8.70 4.51 0.28 "#89DB72" "#111111" 0.9 | Out-Null
    Add-Text $page 8.70 4.51 0.24 0.24 "v" 13 "#111111" 1 $true "Arial" | Out-Null
    Add-Text $page 8.48 4.13 0.75 0.32 "Frame" 11 "#111111" 1 $false | Out-Null

    Add-Text $page 8.52 3.52 0.40 0.43 "T" 25 "#FF7B73" 1 $true "Arial" | Out-Null
    Add-Line $page 8.18 3.25 8.86 3.25 "#111111" 1.0 $true $false | Out-Null
    Add-Text $page 8.52 3.02 0.85 0.32 "Temporal" 10.5 "#111111" 1 $false | Out-Null

    for ($r=0; $r -lt 3; $r++) {
        for ($c=0; $c -lt 3; $c++) {
            $fc = if ($r -eq 1 -and $c -eq 0) {"#FFD0C4"} elseif ($r -eq 2 -and $c -eq 2) {"#FFD65E"} elseif (($r+$c)%3 -eq 0) {"#BEE7B2"} else {"#EAF5EF"}
            Add-Rect $page (8.30+$c*0.17) (2.57-$r*0.17) 0.17 0.17 $fc "#222222" 0 0.6 | Out-Null
        }
    }
    Add-Text $page 8.50 2.05 0.80 0.32 "Spatial" 10.5 "#111111" 1 $false | Out-Null

    Add-Circle $page 8.50 1.14 0.46 "#A96AD4" "#111111" 1.0 | Out-Null
    Add-Text $page 8.50 1.14 0.30 0.30 "?" 20 "#111111" 1 $true | Out-Null
    Add-Line $page 8.67 0.98 8.84 0.78 "#111111" 2.0 | Out-Null
    Add-Lock $page 8.70 0.92 0.75
    Add-Text $page 8.50 0.55 0.70 0.32 "Query" 10.5 "#111111" 1 $false | Out-Null

    # Brackets and arrows into the protected-set gate.
    Add-Line $page 7.98 4.65 7.88 4.65 "#111111" 1.1 | Out-Null
    Add-Line $page 7.88 4.65 7.88 0.88 "#111111" 1.1 | Out-Null
    Add-Line $page 7.88 0.88 7.98 0.88 "#111111" 1.1 | Out-Null
    Add-Arrow $page 8.96 4.55 9.74 4.10 "#111111" 1.2 | Out-Null
    Add-Arrow $page 8.95 3.25 9.62 3.25 "#111111" 1.2 | Out-Null
    Add-Arrow $page 8.95 2.35 9.62 2.35 "#111111" 1.2 | Out-Null
    Add-Arrow $page 8.94 1.05 9.60 1.62 "#111111" 1.2 | Out-Null

    # Funnel / budget gate.
    $funnel = Add-Text $page 9.88 2.72 0.72 2.40 ">" 88 "#D7D9D8" 1 $false "Arial"
    Set-Cell $funnel "Char.Color" "RGB(205,209,210)"
    Add-Text $page 10.70 3.35 0.74 0.65 "Budget`nGate" 10.5 "#111111" 1 $false | Out-Null
    Add-Rect $page 10.72 2.72 0.74 0.34 "#FFFFFF" "#222222" 0 0.9 | Out-Null
    Add-Rect $page 10.48 2.72 0.22 0.34 "#F4A58F" "#222222" 0 0.8 | Out-Null
    Add-Text $page 10.79 2.72 0.42 0.25 "Bc < B" 9.5 "#222222" 1 $false | Out-Null
    Add-Text $page 10.71 1.85 0.90 0.85 "Protected`nMandatory`nSet" 10.5 "#111111" 1 $false | Out-Null
}

function Add-DOptimalPanel {
    param($page)
    $x = 11.52; $w = 3.56
    Add-Panel $page $x $w "#F2EBFF" | Out-Null
    Add-Arrow $page 11.96 0.72 11.96 4.48 "#111111" 1.1 | Out-Null
    Add-Arrow $page 11.96 0.72 14.42 0.72 "#111111" 1.1 | Out-Null
    Add-Text $page 13.05 0.33 1.20 0.34 "Evidence Space" 11 "#111111" 1 $false | Out-Null
    Add-Text $page 13.17 4.55 1.55 0.34 "max log det(M)" 11 "#111111" 1 $false | Out-Null

    $ellipses = @(
        @{x=12.82;y=2.64;w=1.75;h=2.75;rot=18},
        @{x=13.05;y=2.66;w=2.05;h=2.40;rot=-18},
        @{x=13.18;y=2.78;w=1.35;h=2.85;rot=35}
    )
    foreach ($e in $ellipses) {
        $s = $page.DrawOval($e.x-$e.w/2,$e.y-$e.h/2,$e.x+$e.w/2,$e.y+$e.h/2)
        Set-Cell $s "FillPattern" "0"; Set-Cell $s "LineColor" "RGB(18,18,18)"; Set-Cell $s "LineWeight" "1.0 pt"; Set-Cell $s "Angle" "$($e.rot) deg"
    }
    $nodes = @(
        @{x=12.34;y=1.62;c="#FFD052"}, @{x=12.74;y=2.70;c="#66D643"},
        @{x=13.00;y=3.18;c="#0CC5C5"}, @{x=13.48;y=3.62;c="#7E44E1"},
        @{x=13.95;y=3.55;c="#FF895F"}, @{x=13.83;y=2.58;c="#F2A648"},
        @{x=13.33;y=1.88;c="#FF337A"}
    )
    foreach ($n in $nodes) { Add-Circle $page $n.x $n.y 0.34 $n.c "#222222" 0.9 0 $true | Out-Null }
    Add-Lock $page 14.06 3.38 0.72; Add-Lock $page 13.85 2.43 0.72
    $grey = @(@(12.23,4.20),@(12.46,3.94),@(11.96,3.55),@(12.20,2.10),@(12.24,1.13),@(14.38,3.02),@(14.43,1.75),@(14.12,2.04))
    foreach ($p in $grey) { Add-Circle $page $p[0] $p[1] 0.17 "#C9CED3" "#C9CED3" 0.2 | Out-Null }
    Add-Arrow $page 12.55 2.38 13.27 3.45 "#111111" 0.8 | Out-Null
    Add-Arrow $page 12.86 2.92 13.76 3.35 "#111111" 0.8 | Out-Null
    Add-Arrow $page 12.42 1.72 13.24 2.00 "#111111" 0.8 | Out-Null
    Add-Arrow $page 13.52 3.48 13.77 2.78 "#111111" 0.8 | Out-Null
    Add-Text $page 14.37 4.12 0.68 0.65 "Selected`ntokens" 9.5 "#111111" 1 $false | Out-Null
    Add-Arrow $page 14.32 4.05 13.98 3.84 "#111111" 0.8 | Out-Null
    Add-Text $page 14.47 2.44 0.74 0.65 "Discarded`ntokens" 9.5 "#111111" 1 $false | Out-Null
    Add-Text $page 13.88 1.23 1.12 0.62 "Global`nComplementarity" 9.5 "#111111" 1 $false | Out-Null
    Add-Arrow $page 13.84 1.42 13.46 1.74 "#111111" 0.8 | Out-Null
}

function Add-FusionPanel {
    param($page)
    $x = 15.28; $w = 3.66
    Add-Panel $page $x $w "#F0F6EC" | Out-Null
    Add-Text $page 16.24 4.63 1.32 0.34 "Discarded anchors" 10.5 "#111111" 1 $false | Out-Null
    Add-Text $page 18.05 4.63 1.42 0.34 "Certificate anchors" 10.5 "#111111" 1 $false | Out-Null

    $central = Add-Circle $page 16.88 2.62 0.37 "#F1C94A" "#222222" 0.9 0 $true
    $left = @(
        @{x=15.95;y=4.15;c="#65A4E9"}, @{x=15.96;y=3.25;c="#AD73D1"},
        @{x=15.96;y=2.10;c="#A7D68D"}, @{x=15.96;y=1.14;c="#A96BCB"}
    )
    foreach ($n in $left) {
        Add-Circle $page $n.x $n.y 0.34 $n.c "#222222" 0.8 | Out-Null
        Add-Line $page ($n.x+0.17) $n.y 16.69 2.62 "#222222" 1.0 $true $true | Out-Null
    }
    $discarded = @(@(15.53,4.32),@(16.57,4.32),@(16.98,4.18),@(15.52,3.25),@(16.64,1.83),@(16.97,1.76),@(15.53,1.16),@(16.28,0.56),@(16.68,0.50),@(16.86,0.84),@(16.43,0.94))
    foreach ($p in $discarded) {
        Add-Circle $page $p[0] $p[1] 0.20 "#EEF0F2" "#AAB0B7" 0.6 | Out-Null
        Add-Line $page $p[0] $p[1] 16.88 2.62 "#AAB0B7" 0.7 $true $true | Out-Null
    }
    Add-Line $page 17.07 2.62 17.65 2.62 "#222222" 1.2 $true $true | Out-Null

    $cert = @(
        @{x=17.98;y=4.15;c="#65A4E9";tx=18.55;ty=3.35},
        @{x=17.98;y=2.98;c="#9A5DE5";tx=18.55;ty=3.10},
        @{x=17.98;y=2.05;c="#FF895F";tx=18.55;ty=2.82},
        @{x=17.98;y=1.14;c="#FF895F";tx=18.55;ty=1.15}
    )
    foreach ($n in $cert) {
        Add-Circle $page $n.x $n.y 0.34 $n.c "#222222" 0.8 | Out-Null
        Add-Arrow $page ($n.x+0.16) $n.y $n.tx $n.ty "#222222" 1.0 | Out-Null
    }
    $locked = @(
        @{x=18.56;y=4.14;c="#FF9A7D"}, @{x=18.56;y=3.08;c="#84D990"},
        @{x=18.56;y=2.03;c="#FF9A7D"}, @{x=18.56;y=1.14;c="#FF9A7D"}
    )
    foreach ($n in $locked) {
        Add-Circle $page $n.x $n.y 0.34 $n.c "#F4AE84" 1.6 | Out-Null
        Add-Lock $page ($n.x+0.13) ($n.y-0.05) 0.78
    }
}

function Add-OutputTokens {
    param($page)
    Add-Text $page 19.56 2.98 0.80 0.62 "B Output`nTokens" 10.5 "#111111" 1 $false | Out-Null
    $colors = @("#3C7BE0","#18B4B0","#64C64A","#FF7812","#6D40D5","#F0475E","#B44CC8","#2E77D4")
    for ($r=0; $r -lt 2; $r++) {
        for ($c=0; $c -lt 4; $c++) {
            Add-Circle $page (19.22+$c*0.30) (2.43-$r*0.36) 0.20 $colors[$r*4+$c] "#FFFFFF" 0.2 0 $true | Out-Null
        }
    }
}

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "CertVID Architecture"
    Set-Cell $page.PageSheet "PageWidth" "20.1 in"
    Set-Cell $page.PageSheet "PageHeight" "5.25 in"
    Set-Cell $page.PageSheet "PageLeftMargin" "0 in"
    Set-Cell $page.PageSheet "PageRightMargin" "0 in"
    Set-Cell $page.PageSheet "PageTopMargin" "0 in"
    Set-Cell $page.PageSheet "PageBottomMargin" "0 in"
    Set-Cell $page.PageSheet "PageShadowOffsetX" "0 in"
    Set-Cell $page.PageSheet "PageShadowOffsetY" "0 in"

    Add-Filmstrip $page
    Add-DenseTokens $page
    Add-SemanticPanel $page
    Add-CertificatePanel $page
    Add-DOptimalPanel $page
    Add-FusionPanel $page
    Add-OutputTokens $page

    # Major stage arrows.
    Add-Arrow $page 3.36 2.62 3.72 2.62 "#123C82" 4.8 | Out-Null
    Add-Arrow $page 7.60 2.62 7.76 2.62 "#123C82" 4.8 | Out-Null
    Add-Arrow $page 11.32 2.62 11.50 2.62 "#123C82" 4.8 | Out-Null
    Add-Arrow $page 15.10 2.62 15.27 2.62 "#123C82" 4.8 | Out-Null

    $doc.SaveAs($VsdxPath)

    # Avoid Visio's slow raster export filter: copy the editable page as an
    # enhanced metafile and let the clipboard provide the preview bitmap.
    $visio.ActiveWindow.SelectAll()
    $visio.ActiveWindow.Selection.Copy()
    Start-Sleep -Milliseconds 800
    $image = [System.Windows.Forms.Clipboard]::GetImage()
    if ($null -eq $image) { throw "Visio did not provide a preview image" }
    $image.Save($PngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $image.Dispose()
    $doc.Close()
    $doc = $null
    $visio.Quit()
    $visio = $null

    Write-Output "VSDX=$VsdxPath"
    Write-Output "PNG=$PngPath"
}
finally {
    if ($null -ne $doc) { try { $doc.Close() } catch { } }
    if ($null -ne $visio) { try { $visio.Quit() } catch { } }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
