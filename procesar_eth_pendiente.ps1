$ErrorActionPreference = "Stop"

$periodos = @(
    @{ Desde = "2024-01-01"; Hasta = "2025-01-01" },
    @{ Desde = "2023-01-01"; Hasta = "2024-01-01" },
    @{ Desde = "2022-01-01"; Hasta = "2023-01-01" },
    @{ Desde = "2021-01-01"; Hasta = "2022-01-01" },
    @{ Desde = "2026-01-01"; Hasta = "2026-06-01" }
)

foreach ($periodo in $periodos) {
    $desde = $periodo.Desde
    $hasta = $periodo.Hasta

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "PREPARANDO ETHUSDT: $desde hasta $hasta"
    Write-Host "============================================================"

    python -m cripto.corto_plazo.preparar_datos `
        --simbolo ETHUSDT `
        --desde $desde `
        --hasta $hasta

    if ($LASTEXITCODE -ne 0) {
        throw "Falló la preparación de ETHUSDT: $desde hasta $hasta"
    }

    $archivo = "datos/cripto/preparados/corto_plazo/ETHUSDT_1m_4h_${desde}_${hasta}.parquet"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "CREANDO VARIABLES: $archivo"
    Write-Host "============================================================"

    python -m cripto.corto_plazo.variables `
        --archivo $archivo

    if ($LASTEXITCODE -ne 0) {
        throw "Falló la creación de variables: $archivo"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "TODOS LOS PERIODOS PENDIENTES DE ETHUSDT FUERON PROCESADOS"
Write-Host "============================================================"
