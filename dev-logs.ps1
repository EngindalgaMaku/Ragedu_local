# Development Mode - Logları göster
param(
    [string]$Service = ""
)

if ($Service -eq "") {
    Write-Host "📋 Tüm servislerin logları gösteriliyor..." -ForegroundColor Cyan
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
} else {
    Write-Host "📋 $Service servisinin logları gösteriliyor..." -ForegroundColor Cyan
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $Service
}



