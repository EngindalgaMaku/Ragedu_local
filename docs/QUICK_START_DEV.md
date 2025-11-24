# ⚡ Hızlı Başlangıç - Development Mode

## 🎯 Cursor'u Yeniden Başlat!

Task'ları görmek için **Cursor'u tamamen kapatıp tekrar açmanız gerekiyor.**

## 🚀 Kullanım

### Yöntem 1: Task'lar (Cursor'u yeniden başlattıktan sonra)

1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. "🚀 Dev: Development Mode Başlat" seç

### Yöntem 2: Script (Hemen çalışır)

```powershell
.\dev.ps1
```

## 📋 Tüm Komutlar

### Task'lar (Cursor'u yeniden başlattıktan sonra)
- `Ctrl+Shift+P` → "Tasks: Run Task" → Task seç

### Script'ler (Hemen çalışır)
```powershell
.\dev.ps1          # Development mode başlat
.\dev-up.ps1       # Arka planda başlat
.\dev-down.ps1     # Durdur
.\dev-logs.ps1     # Logları göster
```

## ⚠️ Önemli

**Cursor'u yeniden başlatmadan task'lar görünmez!**

1. Cursor'u tamamen kapat
2. Tekrar aç
3. `Ctrl+Shift+P` → "Tasks: Run Task"

## 🔧 Sorun Giderme

### Task'lar görünmüyor?
1. ✅ Cursor'u yeniden başlattın mı?
2. ✅ `.vscode/tasks.json` dosyası var mı?
3. ✅ `Ctrl+Shift+P` → "Tasks: Run Task" yazdın mı?

### Script çalışmıyor?
```powershell
# PowerShell execution policy kontrolü
Get-ExecutionPolicy

# Eğer Restricted ise:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```



