### Frontend'i diğer servisleri etkilemeden hızlıca build etme

- Sadece frontend'i yeniden build ve restart etmek (diğer servisleri hiç dokunmadan):

```bash
docker compose -f docker-compose.yml up -d --no-deps --build frontend
```

- Windows PowerShell için hazır script:

```powershell
.\scripts\frontend-rebuild.ps1 -NoDeps
```

- Mac/Linux için hazır script:

```bash
bash scripts/frontend-rebuild.sh
```

### Neden bu yöntem hızlı?
- `--no-deps`: Frontend sadece kendisi yeniden derlenir, diğer servisler yeniden build edilmez ve yeniden başlatılmaz.
- `frontend/.dockerignore`: Build context küçülür; `node_modules`, `.next` ve gereksiz dosyalar konteynıra gönderilmez.

### Full stack başlatma (ilk kurulum veya backend değiştiğinde)

```bash
docker compose up -d
```

### Faydalı komutlar

- Frontend logları:
```bash
docker compose logs -f frontend
```

- Frontend'i sadece yeniden başlatmak:
```bash
docker compose restart frontend
```

- Sadece frontend imajını yeniden build etmek:
```bash
docker compose build frontend
```



