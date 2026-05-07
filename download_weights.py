#!/usr/bin/env python3
"""
download_weights.py — YOLOPv2 + CLRNet model ağırlıklarını indir
"""

import sys
import os
import urllib.request
import urllib.error
import zipfile
import tempfile
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── İndirilecek modeller ──────────────────────────────────────────────────────

WEIGHTS_DIR = Path(__file__).parent / 'data' / 'weights'

MODELS = [
    {
        'name':   'YOLOPv2',
        'dest':   WEIGHTS_DIR / 'yolopv2.pt',
        'urls':  [
            'https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt',
        ],
        'manual': 'https://github.com/CAIC-AD/YOLOPv2/releases/tag/V0.0.1',
    },
    {
        'name':      'CLRNet culane_r18',
        'dest':      WEIGHTS_DIR / 'culane_r18.pth',
        # Resmi dağıtım .zip arşivi içinde — indirilip çıkartılır
        'urls':     [
            'https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip',
        ],
        'zip_name':  'culane_r18.pth',   # zip içindeki hedef dosya adı
        'manual':    'https://github.com/Turoad/CLRNet/releases/tag/models',
    },
]

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _sizeof_fmt(num):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num < 1024.0:
            return f'{num:.1f} {unit}'
        num /= 1024.0
    return f'{num:.1f} TB'


def _download(url: str, dest: Path) -> bool:
    """
    URL'yi dest'e indirir. Başarılı → True, hata → False.
    İndirme sırasında tqdm (varsa) veya basit % gösterimi kullanır.
    """
    tmp = dest.with_suffix(dest.suffix + '.part')

    if HAS_TQDM:
        pbar = None

        def reporthook(block_num, block_size, total_size):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(
                    total=total_size if total_size > 0 else None,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc=f'  İndiriliyor', leave=False,
                    bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{rate_fmt}]',
                )
            downloaded = block_num * block_size
            if total_size > 0:
                pbar.n = min(downloaded, total_size)
            else:
                pbar.n = downloaded
            pbar.refresh()

        try:
            urllib.request.urlretrieve(url, tmp, reporthook)
        except Exception as e:
            if pbar:
                pbar.close()
            if tmp.exists():
                tmp.unlink()
            return False, str(e)
        finally:
            if pbar:
                pbar.close()
    else:
        # Basit % gösterimi
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                bar  = '#' * int(pct // 2)
                print(f'\r  [{bar:<50}] {pct:5.1f}%  {_sizeof_fmt(downloaded)}', end='')
            else:
                print(f'\r  İndirildi: {_sizeof_fmt(downloaded)}', end='')

        try:
            urllib.request.urlretrieve(url, tmp, reporthook)
            print()
        except Exception as e:
            print()
            if tmp.exists():
                tmp.unlink()
            return False, str(e)

    tmp.rename(dest)
    return True, None


def _extract_zip(zip_path: Path, member_name: str, dest: Path) -> tuple[bool, str]:
    """zip_path içinden member_name dosyasını dest'e çıkartır."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            # Tam eşleşme, sonra suffix eşleşmesi
            match = next((n for n in names if n == member_name), None)
            if match is None:
                match = next((n for n in names if n.endswith('/' + member_name) or n == member_name), None)
            if match is None:
                return False, f'Zip içinde "{member_name}" bulunamadı. İçerik: {names}'
            data = zf.read(match)
        dest.write_bytes(data)
        return True, None
    except Exception as e:
        return False, str(e)


def download_model(model: dict) -> bool:
    dest     = model['dest']
    name     = model['name']
    urls     = model['urls']
    zip_name = model.get('zip_name')          # varsa zip'ten çıkart
    is_zip   = zip_name is not None

    print(f'\n{"─"*55}')
    print(f'  {name}')
    print(f'  Hedef : {dest}')

    if dest.exists():
        size = _sizeof_fmt(dest.stat().st_size)
        print(f'  ✓ Zaten mevcut ({size}) — atlanıyor.')
        return True

    for i, url in enumerate(urls, 1):
        label = f'URL {i}/{len(urls)}' if len(urls) > 1 else 'URL'
        print(f'  {label}: {url}')

        if is_zip:
            # Geçici dosyaya zip indir, sonra çıkart
            tmp_zip = dest.with_suffix('.zip.part')
            ok, err = _download(url, tmp_zip)
            if not ok:
                print(f'  ✗ İndirme başarısız: {err}')
                continue
            zip_size = _sizeof_fmt(tmp_zip.stat().st_size)
            print(f'  ↳ Arşiv indirildi ({zip_size}), çıkartılıyor...')
            ok, err = _extract_zip(tmp_zip, zip_name, dest)
            tmp_zip.unlink(missing_ok=True)
            if not ok:
                print(f'  ✗ Çıkartma başarısız: {err}')
                continue
        else:
            ok, err = _download(url, dest)
            if not ok:
                print(f'  ✗ Başarısız: {err}')
                continue

        size = _sizeof_fmt(dest.stat().st_size)
        print(f'  ✓ Hazır ({size})')
        return True

    print(f'\n  ! Tüm URL\'ler başarısız oldu.')
    print(f'  ! Manuel indirme: {model["manual"]}')
    print(f'  ! Hedef konum   : {dest}')
    return False


# ── Özet ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[tuple]):
    print(f'\n{"═"*55}')
    print('  ÖZET — data/weights/')
    print(f'{"─"*55}')

    # Beklenen dosyalar
    for name, dest, ok in results:
        if dest.exists():
            size  = _sizeof_fmt(dest.stat().st_size)
            durum = f'✓ mevcut  ({size})'
        else:
            durum = '✗ eksik'
        print(f'  {durum:<28}  {dest.name}')

    # Dizinde olan ama listede olmayan dosyalar
    expected = {r[1] for r in results}
    extras   = [f for f in WEIGHTS_DIR.glob('*')
                if f.is_file() and f not in expected and f.suffix != '.part']
    if extras:
        print(f'{"─"*55}')
        print('  Dizinde bulunan diğer dosyalar:')
        for f in sorted(extras):
            print(f'  {"·":<28}  {f.name}  ({_sizeof_fmt(f.stat().st_size)})')

    print(f'{"═"*55}')

    missing = [name for name, dest, ok in results if not dest.exists()]
    if missing:
        print(f'\n  [!] Eksik: {", ".join(missing)}')
        print('      Yukarıdaki manuel indirme bağlantılarını kullanabilirsiniz.\n')
    else:
        print('\n  Tüm ağırlıklar hazır.\n')


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  download_weights.py — Model ağırlıkları')
    print('=' * 55)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'  Hedef dizin: {WEIGHTS_DIR}')

    results = []
    for model in MODELS:
        ok = download_model(model)
        results.append((model['name'], model['dest'], ok))

    print_summary(results)


if __name__ == '__main__':
    main()
