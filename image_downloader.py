# License: Apache-2.0
# image_downloader.py — Fixed version with proper timeouts

import os
import sys
import time
import platform
import argparse
import socket
from pathlib import Path
from typing import List
from functools import partial

import pandas as pd
from tqdm import tqdm

try:
    import multiprocessing
    from multiprocessing import Pool
except Exception as e:
    print(f"Warning: multiprocessing not available: {e}", file=sys.stderr)
    multiprocessing = None

# Set global socket timeout to prevent hanging
socket.setdefaulttimeout(30)  # 30 seconds max per connection

def download_single_image(image_link: str, save_folder: str, retries: int = 2, timeout: int = 15) -> str:
    """
    Download a single image with strict timeout control.
    Returns: 'success', 'exists', 'failed', or 'invalid'
    """
    if not isinstance(image_link, str) or not image_link.strip():
        return "invalid"
    
    try:
        filename = Path(image_link).name
        if not filename:
            return "invalid"
        
        save_path = Path(save_folder) / filename
        
        # Skip if already exists and has content
        if save_path.exists() and save_path.stat().st_size > 100:  # at least 100 bytes
            return "exists"
        
        # Retry loop with timeout
        for attempt in range(retries):
            try:
                import urllib.request
                
                # Create request with timeout
                req = urllib.request.Request(
                    image_link,
                    headers={'User-Agent': 'Mozilla/5.0'}  # Some servers block default user-agent
                )
                
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    with open(save_path, 'wb') as f:
                        f.write(response.read())
                
                # Verify file was written and has content
                if save_path.exists() and save_path.stat().st_size > 100:
                    return "success"
                else:
                    # File too small, probably error page
                    if save_path.exists():
                        save_path.unlink()
                    return "failed"
                    
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                # Network error or timeout
                if attempt < retries - 1:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                else:
                    # Clean up partial downloads
                    if save_path.exists():
                        try:
                            save_path.unlink()
                        except:
                            pass
                    return "failed"
            except Exception as e:
                # Unexpected error
                if save_path.exists():
                    try:
                        save_path.unlink()
                    except:
                        pass
                return "failed"
        
        return "failed"
    
    except Exception:
        return "failed"

def download_batch_sequential(links: List[str], save_folder: str) -> dict:
    """Fallback: download sequentially"""
    results = {"success": 0, "exists": 0, "failed": 0, "invalid": 0}
    for link in tqdm(links, desc="Downloading (sequential)"):
        result = download_single_image(link, save_folder)
        results[result] += 1
    return results

def download_batch_parallel(links: List[str], save_folder: str, workers: int) -> dict:
    """Download batch using multiprocessing with timeout protection"""
    results = {"success": 0, "exists": 0, "failed": 0, "invalid": 0}
    
    download_func = partial(download_single_image, save_folder=save_folder)
    
    try:
        with Pool(processes=workers) as pool:
            # Use imap_unordered with chunksize for better reliability
            for result in tqdm(pool.imap_unordered(download_func, links, chunksize=10), 
                              total=len(links), 
                              desc=f"Downloading (workers={workers})"):
                results[result] += 1
            
            # Force cleanup with timeout
            pool.close()
            pool.join(timeout=60)  # Wait max 60 seconds for cleanup
            
    except Exception as e:
        print(f"\nWarning: Pool error: {e}", file=sys.stderr)
        # Count remaining as failed
        processed = sum(results.values())
        results["failed"] += (len(links) - processed)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Download product images from CSV")
    parser.add_argument("--csv", type=str, default="dataset/train.csv")
    parser.add_argument("--images_dir", type=str, default="dataset/images")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--sequential", action="store_true")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine workers
    if args.workers is not None:
        workers = args.workers
    else:
        if platform.system().lower().startswith("win"):
            workers = 32  # Reduced for stability
        else:
            workers = 32
    
    workers = max(1, min(workers, 32))
    
    print(f"Platform: {platform.system()}")
    print(f"Workers: {workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Socket timeout: 30s")
    
    df = pd.read_csv(csv_path)
    
    if "image_link" not in df.columns:
        print(f"Error: Missing 'image_link' column.", file=sys.stderr)
        sys.exit(1)
    
    all_links = df["image_link"].dropna().astype(str).tolist()
    total_links = len(all_links)
    print(f"Total image links: {total_links}\n")
    
    if total_links == 0:
        print("No links found.")
        return
    
    overall_stats = {"success": 0, "exists": 0, "failed": 0, "invalid": 0}
    batch_num = 0
    
    for start_idx in range(0, total_links, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_links)
        batch = all_links[start_idx:end_idx]
        batch_num += 1
        
        print(f"\n--- Batch {batch_num}: images {start_idx+1} to {end_idx} ---")
        
        if args.sequential or multiprocessing is None:
            batch_stats = download_batch_sequential(batch, str(images_dir))
        else:
            batch_stats = download_batch_parallel(batch, str(images_dir), workers)
        
        for k in batch_stats:
            overall_stats[k] += batch_stats[k]
        
        print(f"Batch results: {batch_stats}")
        
        if end_idx < total_links and args.sleep > 0:
            print(f"Sleeping {args.sleep}s...")
            time.sleep(args.sleep)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total: {total_links}")
    print(f"Downloaded: {overall_stats['success']}")
    print(f"Already existed: {overall_stats['exists']}")
    print(f"Failed: {overall_stats['failed']}")
    print(f"Invalid: {overall_stats['invalid']}")
    print(f"Stored in: {images_dir.absolute()}")

if __name__ == "__main__":
    main()
