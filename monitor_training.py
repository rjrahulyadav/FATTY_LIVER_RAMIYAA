"""
Monitor training progress and auto-test when complete
"""
import time
import subprocess
import os
from pathlib import Path

def monitor_training():
    print("\n" + "=" * 70)
    print("⏳ TRAINING MONITOR")
    print("=" * 70)
    
    model_path = Path('best_model.pth')
    initial_size = model_path.stat().st_size if model_path.exists() else 0
    start_time = time.time()
    last_update = initial_size
    
    print(f"🔍 Monitoring: {model_path}")
    print(f"📁 Initial size: {initial_size / 1024 / 1024:.2f} MB\n")
    
    update_count = 0
    check_interval = 10  # Check every 10 seconds
    
    while True:
        time.sleep(check_interval)
        
        if model_path.exists():
            current_size = model_path.stat().st_size
            elapsed = time.time() - start_time
            
            # Check if file was updated
            if current_size != last_update:
                update_count += 1
                time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                print(f"   [{time_str}] Model updating... ({current_size / 1024 / 1024:.2f} MB)")
                last_update = current_size
            
            # Check if training might be complete (no updates for 2 minutes)
            if update_count > 0 and elapsed > 600:  # At least 10 minutes of training
                print("\n✅ Training appears to be complete!")
                print(f"⏱️  Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
                print("\n🧪 Running model validation tests...\n")
                
                # Run test script
                subprocess.run(['python', 'test_trained_model.py'], cwd='.')
                break
        
        # Safety timeout (prevent infinite loop)
        if time.time() - start_time > 3600:  # 1 hour timeout
            print("\n⏱️  Timeout reached (1 hour). Stopping monitor.")
            break

if __name__ == '__main__':
    monitor_training()
