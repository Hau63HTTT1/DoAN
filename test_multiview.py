import os
import sys
import subprocess

def test_multiview():
    """Test if multiview model can run properly"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    detect_dir = os.path.join(project_root, 'model', 'PIFu-multiview')
    detect_py = os.path.join(detect_dir, 'detect.py')
    
    print(f"Testing multiview model...")
    print(f"Project root: {project_root}")
    print(f"Detect dir: {detect_dir}")
    print(f"Detect py: {detect_py}")
    
    # Check if files exist
    if not os.path.exists(detect_py):
        print(f"ERROR: detect.py not found at {detect_py}")
        return False
    
    # Check if sample images exist
    sample_dir = os.path.join(detect_dir, 'sample_images', 'rp_Man')
    required_files = [
        '0_0_00.png', '90_0_00.png', '180_0_00.png', '270_0_00.png',
        '0_0_00_mask.png', '90_0_00_mask.png', '180_0_00_mask.png', '270_0_00_mask.png'
    ]
    
    for file in required_files:
        file_path = os.path.join(sample_dir, file)
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return False
        else:
            print(f"✓ Found: {file}")
    
    # Check if checkpoints exist
    checkpoint_dir = os.path.join(detect_dir, 'checkpoints')
    net_g_path = os.path.join(checkpoint_dir, 'net_G')
    net_c_path = os.path.join(checkpoint_dir, 'net_C')
    
    if not os.path.exists(net_g_path):
        print(f"ERROR: net_G checkpoint not found at {net_g_path}")
        return False
    else:
        print(f"✓ Found: net_G checkpoint")
    
    if not os.path.exists(net_c_path):
        print(f"ERROR: net_C checkpoint not found at {net_c_path}")
        return False
    else:
        print(f"✓ Found: net_C checkpoint")
    
    # Try to run the model
    try:
        print(f"\nRunning multiview model...")
        process = subprocess.Popen([
            sys.executable, detect_py
        ], cwd=detect_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout, stderr = process.communicate()
        
        print(f"Return code: {process.returncode}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        
        if process.returncode == 0:
            print("✓ Multiview model ran successfully!")
            
            # Check if output was created
            output_dir = os.path.join(detect_dir, 'results', 'pifu_demo', 'rp_Man')
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"Files in output directory: {files}")
                if any(f.endswith('.obj') for f in files):
                    print("✓ Found .obj file in output!")
                    return True
                else:
                    print("✗ No .obj file found in output")
                    return False
            else:
                print("✗ Output directory not created")
                return False
        else:
            print("✗ Multiview model failed to run")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_multiview()
    if success:
        print("\n🎉 Multiview test PASSED!")
    else:
        print("\n❌ Multiview test FAILED!") 