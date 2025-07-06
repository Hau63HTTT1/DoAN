import os
import sys
import subprocess

def run_test():
    # Get the absolute path of the project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Add project directory to Python path
    sys.path.append(project_dir)
    os.environ['PYTHONPATH'] = project_dir
    
    # Set CUDA environment variables for better performance
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting
    
    # Prepare the command with optimized parameters
    cmd = [
        sys.executable,
            os.path.join(project_dir, 'model', 'PiFu-singleview', 'apps', 'simple_test.py'),
    '-i', os.path.join(project_dir, 'model', 'PiFu-singleview', 'sample_images'),
    '-o', os.path.join(project_dir, 'model', 'PiFu-singleview', 'results'),
    '-c', os.path.join(project_dir, 'model', 'PiFu-singleview', 'checkpoints', 'pifuhd.pt'),
        '-r', '128',  # Reduced resolution for faster processing
        '--use_rect'
    ]
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Print any errors
        stderr = process.stderr.read()
        if stderr:
            print("Errors:", stderr)
        
        return process.poll()
    
    except Exception as e:
        print(f"Error running test: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(run_test()) 