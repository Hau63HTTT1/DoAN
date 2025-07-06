import requests
import os
import argparse

def upload_image(image_path, server_url):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    files = {
        'file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        print(f"Uploading image to {server_url}/upload...")
        response = requests.post(f"{server_url}/upload", files=files)
        
        if response.status_code == 200:
            output_path = os.path.splitext(image_path)[0] + "_3d.obj"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"3D model saved to {output_path}")
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    finally:
        files['file'][1].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload an image to PiFuHD server")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("server_url", help="Ngrok URL from Colab (e.g., http://xxxx.ngrok.io)")
    
    args = parser.parse_args()
    upload_image(args.image_path, args.server_url) 