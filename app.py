import os
from flask import Flask, render_template, redirect, url_for
import subprocess
import sys
import threading
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/launch_cat_derma')
def launch_cat_derma():
    # Redirect to the CatDerma app running on port 5001
    return redirect('http://127.0.0.1:5001/')

@app.route('/launch_dog_derma')
def launch_dog_derma():
    # Redirect to the DogDerma app running on port 5002
    return redirect('http://127.0.0.1:5002/')

def start_cat_app():
    cat_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CatDerma', 'app.py')
    subprocess.Popen([sys.executable, cat_app_path], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CatDerma'))

def start_dog_app():
    dog_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DogDerma', 'app.py')
    subprocess.Popen([sys.executable, dog_app_path], cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DogDerma'))

if __name__ == '__main__':
    # Start the CatDerma and DogDerma apps in separate threads
    cat_thread = threading.Thread(target=start_cat_app)
    cat_thread.daemon = True
    cat_thread.start()
    
    dog_thread = threading.Thread(target=start_dog_app)
    dog_thread.daemon = True
    dog_thread.start()
    
    # Open a web browser to the main PetDerma app
    webbrowser.open('http://127.0.0.1:5000/')
    
    # Run the main PetDerma app
    app.run(host='127.0.0.1', port=5000, debug=False)
