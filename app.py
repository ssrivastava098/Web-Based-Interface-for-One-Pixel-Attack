from flask import Flask, request, render_template, send_from_directory,redirect, url_for
import os

# Import the plot_image function from the PythonScriptsOnePixelAttack package
from utils_web import plot_image, real_class, onCLickAttack

app = Flask(__name__)
image_id = None
image_path = None
class_name = None
attack_image_path = None
attack_image_class_name = None
@app.route('/', methods=['GET', 'POST'])
def home():
    global image_id
    global image_path
    global class_name
    global attack_image_path
    global attack_image_class_name
    if request.method == 'POST':
        image_id = int(request.form['image_id'])
        plot_image(image_id)
        class_name = real_class(image_id)
        return render_template('index.html', image_id=image_id, image_path=f"static/images/image_{image_id}.png", class_name =class_name)
    if attack_image_path is not None:
        return render_template('index.html', image_id=image_id, image_path=f"static/images/image_{image_id}.png", class_name =class_name, attack_image_path=f"static/images/attack_image_{image_id}.png", attack_image_class_name = attack_image_class_name)

    return render_template('index.html')

@app.route('/process_button',methods=['GET','POST'])
def attack_image():
    global image_id 
    global image_path 
    global class_name
    global attack_image_path
    global attack_image_class_name
    if image_id is not None:  
        attack_image_class_name = onCLickAttack(image_id)
        attack_image_path=f"static/images/attack_image_{image_id}.png"
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
