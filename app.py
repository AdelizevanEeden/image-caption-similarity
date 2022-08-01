from flask import Flask, render_template, request
import os

from model import Model

app = Flask(__name__)
model = Model()

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # get image and caption from form
    image = request.files['imageFile']
    caption = request.form.get('caption', None)
    if image.filename == '' or caption == None:
        # TODO: handle bad inputs
        raise

    # save image
    filename = 'uploads/' + image.filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)

    # run image and caption through model
    similarity = model.run_model(image, caption)

    return render_template('index.html', similarity=similarity)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
