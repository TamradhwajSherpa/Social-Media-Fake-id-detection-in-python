import pickle
from flask import Flask, render_template, request
import pandas as pd
# import sklearn

app = Flask(__name__, template_folder='templates')

data = pd.read_csv('F://Fake Social Media Account detection Project//Fake social_data.csv')
lr = pickle.load(open("C://Users//sherp//Desktop//Python_Project//py_projects//fake_model.pkl", 'rb'))
print(lr)

with open('F://Fake Social Media Account detection Project//Fake_Social_Media_id_detection.pkl', 'rb') as model_file:
    model = pickle.load(model_file)  # Fixed this line

@app.route("/")
def Hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pp = request.form.get('profile')
    if pp == 'Yes':
        pp = 1
    else:
        pp = 0

    url = request.form.get('url')
    pri = request.form.get('private')
    user = request.form.get('username')
    user_len = len(user)
    bio = request.form.get('lenth_of_bio')  # Fixed variable name
    bio_len = len(bio)
    post = request.form.get('number_of_post')  # Fixed variable name
    foll = request.form.get('foll')
    foloing = request.form.get('following')

    input_data = pd.DataFrame([[pp, url, pri, user_len, bio_len, post, foll, foloing]],
                              columns=['profile', 'url', 'private', 'username', 'lenth_of_bio', 'number_of_post', 'foll',
                                       'following'])
    
    prediction = lr.predict(input_data)
    print(prediction)

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5010)
