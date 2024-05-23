from flask import Flask, request, render_template
from model import recommend_movies

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['userId'])
    already_rated, predictions = recommend_movies(user_id)
    return render_template('recommend.html', predictions=predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
