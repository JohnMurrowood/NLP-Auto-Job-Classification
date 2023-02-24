# Assignmnet 2 job search website flask

from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
jobs_dict = {}
jobs_df = pd.read_excel('Job advertisment spreadsheet.xlsx')


# Create flask instance
app = Flask(__name__)

app.config['SECRET_KEY'] = "John_A2"

@app.route("/", methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        cats_toget = request.form.getlist('cat')
        if 'all' in cats_toget:
            jobs = jobs_df
        else:
            jobs = jobs_df[jobs_df.Category.isin(cats_toget)]
            jobs.reset_index(inplace=True, drop = True)
        return render_template("homepage.html", jobs = jobs, rows = range(jobs.shape[0]))
    else:
        return render_template("homepage.html", jobs = jobs_df, rows = range(jobs_df.shape[0]))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect('/user/'+session['username'])
    else:
        if request.method == 'POST':
            if (request.form['username'] == 'John') and (request.form['password'] == 'Testing'):
                session['username'] = request.form['username']
                return redirect(url_for('user', user = request.form['username']))
            else:
                return render_template('login.html', login_message='Username or password is invalid.')
        else:
            return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')


@app.route("/user/<user>", methods = ['GET', 'POST'])
def user(user):
    if request.method == 'POST':
        cats_toget = request.form.getlist('cat')
        if 'all' in cats_toget:
            jobs = jobs_df
        else:
            jobs = jobs_df[jobs_df.Category.isin(cats_toget)]
            jobs.reset_index(inplace=True, drop = True)
        return render_template("user.html", jobs = jobs, rows = range(jobs.shape[0]), user = user)
    else:
        return render_template("user.html", jobs = jobs_df, rows = range(jobs_df.shape[0]), user = user)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if 'username' in session:
        if request.method == 'POST':
            global jobs_df
            # Read the content
            f_title = request.form['title']
            f_company = request.form['company']
            f_content = request.form['description']

            # Tokenize the content so as to input to the saved model
            pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
            tokenizer = RegexpTokenizer(pattern)
            tokenized_data = tokenizer.tokenize(f_content)
            count = 0
            for i in tokenized_data:
                tokenized_data[count] = i.lower()
                if len(i) < 2:
                    tokenized_data.pop(count)
                count +=1
            # Remove stopwords
            no_stop = []
            stop_words = set(stopwords.words('english'))
            for w in tokenized_data:
                if w not in stop_words:
                    no_stop.append(w)
            tokenized_data = no_stop

            # Create count vector representation
            pkl_filename = "cVectorizer.pkl"
            with open(pkl_filename, 'rb') as file:
                cVectorizer = pickle.load(file)
            count_features = cVectorizer.fit_transform([' '.join(tokenized_data)]) # Get count vector representation for all descriptions
            # Load the LR model
            pkl_filename = "model_lr.pkl"
            with open(pkl_filename, 'rb') as file2:
                model = pickle.load(file2)

            # Predict the label of tokenized_data
            y_pred = model.predict(count_features)
            y_pred = y_pred[0]
            global jobs_df
            jobs_df = jobs_df.append({'Title' : f_title, 'Company' : f_company, 'Description' : f_content, 'Category' : y_pred}, ignore_index=True)
            return redirect(url_for('postjob'))
        else:
            return render_template('classify.html')
    else:
        return redirect('/user/'+session['username'])

@app.route('/postjob', methods=['GET', 'POST'])
def postjob():
    if 'username' in session:
        if request.method == 'POST':
            new_cat = request.form['category']
            global jobs_df
            jobs_df['Category'].iloc[-1] = new_cat
            return redirect('/user/'+session['username']) 
        else:
            title = jobs_df['Title'].iloc[-1]
            company = jobs_df['Company'].iloc[-1]
            description = jobs_df['Description'].iloc[-1]
            predicted_message = "Reccomended Category is {}, This can be changed by selecting below.".format(jobs_df['Category'].iloc[-1])
            return render_template('postjob.html', predicted_message=predicted_message, title=title, description=description, company=company)


@app.route('/job/<title>')
def job(title):
    df_title = jobs_df[jobs_df['Title']==title]
    company = df_title['Company'].values[0]
    category = df_title['Category'].values[0]
    description = df_title['Description'].values[0]
    return render_template("job.html", title=title, company=company, category=category, description=description)


app.run()