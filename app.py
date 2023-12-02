from flask import Flask, render_template, request, session
from sklearn.preprocessing import PolynomialFeatures
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
import random
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'xyzsdfg'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/login_flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

linear_regression_model = pickle.load(open(os.path.join('pickle_files', 'linear_regression.pkl'), 'rb'))
multiple_regression_TV_Radio_Media_model = pickle.load(open(os.path.join('pickle_files', 'multiple_regression_TV_Radio_Media.pkl'), 'rb'))
multiple_regression_model = pickle.load(open(os.path.join('pickle_files', 'multiple_regression.pkl'), 'rb'))
polynomial_regression_TV_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_TV.pkl'), 'rb'))
polynomial_regression_Print_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Print.pkl'), 'rb'))
polynomial_regression_Radio_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Radio.pkl'), 'rb'))
polynomial_regression_Google_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Google.pkl'), 'rb'))
polynomial_regression_Insta_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Insta.pkl'), 'rb'))
polynomial_regression_Youtube_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Youtube.pkl'), 'rb'))
polynomial_regression_Hoarding_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Hoarding.pkl'), 'rb'))
polynomial_regression_Other_model = pickle.load(open(os.path.join('pickle_files', 'polynomial_regression_Other.pkl'), 'rb'))

advertisers = list(linear_regression_model.keys())
advertisers.sort()

class User(db.Model):
    serial_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

class AdvertisingData(db.Model):
    serial_id = db.Column(db.Integer, primary_key=True) 
    advertiser = db.Column(db.String(200))
    year = db.Column(db.Integer)
    month = db.Column(db.Integer)
    Tv = db.Column(db.Numeric(25, 10))
    Print_Media = db.Column(db.Numeric(25, 10))
    Radio = db.Column(db.Numeric(25, 10))
    Google = db.Column(db.Numeric(25, 10))
    Fb_and_Insta = db.Column(db.Numeric(25, 10))
    Youtube = db.Column(db.Numeric(25, 10))
    Hoardings = db.Column(db.Numeric(25, 10))
    Others = db.Column(db.Numeric(25, 10))
    Total = db.Column(db.Numeric(25, 10))
    Sales = db.Column(db.Numeric(25, 10))

class BestAdvertisingModel(db.Model):
    serial_id = db.Column(db.Integer, primary_key=True) 
    advertiser = db.Column(db.String(200))
    Best_Model = db.Column(db.String(200))

@app.route('/')
def index():
    session['loggedin'] = False 
    return render_template('login_html.html')

@app.route('/log')
def index2():
    return render_template('login_html.html')

@app.route('/front')
def front():
    return render_template('front.html', advertisers=advertisers, tv=0, media=0, radio=0, google=0, insta=0, youtube=0, hoardings=0, others=0)

@app.route('/budget')
def budget():
    return render_template('budget.html', advertisers=advertisers, budget=0)

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/logout')
def logout():
    session['loggedin'] = False
    return render_template('login_html.html')
    
@app.route('/csv')
def csv():
    return render_template('add_csv.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email, password=password).first()
    if user:
        session['loggedin'] = True
        message = 'Login Successful !'
    else:
        message = 'Invalid Email or Password !'
    return render_template('login_html.html', message=message)

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password1 = request.form['password1']
    password2 = request.form['password2']
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        message = 'Account already exists !'
    else:
        if password1 != password2:
            message = 'Passwords Do Not Match !'
        else:
            new_user = User(email=email, password=password1)
            db.session.add(new_user)
            db.session.commit()
            message = 'Sign Up Successful !'
    return render_template('login_html.html', message=message)

#tushar lstm start
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#tushar lstm end

@app.route('/predict', methods=['POST'])
def predict():
    advertiser = request.form['advertiser']
    tv = float(request.form['tv'])
    if tv < 0:
        tv = -tv
    if tv == 0:
        tv = 1
    media = float(request.form['media'])
    if media < 0:
        media = -media
    if media == 0:
        media = 1
    radio = float(request.form['radio'])
    if radio < 0:
        radio = -radio
    if radio == 0:
        radio = 1
    google = float(request.form['google'])
    if google < 0:
        google = -google
    if google == 0:
        google = 1
    insta = float(request.form['insta'])
    if insta < 0:
        insta = -insta
    if insta == 0:
        insta = 1
    youtube = float(request.form['youtube'])
    if youtube < 0:
        youtube = -youtube
    if youtube == 0:
        youtube = 1
    hoardings = float(request.form['hoardings'])
    if hoardings < 0:
        hoardings = -hoardings
    if hoardings == 0:
        hoardings = 1
    others = float(request.form['others'])
    if others < 0:
        others = -others
    if others == 0:
        others = 1
    
    best_advertiser_model = BestAdvertisingModel.query.filter(BestAdvertisingModel.advertiser == advertiser).all()
    for i in best_advertiser_model:
        if i.Best_Model.strip() == 'Linear_Regression':
            model = linear_regression_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv]})
            prediction = model.predict(data)
            if prediction < 0:
                prediction = -prediction / 2
            prediction = prediction[0][0]
        elif i.Best_Model.strip() == 'Multiple_Regression_TV_Radio_Media':
            model = multiple_regression_TV_Radio_Media_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv], 'Print Media': [media], 'RADIO': [radio]})
            prediction = model.predict(data)
            if prediction < 0:
                prediction = -prediction / 2
            prediction = prediction[0][0]
        elif i.Best_Model.strip() == 'Multiple_Regression':
            model = multiple_regression_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv], 'Print Media': [media], 'RADIO': [radio], 'GOOGLE': [google], 'FB & Insta': [insta], 'YOUTUBE': [youtube], 'HOARDINGS': [hoardings], 'OTHERS': [others]})
            prediction = model.predict(data)
            if prediction < 0:
                prediction = -prediction / 2
            prediction = prediction[0][0]
        elif i.Best_Model.strip() == 'Polynomial_Regression_TV':
            poly_features = PolynomialFeatures(degree=2)
            model = polynomial_regression_TV_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv]})
            data = poly_features.fit_transform(data)
            prediction = model.predict(data)
            if prediction < 0:
                prediction = -prediction / 2
            prediction = prediction[0][0]
        elif i.Best_Model.strip() == 'Polynomial_Regression_TV_Radio_Media':
            poly_features = PolynomialFeatures(degree=2)
            model = polynomial_regression_TV_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv]})
            data = poly_features.fit_transform(data)
            prediction_tv = model.predict(data)
            if prediction_tv < 0:
                prediction_tv = -prediction_tv / 2
            model = polynomial_regression_Print_model.get(advertiser)
            data = pd.DataFrame({'Print Media': [media]})
            data = poly_features.fit_transform(data)
            prediction_media = model.predict(data)
            if prediction_media < 0:
                prediction_media = -prediction_media / 2
            model = polynomial_regression_Radio_model.get(advertiser)
            data = pd.DataFrame({'RADIO': [radio]})
            data = poly_features.fit_transform(data)
            prediction_radio = model.predict(data)
            if prediction_radio < 0:
                prediction_radio = -prediction_radio / 2
            total = tv + media + radio
            p1 = tv / total
            p2 = media / total
            p3 = radio / total
            prediction = p1 * prediction_tv + p2 * prediction_media + p3 * prediction_radio
            prediction = prediction[0][0]
        elif i.Best_Model.strip() == 'Polynomial_Regression_TV_to_Others':
            poly_features = PolynomialFeatures(degree=2)
            model = polynomial_regression_TV_model.get(advertiser)
            data = pd.DataFrame({'TV': [tv]})
            data = poly_features.fit_transform(data)
            prediction_tv = model.predict(data)
            if prediction_tv < 0:
                prediction_tv = -prediction_tv / 2
            model = polynomial_regression_Print_model.get(advertiser)
            data = pd.DataFrame({'Print Media': [media]})
            data = poly_features.fit_transform(data)
            prediction_media = model.predict(data)
            if prediction_media < 0:
                prediction_media = -prediction_media / 2
            model = polynomial_regression_Radio_model.get(advertiser)
            data = pd.DataFrame({'RADIO': [radio]})
            data = poly_features.fit_transform(data)
            prediction_radio = model.predict(data)
            if prediction_radio < 0:
                prediction_radio = -prediction_radio / 2
            model = polynomial_regression_Google_model.get(advertiser)
            data = pd.DataFrame({'GOOGLE': [google]})
            data = poly_features.fit_transform(data)
            prediction_google = model.predict(data)
            if prediction_google < 0:
                prediction_google = -prediction_google / 2
            model = polynomial_regression_Insta_model.get(advertiser)
            data = pd.DataFrame({'FB & Insta': [insta]})
            data = poly_features.fit_transform(data)
            prediction_insta = model.predict(data)
            if prediction_insta < 0:
                prediction_insta = -prediction_insta / 2
            model = polynomial_regression_Youtube_model.get(advertiser)
            data = pd.DataFrame({'YOUTUBE': [youtube]})
            data = poly_features.fit_transform(data)
            prediction_youtube = model.predict(data)
            if prediction_youtube < 0:
                prediction_youtube = -prediction_youtube / 2
            model = polynomial_regression_Hoarding_model.get(advertiser)
            data = pd.DataFrame({'HOARDINGS': [hoardings]})
            data = poly_features.fit_transform(data)
            prediction_hoardings = model.predict(data)
            if prediction_hoardings < 0:
                prediction_hoardings = -prediction_hoardings / 2
            model = polynomial_regression_Other_model.get(advertiser)
            data = pd.DataFrame({'OTHERS': [others]})
            data = poly_features.fit_transform(data)
            prediction_others = model.predict(data)
            if prediction_others < 0:
                prediction_others = -prediction_others / 2
            total = tv + media + radio + google + insta + youtube + hoardings + others
            p1 = tv / total
            p2 = media / total
            p3 = radio / total
            p4 = google / total
            p5 = insta / total
            p6 = youtube / total
            p7 = hoardings / total
            p8 = others / total
            prediction = p1 * prediction_tv + p2 * prediction_media + p3 * prediction_radio + p4 * prediction_google + p5 * prediction_insta + p6 * prediction_youtube + p7 * prediction_hoardings + p8 * prediction_others
            prediction = prediction[0][0]
        else:
            #tushar lstm start
            current_date = datetime.today()
            current_month = current_date.month
            current_year = current_date.year
            total = tv + media + radio + google + insta + youtube + hoardings + others
            data = pd.DataFrame({'year': [current_year], 'month': [current_month], 'TV': [tv], 'Print Media': [media], 'RADIO': [radio], 'GOOGLE': [google], 'FB & Insta': [insta], 'YOUTUBE': [youtube], 'HOARDINGS': [hoardings], 'OTHERS': [others], 'Total': [total], 'Sales': 0})
            advertiser_data = AdvertisingData.query.filter(AdvertisingData.advertiser == advertiser).all()
            data_list = [
                {'year': item.year, 'month': item.month, 'TV': item.Tv, 'Print Media': item.Print_Media, 'RADIO': item.Radio, 'GOOGLE': item.Google, 'FB & Insta': item.Fb_and_Insta, 'YOUTUBE': item.Youtube, 'HOARDINGS': item.Hoardings, 'OTHERS': item.Others, 'Total': item.Total, 'Sales': item.Sales} 
                for item in advertiser_data
            ]
            existing_data = pd.DataFrame(data_list)
            data = pd.concat([existing_data, data], ignore_index=True)
            num_data_points = len(data)
            values = data.values
            encoder = LabelEncoder()
            values[:,1] = encoder.fit_transform(values[:,1])
            values = values.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            reframed = series_to_supervised(scaled, 1, 1)    
            values = reframed.values
            train = values[:num_data_points-2, :] 
            test = values[num_data_points-2:, :] 
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            saved_model = load_model(os.path.join('h5_files', f'lstm_{advertiser}.h5'))
            pred = saved_model.predict(test_X)
            if train_y[-1] == 0: 
                per_diff = pred[0][0]
            else:
                per_diff = (pred[0][0] - train_y[-1]) * 100 / train_y[-1]
            p_d = per_diff
            dt=existing_data.values
            check=dt[-1:,:]
            last_row_sales=check[:,-1]
            if p_d == 0:
                new_pred = float(last_row_sales[0])
            else:
                new_pred = float(last_row_sales[0]) + p_d * float(last_row_sales[0]) / 100 
            prediction = new_pred
            #tushar lstm end

    attributes = ['Tv', 'Print_Media', 'Radio', 'Google', 'Fb_and_Insta', 'Youtube', 'Hoardings', 'Others']
    values = [tv, media, radio, google, insta, youtube, hoardings, others]
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=attributes, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    advertiser_data = AdvertisingData.query.filter(AdvertisingData.advertiser == advertiser).all()
    months_years_values = [str(item.month) + '-' + str(item.year) for item in advertiser_data]
    current_date = datetime.today()
    current_month = current_date.month
    current_year = current_date.year
    months_years_values.append(str(current_month) + '-' + str(current_year))
    total_values = [item.Total for item in advertiser_data]
    sales_values = [item.Sales for item in advertiser_data]
    total = tv + media + radio + google + insta + youtube + hoardings + others
    total_values.append(total)
    sales_values.append("{:.2f}".format(prediction))
    plt.figure(figsize=(10, 6))
    plt.plot(months_years_values, total_values, marker='o', linestyle='-', color='b', label='Total')
    plt.plot(months_years_values, sales_values, marker='o', linestyle='-', color='r', label='Sales')
    plt.xlabel('Timeline')
    plt.ylabel('Total and Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer_line_chart = io.BytesIO()
    plt.savefig(buffer_line_chart, format='png')
    buffer_line_chart.seek(0)
    line_chart_data = base64.b64encode(buffer_line_chart.getvalue()).decode('utf-8')
    plt.close()

    return render_template('front.html', advertisers=advertisers, advertiser=advertiser, tv=tv, media=media, radio=radio, google=google, insta=insta, youtube=youtube, hoardings=hoardings, others=others, prediction="{:.2f}".format(prediction), plot_image=plot_data, line_chart_image=line_chart_data)

    
@app.route('/predict1', methods=['POST'])
def predict1():
    advertiser = request.form['advertiser']
    budget = float(request.form['budget'])
    budget = int(budget)
    if budget < 0:
        budget = -budget
    if budget == 0:
        budget = 1
    advertiser_data = AdvertisingData.query.filter(AdvertisingData.advertiser == advertiser).all()
    Tv_List = []
    Print_Media_List = []
    Radio_List = []
    Google_List = []
    Fb_and_Insta_List = []
    Youtube_List = []
    Hoardings_List = []
    Others_List = []
    Total_List = []
    for data in advertiser_data:
        Tv_List.append(data.Tv)
        Print_Media_List.append(data.Print_Media)
        Radio_List.append(data.Radio)
        Google_List.append(data.Google)
        Fb_and_Insta_List.append(data.Fb_and_Insta)
        Youtube_List.append(data.Youtube)
        Hoardings_List.append(data.Hoardings)
        Others_List.append(data.Others)
        Total_List.append(data.Total)
        
    Tv_Percentage = [Tv / Total * 100 for Tv, Total in zip(Tv_List, Total_List)]
    Print_Media_Percentage = [Print_Media / Total * 100 for Print_Media, Total in zip(Print_Media_List, Total_List)]
    Radio_Percentage = [Radio / Total * 100 for Radio, Total in zip(Radio_List, Total_List)]
    Google_Percentage = [Google / Total * 100 for Google, Total in zip(Google_List, Total_List)]
    Fb_and_Insta_Percentage = [Fb_and_Insta / Total * 100 for Fb_and_Insta, Total in zip(Fb_and_Insta_List, Total_List)]
    Youtube_Percentage = [Youtube / Total * 100 for Youtube, Total in zip(Youtube_List, Total_List)]
    Hoardings_Percentage = [Hoardings / Total * 100 for Hoardings, Total in zip(Hoardings_List, Total_List)]
    Others_Percentage = [Others / Total * 100 for Others, Total in zip(Others_List, Total_List)]


    Min_Tv_Percentage = float(min(Tv_Percentage))
    Max_Tv_Percentage = float(max(Tv_Percentage))
    Avg_Tv_Percentage = float(sum(Tv_Percentage) / len(Tv_Percentage))

    Min_Media_Percentage = float(min(Print_Media_Percentage))
    Max_Media_Percentage = float(max(Print_Media_Percentage))
    Avg_Media_Percentage = float(sum(Print_Media_Percentage) / len(Print_Media_Percentage))

    Min_Radio_percentage = float(min(Radio_Percentage))
    Max_Radio_percentage = float(max(Radio_Percentage))
    Avg_Radio_Percentage = float(sum(Radio_Percentage) / len(Radio_Percentage))

    Min_Google_percentage = float(min(Google_Percentage))
    Max_Google_percentage = float(max(Google_Percentage))
    Avg_Google_Percentage = float(sum(Google_Percentage) / len(Google_Percentage))

    Min_Insta_percentage = float(min(Fb_and_Insta_Percentage))
    Max_Insta_percentage = float(max(Fb_and_Insta_Percentage))
    Avg_Insta_Percentage = float(sum(Fb_and_Insta_Percentage) / len(Fb_and_Insta_Percentage))

    Min_Youtube_percentage = float(min(Youtube_Percentage))
    Max_Youtube_percentage = float(max(Youtube_Percentage))
    Avg_Youtube_Percentage = float(sum(Youtube_Percentage) / len(Youtube_Percentage))

    Min_Hoardings_percentage = float(min(Hoardings_Percentage))
    Max_Hoardings_percentage = float(max(Hoardings_Percentage))
    Avg_Hoardings_Percentage = float(sum(Hoardings_Percentage) / len(Hoardings_Percentage))

    Min_Others_percentage = float(min(Others_Percentage))
    Max_Others_percentage = float(max(Others_Percentage))
    Avg_Others_Percentage = float(sum(Others_Percentage) / len(Others_Percentage))


                                
    Tv = []
    Media = []
    Radio = []
    Google = []
    Insta = []
    Youtube = []
    Hoardings = []
    Others = []
    for i in range(10):
        total = 100
        Tv_value = round(random.uniform(Min_Tv_Percentage, Max_Tv_Percentage), 2)
        
        Media_value = round(random.uniform(Min_Media_Percentage, Max_Media_Percentage), 2)
        
        Radio_value = round(random.uniform(Min_Radio_percentage, Max_Radio_percentage), 2)

        Google_value = round(random.uniform(Min_Google_percentage, Max_Google_percentage), 2)
        
        Insta_value = round(random.uniform(Min_Insta_percentage, Max_Insta_percentage), 2)

        Youtube_value = round(random.uniform(Min_Youtube_percentage, Max_Youtube_percentage), 2)
        
        Hoardings_value = round(random.uniform(Min_Hoardings_percentage, Max_Hoardings_percentage), 2)
        
        Others_value = round(random.uniform(Min_Others_percentage, Max_Others_percentage), 2)

        total = Tv_value + Media_value + Radio_value + Google_value + Insta_value + Youtube_value + Hoardings_value + Others_value

        if total > 100:
            rem  = total - 100
            Tv_value -= Avg_Tv_Percentage * rem / 100
            Media_value -= Avg_Media_Percentage * rem / 100
            Radio_value -= Avg_Radio_Percentage * rem / 100
            Google_value -= Avg_Google_Percentage * rem / 100
            Insta_value -= Avg_Insta_Percentage * rem / 100
            Youtube_value -= Avg_Youtube_Percentage * rem / 100
            Hoardings_value -= Avg_Hoardings_Percentage * rem / 100
            Others_value -= Avg_Others_Percentage * rem / 100

        else:
            rem = 100 - total
            Tv_value += Avg_Tv_Percentage * rem / 100
            Media_value += Avg_Media_Percentage * rem / 100
            Radio_value += Avg_Radio_Percentage * rem / 100
            Google_value += Avg_Google_Percentage * rem / 100
            Insta_value += Avg_Insta_Percentage * rem / 100
            Youtube_value += Avg_Youtube_Percentage * rem / 100 
            Hoardings_value += Avg_Hoardings_Percentage * rem / 100
            Others_value += Avg_Others_Percentage * rem / 100

        Tv.append(Tv_value)
        Media.append(Media_value)
        Radio.append(Radio_value)
        Google.append(Google_value)
        Insta.append(Insta_value)
        Youtube.append(Youtube_value)
        Hoardings.append(Hoardings_value)
        Others.append(Others_value)
    
    predictions = []
    best_advertiser_model = BestAdvertisingModel.query.filter(BestAdvertisingModel.advertiser == advertiser).all()
    for i in range(10):
        Tv_value = Tv[i] * budget / 100
        Media_value = Media[i] * budget / 100
        Radio_value = Radio[i] * budget / 100
        Google_value = Google[i] * budget / 100
        Insta_value = Insta[i] * budget / 100
        Youtube_value = Youtube[i] * budget / 100
        Hoardings_value = Hoardings[i] * budget / 100
        Others_value = Others[i] * budget / 100
        for i in best_advertiser_model:
            if i.Best_Model.strip() == 'Linear_Regression':
                model = linear_regression_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value]})
                prediction = model.predict(data)
                if prediction < 0:
                    prediction = -prediction / 2
                prediction = prediction[0][0]
            elif i.Best_Model.strip() == 'Multiple_Regression_TV_Radio_Media':
                model = multiple_regression_TV_Radio_Media_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value], 'Print Media': [Media_value], 'RADIO': [Radio_value]})
                prediction = model.predict(data)
                if prediction < 0:
                    prediction = -prediction / 2
                prediction = prediction[0][0]
            elif i.Best_Model.strip() == 'Multiple_Regression':
                model = multiple_regression_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value], 'Print Media': [Media_value], 'RADIO': [Radio_value], 'GOOGLE': [Google_value], 'FB & Insta': [Insta_value], 'YOUTUBE': [Youtube_value], 'HOARDINGS': [Hoardings_value], 'OTHERS': [Others_value]})
                prediction = model.predict(data)
                if prediction < 0:
                    prediction = -prediction / 2
                prediction = prediction[0][0]
            elif i.Best_Model.strip() == 'Polynomial_Regression_TV':
                poly_features = PolynomialFeatures(degree=2)
                model = polynomial_regression_TV_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value]})
                data = poly_features.fit_transform(data)
                prediction = model.predict(data)
                if prediction < 0:
                    prediction = -prediction / 2
                prediction = prediction[0][0]
            elif i.Best_Model.strip() == 'Polynomial_Regression_TV_Radio_Media':
                poly_features = PolynomialFeatures(degree=2)
                model = polynomial_regression_TV_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value]})
                data = poly_features.fit_transform(data)
                prediction_tv = model.predict(data)
                if prediction_tv < 0:
                    prediction_tv = -prediction_tv / 2
                model = polynomial_regression_Print_model.get(advertiser)
                data = pd.DataFrame({'Print Media': [Media_value]})
                data = poly_features.fit_transform(data)
                prediction_media = model.predict(data)
                if prediction_media < 0:
                    prediction_media = -prediction_media / 2
                model = polynomial_regression_Radio_model.get(advertiser)
                data = pd.DataFrame({'RADIO': [Radio_value]})
                data = poly_features.fit_transform(data)
                prediction_radio = model.predict(data)
                if prediction_radio < 0:
                    prediction_radio = -prediction_radio / 2
                total = Tv_value + Media_value + Radio_value
                p1 = Tv_value / total
                p2 = Media_value / total
                p3 = Radio_value / total
                prediction = p1 * prediction_tv + p2 * prediction_media + p3 * prediction_radio
                prediction = prediction[0][0]
            elif i.Best_Model.strip() == 'Polynomial_Regression_TV_to_Others':
                poly_features = PolynomialFeatures(degree=2)
                model = polynomial_regression_TV_model.get(advertiser)
                data = pd.DataFrame({'TV': [Tv_value]})
                data = poly_features.fit_transform(data)
                prediction_tv = model.predict(data)
                if prediction_tv < 0:
                    prediction_tv = -prediction_tv / 2
                model = polynomial_regression_Print_model.get(advertiser)
                data = pd.DataFrame({'Print Media': [Media_value]})
                data = poly_features.fit_transform(data)
                prediction_media = model.predict(data)
                if prediction_media < 0:
                    prediction_media = -prediction_media / 2
                model = polynomial_regression_Radio_model.get(advertiser)
                data = pd.DataFrame({'RADIO': [Radio_value]})
                data = poly_features.fit_transform(data)
                prediction_radio = model.predict(data)
                if prediction_radio < 0:
                    prediction_radio = -prediction_radio / 2
                model = polynomial_regression_Google_model.get(advertiser)
                data = pd.DataFrame({'GOOGLE': [Google_value]})
                data = poly_features.fit_transform(data)
                prediction_google = model.predict(data)
                if prediction_google < 0:
                    prediction_google = -prediction_google / 2
                model = polynomial_regression_Insta_model.get(advertiser)
                data = pd.DataFrame({'FB & Insta': [Insta_value]})
                data = poly_features.fit_transform(data)
                prediction_insta = model.predict(data)
                if prediction_insta < 0:
                    prediction_insta = -prediction_insta / 2
                model = polynomial_regression_Youtube_model.get(advertiser)
                data = pd.DataFrame({'YOUTUBE': [Youtube_value]})
                data = poly_features.fit_transform(data)
                prediction_youtube = model.predict(data)
                if prediction_youtube < 0:
                    prediction_youtube = -prediction_youtube / 2
                model = polynomial_regression_Hoarding_model.get(advertiser)
                data = pd.DataFrame({'HOARDINGS': [Hoardings_value]})
                data = poly_features.fit_transform(data)
                prediction_hoardings = model.predict(data)
                if prediction_hoardings < 0:
                    prediction_hoardings = -prediction_hoardings / 2
                model = polynomial_regression_Other_model.get(advertiser)
                data = pd.DataFrame({'OTHERS': [Others_value]})
                data = poly_features.fit_transform(data)
                prediction_others = model.predict(data)
                if prediction_others < 0:
                    prediction_others = -prediction_others / 2
                total = Tv_value + Media_value + Radio_value + Google_value + Insta_value + Youtube_value + Hoardings_value + Others_value
                p1 = Tv_value / total
                p2 = Media_value / total
                p3 = Radio_value / total
                p4 = Google_value / total
                p5 = Insta_value / total
                p6 = Youtube_value / total
                p7 = Hoardings_value / total
                p8 = Others_value / total
                prediction = p1 * prediction_tv + p2 * prediction_media + p3 * prediction_radio + p4 * prediction_google + p5 * prediction_insta + p6 * prediction_youtube + p7 * prediction_hoardings + p8 * prediction_others
                prediction = prediction[0][0]
            else:
                #tushar start
                current_date = datetime.today()
                current_month = current_date.month
                current_year = current_date.year
                saved_model = load_model(os.path.join('h5_files', f'lstm_{advertiser}.h5'))
                total = Tv_value + Media_value + Radio_value + Google_value + Insta_value + Youtube_value + Hoardings_value + Others_value
                data = pd.DataFrame({'year': [current_year], 'month': [current_month], 'TV': [Tv_value], 'Print Media': [Media_value], 'RADIO': [Radio_value], 'GOOGLE': [Google_value], 'FB & Insta': [Insta_value], 'YOUTUBE': [Youtube_value], 'HOARDINGS': [Hoardings_value], 'OTHERS': [Others_value], 'Total': [total], 'Sales': 0})
                data_list = [
                {'year': item.year, 'month': item.month, 'TV': item.Tv, 'Print Media': item.Print_Media, 'RADIO': item.Radio, 'GOOGLE': item.Google, 'FB & Insta': item.Fb_and_Insta, 'YOUTUBE': item.Youtube, 'HOARDINGS': item.Hoardings, 'OTHERS': item.Others, 'Total': item.Total, 'Sales': item.Sales} 
                for item in advertiser_data
                ]
                existing_data = pd.DataFrame(data_list)
                data = pd.concat([existing_data, data], ignore_index=True)
                num_data_points = len(data)
                values = data.values
                encoder = LabelEncoder()
                values[:,1] = encoder.fit_transform(values[:,1])
                values = values.astype('float32')
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                reframed = series_to_supervised(scaled, 1, 1)    
                values = reframed.values
                train = values[:num_data_points-2, :] 
                test = values[num_data_points-2:, :] 
                train_X, train_y = train[:, :-1], train[:, -1]
                test_X, test_y = test[:, :-1], test[:, -1]
                train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
                test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
                saved_model = load_model(os.path.join('h5_files', f'lstm_{advertiser}.h5'))
                prediction = saved_model.predict(test_X)
                per_diff=(prediction[0][0] - train_y[-1]) * 100 / train_y[-1]
                last_sales = existing_data['Sales'].iloc[-1]
                new_pred = float(last_sales) + per_diff * float(last_sales)/100 
                predictions.append(new_pred)
                #tushar end
                
        predictions.append(prediction)
    index = predictions.index(max(predictions))

    Tv_value = int(Tv[index] * budget / 100)
    Media_value = int(Media[index] * budget / 100)
    Radio_value = int(Radio[index] * budget / 100)
    Google_value = int(Google[index] * budget / 100)
    Insta_value = int(Insta[index] * budget / 100)
    Youtube_value = int(Youtube[index] * budget / 100)
    Hoardings_value = int(Hoardings[index] * budget / 100)
    Others_value = int(Others[index] * budget / 100)
    total = Tv_value + Media_value + Radio_value + Google_value + Insta_value + Youtube_value + Hoardings_value + Others_value
    Tv_value += (budget-total)

    attributes = ['Tv', 'Print_Media', 'Radio', 'Google', 'Fb_and_Insta', 'Youtube', 'Hoardings', 'Others']
    values = [Tv_value, Media_value, Radio_value, Google_value, Insta_value, Youtube_value, Hoardings_value, Others_value]
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=attributes, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    months_years_values = [str(item.month) + '-' + str(item.year) for item in advertiser_data]
    current_date = datetime.today()
    current_month = current_date.month
    current_year = current_date.year
    months_years_values.append(str(current_month) + '-' + str(current_year))
    total_values = [item.Total for item in advertiser_data]
    sales_values = [item.Sales for item in advertiser_data]
    total_values.append(budget)
    sales_values.append("{:.2f}".format(max(predictions)))
    plt.figure(figsize=(10, 6))
    plt.plot(months_years_values, total_values, marker='o', linestyle='-', color='b', label='Total')
    plt.plot(months_years_values, sales_values, marker='o', linestyle='-', color='r', label='Sales')
    plt.xlabel('Timeline')
    plt.ylabel('Total and Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer_line_chart = io.BytesIO()
    plt.savefig(buffer_line_chart, format='png')
    buffer_line_chart.seek(0)
    line_chart_data = base64.b64encode(buffer_line_chart.getvalue()).decode('utf-8')
    plt.close()
        
    return render_template('budget.html', budget=budget, advertiser=advertiser, advertisers=advertisers, Tv_value=Tv_value, Media_value=Media_value, Radio_value=Radio_value, Google_value=Google_value, Insta_value=Insta_value, Youtube_value=Youtube_value, Hoardings_value=Hoardings_value, Others_value=Others_value, Sales="{:.2f}".format(max(predictions)), plot_image=plot_data, line_chart_image=line_chart_data)

@app.route('/upload', methods=['POST'])
def upload_csv():
    uploaded_file = request.files['csv_file']

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, uploaded_file.filename)
    uploaded_file.save(file_path)
    
    df = pd.read_csv(file_path)
    advertiser_data = AdvertisingData.query.all()
    data_list = [
        {'serial_id': item.serial_id, 'advertiser': item.advertiser, 'year': item.year, 'month': item.month, 'Tv': item.Tv, 'Print_Media': item.Print_Media, 'Radio': item.Radio, 'Google': item.Google, 'Fb_and_Insta': item.Fb_and_Insta, 'Youtube': item.Youtube, 'Hoardings': item.Hoardings, 'Others': item.Others, 'Total': item.Total, 'Sales': item.Sales} 
        for item in advertiser_data
    ]
    existing_data = pd.DataFrame(data_list)
    combined_df = pd.concat([existing_data, df], ignore_index=True)
    combined_df = combined_df.sort_values(['advertiser', 'year', 'month'])
    combined_df['serial_id'] = range(1, len(combined_df) + 1)
    db.session.query(AdvertisingData).delete()
    db.session.commit()

    for index, row in combined_df.iterrows():
        new_entry = AdvertisingData(
            serial_id=row['serial_id'],
            advertiser=row['advertiser'],
            year=row['year'],
            month=row['month'],
            Tv=row['Tv'],
            Print_Media=row['Print_Media'],
            Radio=row['Radio'],
            Google=row['Google'],
            Fb_and_Insta=row['Fb_and_Insta'],
            Youtube=row['Youtube'],
            Hoardings=row['Hoardings'],
            Others=row['Others'],
            Total=row['Total'],
            Sales=row['Sales']
        )

        db.session.add(new_entry)
    db.session.commit()
    message = 'File Uploaded Successfully !'
    return render_template('add_csv.html', message=message)

    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run()