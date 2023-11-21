from flask import Flask, render_template, request, jsonify
from flask_mail import Mail,Message
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os

app = Flask(__name__)
mail = Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'pnpfoundations81913@gmail.com'
app.config['MAIL_PASSWORD'] = 'illuminati@123'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

# Load the heart dataset
df_heart = pd.read_csv("heart_2020_cleaned.csv")
selected_cols_heart = ['Smoking', 'AlcoholDrinking', 'Diabetic', 'Asthma', 'HeartDisease']
df_selected_heart = df_heart[selected_cols_heart].copy()
df_selected_heart = df_selected_heart.apply(LabelEncoder().fit_transform)
x_heart = df_selected_heart.drop('HeartDisease', axis=1)
y_heart = df_selected_heart['HeartDisease']
sm_heart = SMOTE(random_state=42)
X_heart, Y_heart = sm_heart.fit_resample(x_heart, y_heart)
x_train_heart, x_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, Y_heart, test_size=0.30, random_state=42)
rf_heart = RandomForestClassifier(n_estimators=50, max_depth=5, max_features=None)
rf_heart.fit(x_train_heart, y_train_heart)

# Load the sleep dataset
df_sleep = pd.read_csv("Sleep.csv")
df_sleep = df_sleep.drop(['Person ID', 'Occupation', 'Quality of Sleep', 'Blood Pressure', 'Age'], axis=1)
df_sleep = df_sleep.dropna()
label_encoder_sleep = preprocessing.LabelEncoder()
df_sleep['Gender'] = label_encoder_sleep.fit_transform(df_sleep['Gender'])
df_sleep['BMI Category'] = label_encoder_sleep.fit_transform(df_sleep['BMI Category'])
X_sleep = df_sleep.drop(['Sleep Disorder'], axis=1)
y_sleep = df_sleep['Sleep Disorder']
classifier_sleep = LogisticRegression(random_state=0)
classifier_sleep.fit(X_sleep, y_sleep)

os.environ['OPENAI_API_KEY'] = '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'  # Replace with your API key

llm_resto = OpenAI(temperature=0.6)

# Set up LangChain
prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
)

chain_resto = LLMChain(llm=llm_resto, prompt=prompt_template_resto)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=["POST"])
def mail():
    if request.method == 'POST':
        reciever = str(request.form['Reciever'])
        msg = Message("Hey", sender="noreply@gmail.com",recipients=[reciever])
        msg.body = "Hey Thank You for Join Us"
        mail.send(msg)
        return "Mail Sent"
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/sleep')
def sleep():
    return render_template('sleep.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')  

# Define a route for heart prediction
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        smoking = int(request.form['Smoking'])
        alcohol_drinking = int(request.form['AlcoholDrinking'])
        diabetic = int(request.form['Diabetic'])
        asthma = int(request.form['Asthma'])
        input_data_heart = {
            'Smoking': [smoking],
            'AlcoholDrinking': [alcohol_drinking],
            'Diabetic': [diabetic],
            'Asthma': [asthma]
        }
        input_df_heart = pd.DataFrame(input_data_heart)
        input_df_heart = input_df_heart.apply(LabelEncoder().fit_transform)
        prediction_heart = rf_heart.predict(input_df_heart)
        if prediction_heart == 0:
            return render_template('heart.html', prediction_heart="Need To take precautions")
        else:
            return render_template('heart.html', prediction_heart="Your Heart is absolutely fine")

# Define a route for sleep prediction
@app.route('/predict_sleep', methods=['POST'])
def predict_sleep():
    if request.method == 'POST':
        features_sleep = [
            int(request.form['Gender']),
            int(request.form['BMI Category']),
            int(request.form['Physical Activity']),
            int(request.form['Mental Health']),
            int(request.form['Alcohol Consumption']),
            int(request.form['Caffeine Consumption']),
            int(request.form['Screen Time'])
        ]
        result_sleep = classifier_sleep.predict([features_sleep])[0]
        return render_template('sleep.html', prediction_sleep=result_sleep)


@app.route('/get_breakfast', methods=['POST'])
def get_breakfast():
    try:
        # Get data from the request
        input_data = request.form.to_dict()

        # Run LangChain to get results
        results = chain_resto.run(input_data)
        sections = results.split('\n\n')

        # Extract breakfast recommendations
        breakfasts = [breakfast.strip() for breakfast in sections[1].split('\n')[1:]]

        return render_template('diet.html', suggest_breakfasts=breakfasts)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)