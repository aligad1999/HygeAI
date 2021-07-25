import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask import Flask, render_template, request
import random
#pip install python-google-places
#from googleplaces import GooglePlaces, types, lang
#import requests
#import json

l=[]
app = Flask(__name__)
app.config["CACHE_TYPE"]="null"

ip={}
training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y
reduced_data = training.groupby(training['prognosis']).max()
#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
model=SVC()
model.fit(x_train,y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
precution_list=[]
symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        return "You should take the consultation from doctor. "
    else:
        return "It might not be that bad but you should take precautions."
		
def getDescription():
    global description_list
    with open('Symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')        
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('Symptom_Severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
			
def getprecautionDict():
    global precautionDictionary
    with open('Symptom_Precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:

        # print(f"comparing {inp} to {item}")
        if regexp.search(item):
            pred_list.append(item)
            # return 1,item
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item
		
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {}
    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])
	
def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return disease

"""

@app.route("/gethospital")
def hospital():
  userlat_lng = request.args.get('msg')	
  latlngList = userlat_lng.split(' ')
  # Use your own API key for making api request calls
  API_KEY = 'AIzaSyC6-gwhsbRMAbtSNhR56y2EBV9S16bZhHE'
  # Initialising the GooglePlaces constructor
  google_places = GooglePlaces(API_KEY)

  query_result = google_places.nearby_search(
      lat_lng ={'lat': latlngList[0]
, 'lng': latlngList[1]},
      radius = 1000,
      types =[types.TYPE_HOSPITAL])

  # Iterate over the search results
  for place in query_result.places:
    return "the nearest hospital to you is " + place.name
"""


@app.route("/")
def index():
   return "Welcome to the first version of HygeAI healthcare chatbot!"	

@app.route("/get")
def get_bot_response():
	       
    userText = request.args.get('msg')
    if "Please enter the symptom you are facing" not in l:
        l.append("Please enter the symptom you are facing")
        return "Hello "+userText+" please enter the symptom you are facing"
    tree=globals()['clf']
    feature_names=globals()['cols']
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    
    while True:
        if "dis_inp" not in ip.keys():
            disease_input = request.args.get('msg').strip()
            ip["dis_inp"]=disease_input
        disease_input=ip["dis_inp"]
        
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        
        shrad=""
        if conf==1:
            for num,it in enumerate(cnf_dis):
                shrad+="\n"+str(num)+")"+it+" "
            if num!=0:
                if "Select the one that you meant" not in l:
                    l.append("Select the one that you meant")
                    return "Select the one that you meant "+shrad
                if "conf_inp" not in ip.keys():
                    conf_inp = request.args.get('msg').strip()
                    ip["conf_inp"]=conf_inp
                conf_inp=ip["conf_inp"]
            else:
                conf_inp=0
            disease_input=cnf_dis[int(conf_inp)]
            break
        else:
            return "Enter valid symptom "
    while True:
        try:
            if "Okay. from how many days?" not in l:
                l.append("Okay. from how many days?")
                return "Okay. from how many days?"
            break
        except:
            print("again")
    if "num_days" not in ip.keys():
        num_days=int(request.args.get('msg').strip())
        ip["num_days"]=num_days
    num_days=ip["num_days"]
    
    recurse(0, 1,tree_,feature_name,disease_input)
    present_disease = print_disease(tree_.value[node])
    red_cols = reduced_data.columns 
    symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
    symptoms_exp=[]
    vir=list(symptoms_given)       
    for syms in list(symptoms_given):
	
	  syms = syms.replace('_',' ')
        inp=""
        if syms not in ip.keys():
            ip[syms]="done"
            return "Are you experiencing any "+syms+" ? : "+"\nProvide proper answers i.e. (yes/no) : "
        else:
            continue
	
        inp=request.args.get('msg')
        if(inp.lower()!="yes" or inp.lower()!="no"):
            print("provide proper answers i.e. (yes/no) : ")
        if(inp.lower()=="yes"):
            symptoms_exp.append(syms)
	
    second_prediction=sec_predict(symptoms_exp)
    calc_condition(symptoms_exp,num_days)
    precution_list=precautionDictionary[present_disease[0]]
    for  i,j in enumerate(precution_list):
        shrad+=str(i+1)+")"+j+"\n"
    if(present_disease[0]==second_prediction[0]):
        return "You may have "+ present_disease[0]+"\n"+description_list[present_disease[0]]+" Take following measures : " + shrad + "\nThank you for using HygeAI chatbot!\n"
		
    else:
        return "You may have "+ present_disease[0]+ "or "+ second_prediction[0]+"\n"+description_list[present_disease[0]]+"\n"+description_list[second_prediction[0]]+"\nTake following measures : \n"+ shrad + "\nThank you for using HygeAI chatbot!"          
node=0

def recurse(n, depth,tree_,feature_name,disease_input):
    globals()['node']=n
    
    indent = "  " * depth
    if tree_.feature[globals()['node']] != _tree.TREE_UNDEFINED:
        name = feature_name[globals()['node']]
        threshold = tree_.threshold[globals()['node']]
        if name == disease_input:
            val = 1
        else:
            val = 0
        if  val <= threshold:
            recurse(tree_.children_left[globals()['node']], depth + 1,tree_,feature_name,disease_input)
        else:
            symptoms_present.append(name)
            recurse(tree_.children_right[globals()['node']], depth + 1,tree_,feature_name,disease_input)
if __name__=='__main__':
    getSeverityDict()
    getDescription()
    getprecautionDict()
    app.run(debug=True, use_reloader=True)
