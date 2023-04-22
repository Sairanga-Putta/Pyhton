import pandas as pd
import numpy as np

data = pd.read_csv("/content/Play Tennis")
features = [feat for feat in data]
features.remove("Play Tennis")
print('DATASET : Play Tennis')
print(data)
days=data.shape[0]
yes=data['Play Tennis'].value_counts()['Yes']
no=data['Play Tennis'].value_counts()['No']
p_yes=yes/days
p_no=no/days

def bayes(examples,attrs):
  s1={}
  n1={}
  for feature in attrs:
    s={}
    n={}
    uniq=np.unique(examples[feature])
    for u in uniq:
      subdata = examples[examples[feature] == u]
      pos=(subdata['Play Tennis'].value_counts()['Yes'])
      if(pos==subdata.shape[0]):
        neg=0
      else:
        neg=(subdata['Play Tennis'].value_counts()['No'])
      s[u]=pos
      n[u]=neg
    s1[feature]=s
    n1[feature]=n    
  return s1,n1


def classify(d):
  s,n=bayes(data,features)
  vy=p_yes
  vn=p_no
  for i in d:
    vy=vy*(s[i][d[i]]/yes)
    vn=vn*(n[i][d[i]]/no)
  vyes=vy/(vy+vn)
  vno=vn/(vn+vy)
  if(vyes>vno):
    return 'Yes'
  else:
    return 'No'

print('-------------------------------------------------')
print("Classifying Given Data : ")
print('-------------------------------------------------')
new=data.iloc[:,:4].sample().to_dict(orient='records')[0]
print('Data: ',new)
print('PLAY TENNIS: ',classify(new))



def Model_Accuracy(Test):
  X=Test.iloc[:,:4].to_dict(orient='records')
  Y=Test['Play Tennis'].to_list()
  print('Test Data :')
  print(Test)
  TP=TN=FP=FN=0
  pred=[]
  act=[]
  for i in range(len(X)):
    Predicted=classify(X[i])
    pred.append(Predicted)
    Actual=Y[i]
    act.append(Actual)
    if(Predicted=='Yes' and Actual=='Yes'):
      TP=TP+1
    elif(Predicted=='No' and Actual=='Yes'):
      FN=FN+1
    elif(Predicted=='Yes' and Actual=='No'):
      FP=FP+1
    elif(Predicted=='No' and Actual=='No'):
      TN=TN+1
  model_accuracy=(TP+TN)/(TP+TN+FP+FN)
  Model=pd.DataFrame(list(zip(pred,act)),columns =['Predicted', 'Actual'])
  print()
  print(Model)
  print()
  return model_accuracy  

Test=data.sample(n=6)
print('-------------------------------------')
print('Naive Bayesian Model Accuracy: ')
print('-------------------------------------')
print('Accuracy : ',Model_Accuracy(Test))
