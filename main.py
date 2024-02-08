import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title("Bank Churn Classification")
@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn-Records.csv')
    df = df.dropna()
    df = df.drop(['Surname', 'CustomerId','RowNumber'], axis=1)
    return df

df = load_data()
percentage = df['Exited'].value_counts(normalize=True) * 100
fig = px.pie(names=percentage.index, values=percentage.values,
             title='Exited Percentage', labels={'names': 'Exited'})
st.plotly_chart(fig)

st.write(df)
# columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#            'EstimatedSalary', 'Exited', 'Complain', 'Satisfaction Score', 'Point Earned']
columns = df.columns.tolist()

selected_column = st.selectbox("Select a Column", columns)

fig = px.histogram(df, x=selected_column, histnorm='percent',
                   labels={selected_column: selected_column},
                   title=f'Histogram of {selected_column} (in %)')

st.plotly_chart(fig)
columns = df.columns.tolist()


selected_x_column = st.selectbox("Select the X-axis", columns, index=columns.index('Tenure'))
selected_y_column = st.selectbox("Select the Y-axis", columns, index=columns.index('Age'))


fig = px.bar(df, x=selected_x_column, y=selected_y_column,
                 labels={selected_x_column: selected_x_column, selected_y_column: selected_y_column},
                 title=f'Scatter Plot of {selected_x_column} vs {selected_y_column}')


st.plotly_chart(fig)
df_encode=pd.get_dummies(df,columns=['Geography', 'Gender', 'Card Type'])
x=df_encode.iloc[:, df_encode.columns != 'Exited'].values
y=df_encode['Exited'].values

smote=SMOTE(random_state=42,sampling_strategy='minority')
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Count before and after SMOTE
count_before = pd.Series(y_train).value_counts()
count_after = pd.Series(y_train_smote).value_counts()


fig = go.Figure(data=[
    go.Bar(name='Before SMOTE', x=count_before.index, y=count_before.values),
    go.Bar(name='After SMOTE', x=count_after.index, y=count_after.values)
])


fig.update_layout(barmode='group', title='Comparison of Target Variable Distribution Before and After SMOTE',
                  xaxis_title='Exited', yaxis_title='Count')

st.plotly_chart(fig)



ada_boost_clf = AdaBoostClassifier(random_state=42).fit(X_train_smote,y_train_smote)
random_forest_clf = RandomForestClassifier().fit(X_train_smote,y_train_smote)
knn_clf = KNeighborsClassifier(n_neighbors=3).fit(X_train_smote,y_train_smote)
mlp_clf = MLPClassifier(random_state=42, max_iter=300).fit(X_train_smote,y_train_smote)
svm_clf = SVC().fit(X_train_smote,y_train_smote)
decision_tree_clf = DecisionTreeClassifier().fit(X_train_smote,y_train_smote)

accuracy_scores = {
    "AdaBoost":accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    "Random Forest": accuracy_score(y_test, random_forest_clf.predict(X_test)),
    "KNN": accuracy_score(y_test, knn_clf.predict(X_test)),
    "MLP": accuracy_score(y_test, mlp_clf.predict(X_test)),
    "SVM": accuracy_score(y_test, svm_clf.predict(X_test)),
    "Decision Tree":accuracy_score(y_test, decision_tree_clf.predict(X_test)),
}
fig_akurasi=px.bar(x=accuracy_scores.keys(),y=accuracy_scores.values())
st.plotly_chart(fig_akurasi)
predictions = {
    "AdaBoost": ada_boost_clf.predict(X_test),
    "Random Forest": random_forest_clf.predict(X_test),
    "KNN": knn_clf.predict(X_test),
    "MLP": mlp_clf.predict(X_test),
    "SVM": svm_clf.predict(X_test),
    "Decision Tree": decision_tree_clf.predict(X_test),
}


predictions_df = pd.DataFrame(predictions)
predictions_df['Actual'] = y_test  #data aktual
predictions_df.index.name = 'Sample'


st.write("Prediction Results:", predictions_df)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  dft = pd.read_csv(uploaded_file)
  dft = dft.dropna()
  dft = dft.drop(['Surname', 'CustomerId', 'RowNumber','Exited'], axis=1)
  dft_encode = pd.get_dummies(dft, columns=['Geography', 'Gender', 'Card Type'])
  X_new = dft_encode[20:60].values
  prediction_top3={
  "predictions Random Forest" : random_forest_clf.predict(X_new),
  "predictions Ada Boost" : ada_boost_clf.predict(X_new),
  "predictions Decision Tree" : decision_tree_clf.predict(X_new)
  }
  prediction_ml=pd.DataFrame(prediction_top3)
  prediction_ml.index.name = 'Sample'


  st.write(dft[20:60])
  st.write("Prediction using top 3 highest ML algorithm for New data : ",prediction_ml)