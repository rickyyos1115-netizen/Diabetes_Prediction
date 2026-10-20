############ Import the required Libraries ##################
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Affected by Diabetes or not ?")
    
    st.sidebar.title("Model Selection Panel")
    st.sidebar.markdown("Choose your model and its parameters")
    
    @st.cache_data
    def load_data(): # Function to load our dataset
        data = pd.read_csv("diabetes.csv")
        return data
     
    # 【修改】增加 test_size 與 use_stratify 參數，支援動態設定分層抽樣
    def split(df, test_size=0.3, use_stratify=True):
        y = df.Outcome
        x = df.drop(columns=['Outcome'])
        
        # 判斷是否開啟 Stratify 分層抽樣
        stratify_param = y if use_stratify else None
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, 
            test_size=test_size, 
            random_state=0, 
            stratify=stratify_param
        )
        return x_train, x_test, y_train, y_test
     
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots()
            disp.plot(cmap="Blues", ax=ax)
            st.pyplot(fig)  # Render matplotlib figure

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            
            y_prob = model.predict_proba(x_test)[:, 1]  # Binary classification probabilities
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)  # Render matplotlib figure

    df = load_data() 
    class_names = ['Non-Diabetes', 'Diabetes']
    
    # --------------------------------------------------
    # 【需求】設定訓練資料比例與 Stratify（分層抽樣）
    # --------------------------------------------------
    st.sidebar.subheader("Dataset Split Settings")
    train_size = st.sidebar.slider("Training Data Size Ratio", 0.1, 0.9, 0.7, step=0.05)
    test_size = round(1.0 - train_size, 2)
    
    # 【新增】側邊欄加入 Stratify 開關
    use_stratify = st.sidebar.checkbox("Stratify Split (Maintain target distribution)", value=True)
    
    st.sidebar.caption(f"Train ratio: {int(train_size*100)}% | Test ratio: {int(test_size*100)}%")
    
    # 切分資料集 (傳入 use_stratify 設定)
    x_train, x_test, y_train, y_test = split(df, test_size=test_size, use_stratify=use_stratify)

    st.sidebar.subheader("Select your Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))

    # --------------------------------------------------
    # 使用 Session State 保存訓練好的模型與狀態
    # --------------------------------------------------
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'current_classifier' not in st.session_state:
        st.session_state.current_classifier = None

    model = None

    # --- 模型 1: Decision Tree ---
    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model parameters")
        criterion = st.sidebar.radio("Criterion(measures the quality of split)", ("gini", "entropy"), key='criterion')
        splitter = st.sidebar.radio("Splitter (How to split at each node?)", ("best", "random"), key='splitter')
        metrics = st.sidebar.multiselect("Select your metrics : ", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            
            st.session_state.trained_model = model
            st.session_state.current_classifier = classifier

    # --- 模型 2: Logistic Regression ---
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Parameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("Select your metrics?", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            
            st.session_state.trained_model = model
            st.session_state.current_classifier = classifier

    # --- 模型 3: Random Forest ---
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=(bootstrap=='True'), n_jobs=-1)
            model.fit(x_train, y_train)
            
            st.session_state.trained_model = model
            st.session_state.current_classifier = classifier

    # --------------------------------------------------
    # 顯示模型評估指標 (包含 F1 指標輸出)
    # --------------------------------------------------
    if st.session_state.trained_model is not None and st.session_state.current_classifier == classifier:
        model = st.session_state.trained_model
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        st.write("Accuracy: ", round(accuracy * 100, 2), "%")
        st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
        st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
        st.write("F1 Score: ", round(f1_score(y_test, y_pred), 2))
        
        plot_metrics(metrics, model, x_test, y_test, class_names)

    # --------------------------------------------------
    # 單筆數據預測區塊 (不需要重新訓練模型)
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("Single Patient Prediction")

    if st.session_state.trained_model is not None:
        st.info(f"Currently using trained model: **{st.session_state.current_classifier}**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 300, 120)
        with col2:
            blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        with col3:
            insulin = st.number_input("Insulin", 0, 900, 79)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        with col4:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
            age = st.number_input("Age", 1, 120, 33)

        if st.button("Predict Single Sample"):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            
            pred_class = st.session_state.trained_model.predict(input_data)[0]
            pred_proba = st.session_state.trained_model.predict_proba(input_data)[0]

            res_label = class_names[pred_class]
            if pred_class == 1:
                st.error(f"Prediction Result: **{res_label}**")
            else:
                st.success(f"Prediction Result: **{res_label}**")

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pred_proba, labels=class_names, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
            ax.axis('equal')
            plt.title("Prediction Probability Distribution")
            st.pyplot(fig)
    else:
        st.warning("Please click 'Classify' in the sidebar to train a model first before making single predictions.")

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Diabetes Raw Dataset")
        st.write(df) 

if __name__ == '__main__':
    main()