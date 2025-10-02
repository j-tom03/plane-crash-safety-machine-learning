import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import string
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier

from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib.colors import ListedColormap

from nltk.util import ngrams
from collections import Counter, defaultdict
import random
import nltk
#nltk.download('punkt')

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

import shap


def avg_of_four(values):
        length = len(values)
        sum = [0,0,0,0]
        for item in values:
            sum[0] += item[0]
            sum[1] += item[1]
            sum[2] += item[2]
            sum[3] += item[3]

        avgs = [sum[0]/length, sum[1]/length, sum[2]/length, sum[3]/length]
        for i in range(4):
            if avgs[i] < 1:
                avgs[i] = 1
            else:
                avgs[i] = round(avgs[i])

        return avgs

def preprocess(crash_df):
    crash_df['Year'] = crash_df['Date'].str.slice_replace(4, 10)
    crash_df['Year'] = crash_df['Year'].astype(int)
    
    crash_df = crash_df.drop(columns=['Registration', 'Schedule', 'MSN', 'Flight no.'])
    crash_df.at[2812, 'Aircraft'] = "Swallow Land Plane"
    crash_df.at[6877, 'Country'] = "United States of America"
    crash_df.at[6877, 'Region'] = "North America"
        
    pax_or_crew_not_na = crash_df.loc[crash_df['Pax on board'].notna() & crash_df['Crew on board'].notna() & crash_df['Crew on board']!=0 & crash_df['PAX fatalities'].notna() & crash_df['Crew fatalities'].notna()]

    plane_types_onb = dict()
    for _, row in pax_or_crew_not_na.iterrows():

        if row['Aircraft'] in plane_types_onb.keys(): 
            plane_types_onb[row['Aircraft']] += [[row['Pax on board'], row['Crew on board'], row['PAX fatalities'], row['Crew fatalities']]]
        else:
            plane_types_onb[row['Aircraft']] = [[row['Pax on board'], row['Crew on board'], row['PAX fatalities'], row['Crew fatalities']]]

    avg_plane_types_onb = dict()
    for key in plane_types_onb.keys():
        avg_plane_types_onb[key] = avg_of_four(plane_types_onb[key])

    condition = crash_df['Pax on board'].isna() | crash_df['Crew on board'].isna() | crash_df['Crew on board']==0 | crash_df['PAX fatalities'].isna() | crash_df['Crew fatalities'].isna()

    mapped_averages = crash_df['Aircraft'].map(avg_plane_types_onb)

    crash_df.loc[condition, ['Pax on board', 'Crew on board', 'PAX fatalities', 'Crew fatalities']] = (
        crash_df.loc[condition, 'Aircraft'].map(avg_plane_types_onb).apply(pd.Series).values
    )

    remaining_survivor_nans_index = crash_df.loc[crash_df['Pax on board'].isna() | crash_df['Crew on board'].isna() | crash_df['Crew on board']==0 | crash_df['PAX fatalities'].isna() | crash_df['Crew fatalities'].isna()].index
    crash_df.drop(remaining_survivor_nans_index, inplace=True)

    crash_df['Survivors'] = (crash_df['Crew fatalities']+crash_df['PAX fatalities'])/(crash_df['Crew on board']+crash_df['Pax on board'])!=1

    crash_df['Death Rate'] = (crash_df['Crew fatalities'] + crash_df['PAX fatalities']) / (crash_df["Crew on board"] + crash_df["Pax on board"])
    bad_death_rate_indexes = crash_df.loc[crash_df['Death Rate'] > 1].index
    crash_df.drop(bad_death_rate_indexes, inplace=True)

    features = ['Crew on board', 'Pax on board', 'Other fatalities']
    for feature in features:
        indexes = crash_df.loc[crash_df[feature].isna()].index
        crash_df.drop(indexes, inplace=True)

    condition = crash_df['Circumstances'].isna() & (crash_df['Crash cause']=='Unknown')
    crash_df.loc[condition, ['Circumstances']] = ""

    condition2 = crash_df['Circumstances'].isna() & (crash_df['Crash cause']!='Unknown')
    crash_df.loc[condition2, 'Circumstances'] = crash_df.loc[condition2, 'Crash cause']

    return crash_df

def crash_counts_year(crash_df):
    counted_chron = crash_df['Year'].value_counts(sort=False).reset_index()
    counted_chron.columns = ['Year', 'Count']
    counted_chron['Year'].astype(int)
    counted_chron = counted_chron.sort_values('Year')

    return counted_chron

def linear_regression_no_split(counted_chron):
    lr = LinearRegression()

    X = counted_chron[['Year']]
    y = counted_chron['Count']

    lr.fit(X, y)

    print("slope: {}, intercept: {}".format(lr.coef_, lr.intercept_))
    print("R^2 Score: {}".format(lr.score(X,y)))
    pred = lr.predict(X)
    counted_chron['y_pred'] = pred

    ax = counted_chron.plot.scatter(x='Year', y='Count')
    ax = counted_chron.plot.line(x='Year', y='y_pred', ax=ax, color='red')
    plt.show()

def linear_regression_split(counted_chron):
    lr = LinearRegression()

    X = counted_chron[['Year']]
    y = counted_chron['Count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))
    y_pred = lr.predict(X_test)

    X_test_copy = X_test.copy()

    X_test_copy['y_pred'] = y_pred
    X_test_copy['y_test'] = y_test

    X_test_copy.sort_values('Year', inplace=True)

    ax = X_test_copy.plot.scatter('Year', 'y_test')
    ax = X_test_copy.plot.line('Year', 'y_pred', ax=ax)

    plt.show()

def nlp(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(crash_df['Circumstances'])
    tfidf_tokens = vectorizer.get_feature_names()



    sparce_matrix = pd.DataFrame(
        data=doc_term_matrix.toarray(), 
        index=crash_df.index, 
        columns=tfidf_tokens
    )
    print(sparce_matrix)

def nlp_pca_tfidf(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(crash_df['Circumstances'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf.toarray())

    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', s=50)

    plt.title('PCA of TF-IDF Matrix')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def nlp_trunc_svd(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(crash_df['Circumstances'])
    
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=19)
    reduced_dim = svd.fit_transform(tfidf)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_dim[:, 0], reduced_dim[:, 1], c=crash_df['Death Rate'], cmap='viridis', edgecolor='k', s=50)

    plt.colorbar(scatter, label='Death Rate')
    plt.title('SVD of TF-IDF Matrix')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def nlp_trunc_svd_dbscan(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(crash_df['Circumstances'])
    
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=19)
    reduced_dim = svd.fit_transform(tfidf)

    db = DBSCAN(eps=0.05, min_samples=5).fit(reduced_dim)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = reduced_dim[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = reduced_dim[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

def nlp_trunc_svd_by_site(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(crash_df['Circumstances'])
    
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=19)
    reduced_dim = svd.fit_transform(tfidf)

    plt.figure(figsize=(10, 7))

    sites = crash_df['Crash site'].unique()
    site_to_color = {operator: i for i, operator in enumerate(sites)}
    site_colors = crash_df['Crash site'].map(site_to_color)

    scatter = plt.scatter(reduced_dim[:, 0], reduced_dim[:, 1], c=site_colors, cmap=ListedColormap(plt.cm.tab10.colors[:len(sites)]), edgecolor='k', s=50)
    
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), markersize=10, label=operator) 
                      for i, operator in enumerate(sites)]
    plt.legend(handles=legend_handles, title='Crash Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SVD of TF-IDF Matrix Coloured by Crash site')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.show()

def nlp_sgd_deaths(crash_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(crash_df['Circumstances'])
    y = crash_df['Total fatalities']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    classifier = SGDClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))

def nlp_word_freq(crash_df):
    count_dict = dict()
    for _, row in crash_df.iterrows():
        text = row['Circumstances']
        text.translate(str.maketrans('', '', string.punctuation))
        text_list = text.split()
        for word in text_list:
            if word in count_dict.keys():
                count_dict[word] += 1
            else:
                count_dict[word] = 1

    df = pd.DataFrame(list(count_dict.items()), columns=['Word', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False)

    df = df[:20]

    plt.figure(figsize=(10, 6))
    plt.bar(df['Word'], df['Frequency'], color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency in Corpus')
    plt.xticks(rotation=45)
    plt.show()

def preprocess_text(text):
    return nltk.word_tokenize(text.lower())

def generate_ngrams(corpus, n=2):
    ngrams_list = []

    for sentence in corpus:
        tokens = preprocess_text(sentence)
        ngrams_list.extend(ngrams(tokens, n))
    return ngrams_list

def n_gram(crash_df, n):
    ngrams = generate_ngrams(crash_df['Circumstances'].tolist(), n=n)
    ngram_counts = Counter(ngrams)

    ngram_model = defaultdict(lambda: defaultdict(int))
    for (w1, w2, w3, w4, w5), count in ngram_counts.items():
        context = (w1,w2,w3,w4)
        ngram_model[context][w5] = count

    for context in ngram_model:
        total_count = float(sum(ngram_model[context].values()))
        for w5 in ngram_model[context]:
            ngram_model[context][w5] /= total_count

    return ngram_model

def generate_sentence(model, start_sequence, max_len=15):
    sentence = list(start_sequence)
    for _ in range(max_len - len(start_sequence)):
        context = tuple(sentence[-4:])
        next_word_probs = model.get(context, {})
        if not next_word_probs:
            break
        next_word = random.choices(list(next_word_probs.keys()), weights=next_word_probs.values())[0]
        sentence.append(next_word)
    return ' '.join(sentence)

def transform_yom(crash_df):
    crash_df.loc[crash_df['YOM'].isna(), 'YOM'] = crash_df.loc[crash_df['YOM'].isna(), 'Year'] - 15
    return crash_df['YOM']

def generate_severity(death_rate, fatalities):
    if death_rate>0.6:
        if fatalities>3:
            return "Very High"
        elif fatalities>1:
            return "High"
        elif fatalities>0:
            return "Medium"
        else:
            return "Medium"
    elif death_rate > 0.3:
        if fatalities>3:
            return "High"
        elif fatalities>1:
            return "High"
        elif fatalities>0:
            return "Low"
        else:
            return "Low"
    elif death_rate > 0:
        if fatalities>3:
            return "Medium"
        elif fatalities>1:
            return "Medium"
        elif fatalities>0:
            return "Low"
        else:
            return "Low"
    elif death_rate==0:
        return "No"
    else:
        return "Low"

def preprocess_for_ann(crash_df):
    # tfidf for the text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(crash_df['Circumstances'])
    # reduce the dimensions then replace the circumstances field
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=19)
    reduced_dim = svd.fit_transform(tfidf)
    crash_df['circumstance_comp1'] = reduced_dim[:,0]
    crash_df['circumstance_comp2'] = reduced_dim[:,1]
    # setting a plane age and estimating yom if not set
    crash_df['YOM'] = transform_yom(crash_df)
    crash_df['Plane Age'] = crash_df['Year'] - crash_df['YOM']


    crash_df['Severity'] = crash_df.apply(lambda row: generate_severity(row['Death Rate'], row['Total fatalities']), axis=1)

    # dropping fields that probably arent useful
    crash_df = crash_df.drop(columns=['Circumstances', 'YOM', 'Time', 'Survivors', 'Crash location', 'PAX fatalities', 'Crew fatalities', 'Death Rate', 'Total fatalities', 'Other fatalities', 'Region'])

    crash_df.loc[crash_df['Flight phase'].isna(), 'Flight phase'] = 'Unknown'
    crash_df.loc[crash_df['Flight type'].isna(), 'Flight type'] = 'Unknown'
    crash_df.loc[crash_df['Crash site'].isna(), 'Crash site'] = 'Unknown'
    crash_df.loc[crash_df['Crash cause'].isna(), 'Crash cause'] = 'Unknown'

    crash_df['Date'] = pd.to_datetime(crash_df['Date'])
    crash_df['Month'] = crash_df['Date'].dt.month_name()
    crash_df = crash_df.drop(columns=['Date'])

    columns_ohe = ['Aircraft', 'Operator', 'Flight phase', 'Flight type', 'Crash site', 'Country', 'Crash cause', 'Month']

    for column in columns_ohe:
        crash_df = pd.get_dummies(crash_df, columns=[column], prefix=column)

    X = crash_df.drop(columns=['Severity'], axis=1)
    y = crash_df['Severity']

    smote = SMOTE(random_state=19)
    X_resample, y_resample = smote.fit_resample(X, y)

    return X_resample, y_resample

def ann_severity_predictor(crash_df):
    X, y = preprocess_for_ann(crash_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    scaler = StandardScaler()

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100)

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss: {} \Accuracy: {}".format(loss, accuracy))

    # Capture the history of the model for plots
    val_losses = history.history['val_loss']
    val_accuracies = history.history['val_accuracy']
    train_losses = history.history['loss']
    train_accuracies = history.history['accuracy']

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def feature_importance_ann_severity_predictor(crash_df):
    X, y = preprocess_for_ann(crash_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    scaler = StandardScaler()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=100)

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss: {} \nAccuracy: {}".format(loss, accuracy))

    results = permutation_importance(model, X_train, y_train, scoring='accuracy')
    importance = results.importances_mean
    print(importance)

    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

def feature_importance_rf(crash_df):
    X, y = preprocess_for_ann(crash_df)
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=19)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance and take the top 20
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(20)

    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color="skyblue")
    plt.gca().invert_yaxis()  # Invert y-axis to display the highest importance at the top
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Features by Importance")
    plt.tight_layout()
    plt.show()

def grid_search_feature_importance_rf(crash_df):
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [20, 40],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [4]
    }
    X, y = preprocess_for_ann(crash_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=19),
                           param_grid=param_grid,
                           scoring='accuracy', verbose=2,
                           cv=3)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")


def bic_pr(counted_chron):
    X = counted_chron[['Year']]
    y = counted_chron['Count']

    bics = dict(enumerate(i for i in range(1,11)))

    for n in range(11):
        poly = PolynomialFeatures(degree=n)
        X_poly = poly.fit_transform(X)
        poly.fit(X_poly, y)
        lr = LinearRegression()
        lr.fit(X_poly, y)

        y_pred = lr.predict(X_poly)


        rss = np.sum((y - y_pred)**2)
        l = len(y)
        k = n+1
        bic = l * math.log(rss / l) + k * math.log(l)
        bics[n] = bic

    plt.plot(bics.keys(), bics.values())
    plt.xlabel("Degree")
    plt.ylabel("BIC")
    plt.show()

def polynomial_regression(counted_chron):
    X = counted_chron[['Year']]
    y = counted_chron['Count']

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, y)
    lr = LinearRegression()
    lr.fit(X_poly, y)

    y_pred = lr.predict(X_poly)

    print("R^2 Score {}".format(lr.score(X_poly, y)))

    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.show()

def polynomial_regression_future_prediction(counted_chron):
    X = counted_chron[['Year']]
    y = counted_chron['Count']

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, y)
    lr = LinearRegression()
    lr.fit(X_poly, y)

    y_pred = lr.predict(X_poly)

    x_2025 = poly.fit_transform([[2025]])
    y_2025 = lr.predict(x_2025)
    print("2025 Crashes: ",y_2025)

    x_onward_years = [[x] for x in range(1960,2090)]
    x_onward = poly.fit_transform(x_onward_years)
    y_onward = lr.predict(x_onward)

    plt.plot(x_onward_years, y_onward, color='red')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.show()

def polynomial_regression_split(counted_chron):
    X = counted_chron[['Year']]
    y = counted_chron['Count']
    maes = 0
    r2s = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=19)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Fold {}".format(i))
        poly = PolynomialFeatures(degree=3)
        X_train = [X.iloc[j] for j in train_index]
        X_test = [X.iloc[h]  for h in test_index]
        y_train = [y.iloc[l]  for l in train_index]
        y_test = [y.iloc[z]  for z in test_index]

        X_train_poly = poly.fit_transform(X_train)
        poly.fit(X_train_poly, y_train)
        lr = LinearRegression()

        lr.fit(X_train_poly, y_train)
        y_pred = lr.predict(poly.fit_transform(X_test))
        sse = 0
        for i in range(len(y_pred)):
            sse += abs(y_pred[i] - y_test[i])

        mae = sse/len(y_pred)
        maes += mae

        print("mae: {}".format(mae))
        print("R^2 Score {}".format(lr.score(poly.fit_transform(X), y)))
        r2s += lr.score(poly.fit_transform(X), y)

    mean_mae = 0.2*maes
    mean_r2 = 0.2*r2s
    print("Mean MAE: {} \nMean R2: {}".format(mean_mae, mean_r2))

def neural_network_regression(counted_chron):
    X = counted_chron[['Year']].values
    y = counted_chron['Count'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=19)

    model = Sequential([
        Dense(64, input_dim=1, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dropout(0.2),
        Dense(32, activation='sigmoid'),
        Dropout(0.2),
        Dense(16),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae}")

    pred_years = np.array([[x] for x in range(1960,2025)])
    pred_years_scaled = scaler.transform(pred_years)
    predictions = model.predict(pred_years_scaled)
    plt.scatter(X, y, color='blue', label='Actual Data')  # Original data points
    plt.plot(pred_years, predictions, color='red', label='NN Prediction Line')   # Prediction line
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def neural_network_regression_future_pred(counted_chron):
    X = counted_chron[['Year']].values
    y = counted_chron['Count'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=19)

    model = Sequential([
        Dense(64, input_dim=1, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dropout(0.2),
        Dense(32, activation='sigmoid'),
        Dropout(0.2),
        Dense(16),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae}")

    pred_years = np.array([[x] for x in range(1960,2090)])
    pred_years_scaled = scaler.transform(pred_years)
    predictions = model.predict(pred_years_scaled)

    plt.plot(pred_years, predictions, color='red', label='NN Prediction Line')   # Prediction line
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def neural_network_regression_cv(counted_chron):
    X = counted_chron[['Year']].values
    y = counted_chron['Count'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    maes = 0
    losses = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=19)

    for i, (train_index, test_index) in enumerate(kf.split(X_scaled)):
        print("Fold {}".format(i))
        X_train = [X_scaled[train_index]]
        X_test = [X_scaled[test_index]]
        y_train = [y[train_index]]
        y_test = [y[test_index]]

        model = Sequential([
            Dense(64, input_dim=1, activation='sigmoid'),
            Dense(128, activation='sigmoid'),
            Dropout(0.2),
            Dense(32, activation='sigmoid'),
            Dropout(0.2),
            Dense(16),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test MAE: {mae}")
        maes += mae
        losses += loss

        print("Fold {} MAE: {} Loss: {}".format(i, mae, loss))

    mean_mae = 0.2*maes
    mean_loss = 0.2*losses
    print("Mean MAE: {}".format(mean_mae))
    print("Mean Loss: {}".format(mean_loss))

def red_bull_plot(counted_chron):
    X = counted_chron[['Year']]
    y = counted_chron['Count']

    red_bull_sales = [4.08, 5.23, 5.39, 5.61, 5.96, 6.06, 7.90, 9.80, 11.60, 12.10]
    red_bull_sales_years = [2011, 2012, 2013, 2014, 2015, 2016, 2020, 2021, 2022, 2023]

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, y)
    lr = LinearRegression()
    lr.fit(X_poly, y)

    X_for_plot = [[x] for x in range(2010, 2023)]
    X_for_plot_poly = poly.fit_transform(X_for_plot)
    
    y_pred = lr.predict(X_for_plot_poly)

    common_years = list(set(counted_chron['Year']).intersection(red_bull_sales_years))
    plane_crash_counts = [counted_chron.loc[counted_chron['Year'] == year, 'Count'].values[0] for year in common_years]
    red_bull_sales_aligned = [red_bull_sales[red_bull_sales_years.index(year)] for year in common_years]

    cov = np.cov(plane_crash_counts, red_bull_sales_aligned)[0][1]

    corr = np.corrcoef(plane_crash_counts, red_bull_sales_aligned)[0][1]

    print(f"Covariance: {cov}")
    print(f"Correlation Coefficient: {corr}")


    fig, ax1 = plt.subplots()
    colour='tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Plane Crash Count (Polynomial Regression)', color=colour)
    ax1.plot(X_for_plot, y_pred, color=colour)
    ax1.tick_params(axis='y', labelcolor=colour)

    ax2 = ax1.twinx()

    colour = 'tab:red'
    ax2.set_ylabel('Red Bull Can Sales (Billions)', color=colour)
    ax2.plot(red_bull_sales_years, red_bull_sales, color=colour)
    ax2.tick_params(axis='y', labelcolor=colour)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def main():
    crash_df = pd.read_csv('Plane Crashes.csv')

    crash_df = preprocess(crash_df)

    #ann_death_rate_predictor(crash_df)

    #nlp_pca_tfidf(crash_df)
    #nlp_sgd_deaths(crash_df)

    feature_importance_rf(crash_df)

    #nlp_word_freq(crash_df)

    #print(generate_sentence(n_gram(crash_df.loc[crash_df['Year']>1980], 5), ['the', 'airplane', 'was', 'flying'], 25))

    #counted_chron = crash_counts_year(crash_df.loc[crash_df['Flight type'] == 'Scheduled Revenue Flight'])
    #neural_network_regression_future_pred(counted_chron.loc[counted_chron['Year']>1960])

main()
