import json
import urllib.request
import pandas as pd
import numpy as np
from konlpy.tag import Okt, Mecab
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter


# 0. 함수 정리
## 토큰화 함수
okt = Okt() # KoNLPy의 Okt 객체 생성
mecab = Mecab()
with open('data/stopwords.txt', 'r', encoding='utf-8') as file: # 불용어 사전 호출
    stopwords = [line.strip() for line in file.readlines()]
    stopwords.extend(['좋', '.', '도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게'])
def tokenize(text):    
    return [word for word in mecab.nouns(text) if word not in stopwords]
    # return [word for word in okt.morphs(text, stem=True) if word not in stopwords]



# 1. 데이터 준비 및 전처리
## 1-(1). 머신러닝을 위한 train data 호출 및 전처리
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt") # train_data 호출
train_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])[:] # 컬럼명 추가
train_data['label'] = np.select([train_data.ratings > 3], [1], default=0) # 평점 1, 2 => 0 / 평점 4, 5 => 1 로 라벨링
train_data.drop_duplicates(subset=['reviews'], inplace=True) # reviews 컬럼에서 중복인 내용이 있다면 중복 제거
train_data['tokenized'] = train_data['reviews'].apply(lambda x: tokenize(x)) # reviews 컬럼을 tokenize한 tokenized 컬럼 생성

## 1-(2). 분석할 test data 호출 및 전처리
test_data = pd.read_excel('data/dataset.xlsx', usecols=['상품평'])[:] # test_data 호출
test_data.rename(columns={'상품평': 'reviews'}, inplace=True)
test_data.drop_duplicates(inplace=True)  # reviews 컬럼에서 중복인 내용이 있다면 중복 제거
test_data['tokenized'] = test_data['reviews'].apply(lambda x: tokenize(x)) # reviews 컬럼을 tokenize한 tokenized 컬럼 생성
test_data['reviews'] = test_data['reviews'].replace(r'\n|\r', ' ', regex=True)



# 2. 머신러닝
## 2-(1). Tfid 사용하기 위해서 tokenized_str 컬럼 생성
train_data['tokenized_str'] = train_data['tokenized'].apply(lambda x: ' '.join(x))
test_data['tokenized_str'] = test_data['tokenized'].apply(lambda x: ' '.join(x))

## 2-(2). 머신러닝 학습 실행
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 1)) # vertorizer 생성
X_train = vectorizer.fit_transform(train_data['tokenized_str'])
y_train = train_data['label']
model = LogisticRegression() # 모델 설정
model.fit(X_train, y_train)



# 3. 머신러닝 기반으로 test data에 긍정/부정 컬럼 추가
X_test = vectorizer.transform(test_data['tokenized_str'])
test_data['predicted'] = model.predict(X_test)



# 4. Clustering
## 4-(1). Clustering 모델 설정
dbscan = DBSCAN(eps=0.4, min_samples=5, metric='cosine')
clusters = dbscan.fit_predict(X_test)
test_data['clusters'] = clusters # test data에 cluster 라벨 추가

## 4-(2). Cluster 라벨로 문장 그룹화
clustered_sentences = {}
for idx, cluster_label in enumerate(clusters):
    if(cluster_label == -1 or cluster_label == 0): 
        continue
    elif cluster_label not in clustered_sentences:
        clustered_sentences[cluster_label] = []
    clustered_sentences[cluster_label].append(test_data.iloc[idx]['reviews'])



# 5. 최종 출력을 위한 추가 연산
## 5-(1). 각 Cluster 별로 KBF 도출
cluster_tokens = {}
review_counts = {}
# Cluster 내의 리뷰 개수 세기 & 토큰 모으기
for idx, row in test_data.iterrows():
    cluster_label = row['clusters']
    tokens = row['tokenized']
    if cluster_label not in cluster_tokens:
        cluster_tokens[cluster_label] = []
        review_counts[cluster_label] = 0
    cluster_tokens[cluster_label].extend(tokens)
    review_counts[cluster_label] += 1
# 최고 빈도 토큰 도출하기
cluster_most_frequent_tokens = {}
for cluster, tokens in cluster_tokens.items():
    token_counts = Counter(tokens)
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    cluster_most_frequent_tokens[cluster] = (most_common_token, most_common_count)

## 5-(2). 긍정/부정 비율 계산
cluster_sentiments = test_data.groupby('clusters')['predicted'].value_counts(normalize=True).unstack(fill_value=0)



# 6. 출력
## 6-(1). test_data: 모델 성능 대충 파악하기 위해서 출력
# print(test_data[['reviews', 'predicted']])

## 6-(2). 각 Cluster 별로 속한 문장 출력
for cluster, sentences in sorted(clustered_sentences.items()):
    num_sentences = review_counts[cluster]
    token, token_count = cluster_most_frequent_tokens[cluster]
    positive_rate = cluster_sentiments.at[cluster, 1] if 1 in cluster_sentiments.columns else 0
    negative_rate = cluster_sentiments.at[cluster, 0] if 0 in cluster_sentiments.columns else 0
    print(f"Cluster {cluster} | 문장 개수: {num_sentences} | KBF: {token} ({token_count}개) | 긍정: {positive_rate*100:.2f}%, 부정: {negative_rate*100:.2f}%")
    for sentence in sentences:
        print(f"  - {sentence}")
    print("")

## 6-(3). 최종 출력
for cluster in sorted(cluster_most_frequent_tokens.keys()):
    if cluster in [-1, 0]:  # Cluster -1과 0은 미포함
        continue
    num_sentences = review_counts[cluster]
    token, token_count = cluster_most_frequent_tokens[cluster]
    positive_rate = cluster_sentiments.at[cluster, 1] if 1 in cluster_sentiments.columns else 0
    negative_rate = cluster_sentiments.at[cluster, 0] if 0 in cluster_sentiments.columns else 0
    print(f"Cluster {cluster} | 문장 개수: {num_sentences} | KBF: {token} ({token_count}개) | 긍정: {positive_rate*100:.2f}%, 부정: {negative_rate*100:.2f}%")