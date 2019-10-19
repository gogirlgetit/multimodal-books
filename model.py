#
# Downloads the training data from a public URL. Defines few features.
# Uses these features and trains models to predict the relevance of a
# video to a given topic from the textbook.
#
import gdown
import os # Good for navigating your computer's files 
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error, mean_squared_log_error, r2_score

# Download the training data
gdown.download('https://drive.google.com/uc?id=11FiTbIEmREYYhqGGQNlDiT_rns1gE2n_', 'training_data.csv', True)
data_path  = 'training_data.csv'
video_data = pd.read_csv(data_path, sep = '\t')

# Define features
video_data['Views/Age'] = video_data[['Views']].values/video_data[['Age']].values
video_data['Likes/Views'] = video_data[['Likes']].values/video_data[['Views']].values
video_data['ED/Length'] = video_data[['EditDistance']].values/video_data[['Length']].values
video_data['Lang^2'] = video_data[['LangMatch']].values*video_data[['LangMatch']].values
video_data['SR/Length'] = video_data[['SearchRank']].values/video_data[['Length']].values
video_data['Views/Age'] = video_data[['Views']].values/video_data[['Age']].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=101)
#X_train = X_train.reshape(133, 7)
#X_test = X_test.reshape(15, 7)
#y_test = y_test.reshape(15,1)
#y_train = y_train.reshape(133, 1)

x = video_data[['Dislikes', 'EditDistance', 'Length', 'Likes', 'SearchRank', 'Views', 'Age', 'Lang^2', 'SR/Length']].values
#x = video_data[['Dislikes', 'SearchRank', 'Lang^2', 'EditDistance', 'SR/Length', 'Views/Age']].values
y = video_data[['Relevance']].values

# setting up the model
multiple = linear_model.LinearRegression(fit_intercept = True, normalize = True)

# training the model 
multiple.fit(X_train, y_train)
y_pred = multiple.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()

# Print model statistics
print('Variance score: %.2f' % multiple.score(x, y))
print('explained_variance_score: %.2f' %explained_variance_score(y_test, y_pred))
print('max_error: %.2f' %max_error(y_test, y_pred))
print('Mean Absolute Error: %.2f' %mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %.2f' %mean_squared_error(y_test, y_pred))
print('Median absolute Error: %.2f' %median_absolute_error(y_test, y_pred))
print('r2 score: %.2f' %r2_score(y_test, y_pred))
print(multiple.coef_)
print(multiple.intercept_)
