def plot_labels(data, data_labels, target_names):
	import numpy as np
	import matplotlib.pyplot as plt
	colors = ['#caeceb', '#9adbd8', '#51c2bc']
	plt.figure(figsize=(8,4))
	index = np.arange(len(target_names))
	i = 0
	for data_set in data:
		plt.bar(index, {label: np.sum(data_set == i) for i, label in enumerate(target_names)}.values(), color=colors[i], label = data_labels[i])
		i+=1
	plt.xticks(index, target_names, rotation=80)
	plt.title("number of documents per class for each data set")
	plt.legend(loc='lower center', ncol=3)
	plt.show()

def plot_words(data, data_labels):
	import matplotlib.pyplot as plt
	colors = ['#caeceb', '#9adbd8', '#51c2bc']
	fig, axes = plt.subplots(figsize=(16,4), nrows=1, ncols=len(data))
	i = 0
	for data_set in data:
		axes[i].hist([len(x.split()) for x in data_set], bins=50, color=colors[i])
		axes[i].set_title(''.join(['words distribution - ', data_labels[i]]), fontsize=10)
		i+=1
	plt.show()

def word_counts(data, text_col_name):
	word_counts = data[text_col_name].apply(lambda x: len(x.split()))
	return word_counts

def stem_helper(word):
	import nltk
	stemmer = nltk.PorterStemmer()
	try:
		x = stemmer.stem(word)
	except:
		x = word
	return x

def clean_text(data, stop_words):
	'''preprocessing for text data: normalizing text, removing unicode characters and numbers, removing english stopwords and performing stemming'''
	import re
	preprocessed_data = []
	for text in data:
		text = re.sub('[^A-Za-z]+', ' ', text)
		text = ' '.join(stem_helper(word) for word in text.split() if word not in stop_words)
		preprocessed_data.append(text.lower().strip())
	return preprocessed_data

def drop_empty(data, text_col_name):
	data.drop(data[data[text_col_name]==0].index, inplace=True)
	return data

def tsne_word_plot(model, word, size):
	'''visualizing word vectors with t-SNE'''
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	word_labels = [word]
	close_words = model.wv.similar_by_word(word, 10)
	array = np.empty((0,size))
	array = np.append(array, np.array([model.wv[word]]), axis=0)
	for word_score in close_words:
		word_vec = model.wv[word_score[0]]
		word_labels.append(word_score[0])
		array = np.append(array, np.array([word_vec]), axis=0)
	tsne = TSNE(n_components=2, random_state=54)
	data = tsne.fit_transform(array)
	x_coords = data[:,0]
	y_coords = data[:,1]
	plt.plot(x_coords, y_coords, 'x', c='#51c2bc')
	for label, x, y in zip(word_labels, x_coords, y_coords):
		plt.annotate(label, xy=(x,y), xytext=(4,-2), textcoords='offset pixels')
	plt.annotate(word_labels[0], xy=(x_coords[0], y_coords[0]), xytext=(4,-2), textcoords='offset pixels', bbox=dict(boxstyle='round, pad=0.7', fc='#51c2bc', alpha=0.5))
	plt.show()

def show_model_performance(model, x_train, y_train, x_test, y_test):
	'''evaluate metrics'''
	import pandas as pd
	from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
	sets = [[x_train, y_train], [x_test, y_test]]
	results = []
	for s in sets:
		predictions = model.predict(s[0])
		results.append([
			precision_score(s[1], predictions, average='weighted').round(4),
			recall_score(s[1], predictions, average='weighted').round(4),
			f1_score(s[1], predictions, average='weighted').round(4),
			accuracy_score(s[1], predictions).round(4) ])
	df = pd.DataFrame(results, index=['train', 'test'], columns=['precision', 'recall', 'f1', 'accuracy'])
	df = df.style.set_caption(' '.join(['model:', model.__class__.__name__])).set_table_styles([{'selector': 'caption', 'props': [('color', 'black'), ('font-weight', 'bold'), ('font-size', '14px')]}])
	return df

def models_comparision(x_train, y_train, x_test, y_test, models):
	import pandas as pd
	from sklearn.metrics import f1_score, accuracy_score
	sets = [[x_train, y_train], [x_test, y_test]]
	results = []
	index = []
	for model in models:
		for s in sets:
			predictions = model.predict(s[0])
			results.append([accuracy_score(s[1], predictions).round(4), f1_score(s[1], predictions, average='weighted').round(4) ])
		index.append(' '.join(['train set', model.__class__.__name__]))
		index.append(' '.join(['test  set', model.__class__.__name__]))
	df = pd.DataFrame(results, index=index, columns=['accuracy', 'f1'])
	return df.sort_index(ascending=False)

def plot_model_history(history):
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)
	axes[0].plot(history.history['categorical_accuracy'], label='train', c='#9adbd8')
	axes[0].plot(history.history['val_categorical_accuracy'], label = 'valid', c='#51c2bc')
	axes[1].plot(history.history['loss'], label='train', c='#9adbd8')
	axes[1].plot(history.history['val_loss'], label = 'valid', c='#51c2bc')
	axes[0].set_title('model accuracy', fontsize=10)
	axes[1].set_title('model loss', fontsize=10)
	plt.setp(axes[:2], xlabel='epoch')
	plt.legend()
	plt.show()

def model_fit(model, x_train, y_train, x_valid, y_valid, epochs, batch_size):
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
	filepath = ''.join([model.name, '_weights.hdf5'])
	early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='max', restore_best_weights=True)
	check_point = ModelCheckpoint(filepath=filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
	history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])
	return history

def model_eval(model, history, x_test, y_test):
	import numpy as np
	plot_model_history(history)
	loss, acc = model.evaluate(x_test, y_test)
	print('\n-----------------------')
	print(model.name, '\nloss:', loss, '\naccuracy:', acc)
	print('-----------------------')
	y_pred = np.argmax(model.predict(x_test), axis=1)
	return y_pred, acc

def model_report(y_test, y_pred, labels):
	from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
	import matplotlib.pyplot as plt
	import pandas as pd
	disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=labels)
	fig, ax = plt.subplots(figsize=(10, 10))
	disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='binary', values_format='.0f')
	plt.show()
	return pd.DataFrame(classification_report(y_test, y_pred, target_names=labels, output_dict=True)).T

def plot_acc(models_names, acc_values):
	import numpy as np
	import matplotlib.pyplot as plt
	index = np.arange(len(acc_values))
	fig, ax = plt.subplots(figsize = (8,5))
	ax.bar(models_names, acc_values, edgecolor='#51c2bc', color='#9adbd8')
	for index, value in enumerate(acc_values):
		plt.text(x=index , y=value-0.1, s=f'{value}')
	plt.xticks(rotation=60)
	plt.title('accuracy scores of different classifiers')
	plt.show()