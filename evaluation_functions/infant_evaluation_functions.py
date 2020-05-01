
import os

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def infant_test_func(*args):
	"""Simple test function"""
	print("Evaluation worked for filename {}!".format(args[5]))

def create_eeg_labels(dataset, model, exp_cfg, revived_cfg, results, filename):

	# Get activation output from model

	# Filter activation output to get trigger points

	# Save to disk

	pass

def create_ranked_filters(dataset, model, exp_cfg, revived_cfg, results, filename):
	"""
	Create a ranked list of filter figures

	Still need to implement saving
	"""

	# Get filter matrix
	filter_matrix = model.get_layer(name="conv").get_weights()[0]

	# Get one subject's data
	subject = dataset.load_subject(exp_cfg.dataset.train_subject_names[0])

	# Parse features from each week
	for key in subject.keys():
		subject[key] = subject[key][revived_cfg.dataset.feature_names]

	# Aggregate predictions for each week
	predictions = {}
	for key in subject.keys():
		print("Key: {}, {}".format(key, subject[key].shape))
		predictions[key] = model.predict(np.expand_dims(subject[key].to_numpy(), axis=0))
		print("Prediction on {}: {}".format(key, predictions[key]))

	# Align ins, outs, and predictions for each week
	dataset.set_fields(revived_cfg)
	ins = np.stack( [subject[key] for key in sorted(subject.keys())] , axis=0)
	outs = np.stack( [dataset._label_week(dataset._parse_week_time(key)) for key in sorted(subject.keys())] , axis=0)
	outs_x = np.stack( [dataset._parse_week_time(key) for key in sorted(subject.keys())], axis=0 )
	preds = np.stack( [np.squeeze(predictions[key]) for key in sorted(predictions.keys())] , axis=0)

	# Get the nan indices for the weeks which aren't present
	def get_nan_indices(x):
		x = x.astype(float)

		nan_indices = []
		for i in range(x.shape[0]-1):
			print("I: {}".format(i))
			if x[i+1]-x[i] != 1:
				nan_indices.append(i+1)
		
		return nan_indices
	nan_indices = get_nan_indices(outs_x)

	# Sort filters based on loss
	def loss(filter_ind):
		print("Out average: {}".format( round(outs.mean(axis=0), 2) ))
		print("Pred average: {}".format( round(preds[:, filter_ind].mean(axis=0), 2) ))
		return round(((outs-preds[:, filter_ind])**2).mean(axis=0), 4)

	ranked_filter_list = list(range(filter_matrix.shape[2]))
	ranked_filter_list.sort(key=loss)

	# Plot
	print("Outs_x: {}".format(np.insert(outs, nan_indices, np.nan)))
	fig, axs = plt.subplots(filter_matrix.shape[2], 2, figsize=(11, 8))
	for i, filter_ind in enumerate(ranked_filter_list):
		# Plot filters with their channels in the first column
		for channel_ind in range(filter_matrix.shape[1]):
			axs[i, 0].plot(filter_matrix[:, channel_ind, filter_ind], label=revived_cfg.dataset.feature_names[channel_ind])
		axs[i, 0].set_xlabel("Time (s)")
		axs[i, 0].set_ylabel("Coefficient")
		axs[i, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))
		axs[i, 0].text(.2, 1.3, 'Filter #: {}'.format(filter_ind), horizontalalignment='center',
			verticalalignment='center', transform=axs[i, 0].transAxes)

		# Plot labels and their corresponding predictions in the right column
		axs[i, 1].plot(np.insert(outs, nan_indices, np.nan), label="True")
		axs[i, 1].plot(np.insert(preds[:, filter_ind], nan_indices, np.nan), label="Predicted")
		axs[i, 1].set_xticks(list(range(outs.shape[0] + len(nan_indices))))
		xticklabels = list(range(outs.shape[0] + len(nan_indices)))
		xticklabels = [x+1 for x in xticklabels]
		axs[i, 1].set_xticklabels(xticklabels)
		axs[i, 1].set_xlabel("Week")
		axs[i, 1].set_ylabel("Rate")
		axs[i, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))
		axs[i, 1].text(.2, 1.3, 'Loss: {}'.format(loss(filter_ind)), horizontalalignment='center',
			verticalalignment='center', transform=axs[i, 1].transAxes)

	fig.tight_layout()
	#plt.show()

	fig.savefig("{}ranked_filters.png".format(filename), dpi=fig.dpi)

def create_stacked_filter_responses(dataset, model, exp_cfg, revived_cfg, results, filename):

	# Get input data
	data = dataset.load_data()

	# Get filter response matrix
	conv_layer_output = model.get_layer(name="conv").output
	filter_response = K.function(inputs=model.inputs, outputs=conv_layer_output)(data)

	# Create directory for results
	filename = "{}stacked_filter_responses/".format(filename)
	if not os.path.exists(filename):
		os.mkdir(filename)

	# Create one figure for each week
	for week_ind in range(filter_response.shape[0]):

		fig, axs = plt.subplots(1, figsize=(11, 8))
		for filter_ind in range(filter_response.shape[2]):
			axs.plot(filter_response[week_ind, :, filter_ind], label="Response {}".format(filter_ind))
			axs.set_ylim([-.1, 1.1])

		fig.tight_layout()
		#plt.show()

		fig.savefig("{}week_{}.png".format(filename, week_ind), dpi=fig.dpi)

def created_stacked_filter_responses_across_weeks(dataset, model, exp_cfg, revived_cfg, results, filename):
	
	# Get input data
	data = dataset.load_data()

	# Get filter response matrix

	# Plot
	fig, axs = plt.subplots(2, sharex=True, sharey=True)

def create_trajectory_responses(dataset, model, exp_cfg, revived_cfg, results, filename):
	
	# Get input data
	data = dataset.load_data()

	# Get filter response matrix

	# Plot
	fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

