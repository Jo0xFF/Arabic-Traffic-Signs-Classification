# Important imports
from app import app
from flask import request, render_template, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from PIL import Image


# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

class_names = {	0: 'front_or_left',
				1: 'front_or_right',
				2: 'hump',
				3: 'left_turn',
				4: 'narro_from_lef',
				5: 'narrows_from_right',
				6: 'no_horron',
				7: 'no_parking',
				8: 'no_u_turn',
				9: 'overtaking_is_forbidden',
				10: 'parking',
				11: 'pedestrian_crossing',
				12: 'right_or_left',
				13: 'right_turn',
				14: 'rotor',
				15: 'slow',
				16: 'speed_100',
				17: 'speed_30',
				18: 'speed_40',
				19: 'speed_50',
				20: 'speed_60',
				21: 'speed_80',
				22: 'stop',
				23: 'u_turn'}

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	# Execute if request is get
	if request.method == "GET":
		full_filename = 'static/uploads/300x300.png'
		predicted_label = "Label of traffic sign..."
		conf_lvl = "the level here.."
		return render_template("index.html", full_filename=full_filename, predicted_label=predicted_label,
			conf_lvl=conf_lvl)

	# Execute if reuqest is post
	if request.method == "POST":
		image_upload = request.files['image_upload']
		imagename = image_upload.filename
		image = Image.open(image_upload)
		print(f"The Debug here: {image} \n")
		# image = tf.io.read_file(imagename)
		# image = tf.image.decode_image(image, channels=3)
		image_org = np.array(image.convert('RGB'))


		image = tf.image.resize(image_org, size=[300, 300])
		image = tf.cast(image, dtype=tf.float32)
		image_final_arr = image/255.
		# print(image_final_arr)
		# print(os.getcwd())


		model_cls = tf.keras.models.load_model("app/static/models/CNN_FINALFIX_68_ACC_86_LOSS_AUG_IMAGEDIR.h5")
		conf_lvl = model_cls.predict(tf.expand_dims(image_final_arr, axis=0))
		predicted_label = class_names[conf_lvl.argmax()]
		confidence_final = f"{round(conf_lvl.max() * 100)}%"
		
		img = Image.fromarray(image_org, 'RGB')
		img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.png'))
		full_filename =  'static/uploads/image.png'
		return render_template("index.html", full_filename=full_filename, predicted_label=predicted_label,
			conf_lvl=confidence_final)
	   
# Main function
if __name__ == '__main__':
	app.run(debug=True)
