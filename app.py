import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from resources.dataset import get_dataset
from resources.preprocessing import preprocessing_input, preprocessing_dataset
from resources.evaluate import evaluate
import json


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello_world():
  return jsonify({"message": "news classification"})

@app.route('/dataset')
@cross_origin()
def end_dataset():
  row = request.args.get('row')

  try:
    if row is None:
      row = 10
    row = int(row)
  except ValueError:
    return jsonify({"message": "wrong input"})

  (df,
   dataset_title,
   dataset_subtitle,
   dataset_link,
   total_dataset_count,
   taken_dataset_count,
   true_label,
   false_label) = get_dataset(num_row=row)
  return jsonify({"dataset": df,
                  "dataset_title": dataset_title,
                  "dataset_subtitle": dataset_subtitle,
                  "dataset_link": dataset_link,
                  "total_dataset": total_dataset_count,
                  "taken_dataset": taken_dataset_count,
                  "true_label": true_label,
                  "false_label": false_label
                  })

@app.route('/model')
@cross_origin()
def end_model():
  model = tf.keras.models.load_model('resources/model.h5')
  model_config = model.get_config()
  model_compile = model.get_compile_config()
  json_model_config = json.loads(json.dumps(model_config))
  json_model_compile = json.loads(json.dumps(model_compile))

  X_train, X_test, y_train, y_test = preprocessing_dataset()
  con_matrix, accuracy, precision, recall = evaluate(X_test, y_test, model)

  lib_info = {
    "name": "tensorflow",
    "source": "https://www.tensorflow.org/",
    "description": "TensorFlow makes it easy to create ML models that can run in any environment. Learn how to use the intuitive APIs through interactive code samples",
    "logo": "https://www.gstatic.com/devrel-devsite/prod/v870e399c64f7c43c99a3043db4b3a74327bb93d0914e84a0c3dba90bbfd67625/tensorflow/images/lockup.svg"
  }

  return jsonify({ "model_config": json_model_config,
                   "model_compile": json_model_compile,
                   "evaluate": {
                     "con_matrix": con_matrix,
                     "accuracy": accuracy,
                     "precision": precision,
                     "recall": recall
                   },
                   "lib_info": lib_info
                 })

@app.route('/predict', methods=['POST'])
@cross_origin()
def end_predict():
  data = request.get_json()
  new_text = data['text']
  model = tf.keras.models.load_model('resources/model.h5')
  processed_text = preprocessing_input(new_text)
  predicted = model.predict(processed_text)
  predicted = float(predicted[0][0])
  return jsonify({"prediction": predicted})

if __name__ == '__main__':
  app.run()